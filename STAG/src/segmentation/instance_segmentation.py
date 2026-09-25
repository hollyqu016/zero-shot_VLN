import numpy as np
from torch.backends.mkl import verbose
from ultralytics import YOLOE
import cv2
from segmentation.object_list import *
import os
import matplotlib.pyplot as plt


MODEL_PATH = os.environ.get("STZS_PATH_YOLOE_CKPT", "../ckpt/yoloe-26l-seg.pt")
category_names = {cat['id']: cat['name'] for cat in categories}

# 模型改为**懒加载**。
#
# 原先是 import 时就 YOLOE(MODEL_PATH)，这带来两个问题：
#   1. 权重路径必须在任何 import 之前就确定，而 config 是在 __main__ 里才读的，
#      顺序对不上——配置项永远来不及生效。
#   2. 任何只想 import 一下这个模块的脚本（比如离线自测）都要付一次加载代价。
# 改成首次调用时加载后，run_experiments.py 可以在读完 config 之后再调
# set_model_path()，也不影响正常推理路径。
_model = None


def set_model_path(path):
    """在首次推理之前调用才有效；已加载则会在下次 _get_model() 时重新加载。"""
    global MODEL_PATH, _model
    if path and path != MODEL_PATH:
        MODEL_PATH, _model = path, None


def _get_model():
    """
    首次调用时加载并绑定词表。

    **只有全部成功才写进 _model**。这一点是踩过坑才这么写的：先赋值再
    set_classes 的话，一旦 set_classes 抛异常（比如 ultralytics 要下载的
    mobileclip 文本编码器损坏），_model 已经是个**没绑定词表**的 YOLOE
    对象，后续调用直接把它返回——整批实验会静默跑在空检测器上，日志里
    只表现为每一帧 `[detect] front=0`，不报任何错。实测 42 次检测全是 0。

    现在的写法保证：要么拿到配好词表的模型，要么每次都抛出同一个异常。
    异常会被 run_experiments 的 episode 级 except 打成 traceback，很显眼。
    """
    global _model
    if _model is None:
        names = list(category_names.values())
        try:
            m = YOLOE(MODEL_PATH)
            m.set_classes(names, m.get_text_pe(names))
        except Exception as e:
            raise RuntimeError(
                f"YOLOE 初始化失败: {type(e).__name__}: {e}\n"
                f"  权重: {MODEL_PATH}\n"
                f"  绑定词表需要 ultralytics 的 mobileclip 文本编码器"
                f"（首次使用会自动下载）。\n"
                f"  若报 'PytorchStreamReader failed reading zip archive'，"
                f"说明该文件下载不完整，删掉让它重新下即可：\n"
                f"    find ~ -name 'mobileclip*' 2>/dev/null   # 找到后删除\n"
                f"  注意不要吞掉这个异常——检测器不工作时地图里不会有任何物体，"
                f"所有依赖接地的判据都会空转。"
            ) from e
        _model = m       # 全部成功之后才发布出去
    return _model


def instance_segmentation(image: np.ndarray):
    """
    Input: image (np.ndarray) - raw image
    Output: List[dict] - per-instance class_id, class_name, mask (same size as input)
    """
    model = _get_model()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    # 边缘过滤从 0.2 放宽到 0.05。
    #
    # 0.2 意味着只保留画面中心 60%×60% 区域内的检测，实测这个过滤器是物体
    # 覆盖稀疏的主要原因：一次真实 episode 里 agent 从头到尾站在卧室中央，
    # 床始终没进地图；另一次指令点名的 "massage table" 全程 not_yet_detected。
    # 物体接不了地，上层所有几何判据(landmark距离、desync、房间推断)就全是空转。
    # 这个过滤器的本意是避免半截物体的点云反投影不准，0.05 已经够挡住紧贴
    # 画面边界的残缺框了。
    edge_threshold_x = w * 0.05
    edge_threshold_y = h * 0.05

    # conf 0.6 -> 0.45：同样是为了提升召回。误检的代价是地图上多几个错标签，
    # 漏检的代价是整条几何链路失效，两者不对称。
    results = model.predict(image, conf=0.45, iou=0.3, retina_masks=True, verbose=False)
    result = results[0]
    output = []
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        class_names = result.names
        for i in range(masks.shape[0]):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            if (center_x < edge_threshold_x or center_y < edge_threshold_y or
                center_x > w - edge_threshold_x or center_y > h - edge_threshold_y):
                continue

            class_id = int(boxes.cls[i].item())
            class_name = class_names.get(class_id, str(class_id))
            mask = masks[i].astype(np.uint8) * 255
            mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            output.append({
                "class_id": class_id,
                "class_name": class_name,
                "mask": mask_resized
            })
    return output

def get_class_color(class_id: int) -> tuple:
    # use matplotlib tab20 colormap for fixed, distinct per-class colors
    color_map = plt.get_cmap('tab20')
    color = color_map(class_id % 20)[:3]  # take first 3 channels (RGB)
    return tuple(int(255 * c) for c in color)

def visualize_instance_segmentation(image: np.ndarray, instances: list, alpha: float = 0.5) -> np.ndarray:
    vis_image = image.copy()
    if vis_image.shape[2] == 4:
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGBA2BGR)
    for inst in instances:
        class_id = inst["class_id"]
        color = get_class_color(class_id)
        mask = inst["mask"]
        colored_mask = np.zeros_like(vis_image)
        for c in range(3):
            colored_mask[:, :, c] = mask // 255 * color[c]
        vis_image = cv2.addWeighted(vis_image, 1, colored_mask, alpha, 0)
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            center_x, center_y = int(xs.mean()), int(ys.mean())
            center_x = np.clip(center_x, 0, vis_image.shape[1] - 1)
            center_y = np.clip(center_y, 0, vis_image.shape[0] - 1)
            cv2.putText(vis_image, inst["class_name"], (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv2.LINE_AA)
            cv2.putText(vis_image, inst["class_name"], (center_x, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    return vis_image