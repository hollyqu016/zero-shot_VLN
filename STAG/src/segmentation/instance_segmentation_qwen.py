import numpy as np
from torch.backends.mkl import verbose
import cv2
import os
import sys

# 路径兜底：这个文件里用的是绝对包路径 "from segmentation.object_list import ..."，
# 这要求 src/ 目录本身在 sys.path 里。正常通过项目其他模块 import 这个文件时没问题
# (调用方早已把 src/ 加进了 sys.path)，但如果直接跑 `python segmentation/instance_segmentation.py`
# (而不是 `python -m segmentation.instance_segmentation`)，Python只会把脚本所在目录
# (src/segmentation/) 加进 sys.path，找不到 segmentation 这个包本身，导致 ModuleNotFoundError。
# 这里手动把 src/ 加进去，兜住这种直接跑脚本做功能测试的场景。
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from segmentation.object_list import *
import matplotlib.pyplot as plt
import logging


# 权重路径：环境变量 > 代码默认。
# 这个模块在 import 时就要加载模型，拿不到 config 对象，所以走环境变量；
# run_experiments.py 会在启动时把 config 的 paths.yoloe_ckpt 写进这个变量。
MODEL_PATH = os.environ.get("STZS_PATH_YOLOE_CKPT", "../ckpt/yoloe-26l-seg.pt")

# ---- YOLOE模型改成懒加载 ----
# 之前是在import这个文件的时候就无条件加载YOLOE模型(装ultralytics + 读取checkpoint)，
# 但项目里实际在用的检测入口是下面的instance_segmentation()(Qwen路径)，
# instance_segmentation_oral()目前没有任何调用点，是保留的备用实现。之前的写法导致
# "哪怕只是想测试/使用Qwen路径"，也被迫要求ultralytics可用、checkpoint存在——现在改成
# 只有真的调用instance_segmentation_oral()时才会触发加载，不影响其行为，只是加载时机推迟。
_yoloe_model = None


def _get_yoloe_model():
    # 与 instance_segmentation.py 同一条规矩：**全部成功才发布**。
    # 先赋值再 set_classes 的话，绑定词表失败时会留下一个没绑词表的模型，
    # 后续调用拿到它照样能 predict，只是什么都检测不出来——静默降级。
    global _yoloe_model
    if _yoloe_model is None:
        from ultralytics import YOLOE
        names = [cat['name'] for cat in categories]
        m = YOLOE(MODEL_PATH)
        m.set_classes(names, m.get_text_pe(names))
        _yoloe_model = m
    return _yoloe_model


def instance_segmentation_oral(image: np.ndarray, landmark: str = None):
    """
    Input: image (np.ndarray) - raw image
    Output: List[dict] - per-instance class_id, class_name, mask, relevance (same size as input)

    [2026-08-09 速度回归排查] 之前mapper.py里实际调用的instance_segmentation()走的是Qwen
    多模态检测路径(本文件下方，通过chat.completions API)，每次调用一次本地Qwen3.6:35b的VLM
    推理(10~90秒/次)。但上游 HSGM(HSGM_public-oral/，本项目基线)用的是这里的YOLOE本地检测——纯本地
    PyTorch推理，不走网络，一次检测毫秒级。这次会话查真实日志确认：现在的pipeline每个物理
    步实际上要付两次VLM调用的代价(一次decide_waypoint决策、一次这里的检测)，而原版只在
    决策这一处用VLM，检测完全不碰VLM——这才是"8 episode/小时掉到2个"这个量级差距的最大
    单一原因，比DestinationNavigator那些开销都大。

    另外核实过：Qwen路径的prompt本身也被约束成"class_name必须从固定词表(222类)里选"(见
    CHANGES第一节)，跟这里YOLOE用set_classes()绑定的词表是同一份(segmentation/object_list.py
    的categories)——也就是说切换检测方式不会损失"开放词表匹配"能力，两边本来就用的是同一个
    封闭词表，Qwen在这一点上并没有额外优势，只是白白多付了VLM调用的延迟和后端不稳定风险。

    landmark参数：Qwen路径用它做"跟当前查找目标的相关性打分"(relevance)，YOLOE路径不需要
    这个语义步骤——这里保留这个形参只是为了跟mapper.py现有调用点(`instance_segmentation(
    image, landmark=landmark)`)保持签名兼容，接了但不使用，不影响调用方代码。
    """
    model = _get_yoloe_model()
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    h, w = image.shape[:2]
    edge_threshold_x = w * 0.2
    edge_threshold_y = h * 0.2

    results = model.predict(image, conf=0.6, iou=0.3, retina_masks=True, verbose=False)
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
            # relevance：mapper.py现有代码期望每个instance带一个relevance字段当置信度用
            # (原本是Qwen路径给的"跟目标地标的相关性"分数)。YOLOE没有这个语义概念，这里用
            # 检测本身的置信度(boxes.conf)填充——比之前注释掉的固定1.0更有信息量，且量纲
            # 一致(0~1，越高越可信)，不需要改mapper.py消费这个字段的下游逻辑。
            confidence = float(boxes.conf[i].item())
            output.append({
                "class_id": class_id,
                "class_name": class_name,
                "mask": mask_resized,
                "relevance": confidence,
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


import os
import re
import base64
import numpy as np
import cv2
from openai import OpenAI

def instance_segmentation(image: np.ndarray, landmark: str) -> list:
    """
    使用 Qwen 模型（通过本地 Ollama API）分析 numpy 数组格式的图片，
    并在同一次调用中输出每个检测物体与给定 landmark 的语义关联度。

    Args:
        image (np.ndarray): 输入的原始图像，形状为 (H, W, C)。
        landmark (str): 参考物体/地标名称，用于计算关联度。

    Returns:
        List[dict]: 包含以下字段的字典列表：
            - class_id (int): 类别索引
            - class_name (str): 类别名称
            - mask (np.ndarray): 与输入图像同尺寸的布尔型矩阵 (shape: H x W)
            - relevance (float): 该物体与 landmark 的关联度，范围 0~1，
                                  完全相同的物体记为 1.0
    """
    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1143/v1")
    api_key = os.getenv("OPENAI_API_KEY", "ollama")
    model_name = os.getenv("OPENAI_MODEL", "qwen3.6:35b")

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Invalid input: image must be a non-empty numpy.ndarray")

    height, width = image.shape[:2]

    success, encoded_image = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Failed to encode numpy array to image bytes.")
    base64_image = base64.b64encode(encoded_image.tobytes()).decode('utf-8')

    # 4. 构造提示词（检测 + 关联度打分 一次完成）
    #
    # 格式说明修复(见CHANGES_STID开发记录)：真实环境验证时发现模型偶尔会把格式理解错，
    # 把"<object>class_name</object>"里的"class_name"当成字面占位符文本直接原样输出，
    # 反而把真实类别名错放到了标签名的位置，比如输出成"<sofa>class_name</sofa>"——这样
    # regex完全匹配不上，虽然模型确实"看到"了东西，最终却被判定成"检测到0个物体"。
    # 现在把"<object>"/"</object>"明确说成是不能改动的固定字面标签，并给一个具体真实
    # 类别的完整示例(而不是只给抽象占位符模板)，减少这种"该替换的东西没替换、不该替换
    # 的东西被替换"的歧义。
    landmark_clean = (landmark or "").strip()
    prompt = (
        "/no_think\n"
        "Please detect all objects in this image.\n\n"
        "Output format rules (follow EXACTLY):\n"
        "- The tags <object>, </object>, <box>, </box>, <relevance>, </relevance> are FIXED "
        "literal tag names. Do NOT change them, and do NOT use the detected category as the "
        "tag name.\n"
        "- Only the TEXT BETWEEN each pair of tags should be replaced with the actual detected "
        "value (the real category name / real coordinates / real score). Never leave the "
        "literal words \"class_name\" or \"score\" in your output — always replace them.\n"
        "- One line per detected object, in this exact template:\n"
        "  <object>REPLACE_WITH_REAL_CATEGORY</object> <box>[xmin, ymin, xmax, ymax]</box> "
        "<relevance>REPLACE_WITH_REAL_SCORE</relevance>\n\n"
        "Concrete example — if you detect a sofa at pixel box [23, 366, 394, 552] with "
        "relevance 0.20 to the reference landmark, the correct output line is exactly:\n"
        "<object>sofa</object> <box>[23, 366, 394, 552]</box> <relevance>0.20</relevance>\n"
        "(NOT <sofa>class_name</sofa> <box>...</box> <relevance>score</relevance> — that is WRONG.)\n\n"
        "The coordinates should be integers from 0 to 1000, where (0, 0) is the top-left "
        "corner and (1000, 1000) is the bottom-right corner.\n\n"
        "IMPORTANT - category vocabulary constraint: the text inside <object></object> MUST be "
        "chosen from EXACTLY this fixed category list (pick the closest matching category, do "
        "not invent new names):\n"
        f"[{category_names_str}]\n"
        "If a detected object truly does not fit any category above, output `other`.\n\n"
        f"The relevance score measures how semantically related each detected object is to "
        f"the reference landmark: \"{landmark_clean}\".\n"
        "- If the object IS the same thing as the landmark (synonym, alias, same entity), "
        "the score must be 1.0.\n"
        "- If completely unrelated, the score should be close to 0.\n"
        "- If partially related (same category, commonly co-occurring, functionally related), "
        "give an intermediate value.\n"
        "The score must be a float between 0 and 1, e.g. 0.85."
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        temperature=0.1,
        max_tokens=8192
    )

    output_text = response.choices[0].message.content

    # [DEBUG] 排查"检测结果为空"问题：output_text为空时，把finish_reason和
    # reasoning_content(如果模型是思考模型，比如qwen3.6可能会把输出分成
    # reasoning_content和content两部分)一起打出来，帮助判断是不是被max_tokens截断了
    # ——加了词表约束后prompt变长，如果模型在"思考"阶段就把token预算耗光，
    # content就会是空字符串，看起来像"检测不到东西"，其实是被截断了。
    if not output_text or not output_text.strip():
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        logging.warning(
            f"[instance_segmentation][DEBUG] output_text为空! finish_reason={finish_reason!r}, "
            f"reasoning_content长度={len(reasoning_content) if reasoning_content else 0}, "
            f"reasoning_content预览={str(reasoning_content)[:300]!r}"
        )

    # 6. 解析输出并生成同尺寸 Mask + relevance
    pattern = (
        r"<object>(.*?)</object>\s*"
        r"<box>\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]</box>\s*"
        r"<relevance>([\d.]+)</relevance>"
    )
    matches = re.findall(pattern, output_text)
    n_dropped_invalid_box = 0  # 记录有多少条regex匹配到了、但坐标换算后被判定为无效框而丢弃

    results = []
    for class_id, match in enumerate(matches):
        # 归一化(小写/下划线转空格/简单单复数)，这样和STID那边同样归一化过的landmark
        # class_name做字符串比较时，不会被大小写/格式差异误判成"没匹配上"。
        class_name = normalize_category_name(match[0].strip()) or match[0].strip()
        xmin, ymin, xmax, ymax = map(int, match[1:5])
        raw_relevance = match[5]

        try:
            relevance = float(raw_relevance)
        except ValueError:
            relevance = 0.0
        relevance = max(0.0, min(1.0, relevance))

        x1 = int((xmin / 1000.0) * width)
        y1 = int((ymin / 1000.0) * height)
        x2 = int((xmax / 1000.0) * width)
        y2 = int((ymax / 1000.0) * height)

        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))

        if x1 >= x2 or y1 >= y2:
            n_dropped_invalid_box += 1
            continue

        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True

        results.append({
            "class_id": class_id,
            "class_name": class_name,
            "mask": mask,
            "relevance": relevance
        })

    # 诊断日志：最终results为空时，把"到底卡在哪一步"打出来——
    # 之前只在regex完全没匹配到东西时才打印，但还有另一种更隐蔽的情况：regex匹配到了
    # N条结果，却在坐标换算后全部被判定为无效框(x1>=x2或y1>=y2)而丢弃，这种情况下
    # matches非空、诊断不会触发，最终却仍然是"检测到0个物体"，看起来和"模型真的什么都
    # 没看到"完全一样，容易误判。现在统一在results为空时打印：regex匹配到几条、
    # 又因为无效框丢了几条、以及模型的原始返回文本，三个信息合在一起基本能定位问题出在
    # prompt理解、坐标格式约定，还是正则本身。
    if not results:
        preview = output_text[:800] + ("...(截断)" if len(output_text) > 800 else "")
        print(
            f"[instance_segmentation] 最终检测结果为空。regex匹配到 {len(matches)} 条，"
            f"其中 {n_dropped_invalid_box} 条因坐标无效被丢弃。模型原始返回内容如下：\n{preview}\n"
        )
        # 之前只在output_text完全为空时才打印reasoning_content，但"格式错乱导致regex
        # 0匹配"这种失败(content非空)完全没有相关诊断——没法判断是不是跟/no_think
        # (跳过思考阶段)有关。这里补上：只要最终没解析出任何结果，就把reasoning_content
        # 一并打出来，方便下次真实环境验证时对照"有没有思考内容"和"格式错乱"是否相关。
        if matches:  # matches非空但全被坐标校验丢弃的情况，前面已经打印过原始内容，这里只补reasoning
            choice = response.choices[0]
            reasoning_content = getattr(choice.message, "reasoning_content", None)
            print(
                f"[instance_segmentation][DEBUG] (regex匹配到{len(matches)}条但全部丢弃) "
                f"reasoning_content长度={len(reasoning_content) if reasoning_content else 0}, "
                f"reasoning_content预览={str(reasoning_content)[:300]!r}"
            )
        elif output_text and output_text.strip():
            # regex完全0匹配，但content不为空(格式错乱这一类，比如<sofa>class_name</sofa>)
            choice = response.choices[0]
            reasoning_content = getattr(choice.message, "reasoning_content", None)
            print(
                f"[instance_segmentation][DEBUG] (content非空但regex 0匹配，疑似格式错乱) "
                f"reasoning_content长度={len(reasoning_content) if reasoning_content else 0}, "
                f"reasoning_content预览={str(reasoning_content)[:300]!r}"
            )

    return results


# ============================================================
# 功能性测试入口：验证 instance_segmentation()（检测器侧）是否真的遵守了
# object_list.py 里那份封闭词表约束——这是和 STID(spatio_temporal_decomposer.py) 那边
# landmark 对齐机制配套的另一半，之前只验证过文本侧(STID)，这里补上检测器侧的验证。
#
# 用法：
#   python instance_segmentation.py                          # 用合成占位图跑通流程(结果无意义)
#   python instance_segmentation.py /path/to/real_image.jpg          # 用真实图片测，默认landmark="chair"
#   python instance_segmentation.py /path/to/real_image.jpg door     # 指定landmark
#
# 需要能连上 OPENAI_BASE_URL 指向的本地LLM服务（和 STID 用的是同一套环境变量配置）。
# ============================================================
if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("instance_segmentation() 检测器侧 —— 词表对齐功能性测试")
    print("=" * 70)

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_image = cv2.imread(image_path)
        if test_image is None:
            print(f"[错误] 无法读取图片: {image_path}")
            sys.exit(1)
        print(f"图片来源: 真实图片 {image_path!r}")
    else:
        print("[提示] 没有提供图片路径，用一张合成占位图(纯色底+文字)跑通流程——")
        print("       这种情况下检测结果没有实际意义，只能验证代码不报错。")
        print("       要验证真实检测效果，请用: python instance_segmentation.py /path/to/real_image.jpg")
        test_image = np.full((480, 640, 3), 200, dtype=np.uint8)
        cv2.putText(test_image, "placeholder image", (60, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    test_landmark = sys.argv[2] if len(sys.argv) > 2 else "chair"
    print(f"参考landmark: {test_landmark!r}\n")

    try:
        detected_instances = instance_segmentation(test_image, test_landmark)
    except Exception as e:
        print(f"[错误] instance_segmentation() 调用失败(检查本地LLM服务是否可达): {e}")
        sys.exit(1)

    print(f"检测到 {len(detected_instances)} 个物体:\n")

    # 合法取值 = 222个类别名(已归一化) + 'other'。检测结果里的class_name如果不在这个
    # 集合里，说明LLM没有遵守prompt里"必须从词表中选"的约束，是需要关注的信号。
    valid_vocab = set(normalize_category_name(n) for n in CATEGORY_NAMES) | {"other"}
    n_in_vocab = 0
    n_violation = 0
    for inst in detected_instances:
        cname = inst["class_name"]
        ok = cname in valid_vocab
        n_in_vocab += int(ok)
        n_violation += int(not ok)
        flag = "OK" if ok else "!! 违反词表约束 !!"
        print(f"  class_name={cname!r:22s} relevance={inst['relevance']:.2f}  [{flag}]")

    print(f"\n统计: {n_in_vocab}/{len(detected_instances)} 落在词表范围内(含'other')，"
          f"{n_violation} 个不在词表内。")
    if len(detected_instances) == 0:
        print("[提示] 没检测到任何物体——如果用的是合成占位图，这是正常的；"
              "如果用的是真实图片，可能需要检查prompt/模型/图片内容。")
    elif n_violation > 0:
        print("[警告] 有detection的class_name没有遵守222类词表约束，"
              "可能需要进一步加强prompt措辞，或者加一层后处理做兜底纠正。")
    else:
        print("[OK] 全部detection都落在词表约束范围内。")

    # 顺带保存一张可视化结果，方便直接打开图看检测框/mask是否合理
    try:
        vis_instances = [
            {**inst, "mask": (inst["mask"].astype(np.uint8) * 255)}
            for inst in detected_instances
        ]
        vis_image = visualize_instance_segmentation(test_image, vis_instances)
        out_path = "instance_segmentation_test_vis.jpg"
        cv2.imwrite(out_path, vis_image)
        print(f"\n可视化结果已保存到: {out_path}")
    except Exception as e:
        print(f"[提示] 可视化保存失败(不影响上面的主测试结果): {e}")
