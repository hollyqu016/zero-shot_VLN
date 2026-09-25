#!/usr/bin/env python3
"""
Landmark grounding 的离线自测。

不需要 habitat_sim / open3d / ultralytics —— 用桩对象替掉点云，直接测三件事：
  1. resolve_category() 在真实 VLN 地标词上的对齐率
  2. Instruct_Mapper.ground_landmarks() 的匹配 + 左右消歧逻辑
  3. create_top_down_map_centered() 的 landmark 渲染分支不崩，并产出一张示例图

测 2 和 3 的时候不是复制一份实现来测，而是把 mapper.py / agent.py 里那两段
函数源码原样抠出来 exec —— 保证测的就是真正会跑的代码，不会出现"测试通过但
线上是另一份实现"的情况。

用法：
    python scripts/test_landmark_grounding.py
产物：
    scripts/_grounding_preview.png
"""

import os
import re
import sys
import textwrap

import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
SRC = os.path.abspath(SRC)
sys.path.insert(0, SRC)

from segmentation.object_list import (  # noqa: E402
    AREA_PROXY_OBJECTS,
    CATEGORY_NAMES,
    GENERIC_AREA_TYPES,
    normalize_category_name,
    resolve_area,
    resolve_category,
)

PASS, FAIL = "PASS", "FAIL"
_failures = []


def check(cond, label, detail=""):
    tag = PASS if cond else FAIL
    if not cond:
        _failures.append(label)
    print(f"  [{tag}] {label}{('  -- ' + detail) if detail and not cond else ''}")


# ======================================================================
# 从源文件里抠出指定的顶层函数/方法源码
# ======================================================================
def extract_block(path, header_regex):
    """按 'def xxx' 的缩进层级切出完整函数体，返回 dedent 后的源码。"""
    src = open(path, encoding="utf-8").read().splitlines()
    start = None
    for i, line in enumerate(src):
        if re.match(header_regex, line):
            start = i
            break
    if start is None:
        raise RuntimeError(f"未在 {path} 找到匹配 {header_regex} 的定义")
    indent = len(src[start]) - len(src[start].lstrip())

    # 先跳过可能跨多行的函数签名：数括号，深度归零且该行以 ':' 结尾才算签名结束。
    # (不这么做的话，多行签名里缩进为 0 的那行 '):' 会被误判成函数体结束)
    depth, sig_end = 0, start
    for j in range(start, len(src)):
        depth += src[j].count("(") - src[j].count(")")
        if depth <= 0 and src[j].rstrip().endswith(":"):
            sig_end = j
            break

    end = len(src)
    for j in range(sig_end + 1, len(src)):
        line = src[j]
        if not line.strip():
            continue
        if len(line) - len(line.lstrip()) <= indent:
            end = j
            break
    return textwrap.dedent("\n".join(src[start:end]))


AGENT_DIR = os.path.join(SRC, "agent")


class HyperDefaults:
    """
    超参的同值默认，给测试桩用。

    这些值原本写死在方法的默认参数上，现在从 config['hyper'] 读（见
    config/vlnce_test.yaml）。真实 agent 在 __init__ 里赋值，测试桩没有
    __init__，所以继承这个基类拿到同一套值。

    **这里的数值必须和 yaml 保持一致**——不一致的话测试测的就不是线上行为了。
    """
    # landmark（粘性状态与远离检测）
    NEAR_M = 1.5
    PASSED_MIN_M = 2.0
    PASSED_GAP_M = 1.0
    LM_RECEDE_MARGIN = 1.2
    LM_RECEDE_STEPS = 6
    # frontier
    FRONTIER_MIN_USEFUL_M = 2.0
    FRONTIER_STEP_LEN = 0.6
    FRONTIER_TOP_K = 3
    # subtask（完成校验与拦截）
    PREMATURE_MIN_RATIO = 0.3
    PREMATURE_SHAKY_RATIO = 0.6
    VERIFY_NEAR_M = 2.0
    TOO_FAR_OBJECT_LIMIT = 2.0
    RECEDE_MARGIN = 1.5
    FORCE_ADVANCE_RATIO = 2.0
    FORCE_ADVANCE_MIN_STEPS = 20
    ROLLBACK_STEP_THRESHOLD = 70


def agent_file(header_regex):
    """
    在 agent 包里找出哪个文件定义了这个方法。

    agent.py 已按 VLN 四阶段拆成子包（perception / planning / action / llm），
    方法会随重构在文件之间移动。这个脚本靠正则抽函数体，硬编码文件名就等于
    每次搬移都要跟着改一遍——按内容找就不用管它在哪。
    """
    hits = []
    for root, _dirs, files in os.walk(AGENT_DIR):
        if "__pycache__" in root:
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(root, fn)
            with open(path, encoding="utf-8") as fh:
                if any(re.match(header_regex, line) for line in fh.read().splitlines()):
                    hits.append(path)
    if not hits:
        raise RuntimeError(f"agent 包里找不到匹配 {header_regex} 的定义")
    if len(hits) > 1:
        # 基类抽象方法 + 子类实现会同时命中，取行数多的那个（真正的实现）
        hits.sort(key=lambda q: -len(open(q, encoding="utf-8").read()))
    return hits[0]


def agent_block(header_regex):
    """从 agent 包里抽出这个方法的源码，不必关心它现在在哪个文件。"""
    return extract_block(agent_file(header_regex), header_regex)


_AGENT_SRC_CACHE = {}


def agent_source():
    """
    agent 包全部源码拼在一起。

    有几处断言是跨方法查字符串的（比如"这个提示词有没有被写进 prompt"），
    拆包之后这些字符串散在不同文件里，按单文件读就查不到了。拼起来查的
    代价是定位不到具体文件，但这些断言本来关心的就是"整个 agent 有没有
    这行逻辑"，而不是它在哪。
    """
    if not _AGENT_SRC_CACHE:
        parts = []
        for root, _dirs, files in os.walk(AGENT_DIR):
            if "__pycache__" in root:
                continue
            for fn in sorted(files):
                if fn.endswith(".py"):
                    parts.append(open(os.path.join(root, fn), encoding="utf-8").read())
        _AGENT_SRC_CACHE["src"] = "\n".join(parts)
    return _AGENT_SRC_CACHE["src"]


# ======================================================================
# 点云桩：只实现渲染/接地代码真正会调到的那几个接口
# ======================================================================
class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a, dtype=float)
        self.shape = self._a.shape

    def cpu(self):
        return self

    def numpy(self):
        return self._a


class _Pt:
    def __init__(self, pos, col=None):
        self.positions = _Arr(pos)
        self.colors = _Arr(col if col is not None else np.zeros_like(np.asarray(pos, float)))


class StubPCD:
    def __init__(self, pos=None, col=None):
        pos = np.zeros((0, 3)) if pos is None else np.asarray(pos, dtype=float)
        self.point = _Pt(pos, col)

    def is_empty(self):
        return self.point.positions.shape[0] == 0


# habitat_rotation(identity) 的真实输出：transform_matrix @ I。
#
# 注意它的行列式是 -1（transform_matrix 是个轴交换，只左乘、没做相似变换），
# 所以 world 系是左手系。测试桩必须用这个矩阵而不是随便一个 det=+1 的正交阵，
# 否则左右消歧的测试会得到相反的结论 —— 第一版就是这么被骗过去的。
HABITAT_ROT_IDENTITY = np.array([[1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0],
                                 [0.0, 1.0, 0.0]])


class StubMapper:
    def __init__(self):
        self.current_position = np.array([0.0, 0.0, 1.2])
        # forward = -R[:,2] = (0,-1,0) 世界系 -y；right = R[:,0] = (1,0,0) 世界系 +x
        self.current_rotation = HABITAT_ROT_IDENTITY.copy()
        self.navigable_pcd = StubPCD()
        self.obstacle_pcd = StubPCD()
        self.frontier_pcd = StubPCD()
        self.object_entities = []
        self.trajectory_position = []
        self.grounded_landmarks = []
        self.unmatched_landmarks = []
        self.inferred_areas = []
        self.room_labels = None
        self.room_origin = None
        self.room_regions = {}
        self._room_calls = 0
        self.local_space = {"state": "unknown"}
        self.space_labels = {}


def _floor(rects, step=0.06):
    """把若干矩形离散成可通行地板点云。rects: [(x0,y0,x1,y1), ...]"""
    pts = []
    for x0, y0, x1, y1 in rects:
        xs = np.arange(x0, x1, step)
        ys = np.arange(y0, y1, step)
        gx, gy = np.meshgrid(xs, ys)
        pts.append(np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1))
    return np.vstack(pts)

def _room_with(*class_names, at=(0.0, -1.0, 0.5)):
    """造一个"单间 + 若干物体"的桩：房间推断现在需要可通行地板才能分区。"""
    m = StubMapper()
    m.navigable_pcd = StubPCD(_floor([(-3.0, -4.0, 3.0, 2.0)]))
    m.segment_rooms(force=True)
    m.object_entities = [make_entity(c, np.array(at) + [i * 0.7, 0, 0])
                         for i, c in enumerate(class_names)]
    return m


def make_entity(class_name, center, n_pts=400, conf=0.9, color=(0.4, 0.6, 0.9)):
    c = np.asarray(center, dtype=float)
    pts = c + np.random.RandomState(abs(hash(class_name)) % 2**31).normal(0, 0.18, (n_pts, 3))
    cols = np.tile(np.asarray(color, dtype=float), (n_pts, 1))
    return {
        "class": abs(hash(class_name)) % 50,
        "class_name": class_name,
        "pcd": StubPCD(pts, cols),
        "confidence": conf,
        "center": c,
    }


# ======================================================================
# 1. resolve_category
# ======================================================================
def test_resolve_category():
    print("\n[1] resolve_category —— 词表对齐")
    print(f"  词表规模: {len(CATEGORY_NAMES)} 类")

    expected = {
        "front door": "door",
        "the wooden door": "door",
        "dining table": "table",
        "coffee table": "table",
        "couch": "sofa",
        "fridge": "refrigerator",
        "chairs": "chair",
        "potted plant": "potted_plant",
        "plant": "potted_plant",
        "bookcase": "bookshelf",
        "staircase": "stairs",
        "TV": "television",
        "kitchen counter": "counter",
        "bathroom sink": "sink",
    }
    for name, want in expected.items():
        got, how = resolve_category(name)
        check(got == want, f"{name!r} -> {want}", f"实际得到 {got!r} ({how})")

    # 从实跑日志里挑出来的高频抽取残渣，修完之后应该都能对上
    for name, want in (("banister rail", "handrail"), ("white lampshade", "lamp"),
                       ("back door", "door")):
        got, _ = resolve_category(name)
        check(got == want, f"{name!r} -> {want}", f"实际 {got!r}")

    # 具体房间类型不能被泛化的 area/space 压过去
    # ("kitchen area" 实测占了 area_no_proxy 的 155 次)
    for name, want in (("kitchen area", "kitchen"), ("living room area", "living_room"),
                       ("bedroom area", "bedroom"), ("dining area", "dining_room"),
                       ("little room", "room"), ("hall", "hallway")):
        got, _ = resolve_area(name)
        check(got == want, f"resolve_area({name!r}) -> {want}", f"实际 {got!r}")

    # 房间名 / 词表外物体必须明确判为 unmatched，而不是硬凑一个类别
    for name in ["kitchen", "hallway", "living room", "window", "bedroom"]:
        got, how = resolve_category(name)
        check(got is None, f"{name!r} 应判为 unmatched", f"却匹配到 {got!r} ({how})")


# ======================================================================
# 2. ground_landmarks
# ======================================================================
def load_ground_landmarks():
    mapper_py = os.path.join(SRC, "mapper.py")
    ns = {
        "np": np,
        "cv2": cv2,
        "normalize_category_name": normalize_category_name,
        "resolve_category": resolve_category,
        "resolve_area": resolve_area,
        "AREA_PROXY_OBJECTS": AREA_PROXY_OBJECTS,
    }
    # infer_area_regions 现在建立在几何分区之上，依赖这几个方法
    for pat in (r"^    def _build_free_grid\(", r"^    def segment_rooms\(",
                r"^    def _region_at\(", r"^    def current_room\(",
                r"^    def area_hypotheses\(",
                r"^    def infer_area_regions\(", r"^    def ground_landmarks\("):
        exec(extract_block(mapper_py, pat), ns)
    for name in ("_build_free_grid", "segment_rooms", "_region_at",
                 "current_room", "area_hypotheses", "infer_area_regions", "ground_landmarks"):
        setattr(StubMapper, name, ns[name])
    for k, v in (("ROOM_GRID_RES", 0.05), ("ROOM_CLOSE_M", 0.35),
                 ("ROOM_SEED_CLEAR", 1.0), ("ROOM_MIN_AREA", 1.5),
                 ("ROOM_REBUILD_EVERY", 8), ("ROOM_MAX_EXPAND_M", 3.0),
                 ("ROOM_DOOR_CUT", 0.55),
                 # 房间命名的两个门槛。min_evidence=1 会让卧室里一个误检的
                 # 冰箱把房间判成厨房——实测发生过，所以必须是 2。
                 ("ROOM_MIN_SCORE", 1.0), ("ROOM_MIN_EVIDENCE", 2)):
        setattr(StubMapper, k, v)


def test_area_regions():
    print("\n[2b] infer_area_regions —— 特征物体反推房间")
    load_ground_landmarks()

    # 房间推断现在建立在几何分区上，所以必须给一张可通行地板。
    # 造两个独立房间 + 中间 0.8m 的门洞：卧室在左，厨房在右。
    m = StubMapper()
    m.navigable_pcd = StubPCD(_floor([
        (-6.5, -5.0, -1.5, -1.0),     # 卧室
        (1.5, -8.5, 7.5, -4.5),       # 厨房
        (-1.5, -3.4, 1.5, -2.6),      # 连接两者的窄通道
    ]))
    m.segment_rooms(force=True)
    m.object_entities = [
        # 卧室：一张床
        make_entity("bed", [-4.0, -3.0, 0.3]),
        make_entity("cupboard", [-3.0, -3.5, 0.6]),   # 确认房间需要两类证据
        # 厨房：冰箱 + 灶台 + 水槽
        make_entity("refrigerator", [5.0, -6.0, 0.6]),
        make_entity("stove", [6.0, -6.5, 0.5]),
        make_entity("sink", [5.5, -7.2, 0.5]),
    ]
    areas = {a["area_type"]: a for a in m.infer_area_regions()}

    check("bedroom" in areas, "床 -> 推出 bedroom")
    check("kitchen" in areas, "冰箱+灶台+水槽 -> 推出 kitchen")
    check(areas.get("kitchen", {}).get("score", 0) > areas.get("bedroom", {}).get("score", 9),
          "厨房证据更多，置信度高于卧室",
          f"kitchen={areas.get('kitchen',{}).get('score')} bedroom={areas.get('bedroom',{}).get('score')}")
    check("dining_room" not in areas, "单独一把 chair 不足以判成 dining_room")
    check(all(a["radius"] <= 4.0 for a in areas.values()), "房间半径受 max_radius 限制")

    # 单张桌子不能凭空造出餐厅：桌子在走廊玄关到处都是，
    # min_score=0.5 时这是个高频误报源
    m_t = _room_with("table", at=[0.0, -3.0, 0.4])
    check("dining_room" not in {a["area_type"] for a in m_t.infer_area_regions()},
          "单张 table 不足以判成 dining_room")
    m_t.object_entities.append(make_entity("chair", [0.8, -3.2, 0.4]))
    m_t.inferred_areas = []
    check("dining_room" in {a["area_type"] for a in m_t.infer_area_regions()},
          "table + chair 才判成 dining_room")

    # 卧室和厨房相距很远，不能被聚成同一个区域
    if "bedroom" in areas and "kitchen" in areas:
        d = np.linalg.norm(areas["bedroom"]["center"] - areas["kitchen"]["center"])
        check(d > 5.0, "两个房间中心分离", f"距离仅 {d:.1f}m")

    m.ground_landmarks(
        [{"name": "the bedroom", "role": "waypoint_marker", "relative_position": "unknown"},
         {"name": "hallway", "role": "destination_marker", "relative_position": "unknown"},
         {"name": "bathroom", "role": "destination_marker", "relative_position": "unknown"}],
        subtask_key="SUBTASK_A")
    got = {g["name"]: g for g in m.grounded_landmarks}
    pend = {u["name"]: u for u in m.unmatched_landmarks}

    check(got.get("the bedroom", {}).get("kind") == "area", "'the bedroom' 接地为 area 类型")
    check(pend.get("hallway", {}).get("reason") == "area_no_proxy",
          "hallway 标为 area_no_proxy（没有特征物体可推）")
    check(pend.get("bathroom", {}).get("reason") == "area_pending",
          "bathroom 标为 area_pending（有判据但还没看到）")

    # 指令没点名的房间也必须留在 inferred_areas 里供渲染层当背景语义层用。
    # 之前 kitchen 算出来了却因为不在当前子任务的 landmark 里而被丢弃，
    # 地图上明明有冰箱+灶台却什么都不显示。
    types = {a["area_type"] for a in m.inferred_areas}
    check("kitchen" in types, "未被指令点名的 kitchen 仍保留在 inferred_areas")

    # current_room：只回答"我在哪个房间"，一个答案
    ns2 = {"np": np}
    exec(extract_block(os.path.join(SRC, "mapper.py"), r"^    def current_room\("), ns2)
    StubMapper.current_room = ns2["current_room"]

    m.current_position = np.array([5.0, -6.5, 1.2])      # 站在厨房那簇证据里
    r = m.current_room()
    check(r is not None and r["area_type"] == "kitchen", "站在厨房里 -> kitchen",
          str(r and r["area_type"]))

    m.current_position = np.array([40.0, 40.0, 1.2])     # 远离一切
    check(m.current_room() is None, "不在任何房间里时返回 None")

    # 只有"房间定义级"物体(权重1.0)才能单独成立。
    # 这组用例直接来自一次实跑：卫生间场景里同时推断出 garage(workbench误检)、
    # 两个 office(按摩床被识别成 desk)、两个 stairwell，地图完全没法看。
    m_w = _room_with("desk")      # 0.9
    m_w.infer_area_regions()
    check("office" not in {a["area_type"] for a in m_w.inferred_areas},
          "单个 desk 不再造出 office（书桌在卧室里也很常见）")

    m_w.object_entities.append(make_entity("computer", [0.5, -1.2, 0.7]))  # +0.7
    m_w.inferred_areas = []
    check("office" in {a["area_type"] for a in m_w.infer_area_regions()},
          "desk + computer 才成立")

    m_g = _room_with("workbench")
    check("garage" not in {a["area_type"] for a in m_g.infer_area_regions()},
          "单个 workbench 不再造出 garage")

    # 确认房间需要两类互相印证的证据。单个强物体也不行——检测阈值降到 0.45
    # 之后误检明显增加，实测一个把白衣柜误检成 refrigerator 的框就在卧室里
    # 确认出一个"厨房"，进而让拦截给出完全错误的理由。
    for cls, want in (("toilet", "bathroom"), ("bed", "bedroom"),
                      ("stove", "kitchen"), ("washing_machine", "laundry")):
        mm = _room_with(cls)
        check(want not in {a["area_type"] for a in mm.infer_area_regions()},
              f"单个 {cls} 不再单独确认 {want}（一个检测可能是错的）")
    for pair, want in ((("toilet", "bathtub"), "bathroom"), (("bed", "cupboard"), "bedroom"),
                       (("stove", "refrigerator"), "kitchen"),
                       (("washing_machine", "dryer"), "laundry")):
        mm = _room_with(*pair)
        check(want in {a["area_type"] for a in mm.infer_area_regions()},
              f"{'+'.join(pair)} 两类证据 -> 确认 {want}")
    # 但单个强证据不会浪费：它仍作为弱假设参与探索引导
    mm = _room_with("bed")
    mm.infer_area_regions()
    check(any(h["area_type"] == "bedroom" and not h["confirmed"]
              for h in mm.area_hypotheses()),
          "单个 bed 仍然是 bedroom 的弱假设（引导用得上）")

    # current_room 额外要求"包含性"：agent 必须落在房间圆内，
    # 光是地图上有这个房间不算（分数门槛和 infer 一致，精度靠权重定标保证）
    m_o = _room_with("bed", "cupboard")
    m_o.infer_area_regions()
    m_o.current_position = np.array([0.0, -1.0, 1.2])
    check(m_o.current_room() is not None, "站在卧室里 -> 报 bedroom")
    m_o.current_position = np.array([0.0, -30.0, 1.2])
    check(m_o.current_room() is None, "房间在地图上但人不在里面 -> 不报")

    # 子任务没有任何 landmark 时（比如 "Turn left"）也要照常推断房间
    m.ground_landmarks([], subtask_key="SUBTASK_TURN")
    check(len(m.inferred_areas) > 0, "空 landmark 子任务仍然更新 inferred_areas")
    return m


def test_ground_landmarks():
    print("\n[2] ground_landmarks —— 接地与消歧")
    load_ground_landmarks()

    check(abs(np.linalg.det(HABITAT_ROT_IDENTITY) - (-1.0)) < 1e-9,
          "测试桩使用 det=-1 的 habitat 约定（左手系）")

    m = StubMapper()
    # forward = -y，所以"前方"的物体 y 为负；right = +x
    m.object_entities = [
        make_entity("sofa", [-1.0, -3.0, 0.4]),
        make_entity("table", [0.5, -2.0, 0.4]),
        make_entity("table", [4.0, -7.0, 0.4]),       # 更远的同类实例
        make_entity("chair", [-2.0, -1.5, 0.4]),      # agent 左侧 (x<0)
        make_entity("chair", [2.2, -1.5, 0.4]),       # agent 右侧 (x>0)
        make_entity("refrigerator", [-3.0, -5.0, 0.6]),
    ]

    landmarks = [
        {"name": "couch", "role": "waypoint_marker", "relative_position": "unknown"},
        {"name": "dining table", "role": "destination_marker", "relative_position": "unknown"},
        {"name": "chair", "role": "avoid_marker", "relative_position": "left"},
        {"name": "window", "role": "destination_marker", "relative_position": "unknown"},
        {"name": "oven", "role": "waypoint_marker", "relative_position": "unknown"},
    ]

    m.ground_landmarks(landmarks, subtask_key="SUBTASK_1")
    by_name = {g["name"]: g for g in m.grounded_landmarks}
    pending = {u["name"]: u for u in m.unmatched_landmarks}

    check(len(m.grounded_landmarks) == 3, "接地 3 个 landmark", f"实际 {len(m.grounded_landmarks)}")
    check("couch" in by_name and by_name["couch"]["class_name"] == "sofa",
          "couch 经同义词接到 sofa 实例")

    # 两个 table，应挑距离更近的那个 (0.5, -2.0)
    check("dining table" in by_name and
          abs(by_name["dining table"]["center"][1] - (-2.0)) < 1e-6,
          "同类多实例时选中更近的 table")
    check(by_name.get("dining table", {}).get("n_candidates") == 2,
          "n_candidates 正确记为 2")

    # right = +x，所以 'left' 应选中 x<0 的那把椅子。
    # 这条断言就是抓到叉积符号写反的那条 —— 在左手系里照搬右手系的
    # "cross>0 即左侧" 会稳定地选错另一侧。
    check("chair" in by_name and by_name["chair"]["center"][0] < 0,
          "relative_position='left' 选中左侧的 chair",
          f"实际 x={by_name.get('chair', {}).get('center', [999])[0]}")

    # 反方向也测一次，避免"符号反了但两边都反"这种自洽的错误蒙混过关
    m.ground_landmarks(
        [{"name": "chair", "role": "waypoint_marker", "relative_position": "right"}],
        subtask_key="SUBTASK_1R")
    right_pick = {g["name"]: g for g in m.grounded_landmarks}.get("chair", {})
    check(right_pick.get("center", [-999])[0] > 0,
          "relative_position='right' 选中右侧的 chair",
          f"实际 x={right_pick.get('center', [-999])[0]}")

    check(pending.get("window", {}).get("reason") == "not_in_vocab",
          "window 标为 not_in_vocab")
    check(pending.get("oven", {}).get("reason") == "not_yet_detected",
          "oven 标为 not_yet_detected")

    # 空输入必须干净返回，不能抛
    m.ground_landmarks([], subtask_key="SUBTASK_2")
    check(m.grounded_landmarks == [] and m.unmatched_landmarks == [],
          "空 landmark 列表安全返回")

    # 地图上一个物体都还没有时也不能崩
    m2 = StubMapper()
    m2.ground_landmarks(landmarks, subtask_key="SUBTASK_1")
    check(len(m2.grounded_landmarks) == 0 and len(m2.unmatched_landmarks) == 5,
          "空地图时全部落入 pending")

    # 上面几个边界用例把 m 的接地结果清空了，交给渲染测试前重新跑一次
    m.ground_landmarks(landmarks, subtask_key="SUBTASK_1")
    return m


# ======================================================================
# 3. 渲染
# ======================================================================
def load_renderer():
    # 签名里有 mapper: Instruct_Mapper 的类型标注，给个占位类即可
    ns = {"np": np, "cv2": cv2, "time": __import__("time"), "Instruct_Mapper": StubMapper}
    exec(agent_block(r"^def draw_direction_markers\("), ns)
    exec(agent_block(r"^def create_top_down_map_centered\("), ns)
    return ns["create_top_down_map_centered"]


def test_external_instances():
    """apply_external_instances 的坐标换算与词表过滤（不含点云反投影）。"""
    print("\n[2c] apply_external_instances —— VLM 检测框回填")

    mapper_py = os.path.join(SRC, "mapper.py")
    ns = {"np": np, "resolve_category": resolve_category}
    exec(extract_block(mapper_py, r"^    def apply_external_instances\("), ns)

    fused = {}

    class M:
        current_depth = np.zeros((480, 640))

        def fuse_instances(self, instances):
            fused["v"] = instances

    M.apply_external_instances = ns["apply_external_instances"]
    m = M()

    n = m.apply_external_instances([
        {"c": "sofa", "b": [0, 0, 500, 500]},        # 左上四分之一
        {"c": "dining table", "b": [500, 500, 1000, 1000]},  # 需要剥修饰词
        {"c": "unicorn", "b": [10, 10, 20, 20]},     # 词表外，应丢弃
        {"c": "chair", "b": [300, 300, 300, 400]},   # 零宽，应丢弃
        {"c": "door", "b": "bad"},                   # 畸形，应丢弃
    ], image_shape=(480, 640, 3))

    check(n == 2, "5 条里只保留 2 条合法检测", f"实际 {n}")
    inst = fused.get("v", [])
    names = [i["class_name"] for i in inst]
    check(names == ["sofa", "table"], "dining table 归一到 table", str(names))

    if inst:
        ys, xs = np.where(inst[0]["mask"] > 0)
        check(xs.min() == 0 and xs.max() == 319 and ys.max() == 239,
              "0-1000 坐标正确换算到 640x480 像素",
              f"x:[{xs.min()},{xs.max()}] y:[{ys.min()},{ys.max()}]")
        check(all(i["confidence"] < 1.0 for i in inst),
              "矩形框置信度被打折（低于本地检测器）")

    check(m.apply_external_instances([], image_shape=(480, 640, 3)) == 0, "空输入返回 0")
    check(m.apply_external_instances(None) == 0, "None 输入返回 0")


def test_desync_detector():
    """
    复现 2026-08-13 15:12 那次运行日志里的真实失败：VLM 在 thought 里宣布
    "Subtask 1 已完成"但 action 给了路点，状态机卡住 25 步，期间
    open door 距离从 1.6m 涨到 3.6m。这里用那串真实距离喂检测器。
    """
    print("\n[2d] _track_landmark_progress —— 状态机滞后检测")
    ns = {"np": np}
    exec(agent_block(r"^    def _reset_landmark_progress\("), ns)
    exec(agent_block(r"^    def _track_landmark_progress\("), ns)

    class A(HyperDefaults):
        pass

    A._reset_landmark_progress = ns["_reset_landmark_progress"]
    A._track_landmark_progress = ns["_track_landmark_progress"]
    A.NEAR_M, A.PASSED_MIN_M, A.PASSED_GAP_M = 1.5, 2.0, 1.0
    # 这些原本是方法的默认参数，现在从 config['hyper'] 读，桩类要显式给同值
    A.LM_RECEDE_MARGIN, A.LM_RECEDE_STEPS = 1.2, 6
    A.enable_interception = A.enable_guidance = True

    class FakeMapper:
        grounded_landmarks = []

    a = A()
    a.mapper = FakeMapper()

    # 真实日志里 open door 的距离序列（step 30 起）
    dists = [1.6, 2.3, 2.3, 2.3, 2.6, 2.6, 3.3, 3.6, 3.6, 3.6, 3.6, 3.6]
    fired_at = None
    for i, d in enumerate(dists):
        a.mapper.grounded_landmarks = [
            {"name": "open door", "role": "waypoint_marker", "distance": d}
        ]
        a._track_landmark_progress("SUBTASK_1")
        if a.passed_landmark_hint and fired_at is None:
            fired_at = i

    check(fired_at is not None, "真实距离序列能触发 desync 提示")
    check(fired_at is not None and fired_at >= 5,
          "不会过早触发（至少积累若干步）", f"第 {fired_at} 步就触发了")

    # 正常靠近目标时绝不能误报
    a2 = A()
    a2.mapper = FakeMapper()
    for d in [5.0, 4.2, 3.5, 2.8, 2.0, 1.5, 1.1, 0.8, 0.5]:
        a2.mapper.grounded_landmarks = [
            {"name": "sofa", "role": "destination_marker", "distance": d}
        ]
        a2._track_landmark_progress("SUBTASK_1")
    check(not a2.passed_landmark_hint, "持续接近目标时不误报")

    # 子任务切换必须清空历史，否则会拿旧landmark继续判定
    a._track_landmark_progress("SUBTASK_2")
    check(not a.passed_landmark_hint, "切换子任务后提示被清空")

    # 一个landmark都没接地时不做判断（距离缺失 != 走过头）
    a3 = A()
    a3.mapper = FakeMapper()
    for _ in range(10):
        a3.mapper.grounded_landmarks = []
        a3._track_landmark_progress("SUBTASK_1")
    check(not a3.passed_landmark_hint, "无接地landmark时不判定")


def test_desync_signals():
    """
    desync 提示的两个来源。重点是复现 episode_2 的失败：
    SUBTASK_1 = "Go out of the room you're in"，唯一 landmark 是 room
    (area_no_proxy，永远接不了地)，几何信号完全哑火，全靠步数预算兜底。
    """
    print("\n[2g] desync 双信号 —— 不依赖 landmark 接地")
    src = agent_source()

    # generate_prompt 太大且依赖一堆实例状态，这里只抽出信号合成那几行的语义做等价检查：
    # 直接断言源码里 desync_str 的触发条件同时包含 over_budget 和 receding。
    seg = src[src.index("desync_str = \"\""):src.index("**PROGRESS CHECK")]
    check("over_budget or receding" in seg,
          "desync 触发条件为 over_budget OR receding（不再只看 landmark）")
    check("over_budget = cur['ratio'] > 1.5" in src,
          "over_budget 由步数预算比例决定，不依赖 landmark")

    # 几何信号在 landmark 接不了地时必须安全地保持 False（而不是报错）
    ns = {"np": np}
    exec(agent_block(r"^    def _reset_landmark_progress\("), ns)
    exec(agent_block(r"^    def _track_landmark_progress\("), ns)

    class A(HyperDefaults):
        pass

    A._reset_landmark_progress = ns["_reset_landmark_progress"]
    A._track_landmark_progress = ns["_track_landmark_progress"]
    A.NEAR_M, A.PASSED_MIN_M, A.PASSED_GAP_M = 1.5, 2.0, 1.0
    # 这些原本是方法的默认参数，现在从 config['hyper'] 读，桩类要显式给同值
    A.LM_RECEDE_MARGIN, A.LM_RECEDE_STEPS = 1.2, 6
    A.enable_interception = A.enable_guidance = True

    class FakeMapper:
        grounded_landmarks = []

    a = A()
    a.mapper = FakeMapper()
    for _ in range(30):  # 模拟 episode_2 的 SUBTASK_1：30步一个landmark都没接上
        a._track_landmark_progress("SUBTASK_1")
    check(a.passed_landmark_hint is False,
          "landmark 全程接不了地时几何信号保持 False（此时只能靠预算兜底）")


def _load_subtask_guards():
    ns = {"np": np}
    for pat in (r"^    def _current_budget_row\(", r"^    def _is_premature_completion\(",
                r"^    def _maybe_force_advance\(", r"^    def _reset_landmark_progress\("):
        exec(agent_block(pat), ns)

    class A(HyperDefaults):
        pass

    for name in ("_current_budget_row", "_is_premature_completion",
                 "_maybe_force_advance", "_reset_landmark_progress"):
        setattr(A, name, ns[name])
    A.NEAR_M, A.PASSED_MIN_M, A.PASSED_GAP_M = 1.5, 2.0, 1.0
    # 这些原本是方法的默认参数，现在从 config['hyper'] 读，桩类要显式给同值
    A.LM_RECEDE_MARGIN, A.LM_RECEDE_STEPS = 1.2, 6
    A.enable_interception = A.enable_guidance = True
    return A


class FakeInstruction:
    """只实现 guard 用到的那几个接口。"""

    def __init__(self, n=4, idx=0, used=5, expected=20.0):
        self.n, self.idx, self.used, self.expected = n, idx, used, expected
        self.records = []
        self.completed = []
        # _is_premature_completion 会读 local_end / boundary_consistent
        self.sub_instruction_dict = {f"SUBTASK_{i+1}": {} for i in range(n)}

    def get_current_subtask_key(self):
        return None if self.idx >= self.n else f"SUBTASK_{self.idx + 1}"

    def is_last_subtask(self):
        return self.idx == self.n - 1

    def get_current_subtask(self):
        k = self.get_current_subtask_key()
        return k, self.sub_instruction_dict.get(k, {})

    def get_progress(self, total):
        return [{'key': self.get_current_subtask_key(), 'status': 'current',
                 'used_steps': self.used, 'expected_steps': self.expected,
                 'ratio': self.used / self.expected}]

    def add_record_to_current_subtask(self, r):
        self.records.append((self.get_current_subtask_key(), r))

    def mark_current_subtask_completed(self, coord=None, rot=None):
        self.completed.append(self.get_current_subtask_key())
        self.idx += 1


def _mk_agent(A, inst):
    a = A()
    a.instruction_obj = inst
    a.total_step_budget = 100

    class M:
        current_position = np.array([0.0, 0.0, 1.2])
        current_rotation = np.eye(3)
        grounded_landmarks = []

    a.mapper = M()
    a.stop_challenge = ""
    return a


def test_premature_guard():
    print("\n[2h] _is_premature_completion —— 提前宣告完成的质询")
    A = _load_subtask_guards()

    # 5/20 = 25% < 30% -> 该质询
    a = _mk_agent(A, FakeInstruction(used=5, expected=20.0))
    check(a._is_premature_completion() is True, "25% 预算时触发质询")
    check("CONFIRM" in a.stop_challenge, "质询文本写入 stop_challenge")

    # 同一子任务只质询一次，否则会死锁
    check(a._is_premature_completion() is False, "同一子任务不重复质询")

    # 正常时长不质询
    b = _mk_agent(A, FakeInstruction(used=15, expected=20.0))
    check(b._is_premature_completion() is False, "75% 预算时不质询")

    # 最后一个子任务不质询（终点stop有自己的双重确认）
    c = _mk_agent(A, FakeInstruction(n=2, idx=1, used=1, expected=20.0))
    check(c._is_premature_completion() is False, "最后一个子任务不质询")


def test_force_advance():
    print("\n[2i] _maybe_force_advance —— 严重超支时强制推进")
    A = _load_subtask_guards()

    # ratio 0.99，远没到阈值
    a = _mk_agent(A, FakeInstruction(used=99, expected=100.0))
    check(a._maybe_force_advance() is False, "未超过 2.0x 时不强推")

    # 短子任务保护：ratio 高但绝对步数少，不该强推
    # ("Go out of the room you're in" 预期9步、实际14~16步是正常的)
    s = _mk_agent(A, FakeInstruction(used=16, expected=7.0))
    check(s._maybe_force_advance() is False, "绝对步数 <20 时不强推（短子任务保护）")

    # ep31 的情形（新标定后）：70 步 vs 预期 24 -> ratio 2.9
    b = _mk_agent(A, FakeInstruction(used=70, expected=24.0))
    check(b._maybe_force_advance() is True, "超过 2.0x 且步数够多时强制推进")
    check(b.instruction_obj.completed == ["SUBTASK_1"], "推进的是当时的当前子任务")
    check(b.instruction_obj.records[0][0] == "SUBTASK_1",
          "system 记录写在被推进的那个子任务上（不是下一个）")
    check("force-advanced" in str(b.instruction_obj.records[0][1]).lower(),
          "记录里说明了是被系统强推的")

    # 每个子任务只强推一次
    b.instruction_obj.used = 300
    check(b._maybe_force_advance() is True, "换到下一个子任务后可以再强推一次")
    b.instruction_obj.idx -= 1  # 手动退回，模拟同一个key再次超支
    check(b._maybe_force_advance() is False, "同一子任务不重复强推")

    # 最后一个子任务无处可推
    c = _mk_agent(A, FakeInstruction(n=2, idx=1, used=999, expected=10.0))
    check(c._maybe_force_advance() is False, "最后一个子任务不强推")

    # ---- 强推必须尊重顺序：不能无视没去过的途经点直接跳过 ----
    d = _mk_agent(A, FakeInstruction(n=3, idx=0, used=70, expected=24.0))
    d.mapper.grounded_landmarks = [
        {"name": "dining table", "role": "waypoint_marker",
         "distance": 6.0, "ever_near": False}]
    check(d._maybe_force_advance() is False,
          "有没去过的途经点时，第一次先宽限不推")
    check("LAST CHANCE" in d.stop_challenge, "宽限时点名了没去的途经点")
    check(d.instruction_obj.completed == [], "宽限阶段没有真的推进")

    check(d._maybe_force_advance() is True, "宽限之后仍未去 -> 真推")
    rec = str(d.instruction_obj.records[-1][1])
    check("skipped WITHOUT" in rec, "记录里写明是跳过的、没访问途经点", rec[:80])

    # 途经点都去过时不需要宽限，直接推
    e = _mk_agent(A, FakeInstruction(n=3, idx=0, used=70, expected=24.0))
    e.mapper.grounded_landmarks = [
        {"name": "dining table", "role": "waypoint_marker",
         "distance": 6.0, "ever_near": True}]
    check(e._maybe_force_advance() is True, "途经点都去过 -> 直接推，不宽限")


def _load_space_classifier():
    ns = {"np": np}
    exec(extract_block(os.path.join(SRC, "mapper.py"), r"^    def classify_local_space\("), ns)
    StubMapper.classify_local_space = ns["classify_local_space"]
    for k, v in (("SPACE_RAY_MAX", 6.0), ("SPACE_SLAB_HALF", 0.25),
                 ("SPACE_CORRIDOR_CLEAR", 1.2), ("SPACE_ROOM_CLEAR", 1.5),
                 ("SPACE_CORRIDOR_ELONG", 2.5),
                 ("SPACE_VOTE_WINDOW", 5), ("SPACE_CELL", 0.25)):
        setattr(StubMapper, k, v)


def _walls(segments, step=0.03, z=0.8):
    """把若干线段离散成障碍点云。segments: [((x1,y1),(x2,y2)), ...]"""
    pts = []
    for (x1, y1), (x2, y2) in segments:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step))
        for t in np.linspace(0, 1, n):
            pts.append([x1 + (x2 - x1) * t, y1 + (y2 - y1) * t, z])
    return np.array(pts)


def test_local_space():
    """
    走廊/房间的局部形态判据。agent 前向是 -y，右向是 +x。
    """
    print("\n[2m] classify_local_space —— 走廊/房间的几何判据")
    _load_space_classifier()

    # 走廊：宽 1.6m（左右各 0.8m），沿 y 方向延伸 20m
    m = StubMapper()
    m.obstacle_pcd = StubPCD(_walls([((-0.8, -10), (-0.8, 10)),
                                     ((0.8, -10), (0.8, 10))]))
    for _ in range(5):
        r = m.classify_local_space()
    check(r['state'] == 'corridor', "1.6m 宽的长通道 -> corridor", str(r))
    check(abs(r['clearance'] - 0.8) < 0.1, "通行余量约 0.8m", f"{r['clearance']:.2f}")
    check(r['elongation'] > 2.5, "细长比 > 2.5", f"{r['elongation']:.1f}")
    check(len(m.space_labels) > 0, "走廊落色进 space_labels")

    # 房间：8x8m 的方形空间
    m2 = StubMapper()
    m2.obstacle_pcd = StubPCD(_walls([((-4, -4), (4, -4)), ((4, -4), (4, 4)),
                                      ((4, 4), (-4, 4)), ((-4, 4), (-4, -4))]))
    for _ in range(5):
        r2 = m2.classify_local_space()
    check(r2['state'] == 'room', "8x8m 方形空间 -> room", str(r2))
    check(len(m2.space_labels) == 0, "房间不落色（保持地板原色）")

    # 家具缝隙：窄但不长（沙发和茶几之间），不该判成走廊
    m3 = StubMapper()
    m3.obstacle_pcd = StubPCD(_walls([((-0.6, -1.2), (-0.6, 1.2)),
                                      ((0.6, -1.2), (0.6, 1.2)),
                                      ((-0.6, -1.2), (0.6, -1.2)),
                                      ((-0.6, 1.2), (0.6, 1.2))]))
    for _ in range(5):
        r3 = m3.classify_local_space()
    check(r3['state'] != 'corridor', "1.2m 见方的家具缝隙不判成走廊", str(r3['state']))

    # 没有障碍点时安全返回
    m4 = StubMapper()
    check(m4.classify_local_space()['state'] == 'unknown', "无障碍点云时返回 unknown")

    # 滑窗投票：单帧误判不该立刻改变结论
    m5 = StubMapper()
    m5.obstacle_pcd = StubPCD(_walls([((-4, -4), (4, -4)), ((4, -4), (4, 4)),
                                      ((4, 4), (-4, 4)), ((-4, 4), (-4, -4))]))
    for _ in range(5):
        m5.classify_local_space()
    before = m5.local_space['state']
    m5.obstacle_pcd = StubPCD(_walls([((-0.5, -10), (-0.5, 10)), ((0.5, -10), (0.5, 10))]))
    r5 = m5.classify_local_space()          # 只喂一帧"走廊"
    check(r5['raw_state'] == 'corridor', "单帧原始判定已变")
    check(r5['state'] == before, "滑窗投票下结论不被单帧翻转")
    return m




def test_room_segmentation():
    """
    自由空间几何分区。造一个"两个房间 + 中间一条走廊 + 两个门口"的平面，
    检查它能不能在门口处把房间切开——这正是墙检测想做但更难做到的事。
    """
    print("\n[2n] segment_rooms —— 自由空间几何分区")

    mapper_py = os.path.join(SRC, "mapper.py")
    ns = {"np": np, "cv2": cv2,
          "normalize_category_name": normalize_category_name,
          "AREA_PROXY_OBJECTS": AREA_PROXY_OBJECTS}
    for pat in (r"^    def _build_free_grid\(", r"^    def segment_rooms\(",
                r"^    def _region_at\(", r"^    def current_room\(",
                r"^    def infer_area_regions\("):
        exec(extract_block(mapper_py, pat), ns)
    for name in ("_build_free_grid", "segment_rooms", "_region_at",
                 "current_room", "infer_area_regions"):
        setattr(StubMapper, name, ns[name])
    for k, v in (("ROOM_GRID_RES", 0.05), ("ROOM_CLOSE_M", 0.35),
                 ("ROOM_SEED_CLEAR", 1.0), ("ROOM_MIN_AREA", 1.5),
                 ("ROOM_REBUILD_EVERY", 8), ("ROOM_MAX_EXPAND_M", 3.0),
                 ("ROOM_DOOR_CUT", 0.55),
                 # 房间命名的两个门槛。min_evidence=1 会让卧室里一个误检的
                 # 冰箱把房间判成厨房——实测发生过，所以必须是 2。
                 ("ROOM_MIN_SCORE", 1.0), ("ROOM_MIN_EVIDENCE", 2)):
        setattr(StubMapper, k, v)

    #   房间A (-5..-1, -4..0)   走廊 (-1..1, -4..0 的一条窄带)  房间B (1..5, -4..0)
    #   门口用 0.8m 宽的缺口连接
    m = StubMapper()
    m.navigable_pcd = StubPCD(_floor([
        (-5.0, -4.0, -1.0, 0.0),      # 房间 A (4x4)
        (1.0, -4.0, 5.0, 0.0),        # 房间 B (4x4)
        (-1.0, -2.4, 1.0, -1.6),      # 中间 0.8m 宽的连接通道（门口）
    ]))
    m.room_labels = None
    regions = m.segment_rooms(force=True)

    check(len(regions) >= 2, f"两个房间被切开（得到 {len(regions)} 个区域）")
    areas = sorted(r['area'] for r in regions.values())
    check(all(a > 3.0 for a in areas[-2:]), "两个主区域面积合理",
          f"{[round(a,1) for a in areas]}")

    # 两个房间中心应该分别落在各自区域里，且标签不同
    la = m._region_at(np.array([-3.0, -2.0]))
    lb = m._region_at(np.array([3.0, -2.0]))
    check(la > 0 and lb > 0, "两个房间中心都能查到区域标签")
    check(la != lb, "门口把两个房间分成了不同标签（不需要检测墙或门）")

    # 命名：A 放一张床，B 放灶台
    m.object_entities = [make_entity("bed", [-3.0, -2.0, 0.4]),
                         make_entity("cupboard", [-3.6, -2.6, 0.6]),
                         make_entity("stove", [3.0, -2.0, 0.5]),
                         make_entity("refrigerator", [3.6, -2.6, 0.6])]
    m.inferred_areas = []
    named = {a['area_type']: a for a in m.infer_area_regions()}
    check("bedroom" in named and "kitchen" in named, "两个区域分别命名成功",
          str(list(named)))
    check(named["bedroom"]["label"] == la, "bedroom 对应房间A 的标签")
    check(named["kitchen"]["label"] == lb, "kitchen 对应房间B 的标签")
    check(named["bedroom"].get("contour") is not None, "区域带轮廓（渲染层画多边形）")

    # 关键回归：同一区域里的 bed + desk 不能再分裂成 bedroom + office
    m2 = StubMapper()
    m2.navigable_pcd = StubPCD(_floor([(-3.0, -5.0, 3.0, 0.0)]))   # 一个大房间
    m2.room_labels = None
    m2.segment_rooms(force=True)
    m2.object_entities = [make_entity("bed", [-2.0, -1.0, 0.4]),
                          make_entity("cupboard", [-1.4, -1.6, 0.6]),
                          make_entity("desk", [2.0, -4.0, 0.5])]   # desk 相距约 5m
    m2.inferred_areas = []
    got = [a['area_type'] for a in m2.infer_area_regions()]
    check(len(got) == 1, "同一个房间只判一种类型，不再产生重叠圈", str(got))
    check(got == ["bedroom"], "bed + desk 同处一室 -> bedroom（不是 office）", str(got))

    # ---- 一个房间不许吞掉整层楼 ----
    # 实测场景 44：地图上一个 [kitchen] 多边形里同时有 desk、四个 workbench、
    # mirror、carpet，agent 标签自相矛盾成 "corridor 0.6m in kitchen"。
    # 原因是走廊余量(0.6~1.0m)够不上种子门槛，于是厨房的标签穿过门口一路
    # 把走廊连同隔壁全吃了。
    big = StubMapper()
    big.navigable_pcd = StubPCD(_floor([
        (-4.0, -4.0, 0.0, 0.0),       # 开阔房间（会产生种子）
        (0.0, -2.4, 12.0, -1.6),      # 0.8m 宽、12m 长的走廊（够不上种子）
        (12.0, -4.0, 15.0, 0.0),      # 走廊尽头的另一个房间
    ]))
    big.room_labels = None
    regs = big.segment_rooms(force=True)
    areas = sorted((r['area'] for r in regs.values()), reverse=True)
    check(areas and areas[0] < 22.0,
          "最大区域没有吞掉整层楼", f"最大 {areas[0]:.1f}m² (总可通行约 25m²)")

    # 走廊远端应当不属于任何已识别房间（label 0），而不是被算成那个房间
    far = big._region_at(np.array([9.0, -2.0]))
    near = big._region_at(np.array([-2.0, -2.0]))
    check(near > 0, "开阔房间内部有区域标签")
    check(far == 0, "12m 走廊的远端不再被并进房间", f"label={far}")

    # 走廊里查 current_room 应当返回 None，而不是谎报一个房间
    big.inferred_areas = []
    big.object_entities = [make_entity("bed", [-2.0, -2.0, 0.4]),
                           make_entity("cupboard", [-2.6, -2.6, 0.6])]
    big.infer_area_regions()
    big.current_position = np.array([9.0, -2.0, 1.2])
    check(big.current_room() is None, "站在走廊里不谎报房间")
    big.current_position = np.array([-2.0, -2.0, 1.2])
    check(big.current_room() is not None, "站在卧室里仍然正常报出")

    # current_room 用精确的标签包含判定
    m2.current_position = np.array([-2.0, -1.0, 1.2])
    check(m2.current_room() is not None, "站在房间里能报出房间")
    m3 = StubMapper()
    m3.room_labels = None
    check(m3.current_room() is None, "没有分区也不崩")


def test_decision_fallback():
    """
    三次解析失败后不能直接放弃整个 episode。
    真实数据：22 个 episode 里 4 个(18%)是这么没的，其中一个只走了 0.00m。
    """
    print("\n[2l] decide_waypoint 失败兜底 —— 不再一失败就终止 episode")
    # 只在 decide_waypoint 所在的文件里切，避免被拼接后的其它模块干扰
    src = agent_source()
    body = agent_block(r"^    def decide_waypoint\(")
    tail = body[body.index("# ---- 三次都失败：不要直接放弃 ----"):]

    check("_decision_failures" in tail, "有跨步的连续失败计数")
    check("'type': 'turn'" in tail, "失败时退化成转向重新观察，而不是 stop")
    check("< 3" in tail, "连续失败 3 次才真的放弃")
    check(tail.index("return ({'type': 'turn'") < tail.index("return None"),
          "转向兜底在放弃之前")
    check("_decision_failures = 0" in src, "成功决策后计数归零")

    # 解析失败必须留下可排查的日志
    check("unparsable" in src, "解析失败时打印模型原始返回")
    check("[vlm] request failed" in src, "网络/后端异常不再静默吞掉")


def test_receded_stop_guard():
    """
    "走到了又走开"的拦截。真实数据：22 个 episode 里 9 个曾进入目标 3m 内，
    只有 4 个停在那里；丢掉的 5 个历史最近 1.2~2.4m，最终停在 3.6~9.2m。
    """
    print("\n[2k] _has_receded_from_destination —— 走到了又走开")
    ns = {"np": np}
    exec(agent_block(r"^    def _is_too_far_to_stop\("), ns)
    exec(agent_block(r"^    def _has_receded_from_destination\("), ns)
    exec(agent_block(r"^    def _challenge_receded_stop\("), ns)

    class A(HyperDefaults):
        pass

    A._is_too_far_to_stop = ns["_is_too_far_to_stop"]
    A._has_receded_from_destination = ns["_has_receded_from_destination"]
    A._challenge_receded_stop = ns["_challenge_receded_stop"]
    A.enable_interception = A.enable_guidance = True

    class FM:
        grounded_landmarks = []

    def mk(now, best, role="destination_marker"):
        a = A()
        a.mapper = FM()
        a.mapper.grounded_landmarks = [
            {"name": "massage table", "role": role, "distance": now}]
        a._lm_min_dist = {"massage table": best}
        a.stop_challenge = ""
        a.instruction_obj = FakeInstruction(n=2, idx=1)
        return a

    # ep37 的情形：最近到过 1.2m，现在 8.0m
    a = mk(8.0, 1.2)
    rec, detail = a._has_receded_from_destination()
    check(rec is True, "历史最近 1.2m、现在 8.0m -> 判定为已走开")
    check("1.2" in detail and "8.0" in detail, "说明文字带上两个距离")

    # 正在接近，不该拦
    b = mk(1.0, 1.2)
    check(b._has_receded_from_destination()[0] is False, "正在接近时不拦")

    # 只远离了一点点（< margin 1.5m），不拦，避免噪声误触
    c = mk(2.5, 1.2)
    check(c._has_receded_from_destination()[0] is False, "小幅波动不拦")

    # 只看 destination_marker，途经点远离是正常的
    d = mk(8.0, 1.2, role="waypoint_marker")
    check(d._has_receded_from_destination()[0] is False, "waypoint_marker 远离不拦")

    # 没有历史最近记录时不能误判
    e = mk(8.0, None)
    e._lm_min_dist = {}
    check(e._has_receded_from_destination()[0] is False, "无历史距离时不拦")

    # 拦截只生效一次，避免和模型顶死
    f = mk(8.0, 1.2)
    check(f._challenge_receded_stop() is True, "首次宣告停止时拦下")
    check("DO NOT STOP HERE YET" in f.stop_challenge, "质询文本已写入")
    check(f._challenge_receded_stop() is False, "同一子任务不重复拦截")

    # ---- 绝对距离判据：复现 ep32 ----
    # 指令"站在按摩床旁边"，agent 自己检测到 table 在 3.6m 处却宣布到达，
    # 成功阈值 3.0m，差 0.6m。历史判据抓不到（接地时 2.7m，只远了 0.9m）。
    g = mk(3.6, 2.7)
    check(g._has_receded_from_destination()[0] is False, "ep32 用历史判据抓不到")
    too_far, d = g._is_too_far_to_stop()
    check(too_far is True, "ep32 用绝对距离判据能抓到")
    check("3.6" in d, "说明文字带上当前距离")
    check(g._challenge_receded_stop() is True, "两条判据任一成立都会拦下")

    # 已经很近了，不能拦
    h = mk(1.2, 1.0)
    check(h._is_too_far_to_stop()[0] is False, "1.2m 已经足够近，不拦")

    # 区域类地标用房间半径当阈值：ep33 的 bedroom 2.2m、ep34 的 0.6m 都该放行
    def mk_area(now, radius):
        a = A()

        class FM2:
            pass

        a.mapper = FM2()
        a.mapper.grounded_landmarks = [{"name": "other bedroom", "role": "destination_marker",
                                        "kind": "area", "radius": radius, "distance": now}]
        a._lm_min_dist = {}
        a.stop_challenge = ""
        a.instruction_obj = FakeInstruction(n=2, idx=1)
        return a

    check(mk_area(2.2, 3.0)._is_too_far_to_stop()[0] is False,
          "区域类：在房间半径内不拦（ep33 的 2.2m）")
    check(mk_area(0.6, 3.0)._is_too_far_to_stop()[0] is False,
          "区域类：0.6m 不拦（ep34）")
    check(mk_area(5.0, 3.0)._is_too_far_to_stop()[0] is True,
          "区域类：超出房间半径才拦")


def test_side_view_pose_swap():
    """
    侧视角分割必须连位姿一起换。

    get_object_entities() 内部用 self.current_position/current_rotation 反投影，
    第一版只换了 depth/点云/内参，导致左视图里的物体被当成正前方的物体放进
    地图——实跑时地图上成片的 bed/desk/sink 落在从未探索的白色区域里。
    """
    print("\n[2o] 侧视角分割的位姿切换")
    src = open(os.path.join(SRC, "mapper.py"), encoding="utf-8").read()
    blk = src[src.index("if do_instance_segmentation and segment_all_views"):]
    blk = blk[:blk.index("print(f\"[detect]")]

    for field in ("current_depth", "_primary_view_pcd", "_primary_view_intrinsic",
                  "current_position", "current_rotation"):
        check(f"self.{field} = proc[" in blk or f"self.{field} = proc.get(" in blk
              or f"self.{field} = proc['" in blk,
              f"侧视角循环里切换了 {field}")
    # 保存/恢复必须覆盖同样这批字段，否则主视图状态会被污染
    check(blk.count("self.current_position") >= 2, "位姿在 finally 里被恢复")
    check("finally:" in blk, "用 finally 保证异常时也恢复")

    # 反投影确实依赖 current_position（这是上面那条约束的理由）
    goe = src[src.index("def get_object_entities"):]
    goe = goe[:goe.index("return entities")]
    check("translate_to_world(camera_points, self.current_position, self.current_rotation)" in goe,
          "get_object_entities 用 self.current_position/rotation 反投影")


def test_landmark_states():
    """landmark 时序状态机：未见 / 已定位 / 就在旁边 / 已走过。"""
    print("\n[2p] landmark 时序状态 —— seen / near / passed")
    ns = {"np": np}
    exec(agent_block(r"^    def _reset_landmark_progress\("), ns)
    exec(agent_block(r"^    def _track_landmark_progress\("), ns)

    class A(HyperDefaults):
        pass
    A._reset_landmark_progress = ns["_reset_landmark_progress"]
    A._track_landmark_progress = ns["_track_landmark_progress"]
    A.NEAR_M, A.PASSED_MIN_M, A.PASSED_GAP_M = 1.5, 2.0, 1.0
    # 这些原本是方法的默认参数，现在从 config['hyper'] 读，桩类要显式给同值
    A.LM_RECEDE_MARGIN, A.LM_RECEDE_STEPS = 1.2, 6
    A.enable_interception = A.enable_guidance = True

    class FM:
        grounded_landmarks = []

    a = A(); a.mapper = FM()

    def step(d):
        a.mapper.grounded_landmarks = [
            {"name": "door", "role": "waypoint_marker", "distance": d}]
        a._track_landmark_progress("SUBTASK_1")
        return a.mapper.grounded_landmarks[0]

    check(step(5.0)["state"] == "seen", "远距离定位到 -> seen")
    check(step(1.2)["state"] == "near", "走到旁边 -> near")
    check(step(3.5)["state"] == "passed", "靠近过又走远 -> passed")
    g = a.mapper.grounded_landmarks[0]
    check(abs(g["min_dist"] - 1.2) < 1e-6, "记录了历史最近距离", str(g["min_dist"]))

    # 从没靠近过就一直远离，不能算 passed（那是还没走到，不是走过了）
    b = A(); b.mapper = FM()
    for d in (8.0, 9.0, 10.0):
        b.mapper.grounded_landmarks = [
            {"name": "sofa", "role": "destination_marker", "distance": d}]
        b._track_landmark_progress("S1")
    check(b.mapper.grounded_landmarks[0]["state"] == "seen",
          "从未靠近过 -> 仍是 seen，不是 passed")


def test_rejected_waypoints():
    """被淘汰的候选 waypoint 要带原因记录下来，并且只画在调试帧上。"""
    print("\n[2q] 被过滤 waypoint 的调试层")
    src = agent_source()

    pre = src[src.index("def _preprocessing_module"):]
    pre = pre[:pre.index("def decide_waypoint")]
    for reason in ("edge", "occlusion", "distance", "no_path", "behind"):
        check(f"'reason': '{reason}'" in pre, f"记录了 {reason} 原因的淘汰点")
    check("self.rejected_waypoints = []" in pre, "每次预处理重置列表")

    ren = src[src.index("# ---------------- 被淘汰的候选 waypoint"):]
    ren = ren[:ren.index("outline_thickness")]
    check("if debug_overlay and rejected_waypoints" in ren, "只在调试帧上画")
    check(all(k in ren for k in ("edge", "occlusion", "distance", "no_path", "behind")),
          "五种原因都有配色")
    check("counts" in ren, "图例带各原因计数")
    # 图例必须画在裁剪之后，否则会被 16m 裁剪窗口整个切掉
    post = src[src.index("被淘汰 waypoint 的图例（裁剪之后"):]
    post = post[:post.index("子任务进度条")]
    check("top_down_map_cropped" in post, "图例画在裁剪后的图上")


def test_order_gate():
    """
    时序约束：指令说"经过 X 再到 Y"，X 从没靠近过就不接受 Y 已完成。
    这是系统里第一条真正的时序约束——此前时间维度只有时长预算(一个超时计数器)。
    """
    print("\n[2r] _challenge_out_of_order —— 时序约束")
    ns = {"np": np, "normalize_category_name": normalize_category_name,
          "resolve_area": resolve_area, "resolve_category": resolve_category,
          "GENERIC_AREA_TYPES": GENERIC_AREA_TYPES}
    exec(agent_block(r"^    def _verify_local_end\("), ns)
    exec(agent_block(r"^    def _challenge_out_of_order\("), ns)

    class A(HyperDefaults):
        pass
    A._challenge_out_of_order = ns["_challenge_out_of_order"]
    A._verify_local_end = ns["_verify_local_end"]
    A.enable_interception = A.enable_guidance = True

    class FM:
        grounded_landmarks = []

    def mk(lms, local_end=None):
        a = A(); a.mapper = FM(); a.mapper.grounded_landmarks = lms
        a.mapper.object_entities = []
        a.mapper.current_position = np.array([0.0, 0.0, 1.2])
        a.mapper.local_space = {"state": "room"}
        a.mapper.current_room = lambda: None
        a.stop_challenge = ""
        a.instruction_obj = FakeInstruction(n=3, idx=1)
        a.instruction_obj.sub_instruction_dict["SUBTASK_2"]["local_end"] = local_end
        return a

    wp_unvisited = {"name": "dining table", "role": "waypoint_marker",
                    "distance": 6.2, "min_dist": 5.8, "ever_near": False}
    wp_visited = dict(wp_unvisited, ever_near=True)
    dest = {"name": "living room", "role": "destination_marker",
            "distance": 1.0, "ever_near": True}

    a = mk([wp_unvisited, dest])
    check(a._challenge_out_of_order() is True, "途经点从没靠近过 -> 拦截")
    check("NOT DONE YET" in a.stop_challenge, "质询文本写入")
    check("dining table" in a.stop_challenge, "点名了没去的途经点")
    check(a._challenge_out_of_order() is False, "同一子任务不重复拦截")

    b = mk([wp_visited, dest])
    check(b._challenge_out_of_order() is False, "途经点已经去过 -> 放行")

    # 防死锁：未接地的途经点不参与判定。
    # 否则 landmark 一辈子检测不到的子任务会永远完不成，而且是静默卡住。
    c = mk([dest])
    check(c._challenge_out_of_order() is False, "没有已接地的途经点 -> 不拦（防死锁）")

    d = mk([])
    check(d._challenge_out_of_order() is False, "什么都没接地 -> 不拦")

    # avoid_marker 不参与顺序判定（它本来就不该被靠近）
    e = mk([{"name": "wet floor", "role": "avoid_marker",
             "distance": 8.0, "ever_near": False}, dest])
    check(e._challenge_out_of_order() is False, "avoid_marker 不参与顺序判定")

    # 途经点之间的先后颠倒："go past the table then past the sofa"
    first = {"name": "table", "role": "waypoint_marker", "distance": 7.0,
             "min_dist": 6.5, "ever_near": False}
    second = {"name": "sofa", "role": "waypoint_marker", "distance": 1.0,
              "min_dist": 0.8, "ever_near": True}
    f = mk([first, second, dest])
    check(f._challenge_out_of_order() is True, "先后颠倒 -> 拦截")
    check("before" in f.stop_challenge, "质询文本说明了颠倒关系")

    # 顺序正确就不该拦
    g = mk([dict(first, ever_near=True), second, dest])
    check(g._challenge_out_of_order() is False, "按顺序经过 -> 放行")


def test_local_end_verification():
    """local_end 从"抄进prompt"升级成"对照地图事实校验"。"""
    print("\n[2t] _verify_local_end —— 终止条件真校验")
    ns = {"np": np, "normalize_category_name": normalize_category_name,
          "resolve_area": resolve_area, "resolve_category": resolve_category,
          "GENERIC_AREA_TYPES": GENERIC_AREA_TYPES}
    exec(agent_block(r"^    def _verify_local_end\("), ns)

    class A(HyperDefaults):
        pass
    A._verify_local_end = ns["_verify_local_end"]

    def mk(local_end, room=None, space="room", objs=()):
        a = A()

        class M:
            pass
        a.mapper = M()
        a.mapper.current_position = np.array([0.0, 0.0, 1.2])
        a.mapper.local_space = {"state": space}
        a.mapper.current_room = lambda: ({"area_type": room} if room else None)
        a.mapper.object_entities = [
            {"class_name": c, "center": np.array([d, 0.0, 0.5])} for c, d in objs]
        a.instruction_obj = FakeInstruction(n=2, idx=0)
        a.instruction_obj.sub_instruction_dict["SUBTASK_1"]["local_end"] = local_end
        return a

    # 抽不出可核对原子 -> unknown，绝不阻拦
    check(mk("somewhere further on")._verify_local_end()[0] == "unknown",
          "抽不出原子 -> unknown（不阻拦）")
    check(mk(None)._verify_local_end()[0] == "unknown", "没有 local_end -> unknown")
    check(mk("unknown")._verify_local_end()[0] == "unknown", "占位符 -> unknown")

    # 区域原子
    check(mk("just inside the hallway", space="corridor")._verify_local_end()[0] == "ok",
          "'inside the hallway' + 走廊判据 -> ok")
    check(mk("just inside the hallway", space="room")._verify_local_end()[0] == "violated",
          "同样的条件但人在开阔房间 -> violated")
    check(mk("Inside the bedroom area", room="bedroom")._verify_local_end()[0] == "ok",
          "'bedroom area' + current_room 匹配 -> ok")

    # 物体原子
    check(mk("At the closet", objs=(("television", 0.5),))._verify_local_end()[0] != "ok",
          "终止条件里的物体没在身边 -> 不判 ok")
    check(mk("near the television", objs=(("television", 1.2),))._verify_local_end()[0] == "ok",
          "'near the television' 且 1.2m 内有电视 -> ok")
    check(mk("near the television", objs=(("television", 9.0),))._verify_local_end()[0] == "violated",
          "电视在 9m 外 -> violated")

    # 任意一个原子满足即 ok：终止条件常同时提房间和物体，从严会大量误拦
    st, dt = mk("living room area near the television", room="living_room",
                objs=(("television", 9.0),))._verify_local_end()
    check(st == "ok", "房间对上但物体没对上 -> 仍判 ok（宽松方向）", f"{st} {dt}")

    # ---- 无法确认 ≠ 确认为否 ----
    # 实跑里 30 次拦截有 23 次栽在这上面：房间推断没攒够证据、current_room
    # 返回 None，被当成"不在卧室"拦下，而且大多拦完下一步就放行，纯属噪音。
    st, _ = mk("Inside the bedroom area", room=None)._verify_local_end()
    check(st == "unknown", "认不出当前房间 -> unknown，不是 violated", st)
    st, _ = mk("Inside the bedroom area", room="kitchen")._verify_local_end()
    check(st == "violated", "明确在别的房间 -> 这才是 violated", st)

    st, _ = mk("near the television", objs=())._verify_local_end()
    check(st == "unknown", "该类物体压根没检测到 -> unknown", st)
    st, _ = mk("near the television", objs=(("television", 9.0),))._verify_local_end()
    check(st == "violated", "物体在图上但很远 -> violated", st)

    # 泛化区域不作为原子：人在卧室里本来就是在"房间"里
    st, dt = mk("Just inside the room after the door", room="bedroom")._verify_local_end()
    check(st != "violated", "'the room' 不该和 bedroom 判矛盾", f"{st} {dt}")
    st, _ = mk("wait in the room", room=None)._verify_local_end()
    check(st == "unknown", "只提泛化 room 时抽不出原子 -> unknown", st)
    # 但具体房间仍然照常校验
    st, _ = mk("inside the bedroom", room="living_room")._verify_local_end()
    check(st == "violated", "具体房间对不上仍判 violated", st)

    # 走廊：局部几何明确判成开阔房间才算反证，unknown 不算
    st, _ = mk("just inside the hallway", space="unknown")._verify_local_end()
    check(st == "unknown", "空间形态未知时不判 violated", st)


def test_dead_fields_wired():
    """分解器产出但长期没人读的字段，现在应该都接上了。"""
    print("\n[2s] 分解器死数据接线")
    src = agent_source()

    check("'boundary_consistent': sub_inst.get('boundary_consistent')" in src,
          "boundary_consistent 存进了 Instruction")
    check("boundary_consistent') is False" in src,
          "boundary_consistent 参与完成判定的宽严")
    check("Completion condition for" in src, "local_end 进了 prompt")
    check("The stated completion condition" in src, "local_end 进了质询文本")
    check("ARRIVED?" in src, "主动到达提示存在")

    # 到达提示必须独立于 action==-1：从不停的 episode 正是靠它兜底
    seg = src[src.index("arrive_str = \"\""):]
    seg = seg[:seg.index("{after_stop_str")] if "{after_stop_str" in seg else seg[:4000]
    check("destination_marker" in seg and "NEAR_M" in seg,
          "到达提示看的是终点距离，不依赖 VLM 是否想停")


def test_compute_frontiers():
    """
    前沿检测。注意 frontier_pcd 这个字段项目里本来就有、渲染层也画、prompt 图例
    还写着，但计算整段是注释掉的——一直是空点云。这里是补上之后的验证。
    """
    print("\n[2u] compute_frontiers —— 已知/未知边界")
    mapper_py = os.path.join(SRC, "mapper.py")
    ns = {"np": np, "cv2": cv2}
    for pat in (r"^    def _build_free_grid\(", r"^    def compute_frontiers\("):
        exec(extract_block(mapper_py, pat), ns)
    StubMapper._build_free_grid = ns["_build_free_grid"]
    StubMapper.compute_frontiers = ns["compute_frontiers"]
    for k, v in (("ROOM_GRID_RES", 0.05), ("ROOM_CLOSE_M", 0.35),
                 ("FRONTIER_MIN_CELLS", 12), ("FRONTIER_REBUILD_EVERY", 4)):
        setattr(StubMapper, k, v)

    # 一条走廊向 -y 延伸，两侧有墙，远端没有墙 -> 远端应该是前沿
    m = StubMapper()
    m.navigable_pcd = StubPCD(_floor([(-0.9, -8.0, 0.9, 1.0)]))
    m.obstacle_pcd = StubPCD(_walls([((-1.0, -8.0), (-1.0, 1.0)),
                                     ((1.0, -8.0), (1.0, 1.0))], z=0.8))
    m.frontiers = []
    fr = m.compute_frontiers(force=True)
    check(len(fr) > 0, "走廊尽头被识别为前沿", f"得到 {len(fr)} 个")
    if fr:
        ys = [f['center'][1] for f in fr]
        check(min(ys) < -6.0 or max(ys) > 0.0, "前沿位于走廊的开口端",
              f"y={[round(y,1) for y in ys]}")

    # 四面围墙的封闭房间：不该有前沿
    m2 = StubMapper()
    m2.navigable_pcd = StubPCD(_floor([(-2.0, -2.0, 2.0, 2.0)]))
    m2.obstacle_pcd = StubPCD(_walls([((-2.2, -2.2), (2.2, -2.2)), ((2.2, -2.2), (2.2, 2.2)),
                                      ((2.2, 2.2), (-2.2, 2.2)), ((-2.2, 2.2), (-2.2, -2.2))],
                                     z=0.8))
    m2.frontiers = []
    check(len(m2.compute_frontiers(force=True)) == 0, "封闭房间没有前沿")

    # 没有地板时安全返回
    m3 = StubMapper(); m3.frontiers = []
    check(m3.compute_frontiers(force=True) == [], "没有可通行点云时返回空")


def test_frontier_scoring():
    """空间项 × 时间项。这是全系统唯一一处两者相乘的决策点。"""
    print("\n[2v] _score_frontiers —— 时空联合打分")
    ns = {"np": np, "AREA_PROXY_OBJECTS": AREA_PROXY_OBJECTS,
          "normalize_category_name": normalize_category_name,
          "resolve_category": resolve_category}
    exec(agent_block(r"^    def _score_frontiers\("), ns)

    class A(HyperDefaults):
        pass
    A._score_frontiers = ns["_score_frontiers"]
    A.FRONTIER_MIN_USEFUL_M = 2.0
    A.FRONTIER_STEP_LEN, A.FRONTIER_TOP_K = 0.6, 3

    def mk(frontiers, areas=(), landmarks=(), used=5, expected=30.0):
        a = A()

        class M:
            pass
        a.mapper = M()
        a.mapper.frontiers = [dict(f) for f in frontiers]
        a.mapper.inferred_areas = [{"area_type": t, "center": np.asarray(c), "score": 1.5,
                                    "evidence": []} for t, c in areas]
        a.mapper.object_entities = []

        def _hyps(_a=a, weak=()):
            out = [{"area_type": x["area_type"], "center": np.asarray(x["center"]),
                    "score": x["score"], "confirmed": True, "evidence": []}
                   for x in _a.mapper.inferred_areas]
            out += list(_a.mapper._weak)
            return out

        a.mapper._weak = []
        a.mapper.area_hypotheses = _hyps
        a.mapper.grounded_landmarks = []
        a.mapper.unmatched_landmarks = []
        a.mapper.current_position = np.array([0.0, 0.0, 1.2])
        a.mapper.current_rotation = HABITAT_ROT_IDENTITY.copy()
        a.instruction_obj = FakeInstruction(used=used, expected=expected)
        a.instruction_obj.sub_instruction_dict["SUBTASK_1"]["landmarks"] = [
            {"name": n, "role": "destination_marker"} for n in landmarks]
        a._current_budget_row = lambda: {"expected_steps": expected, "used_steps": used,
                                         "ratio": used / expected}
        return a

    F_near_kitchen = {"center": np.array([0.0, -4.0]), "width_m": 1.0,
                      "n_cells": 20, "distance": 4.0}
    F_far_other = {"center": np.array([8.0, 2.0]), "width_m": 3.0,
                   "n_cells": 60, "distance": 8.2}

    # 目标是冰箱 -> 属于 kitchen -> 地图上 kitchen 在 (0,-6) -> 朝那边的前沿该赢
    a = mk([F_near_kitchen, F_far_other], areas=[("kitchen", (0.0, -6.0))],
           landmarks=["refrigerator"])
    sf = a._score_frontiers()
    check(sf and np.allclose(sf[0]['center'], F_near_kitchen['center']),
          "朝着已识别厨房的前沿排第一")
    check("kitchen" in sf[0]['why'], "理由里说明了是朝厨房去的", sf[0]['why'])

    # 没有任何房间线索时退化成"宽的优先"——这个退化必须是诚实的
    b = mk([F_near_kitchen, F_far_other], areas=[], landmarks=["refrigerator"])
    sb = b._score_frontiers()
    check(sb and sb[0]['width_m'] == 3.0, "无房间线索时宽的前沿优先")
    check("opening" in sb[0]['why'], "理由里说明是按开口宽度选的", sb[0]['why'])

    # 时间项：预算几乎耗尽时，远处的前沿必须被压下去
    c = mk([F_far_other], areas=[], landmarks=["refrigerator"], used=58, expected=30.0)
    sc = c._score_frontiers()
    check(sc and sc[0]['temporal'] < 0.9, "预算将尽时远处前沿被时间项压低",
          f"temporal={sc[0]['temporal']:.2f}")
    check("budget only covers" in sc[0]['why'], "理由里点明是预算不够")

    d = mk([F_far_other], areas=[], landmarks=["refrigerator"], used=1, expected=60.0)
    check(d._score_frontiers()[0]['temporal'] == 1.0, "预算充足时时间项不惩罚")

    # 分数确实是两项相乘
    e = mk([F_near_kitchen], areas=[("kitchen", (0.0, -6.0))], landmarks=["refrigerator"])
    se = e._score_frontiers()[0]
    check(abs(se['score'] - se['spatial'] * se['temporal']) < 1e-9,
          "score == spatial × temporal")

    # 已经接地的目标不再驱动探索
    f = mk([F_near_kitchen, F_far_other], areas=[("kitchen", (0.0, -6.0))],
           landmarks=["refrigerator"])
    f.mapper.grounded_landmarks = [{"name": "refrigerator"}]
    sfr = f._score_frontiers()
    check(sfr[0]['width_m'] == 3.0, "目标已找到 -> 退回宽度优先，不再往厨房挤")

    check(mk([])._score_frontiers() == [], "没有前沿时返回空")

    # ---- 弱证据：这是"先验只在房间已识别后才可用"那个死结的解法 ----
    # 实测旧版空间项 139 次全部落空、0 次命中：它要求目标房间已被确认，
    # 而一旦确认就不需要引导了。现在瞥见一个水槽也能给方向。
    g = mk([F_near_kitchen, F_far_other], areas=[], landmarks=["refrigerator"])
    g.mapper._weak = [{"area_type": "kitchen", "center": np.array([0.0, -6.0]),
                       "score": 0.3, "confirmed": False, "evidence": ["sink"]}]
    sg = g._score_frontiers()
    check(np.allclose(sg[0]['center'], F_near_kitchen['center']),
          "只凭一个水槽的弱证据也能定方向")
    check(sg[0].get('informed') is True, "弱证据算作 informed（会真的给出建议）")
    check("weak evidence" in sg[0]['why'] and "sink" in sg[0]['why'],
          "理由里如实标明是弱证据及其来源", sg[0]['why'])

    # 强弱要分档：确认的房间应该压过瞥见一眼的线索
    h = mk([F_near_kitchen], areas=[("kitchen", (0.0, -6.0))], landmarks=["refrigerator"])
    strong = h._score_frontiers()[0]['spatial']
    check(strong > sg[0]['spatial'],
          "同样位置，已确认房间的权重高于弱证据",
          f"strong={strong:.3f} weak={sg[0]['spatial']:.3f}")

    # 弱到不能再弱的证据不该盖过距离因素
    i = mk([F_near_kitchen, F_far_other], areas=[], landmarks=["refrigerator"])
    i.mapper._weak = [{"area_type": "kitchen", "center": np.array([9.0, 3.0]),
                       "score": 0.3, "confirmed": False, "evidence": ["sink"]}]
    si = i._score_frontiers()
    check(si[0]['spatial'] < 0.35, "远处的弱证据不会产生高分",
          f"{si[0]['spatial']:.3f}")


def test_area_hypotheses():
    """弱房间假设：单个特征物体也能作为方向线索。"""
    print("\n[2z] area_hypotheses —— 弱证据房间假设")
    ns = {"np": np, "normalize_category_name": normalize_category_name,
          "AREA_PROXY_OBJECTS": AREA_PROXY_OBJECTS}
    exec(extract_block(os.path.join(SRC, "mapper.py"), r"^    def area_hypotheses\("), ns)
    StubMapper.area_hypotheses = ns["area_hypotheses"]

    m = StubMapper()
    m.inferred_areas = []
    m.object_entities = [make_entity("sink", [3.0, -2.0, 0.6])]     # kitchen 0.3 / bathroom 0.3
    h = m.area_hypotheses()
    types = {x["area_type"] for x in h}
    check("kitchen" in types and "bathroom" in types,
          "水槽同时是厨房和卫生间的弱线索（歧义如实保留）", str(types))
    check(all(not x["confirmed"] for x in h), "都标记为未确认")
    check(all(x["score"] < 1.0 for x in h), "权重低于确认房间")

    # 权重太低的 proxy 不出假设（mirror 0.2 < min_weight 0.25）
    m2 = StubMapper()
    m2.inferred_areas = []
    m2.object_entities = [make_entity("mirror", [1.0, -1.0, 1.0])]
    check(m2.area_hypotheses() == [], "过弱的 proxy 不产生假设")

    # 已确认房间内部的同类物体不重复出弱假设
    m3 = StubMapper()
    m3.inferred_areas = [{"area_type": "kitchen", "center": np.array([3.0, -2.0]),
                          "score": 2.3, "evidence": [("stove", 1.0)]}]
    m3.object_entities = [make_entity("sink", [3.2, -2.1, 0.6])]
    h3 = m3.area_hypotheses()
    kitchens = [x for x in h3 if x["area_type"] == "kitchen"]
    check(len(kitchens) == 1 and kitchens[0]["confirmed"],
          "确认房间内的同类物体不重复产生假设", str([(x['area_type'], x['confirmed']) for x in h3]))
    # 但它对别的房间类型仍然是线索（水槽也可能意味着卫生间）
    check(any(x["area_type"] == "bathroom" for x in h3), "对其它房间类型仍保留线索")


def test_frontier_scored_before_map():
    """
    前沿打分必须早于建图。

    实跑时踩过：decide_waypoint 里建图那行直接读 self._last_scored_frontiers，
    而该属性只在 generate_prompt 里赋值，建图又在 generate_prompt 之前——
    第一步必然 AttributeError。而 run_experiments 把 traceback 吞了，表现成
    "每个 episode 跑到第 1 步就悄悄换下一个"，29 个全废。
    """
    print("\n[2w] 前沿打分与建图的先后")
    src = agent_source()

    blk = agent_block(r"^    def decide_waypoint\(")
    i_score = blk.index("self._last_scored_frontiers = self._score_frontiers()")
    i_map = blk.index("top_down_map_np = create_top_down_map_centered")
    i_prompt = blk.index("prompt = self.generate_prompt")
    check(i_score < i_map, "打分在建图之前")
    check(i_map < i_prompt, "建图仍在 generate_prompt 之前（原有顺序未变）")

    # 属性必须在 __init__ 里就存在，不能依赖某个方法先跑过
    init = agent_block(r"^    def __init__\(self, sim_wrapper")
    check("self._last_scored_frontiers = None" in init, "__init__ 里初始化了打分结果")
    check("self.rejected_waypoints = []" in init, "__init__ 里初始化了淘汰点列表")

    # 异常不能再被静默吞掉
    rex = open(os.path.join(SRC, "run_experiments.py"), encoding="utf-8").read()
    check(rex.count("[episode] EXCEPTION") >= 4, "episode 级异常会打印 traceback")
    check("tb = traceback.format_exc()\n                    pass" not in rex,
          "不再有 format_exc 之后直接 pass 的写法")


def test_explore_suppressed_when_arrived():
    """
    目标已找到时不得再给探索建议。

    实测：_score_frontiers 在目标全部接地后会退化成"哪个开口最大往哪走"，
    却仍顶着"ranked by how likely they lead to what this subtask needs"送进
    prompt。同一批 9 个 episode 里 4 个曾进到目标 1.5m 内，最后全停在刚超出
    3m 阈值处，行程比上一版多 39%，SR 从 3/9 掉到 1/9。
    引导和到达是互斥信号，不该同时出现在 prompt 里。
    """
    print("\n[2x] 到达后抑制探索建议")
    src = agent_source()
    seg = src[src.index("explore_str = \"\""):]
    seg = seg[:seg.index("# 主动到达提示")]

    check("dest_found" in seg, "计算了'终点是否已接地且在附近'")
    check("destination_marker" in seg, "只看 destination_marker")
    check("if dest_found" in seg and "self._last_scored_frontiers = None" in seg,
          "已到达时清空打分结果（地图上也不再画 F1/F2/F3）")
    i_flag = seg.index("dest_found = any(")
    i_use = seg.index("sf = None if dest_found")
    check(i_flag < i_use, "先判断再决定是否给建议")

    # 两个提示都要能在日志里看到，否则无法判断有没有触发
    check("[hint] ARRIVED" in src, "到达提示会打日志")
    check("[hint] EXPLORE" in src, "探索建议会打日志")


def test_ablation_switches():
    """
    消融开关。连续两轮改动都让指标变差，而每次都只能猜是哪一层的锅，
    所以把"关掉某一层再跑"做成一条命令。
    """
    print("\n[2y] 消融开关")
    src = agent_source()

    check("STZS_GUIDANCE" in src and "STZS_INTERCEPTION" in src, "两个环境变量都支持")
    check("enable_guidance" in src and "enable_interception" in src, "两个开关字段都存在")
    check("[config] guidance=" in src, "启动时把配置打进日志（否则事后分不清哪次是哪个模式）")

    # 三个拦截入口都要受开关控制，漏一个消融就不干净
    for fn in ("_challenge_out_of_order", "_is_premature_completion",
               "_challenge_receded_stop"):
        i = src.index(f"def {fn}(")
        head = src[i:i + 2000]
        check("if not self.enable_interception" in head, f"{fn} 受拦截开关控制")

    # 引导层：关掉时不能只是不打印，打分结果也要清空，否则地图上还画 F 环
    seg = src[src.index('explore_str = ""'):]
    seg = seg[:seg.index("# 主动到达提示")]
    check("if not self.enable_guidance" in seg, "引导层受开关控制")
    check("self._last_scored_frontiers = None" in seg, "关掉时同时清空打分（地图不画 F 环）")

    # 没有房间先验时应当沉默，而不是给"哪个洞最大"的默认建议
    check("informed" in src, "打分结果标注了是否用到了指令信息")
    check("EXPLORE suppressed" in src, "无先验时抑制建议并留痕")


def test_frontier_as_action():
    """
    前沿必须是**可选动作**，不能只是 prompt 里的描述。

    实测动机：推荐前沿距离中位数 5m，而 max_move_distance=3m，67% 的建议
    在动作空间里没有对应候选点；同时 40% 的候选点因"距离"被丢弃。系统一边
    说"往那边走 5 米"，一边只给 3 米内的点选，agent 只能原地打转。
    """
    print("\n[2A] 前沿作为可选动作")
    src = agent_source()

    blk = src[src.index("# ---- 把前沿变成可选动作 ----"):]
    blk = blk[:blk.index("top_down_map_np = create_top_down_map_centered")]
    check("actions[f\"F{i + 1}\"]" in blk, "F1/F2/F3 被注入 actions 字典")
    check("'type': 'frontier'" in blk, "动作类型标为 frontier")
    check("f.get('point', f['center'])" in blk,
          "用可导航代表点而不是质心（弧形前沿的质心可能在墙里）")
    check("if self.enable_frontier_action" in blk,
          "受独立的前沿动作开关控制（和'建议'拆开，三臂消融才分得清）")
    # 动作没开时不能在 prompt 里说"可以选 F1"，否则模型会输出不存在的动作、
    # 触发重试白烧一次调用
    gp = src[src.index('if self.enable_frontier_action:\n                    head ='):]
    gp = gp[:gp.index('explore_str = head')]
    check("for reference" in gp, "动作关闭时改成'仅供参考'的措辞")
    check("self.last_actions = actions" in blk, "更新 last_actions，渲染层能看到")

    # 执行层要认这个类型，否则选了也走不了
    step = src[src.index("elif decision['type'] in ('waypoint', 'frontier')"):]
    step = step[:step.index("elif decision['type'] == 'rollback'")]
    check("plan_path_to_target" in step, "frontier 走和 waypoint 相同的 A* 规划")

    # prompt 要明确告诉模型这是可以选的，否则它只会当参考信息
    check('"action": "F1"' in src, "prompt 里给出了选择 F1 的具体写法")
    check("do NOT pick an F option" in src, "已到达时提示不要再选 F")

    # 代表点必须真的在前沿格上
    mapper_src = open(os.path.join(SRC, "mapper.py"), encoding="utf-8").read()
    cf = mapper_src[mapper_src.index("def compute_frontiers"):]
    cf = cf[:cf.index("def _region_at")]
    check("'point':" in cf, "compute_frontiers 输出可导航代表点")
    check("argmin" in cf, "代表点取离质心最近的前沿格")


def test_metric_invariant_and_flags():
    """
    指标恒不变式 + 开关从配置文件读取。
    """
    print("\n[2B] 指标恒不变式与配置开关")
    rex = open(os.path.join(SRC, "run_experiments.py"), encoding="utf-8").read()

    # oracle_spl 恒 >= spl：必须靠结构保证，不能靠两处表达式手工保持一致
    seg = rex[rex.index("spl_now = 0.0"):]
    seg = seg[:seg.index("metrics['oracle_navigation_error']")]
    check("self.oracle_best_spl = max(self.oracle_best_spl, spl_now)" in seg,
          "oracle_best_spl 由 spl_now 更新")
    check("metrics['spl'] = spl_now" in seg, "spl 直接用同一个 spl_now，不重算")
    check(seg.count("geodesic_path) / max(") == 1,
          "同一个式子只出现一次（重复书写正是不一致的来源）")

    check("oracle_spl < spl" in rex, "聚合时逐 episode 校验恒不变式")
    check('summary["counts"]' in rex, "输出各指标的样本数（分母不同则不可直接比较）")

    # 开关走 config 文件
    check('config.get("features", {})' in rex, "从 config 的 features 段读开关")
    import yaml
    cfg = yaml.safe_load(open(os.path.join(SRC, "..", "config", "vlnce_test.yaml"),
                              encoding="utf-8"))
    feat = cfg.get("features", {})
    for k in ("enable_guidance", "enable_frontier_action", "enable_interception",
              "enable_vlm_detection", "budget_ratio"):
        check(k in feat, f"配置文件里有 {k}")
    check(feat.get("enable_vlm_detection") == 0, "VLM 检测默认关闭（投入产出比为负）")

    # 关掉时不能只是不回填——schema 和 system prompt 里都不该再提，
    # 否则模型照样生成一堆用不上的框，纯浪费输出 token
    # 拆包之后这几处散在 perception/mapping.py 和 planning/prompt.py 里，
    # 断言关心的是"整个 agent 有没有这行逻辑"，所以按全包源码查
    ag = agent_source()
    check("if not self.enable_vlm_detection:\n                return" in ag,
          "关闭时不回填检测结果")
    check("det_schema = \"\"" in ag, "关闭时 JSON schema 里不出现 detections 字段")
    check("_DETECTION_SIDE_TASK" in ag and "if self.enable_vlm_detection else \"\"" in ag,
          "关闭时 system prompt 里不出现检测副任务")
    check(ag.count("Object detection side-task") == 1,
          "检测副任务只在常量里定义一处")



def test_hyper_available_before_mapping():
    """
    所有超参必须在首帧建图之前就可读。

    实跑踩过：_apply_hyper 被放在消融开关之后（也就是首帧 update_map 之后），
    第一步就报 'GPTAgent' object has no attribute 'LM_RECEDE_MARGIN'。而地标
    接地外面包着 except，异常被降级成一行 "skipped due to error"，首帧接地
    静默失效——SR 不会归零，只是悄悄变差，这种 bug 最难查。

    两道防线，这里都要验：
      1. 每个超参在所属 mixin 上有类级默认（缺属性时退化成默认值，不崩）
      2. __init__ 里 _apply_hyper() 的调用早于 update_map()（配置真正生效）
    """
    print("\n[2D] 超参可用性与初始化顺序")

    HYPER = {
        "perception/grounding.py": ["NEAR_M", "PASSED_MIN_M", "PASSED_GAP_M",
                                    "LM_RECEDE_MARGIN", "LM_RECEDE_STEPS"],
        "planning/frontier.py": ["FRONTIER_MIN_USEFUL_M", "FRONTIER_STEP_LEN",
                                 "FRONTIER_TOP_K"],
        "planning/verification.py": ["PREMATURE_MIN_RATIO", "PREMATURE_SHAKY_RATIO",
                                     "VERIFY_NEAR_M", "TOO_FAR_OBJECT_LIMIT",
                                     "RECEDE_MARGIN", "FORCE_ADVANCE_RATIO",
                                     "FORCE_ADVANCE_MIN_STEPS", "ROLLBACK_STEP_THRESHOLD"],
    }
    for rel, names in HYPER.items():
        mod = open(os.path.join(AGENT_DIR, rel), encoding="utf-8").read()
        missing = [n for n in names if not re.search(rf"^    {n}\s*=", mod, re.M)]
        check(not missing, f"{rel} 的类级默认齐全" + (f"（缺 {missing}）" if missing else ""))

    # _apply_hyper 必须覆盖到全部超参
    body = agent_block(r"^    def _apply_hyper\(")
    allnames = [n for names in HYPER.values() for n in names]
    unset = [n for n in allnames if f"self.{n} = " not in body]
    check(not unset, "_apply_hyper 覆盖全部超参" + (f"（漏 {unset}）" if unset else ""))

    # 调用顺序：_apply_hyper 早于首帧 update_map。
    # 必须先剥掉注释——__init__ 里正好有一段注释在解释这个顺序，里面
    # 引用了 self.update_map(obs) 这个字面串，直接 index 会命中注释。
    init = agent_block(r"^    def __init__\(self, sim_wrapper")
    code = "\n".join(l.split("#", 1)[0] for l in init.splitlines())
    i_hyper = code.index("self._apply_hyper(config)")
    i_map = code.index("self.update_map(obs)")
    check(i_hyper < i_map, "_apply_hyper() 在首帧 update_map() 之前调用")

    # 每个超参的类级默认要和 yaml 里的值一致，否则测试测的不是线上行为
    import yaml as _yaml
    cfg = _yaml.safe_load(open(os.path.join(SRC, "..", "config", "vlnce_test.yaml"),
                               encoding="utf-8"))
    groups = {"landmark": "perception/grounding.py", "frontier": "planning/frontier.py",
              "subtask": "planning/verification.py"}
    mismatched = []
    for grp, rel in groups.items():
        mod = open(os.path.join(AGENT_DIR, rel), encoding="utf-8").read()
        for key, val in (cfg.get("hyper", {}).get(grp, {}) or {}).items():
            # yaml 的键名 → 代码里的常量名。landmark 组和 subtask 组都有
            # recede_margin，但对应两个不同的常量，所以要连组名一起查。
            alias = {("landmark", "recede_margin"): "LM_RECEDE_MARGIN",
                     ("landmark", "recede_steps"): "LM_RECEDE_STEPS",
                     ("subtask", "recede_margin"): "RECEDE_MARGIN"}
            const = alias.get((grp, key),
                              ("FRONTIER_" if grp == "frontier" else "") + key.upper())
            m = re.search(rf"^    {const}\s*=\s*([0-9.]+)", mod, re.M)
            if m and abs(float(m.group(1)) - float(val)) > 1e-9:
                mismatched.append(f"{const}: 代码 {m.group(1)} vs yaml {val}")
    check(not mismatched, "类级默认与 yaml 取值一致" + (f"（{mismatched}）" if mismatched else ""))



def test_init_hooks_in_right_method():
    """
    初始化钩子必须落在 __init__ 里，而不是 reset() 里。

    实测踩过：mapper 的 __init__ 和 reset() 末尾各有一句 _switch_to_floor(0,0)，
    按上下文做字符串替换时匹配到了 reset() 那句，于是 _apply_hyper 被塞进
    reset()。后果不是崩，而是超参延迟到第一次 reset 才生效，并且每 reset
    一次就重打一遍日志——一个 episode 打两遍，日志里数一下就能发现。

    这类错误靠语法检查和 import 测试都抓不到，只能按 AST 定位方法归属。
    """
    print("\n[2E] 初始化钩子的方法归属")
    import ast as _ast

    mp = os.path.join(SRC, "mapper.py")
    tree = _ast.parse(open(mp, encoding="utf-8").read())
    rng = {}
    for n in _ast.walk(tree):
        if isinstance(n, _ast.ClassDef) and n.name == "Instruct_Mapper":
            for m in n.body:
                if isinstance(m, _ast.FunctionDef):
                    rng[m.name] = (m.lineno, m.end_lineno)

    calls = [i + 1 for i, l in enumerate(open(mp, encoding="utf-8").read().splitlines())
             if "self._apply_hyper(" in l]
    check(len(calls) == 1, f"_apply_hyper 只被调用一处（实际 {len(calls)} 处）")
    owner = next((k for k, (a, b) in rng.items() if a <= calls[0] <= b), None) if calls else None
    check(owner == "__init__", f"_apply_hyper 在 __init__ 里调用（实际在 {owner}）")

    # 而且要在 _switch_to_floor 之前——后者会读区域相关阈值
    if owner == "__init__":
        a, b = rng["__init__"]
        body = open(mp, encoding="utf-8").read().splitlines()[a - 1:b]
        code = [l.split("#", 1)[0] for l in body]
        i_h = next(i for i, l in enumerate(code) if "self._apply_hyper(" in l)
        i_s = next((i for i, l in enumerate(code) if "self._switch_to_floor(" in l), 10 ** 9)
        check(i_h < i_s, "_apply_hyper 早于 _switch_to_floor")


def test_no_silent_episode_failure():
    """
    episode 级异常一律要打印 traceback，一处都不能漏。

    _initialize_episode()（构造 agent、建 mapper、跑首帧 update_map）在内层
    try 之外，它抛异常时只有最外层那个 except 能接住。之前那处只把 str(e)
    存进 metrics，终端一个字不打，表现成"某个 episode 连 Step 1 都没有就
    跳过去了"——一个必现的错误能这样悄悄报废整批实验。
    """
    print("\n[2F] episode 级异常不得静默")
    rex = open(os.path.join(SRC, "run_experiments.py"), encoding="utf-8").read()

    check("import traceback" in rex.split("def ")[0],
          "traceback 在模块级 import（否则外层作用域会 NameError）")

    # 每个捕获 episode 级异常的 except 都要有打印
    bad = []
    lines = rex.splitlines()
    for i, l in enumerate(lines):
        if 'finish_status": "error"' not in l and "'finish_status': 'error'" not in l:
            continue
        window = "\n".join(lines[max(0, i - 12):i + 4])
        if "EXCEPTION" not in window and "format_exc" not in window:
            bad.append(i + 1)
    check(not bad, f"所有 error 分支都打印了 traceback（缺失于行 {bad}）")

    check(rex.count("[episode") >= 5, "episode 级异常打印点齐全")



def test_detector_fails_loudly():
    """
    检测器初始化失败必须**每次都抛**，不能留下半成品。

    实测事故：懒加载写成
        _model = YOLOE(path)                       # 先赋值
        _model.set_classes(names, get_text_pe())   # 这里抛异常
    时，_model 已经是个没绑定词表的模型。第一个 episode 崩掉之后，后面每个
    episode 拿到的都是它——照样能 predict，只是什么都检测不出来。日志里
    只有一行 `[detect] front=0`，不报任何错。那一批 42 次检测全是 0，
    地图里一个物体都没有，所有依赖接地的判据全程空转。

    正确语义：要么拿到配好词表的模型，要么每次调用都抛出同一个异常。
    """
    print("\n[2G] 检测器失败必须显式")
    seg_py = os.path.join(SRC, "segmentation", "instance_segmentation.py")
    src = open(seg_py, encoding="utf-8").read()

    body = extract_block(seg_py, r"^def _get_model\(")
    # 赋值给全局 _model 的那一行，必须在 set_classes 之后
    i_set = body.find("set_classes")
    i_pub = body.find("_model = m")
    check(i_pub > i_set > 0, "先 set_classes 成功、再把模型赋给 _model")
    check("_model = YOLOE(" not in body, "不得先把裸模型赋给 _model")

    # 另一个入口（qwen 路径的 YOLOE 备用实现）同样不能先赋值
    q = os.path.join(SRC, "segmentation", "instance_segmentation_qwen.py")
    qbody = extract_block(q, r"^def _get_yoloe_model\(")
    check("_yoloe_model = YOLOE(" not in qbody, "qwen 路径同样不先赋值裸模型")
    check(qbody.find("_yoloe_model = m") > qbody.find("set_classes"),
          "qwen 路径也是 set_classes 成功后才发布")

    # 启动时预热：失败要在第 0 秒退出，而不是跑几小时才发现数据作废
    rex = open(os.path.join(SRC, "run_experiments.py"), encoding="utf-8").read()
    check("[startup] 预热检测器" in rex, "启动时预热检测器")
    check("sys.exit(2)" in rex, "预热失败直接退出，不带病开跑")


def test_challenge_keeps_path():
    """
    拦截不得清空已规划路径。

    动机是纯性能：耗时几乎全在 VLM（单次约 30s，占总时长 90%+），而清空路径
    会让下一步必然重新决策——一次拦截等于付两次调用的钱。实测每次 VLM 调用
    之间的步数中位数只有 2，路径本来就短，再被拦截清掉就更碎。
    """
    print("\n[2C] 拦截时保留路径")
    src = agent_source()
    blk = src[src.index("elif decision['type'] == 'challenge_stop':"):]
    blk = blk[:blk.index("elif decision['type'] == 'turn':")]

    check("self.current_path = []" not in blk, "不再清空 current_path")
    check("_get_action_for_next_waypoint" in blk, "有在途路径时继续沿原路走")
    check("PolarAction.pause()" in blk, "确实没路径了才原地停一步")
    i_path = blk.index("if self.current_path:")
    i_pause = blk.index("PolarAction.pause()")
    check(i_path < i_pause, "先尝试续走，再退化成 pause")


def test_retry_order():
    """
    钉死 decide_waypoint 里的语句顺序：动作合法性校验必须先于任何状态修改。
    真实日志里出现过 "action": "F" 和只有10个候选却选 18，两次都导致
    子任务被推进后又重试、一次决策吃掉两个子任务。
    """
    print("\n[2j] decide_waypoint 语句顺序 —— 非法动作不得污染状态")
    src = agent_source()
    body = src[src.index("action_key = str(parsed_json['action']).upper()"):]
    body = body[:body.index("return actions[action_key]")]

    i_guard = body.index("if action_key not in actions:")
    i_done = body.index("done_flag = parsed_json.get('subtask_done'")
    i_mark = body.index("mark_current_subtask_completed")
    check(i_guard < i_done, "非法动作校验在读取 subtask_done 之前")
    check(i_guard < i_mark, "非法动作校验在推进子任务之前")
    check("continue" in body[i_guard:i_guard + 120], "非法动作走 continue 重试")


def test_avoid_penalty():
    print("\n[2e] _avoid_penalty —— avoid 代价场")

    mapper_py = os.path.join(SRC, "mapper.py")
    ns = {"np": np}
    exec(extract_block(mapper_py, r"^    def get_avoid_zones\("), ns)
    exec(extract_block(mapper_py, r"^    def _avoid_penalty\("), ns)

    class M:
        AVOID_RADIUS = 1.5
        AVOID_WEIGHT = 6.0
        grounded_landmarks = []

    M.get_avoid_zones = ns["get_avoid_zones"]
    M._avoid_penalty = ns["_avoid_penalty"]
    m = M()

    m.grounded_landmarks = [
        {"name": "wet floor", "role": "avoid_marker", "kind": "object",
         "center": np.array([2.0, 0.0, 0.0])},
        {"name": "sofa", "role": "waypoint_marker", "kind": "object",
         "center": np.array([5.0, 0.0, 0.0])},
    ]
    m._avoid_zones_cache = m.get_avoid_zones()

    check(len(m._avoid_zones_cache) == 1, "只有 avoid_marker 进入禁区列表")
    check(m._avoid_penalty(np.array([2.0, 0.0, 0.0])) == 6.0, "圆心处惩罚等于 AVOID_WEIGHT")
    check(abs(m._avoid_penalty(np.array([2.75, 0.0, 0.0])) - 3.0) < 1e-6, "半径一半处惩罚减半")
    check(m._avoid_penalty(np.array([9.0, 0.0, 0.0])) == 0.0, "区域外惩罚为 0")
    check(m._avoid_penalty(np.array([3.5, 0.0, 0.0])) == 0.0, "恰好在半径外惩罚为 0")

    # 区域类 avoid 用自己的半径，不是固定的 AVOID_RADIUS
    m.grounded_landmarks = [
        {"name": "dining room", "role": "avoid_marker", "kind": "area",
         "center": np.array([0.0, 0.0, 0.0]), "radius": 4.0},
    ]
    m._avoid_zones_cache = m.get_avoid_zones()
    check(m._avoid_zones_cache[0][1] == 4.0, "区域类 avoid 使用自身半径")
    check(m._avoid_penalty(np.array([3.0, 0.0, 0.0])) > 0, "4m 半径内仍有惩罚")

    # 没有 avoid 时必须零开销地返回 0（这个函数在A*内层循环里）
    m.grounded_landmarks = []
    m._avoid_zones_cache = m.get_avoid_zones()
    check(m._avoid_penalty(np.array([0.0, 0.0, 0.0])) == 0.0, "无 avoid 时恒为 0")


def test_budget():
    print("\n[2f] get_progress / get_current_step_budget —— 时长预算")
    ns = {"np": np}
    exec(agent_block(r"^    def get_progress\("), ns)
    exec(agent_block(r"^    def get_current_step_budget\("), ns)

    class I:
        pass

    I.get_progress = ns["get_progress"]
    I.get_current_step_budget = ns["get_current_step_budget"]

    inst = I()
    inst.sub_instruction_keys = ["SUBTASK_1", "SUBTASK_2", "SUBTASK_3"]
    inst.num_subtasks = 3
    inst.current_subtask_index = 1
    inst.current_step_count = 30
    inst.sub_instruction_dict = {
        "SUBTASK_1": {"completed": True, "relative_duration": 0.15},
        "SUBTASK_2": {"completed": False, "relative_duration": 0.60},
        "SUBTASK_3": {"completed": False, "relative_duration": 0.25},
    }
    inst.get_current_subtask_key = lambda: inst.sub_instruction_keys[inst.current_subtask_index]

    rows = inst.get_progress(total_step_budget=100)
    st = {r["key"]: r for r in rows}
    check(st["SUBTASK_1"]["status"] == "done", "已完成段标记为 done")
    check(st["SUBTASK_2"]["status"] == "current", "当前段标记为 current")
    check(st["SUBTASK_3"]["status"] == "todo", "未开始段标记为 todo")
    check(abs(st["SUBTASK_2"]["expected_steps"] - 60) < 1e-6,
          "0.60 * 100 = 60 步预算", str(st["SUBTASK_2"]["expected_steps"]))
    check(abs(st["SUBTASK_2"]["ratio"] - 0.5) < 1e-6, "30/60 = 50%")

    # 长子任务拿到比短子任务更大的阈值 —— 这正是替掉硬编码 70 的意义
    b_long = inst.get_current_step_budget(total_step_budget=100)
    inst.current_subtask_index = 0
    b_short = inst.get_current_step_budget(total_step_budget=100)
    check(b_long > b_short, "长子任务阈值 > 短子任务阈值", f"{b_long} vs {b_short}")
    check(b_short >= 25, "短子任务阈值不低于 hard_min", str(b_short))
    check(b_long <= 90, "长子任务阈值不超过 hard_max", str(b_long))

    # relative_duration 缺失/非法时退化成均分，不能崩
    inst.sub_instruction_dict["SUBTASK_1"]["relative_duration"] = None
    inst.sub_instruction_dict["SUBTASK_2"]["relative_duration"] = -1
    rows2 = inst.get_progress(total_step_budget=100)
    check(all(r["expected_steps"] > 0 for r in rows2), "非法 relative_duration 安全退化")


def test_render(m):
    print("\n[3] create_top_down_map_centered —— landmark 图层渲染")
    render = load_renderer()

    # 造一片地板 + 一堵墙，让底图不至于全白。前方是 -y。
    xs, ys = np.meshgrid(np.arange(-6, 6, 0.05), np.arange(-9, 2, 0.05))
    floor = np.stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)], axis=1)
    m.navigable_pcd = StubPCD(floor)

    wx = np.arange(-6, 6, 0.04)
    wall = np.stack([wx, np.full_like(wx, -8.5), np.full_like(wx, 0.5)], axis=1)
    m.obstacle_pcd = StubPCD(wall)

    # 模拟 agent 沿一条走廊走过来：给 space_labels 铺一段走廊染色，
    # 并造一个当前的 local_space 状态用于标签和调试射线
    _load_space_classifier()
    m.space_labels = {}
    for yy in np.arange(-1.0, 2.0, 0.25):
        for xx in np.arange(-0.75, 0.76, 0.25):
            m.space_labels[(int(round(xx / 0.25)), int(round(yy / 0.25)))] = 'corridor'
    m.local_space = {'state': 'corridor', 'raw_state': 'corridor', 'clearance': 0.85,
                     'along': 9.4, 'across': 1.7, 'elongation': 5.5,
                     'rays': {'f': 4.2, 'b': 5.2, 'l': 0.85, 'r': 0.85}}

    m.trajectory_position = [np.array([0.0, t, 1.2]) for t in np.arange(1.5, -0.1, -0.15)]

    out = render(
        m, 90.0,
        target_point=np.array([0.5, -2.0, 0.4]),
        action_candidates={
            "1": {"type": "waypoint", "target_point_world": [0.0, -1.2, 0.0]},
            "2": {"type": "waypoint", "target_point_world": [1.6, -2.4, 0.0]},
            "L": {"type": "turn", "angle": 1.57},
        },
        subtasks={},
        progress=[
            {"key": "SUBTASK_1", "rel": 0.15, "status": "done", "ratio": 1.0},
            {"key": "SUBTASK_2", "rel": 0.55, "status": "current", "ratio": 1.35},
            {"key": "SUBTASK_3", "rel": 0.30, "status": "todo", "ratio": 0.0},
        ],
        debug_overlay=True,
        rejected_waypoints=[
            {"world": [-1.2, -3.0, 0.0], "reason": "no_path"},
            {"world": [-2.0, -2.2, 0.0], "reason": "no_path"},
            {"world": [1.8, -1.0, 0.0], "reason": "edge"},
            {"world": [2.4, -3.2, 0.0], "reason": "occlusion"},
            {"world": [0.2, -0.4, 0.0], "reason": "distance"},
            {"world": [-0.6, 1.2, 0.0], "reason": "behind"},
        ],
    )

    check(out is not None and out.ndim == 3, "渲染返回有效图像")
    check(out.shape[0] > 100 and out.shape[1] > 100, "输出尺寸合理", str(out.shape))
    check(not np.all(out == 255), "图像非全白（确实画了内容）")

    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_grounding_preview.png")
    cv2.imwrite(dst, out)
    print(f"  预览图(录像帧, 含调试层): {dst}  ({out.shape[1]}x{out.shape[0]})")

    # 送给 VLM 的那一版：没有调试射线、没有背景房间灰圈
    out_vlm = render(
        m, 90.0, target_point=np.array([0.5, -2.0, 0.4]),
        action_candidates={"1": {"type": "waypoint", "target_point_world": [0.0, -1.2, 0.0]}},
        subtasks={},
        progress=[{"key": "SUBTASK_1", "rel": 0.4, "status": "current", "ratio": 0.6},
                  {"key": "SUBTASK_2", "rel": 0.6, "status": "todo", "ratio": 0.0}],
        debug_overlay=False,
    )
    dst2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_grounding_preview_vlm.png")
    cv2.imwrite(dst2, out_vlm)
    print(f"  预览图(给VLM的地图):     {dst2}")
    check(np.count_nonzero(np.all(out_vlm == (0, 200, 255), axis=-1)) == 0,
          "给VLM的地图里没有调试射线")

    # grounding 为空时必须完全退化成原来的行为
    m.grounded_landmarks = []
    m.unmatched_landmarks = []
    out2 = render(m, 90.0, target_point=None, action_candidates=None, subtasks={})
    check(out2 is not None and out2.shape == out.shape, "无 grounding 时正常退化")
    return dst


if __name__ == "__main__":
    print("=" * 68)
    print("Landmark Grounding 离线自测")
    print("=" * 68)
    test_resolve_category()
    mapper = test_ground_landmarks()
    test_area_regions()
    test_external_instances()
    test_desync_detector()
    test_desync_signals()
    test_premature_guard()
    test_force_advance()
    test_receded_stop_guard()
    test_decision_fallback()
    test_local_space()
    test_room_segmentation()
    test_side_view_pose_swap()
    test_landmark_states()
    test_order_gate()
    test_local_end_verification()
    test_compute_frontiers()
    test_frontier_scoring()
    test_area_hypotheses()
    test_frontier_scored_before_map()
    test_explore_suppressed_when_arrived()
    test_ablation_switches()
    test_frontier_as_action()
    test_hyper_available_before_mapping()
    test_init_hooks_in_right_method()
    test_no_silent_episode_failure()
    test_detector_fails_loudly()
    test_challenge_keeps_path()
    test_metric_invariant_and_flags()
    test_dead_fields_wired()
    test_rejected_waypoints()
    test_retry_order()
    test_avoid_penalty()
    test_budget()

    # 渲染测试里放一个被点名的区域(bedroom)和一个没被点名的(kitchen)，
    # 覆盖"命名区域"和"背景推断区域"两条渲染分支
    mapper.object_entities.append(make_entity("bed", [-4.0, -4.0, 0.3]))
    mapper.object_entities.append(make_entity("lamp", [-3.4, -4.4, 0.9]))
    mapper.object_entities.append(make_entity("refrigerator", [5.5, -5.0, 0.6]))
    mapper.object_entities.append(make_entity("stove", [6.2, -5.6, 0.5]))
    mapper.ground_landmarks(
        [{"name": "couch", "role": "waypoint_marker", "relative_position": "unknown"},
         {"name": "dining table", "role": "destination_marker", "relative_position": "unknown"},
         {"name": "chair", "role": "avoid_marker", "relative_position": "left"},
         {"name": "the bedroom", "role": "waypoint_marker", "relative_position": "unknown"},
         {"name": "hallway", "role": "destination_marker", "relative_position": "unknown"},
         {"name": "window", "role": "destination_marker", "relative_position": "unknown"}],
        subtask_key="SUBTASK_1")
    test_render(mapper)

    print("\n" + "=" * 68)
    if _failures:
        print(f"{len(_failures)} 项失败:")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    print("全部通过")