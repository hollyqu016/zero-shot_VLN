"""
规划 · 前沿打分。

score(f) = spatial(f) × temporal(f)

这是全系统**唯一**一处空间项与时间项相乘的决策点。空间项来自房间假设
（AREA_PROXY_OBJECTS 倒查），时间项来自剩余步数换算的可达半径。
"""

import json
import math
import re
from abc import ABC, abstractmethod

import PIL.Image

from simWrapper import SimWrapper, PolarAction
from mapper import Instruct_Mapper
from config_utils import hyper
from segmentation.object_list import (AREA_PROXY_OBJECTS, GENERIC_AREA_TYPES, category_names_str,
                                      normalize_category_name, resolve_area, resolve_category)
import time
from typing import Optional, List
from PIL.Image import Image
from utils import *
from scipy.spatial.transform import Rotation as R
from ..decomposer.spatio_temporal_decomposer import SpatioTemporalInstructionDecomposer
import logging



class FrontierMixin:
    # 兜底默认，运行时由 _apply_hyper() 从 config['hyper']['frontier'] 覆盖
    FRONTIER_MIN_USEFUL_M = 2.0   # 比这更近的前沿不值得当探索目标
                                  # 1m 处的前沿会让 VLM 以为目标房间就在眼前
    FRONTIER_STEP_LEN = 0.6       # 估算可达半径用的每步位移(米)
    FRONTIER_TOP_K = 3            # 最多向 VLM 提供几个前沿（F1..F3）

    def _score_frontiers(self, step_len=None, top_k=None):
        """
        给每个前沿打分：score = 空间项 × 时间项。

        这是全系统唯一一处空间与时间**相乘**的决策点。此前两条线一直是平行的：
        空间那边算距离、房间、走廊，时间那边算预算、进度，从来没有在同一个
        决策里共同参与过。一个叫 spatio-temporal 的系统，中间那个连字符总得
        有地方兑现。

        空间项 —— "下一个目标在这个前沿后面的可能性"
          用 AREA_PROXY_OBJECTS **反过来**推理：当前子任务还没找到的目标物体
          属于哪类房间？地图上那类房间已经识别出来了吗？在哪个方向？
          朝那个方向的前沿加分。房间还没找到时退化成"宽的前沿优先"——
          这个退化是诚实的：没有线索时唯一合理的先验就是"大开口通向大空间"。

        时间项 —— "剩下的预算够不够走到那儿"
          剩余步数换算成可达距离，超出的按距离衰减。只剩 20 步时，15 米外
          再有希望的前沿也不该去。

        返回按分数排序的前 top_k，每项加了 'score' / 'why' / 'bearing'。
        """
        step_len = self.FRONTIER_STEP_LEN if step_len is None else step_len
        top_k = self.FRONTIER_TOP_K if top_k is None else top_k
        # 太近的前沿不是探索目标，而且会帮倒忙。
        #
        # 实测 ep34：前沿在正前方 1m，提示写成"a bed was seen that way — weak
        # evidence for a bedroom (1m)"，而指令正是"走到主卧"。等于告诉 VLM
        # "你要找的卧室就在一米外"，它立刻宣布到达、原地停下，全程位移 0。
        # 一米外的开口用普通 waypoint 就能走过去，不需要专门指路。
        fr = [f for f in (getattr(self.mapper, 'frontiers', []) or [])
              if float(f.get('distance', 0.0)) >= self.FRONTIER_MIN_USEFUL_M]
        if not fr:
            return []

        # ---- 目标：当前子任务里还没接地的物体类 landmark ----
        targets = []
        try:
            _k, st = self.instruction_obj.get_current_subtask()
            for lm in (st or {}).get('landmarks', []):
                if lm.get('role') == 'avoid_marker':
                    continue
                c, _ = resolve_category(lm.get('name', ''))
                if c:
                    targets.append((lm.get('name', ''), c))
        except Exception:
            targets = []
        grounded_names = {g.get('name') for g in (self.mapper.grounded_landmarks or [])}
        targets = [t for t in targets if t[0] not in grounded_names]

        # ---- 目标物体可能在哪类房间里（AREA_PROXY_OBJECTS 的反向索引）----
        want_rooms = set()
        for _nm, cls in targets:
            for room_type, proxies in AREA_PROXY_OBJECTS.items():
                if any(normalize_category_name(p) == normalize_category_name(cls)
                       for p in proxies):
                    want_rooms.add(room_type)
        # 子任务直接点名了某个房间（"go into the kitchen"）也算
        try:
            for u in (self.mapper.unmatched_landmarks or []):
                if u.get('area_type'):
                    want_rooms.add(u['area_type'])
        except Exception:
            pass

        # ---- 目标房间可能在哪（含弱证据）----
        # 不再只看"已确认的房间"。那个版本要求目标房间已经被识别，
        # 而一旦识别了就不需要引导——实测 139 次全部落空、0 次命中。
        # 现在单个特征物体（瞥见的水槽、灶台一角）也算方向线索，
        # 只是权重按 proxy 权重打折。
        try:
            hyps = self.mapper.area_hypotheses()
        except Exception:
            hyps = []
        room_pts = [h for h in hyps if h.get('area_type') in want_rooms]

        agent_xy = np.asarray(self.mapper.current_position[:2], dtype=float)

        # ---- 时间项的可达半径 ----
        remaining = None
        try:
            cur = self._current_budget_row()
            if cur:
                remaining = max(0.0, cur['expected_steps'] * 2.0 - cur['used_steps'])
        except Exception:
            remaining = None
        reach_m = (remaining * step_len) if remaining is not None else 25.0

        max_w = max(f['width_m'] for f in fr) or 1.0
        scored = []
        for f in fr:
            c = np.asarray(f['center'], dtype=float)

            # 空间项：取所有假设里得分最高的一个。
            # 距离衰减 × 假设强度——强证据的房间和瞥见一眼的水槽不该等价。
            if room_pts:
                best = None
                for h in room_pts:
                    d = float(np.linalg.norm(c - np.asarray(h['center'], dtype=float)))
                    conf = min(1.0, float(h.get('score', 1.0)))
                    val = conf * float(np.exp(-d / 6.0))
                    if best is None or val > best[0]:
                        best = (val, d, h)
                spatial, best_d, h = best
                if h.get('confirmed'):
                    why = f"heads toward the {h['area_type']} ({best_d:.0f}m from it)"
                else:
                    ev = "/".join(h.get('evidence', [])) or "a cue"
                    why = (f"a {ev} was seen that way — weak evidence for a "
                           f"{h['area_type']} ({best_d:.0f}m)")
            else:
                # 没有任何房间线索时，这个打分和指令毫无关系，纯粹是几何启发式。
                # 标成 informed=False，上层据此**不给建议**——见下面的说明。
                spatial = 0.35 * (f['width_m'] / max_w)
                why = f"widest unexplored opening ({f['width_m']:.1f}m)" \
                    if f['width_m'] >= max_w * 0.9 else f"{f['width_m']:.1f}m opening"

            # 时间项
            d_go = float(np.linalg.norm(c - agent_xy))
            if d_go <= reach_m:
                temporal = 1.0
            else:
                temporal = float(np.exp(-(d_go - reach_m) / max(reach_m, 1.0)))
                why += f"; but {d_go:.0f}m away and budget only covers ~{reach_m:.0f}m"

            g = dict(f)
            g['score'] = spatial * temporal
            g['spatial'] = spatial
            g['temporal'] = temporal
            g['why'] = why
            # 这个排名到底有没有用到指令信息。没有的话它就只是"哪个洞最大"，
            # 不该冒充"最可能通向目标的方向"。
            g['informed'] = bool(room_pts)
            # 相对 agent 的方位，用于在 prompt 里说人话
            v = c - agent_xy
            rot = np.asarray(self.mapper.current_rotation, dtype=float)
            fwd = -rot[:, 2][:2]
            nf = np.linalg.norm(fwd)
            fwd = fwd / nf if nf > 1e-6 else np.array([0.0, -1.0])
            right = rot[:, 0][:2]
            nr = np.linalg.norm(right)
            right = right / nr if nr > 1e-6 else np.array([1.0, 0.0])
            ahead, side = float(np.dot(v, fwd)), float(np.dot(v, right))
            g['bearing'] = (("ahead" if ahead > abs(side) else
                             "behind" if -ahead > abs(side) else
                             "right" if side > 0 else "left"))
            scored.append(g)

        scored.sort(key=lambda x: -x['score'])
        return scored[:top_k]
