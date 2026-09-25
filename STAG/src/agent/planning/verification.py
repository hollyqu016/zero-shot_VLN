"""
规划 · 子任务完成校验与停止拦截。

八个判据。贯穿它们的一条原则：**「无法确认」不等于「确认为否」**。
证据不足时返回 unknown 并保持沉默，而不是投反对票——这条在三处独立
出现过同一种失败，修法也相同。见 docs/PAPER_CONTEXT.md §4.1。
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



class VerificationMixin:
    # 兜底默认，运行时由 _apply_hyper() 从 config['hyper']['subtask'] 覆盖
    PREMATURE_MIN_RATIO = 0.3       # 步数用量低于预算这个比例，视为可疑的提前完成
    PREMATURE_SHAKY_RATIO = 0.6     # 介于两者之间时做质询而非直接拦截
    VERIFY_NEAR_M = 2.0             # 校验 local_end 时"到达"的判定距离
    TOO_FAR_OBJECT_LIMIT = 2.0      # 目的地是物体时，超过这么远不许停
    RECEDE_MARGIN = 1.5             # 已从终点退开多远就质询停止
    FORCE_ADVANCE_RATIO = 2.0       # 花掉预算这么多倍后强制推进
    FORCE_ADVANCE_MIN_STEPS = 20    # 但至少要花掉这么多步才允许强推
    ROLLBACK_STEP_THRESHOLD = 70    # 超过这一步数不再回退，避免来回拉锯

    def _current_budget_row(self):
        """当前子任务的预算行；拿不到就返回 None。"""
        try:
            rows = self.instruction_obj.get_progress(self.total_step_budget)
            return next((r for r in rows if r['status'] == 'current'), None)
        except Exception:
            return None

    def _is_premature_completion(self, min_ratio=None, shaky_ratio=None):
        """
        判断"宣告完成"是否来得太早：步数还不到预算的 min_ratio，且不是最后一个
        子任务。每个子任务只质询一次——质询过一次还坚持说完成，就认了，
        毕竟模型可能真的对，硬拦会死锁。

        实测动机：一批 episode 在末尾出现雪崩式清空，剩下的子任务 1~5 步就被
        连续的 -1 烧完（ep46 的 SUBTASK_3 只有 1 步，预算 15 步）。那些子任务
        根本没被执行过，只是VLM认定"我已经到终点了"之后的连锁反应。
        """
        min_ratio = self.PREMATURE_MIN_RATIO if min_ratio is None else min_ratio
        shaky_ratio = self.PREMATURE_SHAKY_RATIO if shaky_ratio is None else shaky_ratio
        if not self.enable_interception:
            return False
        io = self.instruction_obj
        key = io.get_current_subtask_key()
        if key is None or io.is_last_subtask():
            return False
        if not hasattr(self, '_challenged_subtasks'):
            self._challenged_subtasks = set()
        if key in self._challenged_subtasks:
            return False

        # 分解器判定"本段终点和下一段起点衔接不上"时，把质询区间放宽。
        # 那意味着分段本身就不太可靠，此处的完成判定更容易出错。
        st = io.sub_instruction_dict.get(key, {})
        shaky = st.get('boundary_consistent') is False
        limit = shaky_ratio if shaky else min_ratio

        cur = self._current_budget_row()
        if not cur or cur['ratio'] >= limit:
            return False

        self._challenged_subtasks.add(key)
        # local_end 是分解器给出的显式终止条件，之前存了从没读过。
        # 质询的时候把它摆出来，比空泛地问"你真的到了吗"有用得多。
        end_cond = st.get('local_end')
        end_str = (f" The stated completion condition for {key} is: \"{end_cond}\"."
                   if end_cond and end_cond != 'unknown' else "")
        shaky_str = (" (Note: this subtask's boundary was already flagged as inconsistent "
                     "with the next one, so be extra careful here.)" if shaky else "")
        self.stop_challenge = (
            f"\n**CONFIRM: you just declared {key} finished after only "
            f"{cur['used_steps']} steps, but it was expected to take about "
            f"{cur['expected_steps']:.0f}.{end_str} Look again: have you actually reached that, "
            f"or are you jumping ahead because the FINAL destination looks close? If it really is "
            f"done, say so again this turn and it will be accepted. If not, keep working on it."
            f"{shaky_str}**")
        print(f"[subtask] challenged early completion of {key} "
              f"({cur['used_steps']} steps vs ~{cur['expected_steps']:.0f} expected)")
        return True

    def _challenge_receded_stop(self):
        """
        宣告停止时如果检测到"之前离目标更近过"，拦一次并要求它回头。
        每个子任务只拦一次，避免和模型顶死。
        """
        if not self.enable_interception:
            return False
        io = self.instruction_obj
        key = io.get_current_subtask_key()
        if key is None:
            return False
        if not hasattr(self, '_receded_challenged'):
            self._receded_challenged = set()
        if key in self._receded_challenged:
            return False

        # 两条独立判据，任一成立就拦：
        #   a) 目标还很远  —— 只看当前距离，不需要历史，命中率最高
        #   b) 曾经更近过  —— 走过头的情况，a 抓不到时由它兜底
        too_far, detail = self._is_too_far_to_stop()
        if not too_far:
            receded, detail = self._has_receded_from_destination()
            if not receded:
                return False

        self._receded_challenged.add(key)
        self.stop_challenge = (
            f"\n**DO NOT STOP HERE YET: {detail}. Move closer to it first and stop only when you "
            "are right next to it. If you are certain this is the correct place despite the "
            "distance (for example the object on the map is a mis-detection), say so again next "
            "turn and the stop will be accepted.**")
        print(f"[subtask] blocked stop on {key}: {detail}")
        return True

    def _verify_local_end(self, near_m=None):
        """
        把分解器给的终止条件（自由文本）和地图事实对照。
        返回 (status, detail)，status ∈ {'ok', 'violated', 'unknown'}。

        这是"校验"和"展示"的区别所在：之前 local_end 只是抄进 prompt 让 VLM
        自己解释，代码没有拿它核对任何东西。

        做法不是模糊语义匹配，而是复用系统里已有的两个解析器，从终止条件里
        抽出**可核对的原子**：
          - resolve_area("just inside the hallway")  -> hallway  -> 查 current_room / 走廊判据
          - resolve_category("near the television")  -> television -> 查该类实体是否在身边
        抽不出任何原子就返回 unknown，绝不阻拦——终止条件是自由文本，覆盖不全
        是常态，宁可放过也不能凭"没看懂"就卡住。

        满足**任意一个**原子即判 ok（而不是全部）。终止条件常同时提到房间和
        物体("living room area near the television")，物体没检测到不代表人没到
        客厅，从严会造成大量误拦。
        """
        near_m = self.VERIFY_NEAR_M if near_m is None else near_m
        try:
            _key, st = self.instruction_obj.get_current_subtask()
        except Exception:
            return 'unknown', ''
        text = (st or {}).get('local_end')
        if not text or text == 'unknown':
            return 'unknown', ''

        toks = normalize_category_name(text).split()
        area_atoms, obj_atoms = [], []
        for n in (2, 1):
            for i in range(len(toks) - n + 1):
                phrase = " ".join(toks[i:i + n])
                a, _ = resolve_area(phrase)
                # 泛化区域("room"/"area"/"space")不作为可核对原子。
                # 它不携带任何区分信息：人在卧室里本来就是在"房间"里，
                # 拿 bedroom != room 判矛盾纯属误拦——实测 19 次 local_end
                # 拦截里有相当一部分是 "you are in the bedroom, not the room"。
                if a and a in GENERIC_AREA_TYPES:
                    continue
                if a and a not in area_atoms:
                    area_atoms.append(a)
                c, _ = resolve_category(phrase)
                if c and c not in obj_atoms:
                    obj_atoms.append(c)
        if not area_atoms and not obj_atoms:
            return 'unknown', ''

        # 关键区分：**无法确认** 和 **确认为否** 不是一回事。
        #
        # 第一版把两者混为一谈：local_end 说 "inside the bedroom"，而
        # current_room() 返回 None（房间推断还没攒够证据）时，判成"不在卧室"
        # 于是拦截。房间推断本来就稀疏，None 是常态——结果实跑里 30 次拦截有
        # 23 次来自这里，而且大多拦完下一步就放行，纯属噪音。
        # 现在只有拿到**反向证据**才算 violated：
        #   - current_room 明确返回了另一个房间
        #   - 该类物体在地图上存在，但离得很远
        # 查不到 / 没检测到 -> unknown，不阻拦。
        satisfied, contradicted = [], []

        room = None
        try:
            room = self.mapper.current_room()
        except Exception:
            room = None
        space = (getattr(self.mapper, 'local_space', {}) or {}).get('state')
        for a in area_atoms:
            if room is not None and room.get('area_type') == a:
                satisfied.append(f"you are in the {a}")
            elif a in ('hallway', 'stairwell') and space == 'corridor':
                satisfied.append(f"you are in a corridor-like space ({a})")
            elif room is not None:
                contradicted.append(f"you are in the {room.get('area_type')}, not the {a}")
            elif a in ('hallway', 'stairwell') and space == 'room':
                # 局部几何明确判成开阔房间，这是"不在走廊"的正面证据
                contradicted.append(f"the space around you is open, not a {a}")
            # 其余情况：认不出当前房间，无从判断，不作数

        for c in obj_atoms:
            hit = None
            for e in (self.mapper.object_entities or []):
                try:
                    if normalize_category_name(e['class_name']) != normalize_category_name(c):
                        continue
                    d = float(np.linalg.norm(
                        np.asarray(e['center'][:2], dtype=float)
                        - np.asarray(self.mapper.current_position[:2], dtype=float)))
                    if hit is None or d < hit:
                        hit = d
                except Exception:
                    continue
            if hit is not None and hit <= near_m:
                satisfied.append(f"'{c}' is {hit:.1f}m away")
            elif hit is not None:
                contradicted.append(f"nearest '{c}' is {hit:.1f}m away")
            # hit is None：这类物体压根没检测到，不能据此判定人没到

        if satisfied:
            return 'ok', "; ".join(satisfied)
        if contradicted:
            return 'violated', "; ".join(contradicted)
        return 'unknown', ''

    def _challenge_out_of_order(self):
        """
        顺序约束：指令说"经过 X 再到 Y"，那么在 X 从没被靠近过之前，
        不接受 Y 已完成的声明。

        这是整个系统里第一条真正的**时序**约束。此前时间维度只有
        relative_duration 这个时长预算——本质是个超时计数器，和"顺序"无关。
        而分解器一直在输出顺序信息（stage 次序 + waypoint/destination 的角色
        语义），从来没有被消费过。

        只对**已接地**的 waypoint_marker 生效。未接地的不参与判定——否则
        landmark 一辈子检测不到的子任务会永远完不成，是个静默死锁。
        同一子任务只拦一次，拦过还坚持就认。
        """
        if not self.enable_interception:
            return False
        io = self.instruction_obj
        key = io.get_current_subtask_key()
        if key is None:
            return False
        if not hasattr(self, '_order_challenged'):
            self._order_challenged = set()
        if key in self._order_challenged:
            return False

        grounded = self.mapper.grounded_landmarks or []
        # grounded_landmarks 的顺序沿用 landmarks 在句子里出现的顺序，
        # 所以这个列表本身就是指令要求的先后。
        wps = [g for g in grounded if g.get('role') == 'waypoint_marker']

        reasons = []

        # (1) 该经过的途经点从没靠近过
        pending = [g for g in wps if not g.get('ever_near')]
        if pending:
            reasons.append("you have never actually been next to " + ", ".join(
                f"'{g.get('name','')}' (still {float(g.get('distance',0.0)):.1f}m away, "
                f"closest ever {float(g.get('min_dist',0.0)):.1f}m)" for g in pending))

        # (2) 途经点之间的先后颠倒："go past the table then past the sofa"
        #     只检查已经到过的那些，谁先谁后
        for i in range(len(wps)):
            for j in range(i + 1, len(wps)):
                if wps[j].get('ever_near') and not wps[i].get('ever_near'):
                    reasons.append(
                        f"you reached '{wps[j].get('name','')}' before "
                        f"'{wps[i].get('name','')}', but the instruction lists them the other "
                        "way round")
                    break
            else:
                continue
            break

        # (3) 终止条件和地图事实对不上
        status, detail = self._verify_local_end()
        if status == 'violated':
            reasons.append(f"the stated completion condition is not met ({detail})")

        if not reasons:
            return False

        self._order_challenged.add(key)
        why = "; and ".join(reasons)
        self.stop_challenge = (
            f"\n**NOT DONE YET: for {key}, {why}. Finish it properly before moving on. "
            "If an object marked on the map is a mis-detection and the real one is elsewhere, "
            "say so again next turn and this will be accepted.**")
        print(f"[subtask] blocked completion of {key}: {why}")
        return True

    def _is_too_far_to_stop(self, object_limit=None):
        """
        目标近在眼前却宣布到达。返回 (是否太远, 说明文字)。

        比"曾经更接近"更直接、也更常命中：不需要任何历史，只看当前子任务的
        destination_marker 现在有多远。实测 ep32 的指令是"站在按摩床旁边"，
        agent 自己已经检测到 massage table 在 3.6m 处，却在那里宣布到达——
        成功阈值是 3.0m，差 0.6m。这种情况用历史距离判据抓不到（它接地时
        是 2.7m，只远了 0.9m），但用绝对距离一眼就能看出不对。

        区域类地标用房间半径当阈值：人在房间圆内就算到了，不能拿 2m 卡。
        """
        object_limit = self.TOO_FAR_OBJECT_LIMIT if object_limit is None else object_limit
        for g in getattr(self.mapper, 'grounded_landmarks', []) or []:
            if g.get('role') != 'destination_marker':
                continue
            dist = float(g.get('distance', 0.0))
            if g.get('kind') == 'area':
                limit = float(g.get('radius', 3.0))
            else:
                limit = object_limit
            if dist > limit:
                return True, (f"'{g.get('name', '')}' is the destination of this subtask and it is "
                              f"still {dist:.1f}m away (you should be within about {limit:.1f}m)")
        return False, ""

    def _has_receded_from_destination(self, margin=None):
        """
        "你之前离目标更近过"。返回 (是否远离了, 说明文字)。

        动机来自实测：22 个 episode 里 9 个曾经进入目标 3 米内，只有 4 个停在
        那里。剩下 5 个的历史最近距离是 1.2 / 1.8 / 1.9 / 2.2 / 2.4 米，最终却
        停在 3.6~9.2 米——走到了又走开，白丢 22.7 个百分点。

        agent 不知道真实 goal 在哪(那是评测信息)，但它知道当前子任务的
        destination_marker 在哪，以及自己历史上离它最近过多少。这个信息完全
        合法，而且恰好对应上面那个缺口。
        """
        margin = self.RECEDE_MARGIN if margin is None else margin
        mins = getattr(self, '_lm_min_dist', None) or {}
        worst = None
        for g in getattr(self.mapper, 'grounded_landmarks', []) or []:
            if g.get('role') != 'destination_marker':
                continue
            name = g.get('name', '')
            now = float(g.get('distance', 0.0))
            best = mins.get(name)
            if best is None or now <= best + margin:
                continue
            if worst is None or (now - best) > worst[2] - worst[1]:
                worst = (name, best, now)
        if worst is None:
            return False, ""
        name, best, now = worst
        return True, (f"you were {best:.1f}m from '{name}' earlier and you are now {now:.1f}m "
                      f"away — you have walked past or away from it")

    def _maybe_force_advance(self, ratio_limit=None, min_abs_steps=None):
        """
        子任务严重超支时由代码强制推进，不再等VLM同意。

        实测动机：一批 episode 整个报废在第一个子任务上（99/100、84/100、
        76/100 步），desync 提示发了也没用。相比"可能推错"，"整个 episode
        白跑"的代价明显更大。

        每个子任务最多强制一次，并把这件事写进 record，下一轮prompt里VLM
        会看到——它需要知道自己是被系统推着走的，而不是自己完成的。
        """
        ratio_limit = self.FORCE_ADVANCE_RATIO if ratio_limit is None else ratio_limit
        min_abs_steps = self.FORCE_ADVANCE_MIN_STEPS if min_abs_steps is None else min_abs_steps
        io = self.instruction_obj
        key = io.get_current_subtask_key()
        if key is None or io.is_last_subtask():
            return False
        if not hasattr(self, '_forced_subtasks'):
            self._forced_subtasks = set()
        if key in self._forced_subtasks:
            return False

        cur = self._current_budget_row()
        if not cur or cur['ratio'] <= ratio_limit:
            return False
        # 绝对步数下限：短子任务的预期只有 8~10 步，光看比例会在十几步就强推。
        # 实测 "Go out of the room you're in" 这种正常也要 14~16 步。
        if cur['used_steps'] < min_abs_steps:
            return False

        # 强推也要尊重顺序约束。
        #
        # 之前这里完全不看 landmark 状态：只要超预算 2 倍就把子任务推过去，
        # 哪怕它的途经点一次都没去过。那等于代码自己承认顺序约束不存在——
        # 比 VLM 违反顺序更糟。
        #
        # 但也不能直接不推，否则就是死锁(强推存在的意义就是打破卡死)。
        # 折中：第一次先给一轮宽限，把没去的途经点点名塞进 prompt 让它去；
        # 下次再触发才真推，并在记录里写明"是跳过的，不是完成的"。
        if not hasattr(self, '_force_deferred'):
            self._force_deferred = set()
        unreached = [g for g in (self.mapper.grounded_landmarks or [])
                     if g.get('role') == 'waypoint_marker' and not g.get('ever_near')]
        if unreached and key not in self._force_deferred:
            self._force_deferred.add(key)
            names = ", ".join(f"'{g.get('name','')}' ({float(g.get('distance',0.0)):.1f}m)"
                              for g in unreached)
            self.stop_challenge = (
                f"\n**LAST CHANCE: you are well over budget on {key} and you still have not "
                f"visited {names}, which the instruction says to pass. Go there now. If you do "
                "not, the system will move on without it.**")
            print(f"[subtask] force-advance deferred on {key}: unreached waypoints {names}")
            return False

        self._forced_subtasks.add(key)
        skipped = ", ".join(f"'{g.get('name','')}'" for g in unreached) if unreached else None
        # record 必须在 mark 之前写，否则会记到下一个子任务头上
        io.add_record_to_current_subtask({
            'system': (f"{key} was force-advanced by the system after {cur['used_steps']} steps "
                       f"(~{cur['expected_steps']:.0f} expected). It was NOT confirmed complete."
                       + (f" It was skipped WITHOUT ever visiting {skipped}, which the instruction "
                          "asked you to pass — keep that in mind, you may need to come back."
                          if skipped else "")
                       + " If you have not actually done it, fold it into your current plan.")})
        io.mark_current_subtask_completed(
            self.mapper.current_position - np.array([0, 0, 1.2]), self.mapper.current_rotation)
        print(f"[subtask] FORCE-ADVANCED {key} after {cur['used_steps']} steps "
              f"(~{cur['expected_steps']:.0f} expected) -> now {io.get_current_subtask_key()}")
        self._reset_landmark_progress()
        return True
