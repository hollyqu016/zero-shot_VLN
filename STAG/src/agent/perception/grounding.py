"""
感知 · 地标接地与进度追踪。

指令里的地标名 → 地图上的实体/区域。以及粘性状态 ever_near / ever_passed：
瞬时的 seen/near/passed 会因为一次转身就丢，粘性状态不会。

注意 _track_landmark_progress 必须**单趟**遍历完成——拆成两个循环会让
prev_min 被覆写，远离检测就失效了。这是踩过的坑。
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


class GroundingMixin:
    # 下面这些既是文档也是**兜底默认**。运行时由 agent 的 _apply_hyper() 从
    # config['hyper']['landmark'] 覆盖成实例属性。
    #
    # 类级默认不能省：本模块的方法在 agent 的 __init__ 里（首帧建图）就会被
    # 调到，而外层包着 except，缺属性会被降级成一行 "skipped due to error"，
    # 首帧接地静默失效——实跑踩过这个坑。
    NEAR_M = 1.5             # 视为"就在旁边"的距离
    PASSED_MIN_M = 2.0       # 曾经至少靠到这么近，才谈得上"走过了"
    PASSED_GAP_M = 1.0       # 比历史最近再远出这么多，判定为已走过
    LM_RECEDE_MARGIN = 1.2   # 判定"正在远离目标"的距离裕度
    LM_RECEDE_STEPS = 6      # 连续多少步在远离才算数

    def _ground_current_landmarks(self):
        """
        用当前子任务的landmark列表刷新mapper里的接地结果。

        landmarks字段由 Instruction.__init__ 从时空分解器的输出里存下来
        (agent.py 里 'landmarks': sub_inst['semantic_anchors'].get('landmarks', [])),
        在此之前整个项目没有任何地方读过它。

        这里整段用try包住是有意的：grounding纯粹是给地图加一层可视化/语义标注，
        任何情况下都不应该让导航主循环挂掉。失败时mapper里的列表保持为空，
        渲染层会自然退化成原来的行为。
        """
        try:
            subtask_key, subtask = self.instruction_obj.get_current_subtask()
            if subtask is None:
                self.mapper.grounded_landmarks = []
                self.mapper.unmatched_landmarks = []
                return
            self.mapper.ground_landmarks(subtask.get('landmarks', []), subtask_key=subtask_key)
            self._track_landmark_progress(subtask_key)
        except Exception as e:
            print(f"[grounding] skipped due to error: {e}")
            self.mapper.grounded_landmarks = []
            self.mapper.unmatched_landmarks = []

    def _reset_landmark_progress(self):
        """子任务切换时清空距离历史，否则会拿旧子任务的landmark继续判定。"""
        self._lm_min_dist = {}
        self._lm_receding_steps = 0
        self.passed_landmark_hint = False
        # 粘性状态：一旦成立就不再撤销，直到子任务切换。
        # 顺序约束要的是"曾经到达过"这个事实，而不是"此刻在不在旁边"——
        # 瞬时状态会因为折返而翻回 near，拿它做顺序判定会漏掉已经走过的点。
        self._lm_ever_near = set()
        self._lm_ever_passed = set()

    def _track_landmark_progress(self, subtask_key, recede_margin=None, recede_steps=None):
        """
        检测"状态机落后于实际进度"：如果当前子任务的所有已接地landmark都比
        历史最近距离远出 recede_margin 米，且连续 recede_steps 步都是这样，
        说明agent已经走过了这个子任务的目标却没人宣告完成。

        这个信号来自一次真实失败：VLM在第33步的thought里写了"Subtask 1 已完成"
        但action给的是路点，状态机于是卡在SUBTASK_1整整25步，期间 open door
        的距离从1.6m一路涨到3.6m。距离单调变远是纯几何量，不依赖读懂散文，
        可以直接机器判定，所以拿它当兜底提示塞回prompt里。
        """
        recede_margin = self.LM_RECEDE_MARGIN if recede_margin is None else recede_margin
        recede_steps = self.LM_RECEDE_STEPS if recede_steps is None else recede_steps
        if not hasattr(self, '_lm_min_dist'):
            self._reset_landmark_progress()
        if getattr(self, '_lm_subtask_key', None) != subtask_key:
            self._reset_landmark_progress()
            self._lm_subtask_key = subtask_key

        # 顺便给每个已接地的 landmark 标一个时序状态，供清单和 pin 显示。
        # 数据本来就在算（历史最近距离），之前只喂给了 desync 检测，
        # 没有变成人和 VLM 都能看到的东西。
        #   seen   已经在图上定位，但还没靠近过
        #   near   正在旁边（当前距离 <=1.5m）
        #   passed 曾经靠近过(<=2m)，现在明显远离(+1m)——多半已经走过去了
        # 单次遍历同时算三件事：粘性状态、显示状态、远离检测。
        #
        # 必须合成一趟：远离检测要的是**本步更新之前**的历史最近值，如果分成
        # 两个循环、前一个已经把 min 更新过了，后一个拿到的 prev 恒 <= 当前距离，
        # all_receding 永远为假，desync 检测会静默失效。
        grounded = []
        all_receding = True
        for g in (self.mapper.grounded_landmarks or []):
            nm, dd = g.get('name', ''), float(g.get('distance', 0.0))
            prev_min = self._lm_min_dist.get(nm)          # 更新前的历史最近
            best = dd if prev_min is None else min(prev_min, dd)
            self._lm_min_dist[nm] = best

            if dd <= self.NEAR_M:
                self._lm_ever_near.add(nm)
            if best <= self.PASSED_MIN_M and dd > best + self.PASSED_GAP_M:
                self._lm_ever_passed.add(nm)

            # 显示用的瞬时状态：此刻在旁边就是 near，哪怕之前走过又折返回来
            if dd <= self.NEAR_M:
                g['state'] = 'near'
            elif nm in self._lm_ever_passed:
                g['state'] = 'passed'
            else:
                g['state'] = 'seen'
            # 粘性事实：顺序约束用这两个，不用上面的瞬时 state
            g['ever_near'] = nm in self._lm_ever_near
            g['ever_passed'] = nm in self._lm_ever_passed
            g['min_dist'] = best

            if g.get('role') in ('destination_marker', 'waypoint_marker'):
                grounded.append(g)
                if prev_min is None or dd <= prev_min + recede_margin:
                    all_receding = False

        if not grounded:
            # 一个都没接地时不做判断——距离信息缺失不等于"走过头了"
            self._lm_receding_steps = 0
            self.passed_landmark_hint = False
            return

        self._lm_receding_steps = self._lm_receding_steps + 1 if all_receding else 0
        hint = self._lm_receding_steps >= recede_steps
        if hint and not self.passed_landmark_hint:
            detail = ", ".join(
                f"{g['name']} {self._lm_min_dist.get(g['name'], 0):.1f}->{g['distance']:.1f}m"
                for g in grounded)
            print(f"[subtask] possible desync on {subtask_key}: receding for "
                  f"{self._lm_receding_steps} steps ({detail})")
        self.passed_landmark_hint = hint
