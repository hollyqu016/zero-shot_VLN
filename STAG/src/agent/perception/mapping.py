"""
感知 · 建图。

把一帧观测灌进 mapper，并决定要不要额外跑侧视角分割。

跨阶段读写的 self 字段（未来解耦时这就是接口）：
    读  self.mapper / self.instruction / self.enable_vlm_detection
    写  self.mapper 的内部状态
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


class MappingMixin:
    def update_map(self, obs: dict):
        """Update map using four-direction observations (batched multiview)."""

        views = []

        if 'left' in obs and 'agent_state' in obs['left']:
            s = obs['left']['agent_state']
            views.append({
                'rgb': obs['left']['color_sensor'],
                'depth': obs['left']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        if 'right' in obs and 'agent_state' in obs['right']:
            s = obs['right']['agent_state']
            views.append({
                'rgb': obs['right']['color_sensor'],
                'depth': obs['right']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        if 'back' in obs and 'agent_state' in obs['back']:
            s = obs['back']['agent_state']
            views.append({
                'rgb': obs['back']['color_sensor'],
                'depth': obs['back']['depth_sensor'],
                'position': np.array(s.position),
                'rotation': s.sensor_states['color_sensor'].rotation,
            })

        agent_state = obs['agent_state']
        # Primary (front) view — always last so primary_index defaults to it
        views.append({
            'rgb': obs['color_sensor'],
            'depth': obs['depth_sensor'],
            'position': np.array(agent_state.position),
            'rotation': agent_state.sensor_states['color_sensor'].rotation,
        })

        # 只在"当前子任务还有物体类 landmark 没接地"时才多跑侧方视角的检测。
        # 全接上了就没必要多付这个开销；一个都没有(比如landmark全是房间)也不必，
        # 因为多看两眼也接不了地。
        instances = self.mapper.update_multiview(
            views, segment_all_views=self._needs_side_segmentation())

        # 必须在 grounding 之前：hallway 类地标要靠这一步的结果才能接地
        self.mapper.classify_local_space()
        self._ground_current_landmarks()
        # grounding 之后再算前沿：打分要用到"哪些 landmark 还没接地"
        self.mapper.compute_frontiers()

        return instances

    def _needs_side_segmentation(self):
        """当前子任务是否还有"词表里有、但地图上还没找到"的物体类 landmark。"""
        try:
            pending = getattr(self.mapper, 'unmatched_landmarks', []) or []
            return any(u.get('reason') == 'not_yet_detected' for u in pending)
        except Exception:
            return False

    def _apply_vlm_detections(self, parsed_json, obs):
        """
        把决策VLM在 'detections' 字段里顺带返回的 front view 检测框并进地图。

        为什么值得这么做：决策调用本来就已经把 front view 以 detail=high 发过去了，
        让模型在同一次回复里额外吐一组框，边际成本只有几十个 output token，
        而单独跑一次检测VLM要付一整次请求的延迟。中间步仍然走本地YOLOE，
        所以物体地图的更新频率不会下降，这里只是在决策帧上叠加一层
        更强的检测结果。

        整段 try 包住：检测是锦上添花，解析失败绝不能影响导航决策。
        """
        try:
            if not self.enable_vlm_detection:
                return
            dets = parsed_json.get('detections')
            if not dets or not isinstance(dets, list):
                return
            shape = obs['color_sensor'].shape if obs is not None else None
            n = self.mapper.apply_external_instances(dets, image_shape=shape)
            if n:
                print(f"[vlm-det] fused {n}/{len(dets)} detections from decision call")
        except Exception as e:
            print(f"[vlm-det] skipped: {e}")
