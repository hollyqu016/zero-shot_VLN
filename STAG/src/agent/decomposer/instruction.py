"""
指令分解的产物：Instruction 对象。

它是子任务状态机的载体——当前进行到第几个子任务、每个子任务花了多少步、
完成时 agent 在哪。放在 decomposer 下是因为它由分解产出，但四个阶段都读它。
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
from .spatio_temporal_decomposer import SpatioTemporalInstructionDecomposer
import logging


class Instruction:
    """
    Class encapsulating task instructions, decomposition, and subtask management.
    Each subtask has a final_coord attribute, recorded only when marked complete.
    """
    def __init__(self, full_instruction: str, sub_instructions: list = None):
        """
        Initialize Instruction object.
        Contains the full task instruction and an optional list of subtask instructions.
        :param full_instruction: full task instruction text.
        :param sub_instructions: decomposed subtask instruction list.
        """
        self.full_instruction = full_instruction
        self.sub_instruction_dict = {}
        for sub_inst in sub_instructions:
            key = f"SUBTASK_{sub_inst['stage_index']}"
            if key not in self.sub_instruction_dict:
                inst_body = {
                    'instruction': sub_inst['raw_text'],
                    'local_start': sub_inst['semantic_anchors']['local_start'],
                    'local_end': sub_inst['semantic_anchors']['local_end'],
                    'landmarks': sub_inst['semantic_anchors'].get('landmarks', []),
                    'spatial_geometry': sub_inst['kinematic_tube']['spatial_geometry'],
                    'temporal_duration': sub_inst['kinematic_tube']['temporal_duration'],
                    'relative_duration': sub_inst.get('relative_duration'),
                    # 分解器自己算的"本段终点和下一段起点是否语义一致"。
                    # 之前连存都没存。衔接不自洽的边界，完成判定该更保守。
                    'boundary_consistent': sub_inst.get('boundary_consistent'),
                    'boundary_overlap_score': sub_inst.get('boundary_overlap_score'),
                    'completed': False,
                    'plan': "",
                    'record': [],
                    'final_coord': None
                }
                self.sub_instruction_dict[key] = inst_body

        self.sub_instruction_keys = list(self.sub_instruction_dict.keys())
        self.current_subtask_index = 0
        self.current_step_count = 0
        self.num_subtasks = len(self.sub_instruction_keys)

    def get_current_subtask_key(self):
        """Get the current subtask key."""
        if self.current_subtask_index < self.num_subtasks:
            return self.sub_instruction_keys[self.current_subtask_index]
        return None

    def is_last_subtask(self):
        """Check if the current subtask is the last one."""
        return self.current_subtask_index == self.num_subtasks - 1

    def get_current_subtask(self):
        """Get current subtask instruction (key + dict)."""
        key = self.get_current_subtask_key()
        if key:
            return key, self.sub_instruction_dict[key]
        return None, None

    def update_plan_for_current_subtask(self, plan: str):
        """Update the plan text for the current subtask."""
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['plan'] = plan

    def add_record_to_current_subtask(self, record: dict):
        """Append an execution record to the current subtask."""
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['record'].append(record)

    def reset_subtask(self, key: str):
        """Reset the subtask for the given key to incomplete, clearing plan/records/coords/rotation."""
        key_upper = key.upper()
        if key_upper in self.sub_instruction_dict:
            self.sub_instruction_dict[key_upper]['completed'] = False
            self.sub_instruction_dict[key_upper]['plan'] = ""
            self.sub_instruction_dict[key_upper]['record'] = []
            self.sub_instruction_dict[key_upper]['final_coord'] = None

            self.sub_instruction_dict[key_upper]['final_rotation'] = None
        self.current_step_count = 0

    def get_subtask_by_key(self, key: str):
        """Get subtask instruction by key (case-insensitive)."""
        key_upper = key.upper()
        return self.sub_instruction_dict.get(key_upper, None)

    def get_all_subtasks_str(self):
        """
        Get JSON string representation of all subtasks (only current has record + final_coord).
        """
        result = []
        for idx, key in enumerate(self.sub_instruction_keys):
            subtask = self.sub_instruction_dict[key]

            if subtask['completed']:
                status = "completed"
            elif idx == self.current_subtask_index:
                status = "ongoing(current task)"
            else:
                status = "queuing"

            if idx == self.current_subtask_index:
                record_dict = {}
                for i, rec in enumerate(subtask['record']):
                    record_key = f"{key.lower()}.{i + 1}"
                    record_dict[record_key] = rec
            else:
                record_dict = {}

            task_info = {
                "task_name": key.lower(),
                "instruction": subtask['instruction'],
                "status": status,
                "record": record_dict,
                "your_last_plan": subtask['plan'],
            }
            result.append(task_info)

        all_tasks = json.dumps(result, ensure_ascii=False, indent=0)
        cur_task = f"\n\nCurrent task: {self.get_current_subtask_key()}: {self.get_current_subtask()[1]['instruction']}"
        return f"Full instruction: {self.full_instruction}\n" + all_tasks + cur_task

    def mark_current_subtask_completed(self, coord=None, rotation=None):
        """
        Mark the current subtask as completed and advance to the next.
        coord: final coordinate (x, y, z).
        rotation: final orientation quaternion (supports mn.Quaternion / [w,x,y,z] / objects with w,x,y,z).
        """
        key = self.get_current_subtask_key()
        if key:
            self.sub_instruction_dict[key]['completed'] = True

            if coord is not None:
                try:
                    self.sub_instruction_dict[key]['final_coord'] = [
                        float(coord[0]), float(coord[1]), float(coord[2])
                    ]
                except Exception:
                    self.sub_instruction_dict[key]['final_coord'] = None

            if rotation is not None:
                self.sub_instruction_dict[key]['final_rotation'] = rotation

            if self.current_subtask_index < self.num_subtasks - 1:
                self.current_subtask_index += 1
            self.current_step_count = 0


    def is_all_completed(self):
        """Check if all subtasks are completed."""
        return all(subtask['completed'] for subtask in self.sub_instruction_dict.values())

    def roll_back_to_specific_subtask(self, key: str):
        """Roll back to the subtask identified by key."""
        key_upper = key.upper()
        if key_upper in self.sub_instruction_keys:
            target_index = self.sub_instruction_keys.index(key_upper)
            self.current_subtask_index = target_index

            for i in range(target_index, self.num_subtasks):
                reset_key = self.sub_instruction_keys[i]
                self.reset_subtask(reset_key)

    def get_progress(self, total_step_budget=100, min_steps=8):
        """
        每个子任务的步数预算与当前消耗，供进度条渲染和prompt自省使用。

        预算来自分解器的 relative_duration（已归一化，求和为1，代表该子任务
        大约占整条指令总移动量的比例）。这个字段一直躺在 sub_instruction_dict
        里没人读——用它做预算比给所有子任务一个统一的硬编码阈值合理得多：
        "Turn left" 和 "walk down the long hallway" 显然不该有同样的步数配额。

        ratio 允许 >1（超预算），渲染层据此变红。
        """
        rows = []
        n = max(self.num_subtasks, 1)
        for idx, key in enumerate(self.sub_instruction_keys):
            st = self.sub_instruction_dict[key]
            rel = st.get('relative_duration')
            if not isinstance(rel, (int, float)) or rel <= 0:
                rel = 1.0 / n  # 分解器没给或给了非法值时退化成均分
            expected = max(float(min_steps), float(rel) * float(total_step_budget))

            if st['completed']:
                status, ratio = 'done', 1.0
            elif idx == self.current_subtask_index:
                status, ratio = 'current', self.current_step_count / expected
            else:
                status, ratio = 'todo', 0.0

            rows.append({
                'key': key,
                'rel': float(rel),
                'expected_steps': expected,
                'used_steps': self.current_step_count if status == 'current' else None,
                'status': status,
                'ratio': float(ratio),
            })
        return rows

    def get_current_step_budget(self, total_step_budget=100, min_steps=8,
                                slack=2.5, hard_min=25, hard_max=90):
        """
        当前子任务的 rollback 步数阈值。用 relative_duration 分配，再乘一个
        slack 容忍系数（预算是"理想路程"，实际总要多走一些绕路和转向）。

        夹在 [hard_min, hard_max] 之间：太小会让短子任务动不动就触发回退，
        太大就退化回原来那个对所有子任务一视同仁的硬编码 70。
        """
        key = self.get_current_subtask_key()
        if key is None:
            return hard_max
        st = self.sub_instruction_dict[key]
        rel = st.get('relative_duration')
        if not isinstance(rel, (int, float)) or rel <= 0:
            rel = 1.0 / max(self.num_subtasks, 1)
        expected = max(float(min_steps), float(rel) * float(total_step_budget))
        return int(max(hard_min, min(hard_max, expected * slack)))

    def get_subtasks_key_coord(self):
        """Get a dict of all completed subtask keys with their final coordinates."""
        result = {}
        for key in self.sub_instruction_keys:
            subtask = self.sub_instruction_dict[key]
            if subtask['completed'] and subtask['final_coord'] is not None:
                result[key] = subtask['final_coord']
        return result

    def get_subtask_pos_rot_by_key(self, key: str):
        """Get the final coordinate and rotation for the given subtask key. Returns (coord, rotation) tuple."""
        key_upper = key.upper()
        subtask = self.sub_instruction_dict.get(key_upper, None)
        if subtask and subtask['completed']:
            coord = subtask.get('final_coord', None)
            rotation = subtask.get('final_rotation', None)
            return coord, rotation
        return None, None
