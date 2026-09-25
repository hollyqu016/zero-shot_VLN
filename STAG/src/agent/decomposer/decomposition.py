"""
指令分解 · 调度层。

真正的分解逻辑在 spatio_temporal_decomposer.py，这里只负责调用它、
兜底、以及把结果装进 Instruction 对象。
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



class DecompositionMixin:
    def decompose_instruction(self, instruction: str) -> List[str]:
        """
        Call LLM to decompose a complex instruction into subgoals. Parses JSON {"subgoals": [...]} first;
        falls back to heuristic splitting on failure. Returns list of subgoal strings.
        """
        prompt = """
        Here is a visual language navigation instruction. The agent only needs to move to the designated position according to the instruction and does not operate any object.
        Decompose the following navigation instruction. Break down a complex instruction into multiple subtasks in sequence.
        The decomposition principle requires ensuring each subtask has a clear completion condition(end position); otherwise, it cannot be an independent subtask and you can combine multiple subtasks into one subtask if necessary.
        Each subtask must contain at least 3 objects or places to ensure clarity. Under the above condition, each sub-task should be as small as possible.
        **You can only split and merge based on the original instructions, and must not alter the original expression.** Do not add any additional explanations.
        For example, only "walk through the hallway" lacks a clear end position and should therefore be merged with subsequent subtasks until it's clear. But a subtask cannot contain more than three actions.
        First, analyze the instruction and write a draft.
        Finally, output in JSON format. Return JSON in the exact form: {\"analysis\": \"\", \"subtasks\": [\"...\", \"...\", ...]}\n\n
        SELF-CHECK BEFORE OUTPUT:
        - For every subtask, verify it is an exact substring of the original instruction.
        - Count objects/places (must be >= 2).
        - Count actions (must be <= 3).
        - If a subtask violates rules, MERGE it with adjacent text until it satisfies all rules.

        Instruction: {instruction}\n\n
        """

        prompt = prompt.replace("{instruction}", instruction)

        try:
            raw = self._get_llm_response(prompt)
        except Exception as e:
            raw = ""
        subs: List[str] = []

        try:
            m = re.search(r'\{.*\}', raw, flags=re.S)
            candidate = m.group(0) if m else raw
            data = json.loads(candidate)
            if isinstance(data, dict) and "subtasks" in data and isinstance(data["subtasks"], list):
                for s in data["subtasks"]:
                    if not isinstance(s, str):
                        continue

                    s_clean = re.sub(r'^\s*\d+\s*[\)\.\-:]*\s*', '', s).strip()
                    if s_clean:
                        subs.append(s_clean)
        except Exception:
            subs = []

        if not subs:
            parts = re.split(r'(?:then|and then|after that|;|\n|,|\.)+', instruction, flags=re.I)
            candidates = [p.strip() for p in parts if p and p.strip()]

            merged: List[str] = []
            for p in candidates:
                if len(p) < 6 and merged:
                    merged[-1] = (merged[-1] + ' ' + p).strip()
                else:
                    merged.append(p)
            subs = merged if merged else [instruction.strip()]

        if not subs:
            subs = [instruction.strip()]

        print(subs)

        return subs
