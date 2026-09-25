"""
VLM / LLM 调用接口（抽象层）。

这一层不属于 VLN 的四个阶段，单独放。基类只定义签名，具体实现在
GPTAgent 里——换模型后端只需要改那一处。
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



class LLMMixin:
    def _get_vlm_response(self, front_rgb_image: PIL.Image.Image, map_image: PIL.Image.Image, prompt: str, img_buffer: list) -> str:
        """
        Call VLM API to get response for image and prompt.
        Subclasses must implement to call a specific VLM (e.g., GPT-4V).

        :param front_rgb_image:
        :param map_image:
        :param prompt:
        :param img_buffer:
        :return: VLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_vlm_response.")

    def _get_vlm_response_multiview(self, front_rgb_image: PIL.Image.Image, left_rgb_image: PIL.Image.Image, right_rgb_image: PIL.Image.Image, back_rgb_image: PIL.Image.Image, map_image: PIL.Image.Image, prompt: str, img_buffer: list) -> str:
        """
        Call VLM API with multi-view images and prompt.
        Subclasses must implement to call a specific VLM (e.g., GPT-4V).

        :param front_rgb_image:
        :param left_rgb_image:
        :param right_rgb_image:
        :param back_rgb_image:
        :param map_image:
        :param prompt:
        :param img_buffer:
        :return: VLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_vlm_response_multiview.")

    def _get_llm_response(self, prompt: str) -> str:
        """
        Call text LLM API for prompt response.
        Subclasses must implement to call a specific LLM.

        :param prompt:
        :return: LLM text response.
        """
        raise NotImplementedError("Subclasses must implement _get_llm_response.")
