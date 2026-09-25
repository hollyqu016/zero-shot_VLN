"""
STAG 导航 agent。

按 VLN 的四个阶段拆成了子包，每个阶段一个 mixin：

    decomposer/   指令分解  —— 指令 → 时空约束
    perception/   感知      —— 观测 → 时空事实（建图、地标接地、俯视图）
    planning/     规划      —— 候选点、前沿打分、完成校验、prompt 组装
    action/       行动      —— 动作执行与主循环
    llm/          VLM 接口  —— 不属于四阶段，单独放

为什么用 mixin 而不是拆成协作对象：这几十个方法共享大量 self 状态
（mapper、instruction、current_path、各种粘性标志）。mixin 是**纯粹的
文件层面切分**，方法体一字未改，self.xxx 的读法完全照旧，因此这次重构
不可能改变任何运行时行为。真正的解耦要先盘清跨阶段字段的归属，那是
下一步的事——每个 mixin 的文件头已经记下了它读写哪些跨阶段字段。
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
from .decomposer.spatio_temporal_decomposer import SpatioTemporalInstructionDecomposer
import logging

from .common import robust_json_parse, _setup_proxy  # noqa: F401
from .decomposer import Instruction, DecompositionMixin
from .perception import (MappingMixin, GroundingMixin,
                         create_top_down_map_centered, create_top_down_map_global,
                         draw_direction_markers)  # noqa: F401
from .planning import WaypointMixin, FrontierMixin, VerificationMixin, PromptMixin, _SkipGuidance  # noqa: F401
from .action import ActionMixin
from .llm import LLMMixin

# VLM 侧的检测子任务提示。由 enable_vlm_detection 开关决定是否拼进 prompt。
_DETECTION_SIDE_TASK = """
        **Object detection side-task**
        In every reply you must also fill a `detections` field: the objects you can see in the
        FRONT VIEW image, with bounding boxes. These are fused into the top-down map you receive
        next turn, so the more accurate they are, the better your own map becomes.

        - Report objects from the FRONT VIEW only. Ignore the left/right views and the map for this field.
        - Coordinates are integers from 0 to 1000, `[xmin, ymin, xmax, ymax]`, where (0,0) is the
          top-left corner of the front view and (1000,1000) is the bottom-right corner.
        - `c` MUST be chosen from this fixed vocabulary (pick the closest match, never invent a name;
          strip modifiers — "dining table" is `table`, "front door" is `door"):
          [{vocab}]
          Skip any object that does not fit the vocabulary — do not output `other`.
        - Report at most 10 objects, prioritising large, clearly visible, and navigationally
          meaningful ones. An empty list `[]` is fine if you see nothing worth reporting.
"""


_setup_proxy()


_setup_proxy()


class PathPlannerAgent(
    DecompositionMixin,     # 指令分解
    MappingMixin,           # 感知 · 建图
    GroundingMixin,         # 感知 · 地标接地
    WaypointMixin,          # 规划 · 候选点与决策调度
    FrontierMixin,          # 规划 · 前沿打分
    VerificationMixin,      # 规划 · 完成校验与停止拦截
    PromptMixin,            # 规划 · prompt 组装
    ActionMixin,            # 行动 · 动作执行
    LLMMixin,               # VLM 接口
    ABC,
):
    """
    An Agent whose decision logic is:
    1. Generate waypoints in current view plus basic actions (L, R, 0).
    2. Let the VLM choose a target waypoint or action.
    3. Use A* to plan a path to that point.
    4. Decompose the path into actions and execute step-by-step, updating the map each step.
    """
    def __init__(self, sim_wrapper: SimWrapper, config: dict, instruction: str, init_position=None, init_rotation=None, mode='objnav'):
        self.sim_wrapper = sim_wrapper
        self.config = config
        self.forward_step = config.get('forward_step', 1.0)
        self.turn_angle_rad = np.deg2rad(config.get('turn_angle_deg', 90))
        self.instruction = instruction
        self.curr_obs = None
        self.last_action = PolarAction.null
        self.last_actions = {}
        self._recent_positions = []
        self.resolution = (config['camera']['height'], config['camera']['width'])
        self.focal_length = calculate_focal_length(config['camera']['fov'], self.resolution[1])
        self.image_edge_threshold = 0.04
        self.traveled_distance = 0.0
        self.prev_agent_position = None
        self.vlm_responses = []
        self.mode = mode
        self.error = False
        self.vln_instruction_planner = SpatioTemporalInstructionDecomposer()
        sub_insts = self.vln_instruction_planner.decompose(instruction)
        logging.info(f"第一版指令分解: {sub_insts}")
        self.instruction_obj = Instruction(full_instruction=instruction, sub_instructions=sub_insts)

        if init_position is not None and init_rotation is not None:
            self.sim_wrapper.set_initial_state(init_position, init_rotation)

        # ---- 算法超参 ----
        # 全部从 config['hyper'] 取，取不到就用类常量的同值默认（见各 mixin 的类定义）。
        # 分组与 config/vlnce_test.yaml 的 hyper: 段一一对应。
        #
        # 位置很关键：**必须在首帧建图之前**。下面 self.update_map(obs) 会走
        # 地标接地和进度追踪，那两条路径都会读这些属性。之前把这段放在消融
        # 开关之后（也就是建图之后），实跑第一帧就报
        #     'GPTAgent' object has no attribute 'LM_RECEDE_MARGIN'
        # 而 grounding 的 except 把它降级成一行 "skipped due to error"，
        # 整个 episode 的首帧接地静默失效。
        self._apply_hyper(config)

        camera_params = config['camera']
        cam_intrinsics = np.array([
            [camera_params['fx'], 0, camera_params['width'] / 2],
            [0, camera_params['fy'], camera_params['height'] / 2],
            [0, 0, 1]
        ])

        _map = config.get('map', {})
        self.mapper = Instruct_Mapper(camera_intrinsic=cam_intrinsics,
                                      grid_resolution=_map.get('grid_resolution', 0.05),
                                      floor_height=_map.get('floor_height', -1.2),
                                      ceiling_height=_map.get('ceiling_height', 0.6),
                                      pcd_resolution=_map.get('pcd_resolution', 0.025),
                                      resolution=(config['camera']['height'], config['camera']['width']),
                                      config=config)

        initial_state = self.sim_wrapper.sim.get_agent(0).get_state()
        self.mapper.reset(initial_state.position, initial_state.rotation)

        obs = sim_wrapper.step(PolarAction(r=0, theta=0))
        self.mapper.reset(obs['agent_state'].position, obs['agent_state'].rotation)
        self.instances = self.update_map(obs)
        self.curr_obs = obs
        self.turn_cooldown = 0
        self.turn_count = 0
        self.current_plan = None
        self.last_vlm_response = "Executing initial action."
        self.target_world_position = [0, 0, 0]
        self.img_buffer = []
        self.min_move_distance = 0.4
        self.max_move_distance = 3

        self.init_pos = self.mapper.current_position - [0, 0, 1.2]
        self.init_rot = self.mapper.current_rotation

        # 子任务步数预算的分配总量。
        #
        # 注意这里不能直接用 max_steps：max_steps 是硬上限，不是"预期花费"。
        # 早先直接令 total_step_budget = max_steps = 100，后果是"用完整个 episode"
        # 才算刚好在预算内，于是所有下游阈值都够不着：
        #   2 个子任务时 rel=0.4 -> 预期 40 步 -> 强制推进要 2.5x = 100 步 = 整个episode
        #   实测 FORCE-ADVANCED 和 ROLLBACK 在 22 个 episode 里触发 0 次，
        #   而同期有 episode 在第一个子任务上耗掉 70 步。
        # 现在只把上限的 60% 当作预期路程，剩下 40% 留给绕路、转向和纠错。
        _max_steps = int(config.get('max_steps', 100))
        self.total_step_budget = max(10, int(_max_steps * float(config.get('budget_ratio', 0.6))))

        self._last_scored_frontiers = None
        self.rejected_waypoints = []

        # ---- 消融开关 ----
        # 连续两轮改动都让指标变差，而每次我都只能猜是哪一层的锅。与其继续
        # 盲改，不如让"关掉某一层再跑"变成一条命令。默认全开，行为不变。
        #   STZS_GUIDANCE=0      关掉前沿引导（explore 提示 + 地图上的 F 环）
        #   STZS_INTERCEPTION=0  关掉完成/停止拦截（顺序门、提前完成、走开、太远）
        def _flag(env, key, default=1):
            v = os.environ.get(env)
            return bool(int(v)) if v is not None else bool(int(config.get(key, default)))

        self.enable_guidance = _flag('STZS_GUIDANCE', 'enable_guidance')
        self.enable_interception = _flag('STZS_INTERCEPTION', 'enable_interception')
        # 前沿"建议"和前沿"动作"是两件不同的事：前者影响 VLM 的判断，
        # 后者扩大它的动作空间。捆在一个开关里，赢了也不知道是哪一半的功劳，
        # 所以单独拆出来。默认跟随 guidance。
        self.enable_frontier_action = _flag(
            'STZS_FRONTIER_ACTION', 'enable_frontier_action',
            1 if self.enable_guidance else 0)
        # 让决策VLM顺带输出检测框。实测每次只吐 1 个框，融进地图的贡献很小，
        # 却让 schema 变长、输出变长——而时间几乎全花在 VLM 上（单次约 30s，
        # 占总时长 90%+）。默认关掉，需要时再开。
        self.enable_vlm_detection = _flag('STZS_VLM_DETECTION',
                                          'enable_vlm_detection', 0)
        print(f"[config] guidance={self.enable_guidance} "
              f"frontier_action={self.enable_frontier_action} "
              f"interception={self.enable_interception} "
              f"vlm_detection={self.enable_vlm_detection} "
              f"budget={self.total_step_budget}")

        self.first_stop = False
        self.second_stop = False

        self.rolling_back = False

        stair_words = ['stair', 'step']
        if any(word in instruction.lower() for word in stair_words):
            pass
            self.mapper.enable_stair_detection = True
        else:
            self.mapper.enable_stair_detection = False

    def _apply_hyper(self, config):
        """
        用 config['hyper'] 覆盖各 mixin 的类级默认。

        **必须在首帧建图之前调用**——update_map 会走地标接地与进度追踪，
        那两条路径都读这些属性。

        为什么类级默认也要保留：测试桩和离线脚本会直接构造对象而不走这个
        方法；更要紧的是地标接地那条路径外面包着 except，一旦 AttributeError
        会被降级成一行 "skipped due to error"，静默失效比直接崩难查得多。
        """
        lm = hyper(config, 'landmark')
        self.NEAR_M = lm('near_m', self.NEAR_M)
        self.PASSED_MIN_M = lm('passed_min_m', self.PASSED_MIN_M)
        self.PASSED_GAP_M = lm('passed_gap_m', self.PASSED_GAP_M)
        self.LM_RECEDE_MARGIN = lm('recede_margin', self.LM_RECEDE_MARGIN)
        self.LM_RECEDE_STEPS = lm('recede_steps', self.LM_RECEDE_STEPS)

        fr = hyper(config, 'frontier')
        self.FRONTIER_MIN_USEFUL_M = fr('min_useful_m', self.FRONTIER_MIN_USEFUL_M)
        self.FRONTIER_STEP_LEN = fr('step_len', self.FRONTIER_STEP_LEN)
        self.FRONTIER_TOP_K = fr('top_k', self.FRONTIER_TOP_K)

        st = hyper(config, 'subtask')
        self.PREMATURE_MIN_RATIO = st('premature_min_ratio', self.PREMATURE_MIN_RATIO)
        self.PREMATURE_SHAKY_RATIO = st('premature_shaky_ratio', self.PREMATURE_SHAKY_RATIO)
        self.VERIFY_NEAR_M = st('verify_near_m', self.VERIFY_NEAR_M)
        self.TOO_FAR_OBJECT_LIMIT = st('too_far_object_limit', self.TOO_FAR_OBJECT_LIMIT)
        self.RECEDE_MARGIN = st('recede_margin', self.RECEDE_MARGIN)
        self.FORCE_ADVANCE_RATIO = st('force_advance_ratio', self.FORCE_ADVANCE_RATIO)
        self.FORCE_ADVANCE_MIN_STEPS = st('force_advance_min_steps', self.FORCE_ADVANCE_MIN_STEPS)
        self.ROLLBACK_STEP_THRESHOLD = st('rollback_step_threshold', self.ROLLBACK_STEP_THRESHOLD)

        for line in (lm.describe(), fr.describe(), st.describe()):
            if line:
                print(f"[agent] {line}")

    def close(self):
        """Close and release GPU memory/large object refs; safe to call repeatedly."""
        if getattr(self, "_ended", False):
            return

        try:
            if hasattr(self, "mapper") and self.mapper is not None:

                try:
                    release = getattr(self.mapper, "release_all_vram", None)
                    if callable(release):
                        release()
                except Exception:
                    pass
                finally:

                    self.mapper = None
        except Exception:
            pass

        for attr in ("curr_obs", "instances", "last_actions", "img_buffer"):
            try:
                setattr(self, attr, None)
            except Exception:
                pass

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        try:
            import gc
            gc.collect()
        except Exception:
            pass

        self._ended = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


import base64
from io import BytesIO
from openai import AzureOpenAI, OpenAI


class GPTAgent(PathPlannerAgent):
    """
    VLM agent implementation using OpenAI / Azure OpenAI compatible API.
    Configured via environment variables:
      - OPENAI_API_KEY   API key
      - OPENAI_BASE_URL  (optional) custom base_url
      - AZURE_OPENAI_ENDPOINT   Azure endpoint
      - AZURE_OPENAI_API_KEY   Azure API key
      - AZURE_OPENAI_API_VERSION  Azure API version
    """

    def __init__(self, sim_wrapper, config, instruction, initial_position=None, initial_rotation=None, model_name=None, mode='vln'):
        self.model_name = model_name or os.environ.get("OPENAI_MODEL", "gpt-5")

        azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")

        if azure_endpoint and azure_api_key:
            self.client = AzureOpenAI(api_version=azure_api_version, azure_endpoint=azure_endpoint, api_key=azure_api_key)
        else:
            self.client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                base_url=os.environ.get("OPENAI_BASE_URL"),
            )
        sys_prompt = """

        """
        self.history_msgs = [
            {"role": "system", "content": "You are an agent good at navigating in an indoor environment."}
        ]
        super().__init__(sim_wrapper, config, instruction, initial_position, initial_rotation, mode)

    def _pil_to_base64(self, pil_img):
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return img_str

    def _get_vlm_response(self, rgb_image: Image.Image, map_image: Image.Image, prompt: str, img_buffer=None) -> str:
        if img_buffer is None:
            img_buffer = []
        try:
            rgb_img_b64 = self._pil_to_base64(rgb_image)
            map_img_b64 = self._pil_to_base64(map_image)

            content_for_request = [{"type": "text", "text": prompt}]
            for img in img_buffer:
                img_b64 = self._pil_to_base64(img)
                content_for_request.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
                })
            content_for_request.extend([
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_img_b64}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "high"}}
            ])
            current_msg_for_api = {"role": "user", "content": content_for_request}

            content_for_history = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rgb_img_b64}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "auto"}},
            ]
            current_msg_for_history = {"role": "user", "content": content_for_history}

            if not hasattr(self, "history_msgs"):
                self.history_msgs = [
                    {"role": "system", "content": "You are an agent good at navigating in an indoor environment."}
                ]

            messages_for_api = self.history_msgs + [current_msg_for_api]

            user_msgs = [msg for msg in messages_for_api if msg["role"] == "user"]
            if len(user_msgs) > 7:
                user_count = 0
                for msg in messages_for_api:
                    if msg["role"] == "user":
                        user_count += 1
                        if user_count <= len(user_msgs) - 7:
                            if isinstance(msg["content"], list):
                                msg["content"] = [c for c in msg["content"] if c["type"] == "text"]

            max_turn = 30
            if len(messages_for_api) > max_turn:
                messages_for_api = messages_for_api[-max_turn:]

                if messages_for_api[0]["role"] != "system":
                    messages_for_api.insert(0, {"role": "system",
                                                "content": "You are an agent good at navigating in an indoor environment."})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,

            )

            result = response.choices[0].message.content
            print(result)

            self.history_msgs.append(current_msg_for_history)
            self.history_msgs.append({
                "role": "assistant",
                "content": result
            })
            return result
        except Exception as e:
            # 之前这里静默返回 "stop"，上层解析失败后完全看不出是网络挂了、
            # 后端 500 还是别的。一次实跑里模型服务不可达，29 个 episode 全在
            # 第 2 步死掉，日志里除了 Connection error 什么线索都没有。
            print(f"[vlm] request failed: {type(e).__name__}: {e}")
            return "stop"

    def _get_vlm_response_multiview(self, front_rgb_image: PIL.Image.Image, left_rgb_image: PIL.Image.Image,
                                    right_rgb_image: PIL.Image.Image, back_rgb_image: PIL.Image.Image,
                                    map_image: PIL.Image.Image, prompt: str, img_buffer: list) -> str:
        if img_buffer is None:
            img_buffer = []
        try:

            front_img_b64 = self._pil_to_base64(front_rgb_image)
            left_img_b64 = self._pil_to_base64(left_rgb_image)
            right_img_b64 = self._pil_to_base64(right_rgb_image)
            # back_img_b64 = self._pil_to_base64(back_rgb_image)
            map_img_b64 = self._pil_to_base64(map_image)

            content_for_request = [{"type": "text", "text": prompt}]
            for img in img_buffer[-20:]:
                img_b64 = self._pil_to_base64(img)
                content_for_request.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "auto"}
                })

            content_for_request.extend([

                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{left_img_b64}", "detail": "high"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{right_img_b64}", "detail": "high"}},
                # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{back_img_b64}", "detail": "high"}},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{front_img_b64}", "detail": "high"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "high"}},
            ])
            current_msg_for_api = {"role": "user", "content": content_for_request}

            content_for_history = [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{front_img_b64}", "detail": "auto"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{map_img_b64}", "detail": "auto"}},
            ]
            current_msg_for_history = {"role": "user", "content": content_for_history}

            # f-string：下面的检测词表约束需要把 category_names_str 插进来。
            # 注意本块里原有的 markdown 反引号内容不含花括号，改成 f-string 是安全的。
            det_task = (_DETECTION_SIDE_TASK.format(vocab=category_names_str)
                        if self.enable_vlm_detection else "")
            sys_prompt = f"""
                    **Image Inputs**:
        1.  **Top-Down Map**: This is your memory. It will be updated as you explore. It's oriented with you facing upwards.
            -   `Gray`: Navigable floor you have seen.
            -   `Black`: Obstacles or walls that cannot be passed.
            -   `Red Arrow`: Your current position and direction.
            -   `Red Trail`: Your recent trajectory.
            -   `Start Point`: Your starting position.
            -   `Text Labels`: Automatically detected objects (may be inaccurate).
            -   `F1 / F2 / F3 rings`: unexplored openings (frontiers), ranked by how likely they
                lead to what the current subtask needs. The ranking combines WHERE the target
                probably is (which room type contains it, and whether such a room is already on
                the map) with WHETHER you can still get there on the remaining step budget.
                F1 is the system's best guess. **You can select these directly** — reply with
                "action": "F1" and a multi-step path will be planned all the way there. Prefer that
                over nudging forward one waypoint at a time whenever the thing you need is not yet
                in sight. Ring size = how wide the opening is.
            -   `Numbered Circles`: Candidate waypoints projected into your view.
            -   `Landmark Pins`: Objects from the CURRENT subtask that have been located on the map.
                `DEST:` (green filled circle, green dashed line to you) = the destination object of this subtask.
                `VIA:` (blue hollow circle) = an object you should pass by on the way.
                `AVOID:` (red cross in a red shaded circle) = an object/area to stay away from.
                Each pin shows its distance from you. `(1/N)` means N objects of that class were found
                and this is the one picked — treat such pins with suspicion and verify in your view.
            -   `Cool-grey tinted floor`: narrow, corridor-like space you have already walked through
                (measured from the geometry, not from furniture). Normal grey floor = open room-like
                space or somewhere you have not walked yet.
            -   `Label under the red arrow`: what kind of space you are standing in right now
                (`corridor` or `room`), how much clearance you have, and — when there is enough
                furniture evidence — which room it is, e.g. `corridor 0.8m` or `room 2.4m in bedroom`.
                Use this to answer "am I in the hallway yet?" / "which room is this?" — it is
                measured from the map, not guessed from the image.
            -   `Bottom progress bar`: one segment per subtask, width proportional to how much of the
                whole route that subtask is expected to take. Green = finished, blue = current
                (filled by steps used / steps expected), RED fill = current subtask is over budget,
                pale grey = not started.
            -   `Area Blobs`: large translucent shaded circles marking ROOMS, inferred from the
                furniture seen inside them (a bed implies a bedroom, a fridge+stove implies a kitchen).
                The circle is an approximate extent, not an exact room boundary — walls still apply,
                so route through doorways, not across the circle edge.
                Only rooms named by the CURRENT subtask are drawn (a target, or an area to avoid).
                To find out which room you are standing in right now, read the label under the red
                arrow instead — it is more reliable than guessing from the first-person view.
            -   `Top-left checklist`: current subtask landmarks, with how far along you are with each:
                `*` = located on the map but you have not been near it yet,
                `>` = you are right next to it now,
                `V` = you were close to it earlier and have since moved away (you have PASSED it),
                `#` = room/area located on the map,
                `o` = not found yet (keep exploring to find it),
                `x` = can never appear as a pin (either not in the detector's vocabulary, or a
                corridor-like area with no furniture to identify it) — for these, rely purely on
                your first-person view.
        2.  **First-Person View**: This is what you see right now. It is divided into 3 perspectives: front, left, right.
            -   `Numbered Circles`: Candidate waypoints projected into your front view.
            -   `L, R, B Circles`: Turn actions (L: Left 90°, R: Right 90°, B: Turn Around 180°).
            -   `Action Text`: The last action you executed.

        **Important Notes**:
        1. Labels on the map can be wrong; trust your first-person view more.
        2. You cannot walk through or open closed doors. The target will not be in such places. There's no need to try to open doors.
        3. You can turn right/left/round in place to observe the surrounding environment.
        4. Avoid going back to places you've already been unless necessary!!!
        5. There's no need to operate objects, just move to the target position.
        6. Red waypoints indicate locations that have been visited, and white ones indicate those that haven’t.

{det_task}            """

            if not hasattr(self, "history_msgs"):
                self.history_msgs = [
                    {"role": "system", "content": sys_prompt}
                ]

            messages_for_api = self.history_msgs + [current_msg_for_api]

            user_msgs = [msg for msg in messages_for_api if msg["role"] == "user"]
            if len(user_msgs) > 8:
                user_count = 0
                for msg in messages_for_api:
                    if msg["role"] == "user":
                        user_count += 1
                        if user_count <= len(user_msgs) - 8:
                            msg["content"] = [c for c in msg["content"] if c["type"] == "text"]

            max_turn = 30
            if len(messages_for_api) > max_turn:
                messages_for_api = messages_for_api[-max_turn:]
                if messages_for_api[0]["role"] != "system":
                    messages_for_api.insert(0, {"role": "system",
                                                "content": "You are an agent good at navigating in an indoor environment."})

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages_for_api,
                # temperature=0.9,
                # extra_body={"vl_high_resolution_images": False},
            )
            result = response.choices[0].message.content
            print(result)

            self.history_msgs.append(current_msg_for_history)
            self.history_msgs.append({
                "role": "assistant",
                "content": result
            })
            return result
        except Exception as e:
            # 之前这里静默返回 "stop"，上层解析失败后完全看不出是网络挂了、
            # 后端 500 还是别的。一次实跑里模型服务不可达，29 个 episode 全在
            # 第 2 步死掉，日志里除了 Connection error 什么线索都没有。
            print(f"[vlm] request failed: {type(e).__name__}: {e}")
            return "stop"

    def _get_llm_response(self, prompt: str) -> str:
        """
        Call the text LLM (no image, no chat history) for instruction decomposition.
        Returns the model's plain-text response, or "stop" on failure.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            if hasattr(response, "choices") and len(response.choices) > 0:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    return choice.message.content
                if hasattr(choice, "text"):
                    return choice.text
                return str(choice)
            if hasattr(response, "output_text"):
                return response.output_text
            return str(response)
        except Exception as e:
            pass
            return "stop"
