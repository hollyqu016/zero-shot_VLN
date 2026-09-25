"""
行动 · 动作执行与主循环。

step() 是每个物理步的入口：拿决策 → 转成 PolarAction → 送进仿真器。

注意 challenge_stop 分支**不清空 self.current_path**。清空会让 agent 在
被拦截后原地重规划，白白多花 VLM 调用；保留路径则继续沿原路走。
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

from ..perception.topdown_map import create_top_down_map_centered


class ActionMixin:
    def _stop(self):
        """Execute stop action and update sub-goal state."""
        if not self.first_stop:
            pass
            self.first_stop = True
            return PolarAction.stop
        elif not self.second_stop:
            pass
            self.second_stop = True
            return PolarAction.stop
        else:
            pass
            return PolarAction.null()

    def _get_agent_yaw(self):
        """
        Extract yaw angle (around z) from self.mapper.current_rotation (3x3 rotation matrix).
        """
        R = np.asarray(self.mapper.current_rotation)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _rot_to_yaw(self, rotation):
        """
        Convert mapper.rotation to yaw angle.
        :param rotation:
        :return:
        """
        R = np.asarray(rotation)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _normalize_angle(self, a):
        return float((a + np.pi) % (2 * np.pi) - np.pi)

    def step(self, step, rollback_step_threshold=None):
        """
        Execute one step based on prior agent decision; returns obs, top-down map, VLM response, labeled PIL image.
        """
        rollback_step_threshold = (self.ROLLBACK_STEP_THRESHOLD if rollback_step_threshold is None
                                   else rollback_step_threshold)
        action_to_execute = None
        pil_labeled_img = None

        if not hasattr(self, '_position_history'):
            self._position_history = []
        if not hasattr(self, '_stuck_threshold'):
            self._stuck_threshold = 0.2
        if not hasattr(self, '_stuck_check_window'):
            self._stuck_check_window = 8
        if not hasattr(self, 'action_sequence'):
            self.action_sequence = []
        if not hasattr(self, 'current_path'):
            self.current_path = []

        if not hasattr(self, '_heading_before_nav'):
            self._heading_before_nav = 0.0
        if not hasattr(self, '_need_restore_heading'):
            self._need_restore_heading = False
        if not hasattr(self, '_no_move_steps'):
            self._no_move_steps = 0
        if not hasattr(self, '_global_stuck_limit'):
            self._global_stuck_limit = 20
        if not hasattr(self, '_no_move_epsilon'):
            self._no_move_epsilon = 1e-1

        current_pos = np.array(self.mapper.current_position[:2])
        self._position_history.append(current_pos)
        agent_state = self.curr_obs['agent_state']

        if len(self._position_history) > self._stuck_check_window + 1:
            self._position_history.pop(0)

        if self.prev_agent_position is not None:
            prev2d = np.array(self.prev_agent_position[:2])
            if np.linalg.norm(agent_state.position[:2] - prev2d) < self._no_move_epsilon:
                self._no_move_steps += 1
            else:
                self._no_move_steps = 0
        else:

            self._no_move_steps = 0

        if self._no_move_steps >= self._global_stuck_limit and action_to_execute is None:
            pass
            self.last_vlm_response = "(Auto)Global-stuck detected: stopping."
            self.vlm_responses.append(self.last_vlm_response)

            self.current_path = []
            self.action_sequence = []
            self._need_restore_heading = False
            action_to_execute = PolarAction.stop
            self.second_stop = True

        is_stuck = False
        if len(self._position_history) >= self._stuck_check_window:
            position_changes = [
                np.linalg.norm(self._position_history[i] - self._position_history[i - 1])
                for i in range(1, len(self._position_history))
            ]
            if len(position_changes) >= self._stuck_check_window and all(
                    change < self._stuck_threshold for change in position_changes[-self._stuck_check_window:]
            ):
                is_stuck = True
                print(
                    f"detected agent stuck, last{self._stuck_check_window} steps position changes: {position_changes[-self._stuck_check_window:]}")

                if self._need_restore_heading:
                    self.current_path = []
                    self.action_sequence = []

                # self._need_restore_heading = False

        if self._need_restore_heading and not self.current_path and not self.action_sequence:
            curr_yaw = self._get_agent_yaw()
            delta = - self._normalize_angle(self._heading_before_nav - curr_yaw)
            restore_threshold = math.radians(30)
            if abs(delta) > restore_threshold:
                if not self.rolling_back:
                    action_to_execute = PolarAction(r=0, theta=math.copysign(abs(delta) - restore_threshold, delta),
                                                type='turn')
                else:
                    action_to_execute = PolarAction(r=0, theta=delta, type='turn')
                    self.instruction_obj.current_step_count = 0
                    self.rolling_back = False

            self._need_restore_heading = False

        if action_to_execute is None and ((not self.current_path and not self.action_sequence) or is_stuck):
            if is_stuck:
                pass
                self.current_path, self.action_sequence, self._position_history = [], [], [current_pos]

            # 按 relative_duration 给当前子任务单独算阈值，而不是所有子任务
            # 共用一个硬编码的 70。step() 的形参保留作为上限兜底。
            budget = self.instruction_obj.get_current_step_budget(
                total_step_budget=self.total_step_budget, hard_max=rollback_step_threshold)

            # 强制推进要在决策之前，这样本轮决策就已经面向新的子任务了。
            # 放在 rollback 判断之前也是有意的：能往前推就别往回退——
            # "已经做完了只是不肯承认" 比 "彻底迷路" 常见得多。
            self._maybe_force_advance()

            if self.instruction_obj.current_step_count <= budget:
                decision, vlm_response, pil_labeled_img = self.decide_waypoint(
                    is_stuck=is_stuck, is_first_step=step == 2
                )
            else:
                # 这条以前只写进 vlm_responses（进可视化面板），日志里完全看不到，
                # 导致排查时无法判断rollback到底有没有触发过。补一行print。
                print(f"[subtask] ROLLBACK {self.instruction_obj.get_current_subtask_key()}: "
                      f"step_count {self.instruction_obj.current_step_count} > budget {budget}")
                decision, vlm_response, pil_labeled_img = {"type": "rollback", "subtask_key": self.instruction_obj.get_current_subtask_key()}, f"(Auto)Exceeded step budget for this subtask ({self.instruction_obj.current_step_count} > {budget}), initiating rollback to subtask start.", None

            self.vlm_responses.append(vlm_response)
            self.last_vlm_response = vlm_response
            self.img_buffer = []

            if decision is None:
                self._stop()
                action_to_execute = PolarAction.stop
                self.error = True

            elif decision['type'] == 'stop':
                if self.instruction_obj.is_last_subtask():
                    action_to_execute = PolarAction.stop
                    if self.first_stop:
                        self.instruction_obj.mark_current_subtask_completed(self.mapper.current_position - [0, 0, 1.2], self.mapper.current_rotation)
                    self._stop()
                else:
                    action_to_execute = PolarAction.pause()
                    self.instruction_obj.mark_current_subtask_completed(self.mapper.current_position - [0, 0, 1.2], self.mapper.current_rotation)

            elif decision['type'] == 'challenge_stop':
                # 提前宣告完成被质询：不推进子任务，下一轮 prompt 带上
                # stop_challenge 文本要求它复核。
                #
                # 关键是**不清空 current_path**。之前这里清了路径，导致下一步
                # 必然重新决策——每次拦截都要多烧一次 VLM 调用。而耗时几乎
                # 全在 VLM 上（单次约 30s，占总时长 90%+），一次拦截等于付
                # 两次调用的钱（被拦那次 + 重新决策那次）。
                # 保留路径的话，被拦之后沿原路继续走，质询照样在下一轮送到。
                if self.current_path:
                    action_to_execute = self._get_action_for_next_waypoint()
                if action_to_execute is None:
                    # 确实没有在途路径了，才原地停一步
                    action_to_execute = PolarAction.pause()
                    self._need_restore_heading = False

            elif decision['type'] == 'turn':
                action_to_execute = PolarAction(r=0, theta=decision['angle'], type='turn')
                self.current_path = []
                self._need_restore_heading = False

            elif decision['type'] == 'look_around':
                pass
                turn_right_action = PolarAction(r=0, theta=-self.turn_angle_rad*2, type='turn')
                self.action_sequence = [turn_right_action] * 2
                action_to_execute = self.action_sequence.pop(0)
                self._need_restore_heading = False

            elif decision['type'] in ('waypoint', 'frontier'):
                # 前沿和路点走同一套执行逻辑：A* 规划 + 逐点跟随。
                # 区别只在前沿通常远得多，一次决策会展开成很多步。
                target_world = decision['target_point_world']
                path = self.mapper.plan_path_to_target(target_world)
                self.target_world_position = target_world

                if path and len(path) > 1:

                    self._heading_before_nav = self._get_agent_yaw()
                    self._need_restore_heading = True

                    self.current_path = path[1:]
                    action_to_execute = self._get_action_for_next_waypoint()
                else:
                    pass
                    self.current_path = []
                    self._need_restore_heading = False
                    action_to_execute = PolarAction(r=0, theta=np.pi, type='turn')
            elif decision['type'] == 'rollback':

                subtask_key = decision.get('subtask_key', None)
                path, target_rot = self.plan_rollback_path(subtask_key=subtask_key)
                if path and len(path) > 1:
                    self.current_path = path[1:]
                    self.target_world_position = path[-1]

                    self._heading_before_nav = self._rot_to_yaw(target_rot)
                    self._need_restore_heading = True
                    action_to_execute = self._get_action_for_next_waypoint()
                    self.instruction_obj.add_record_to_current_subtask({"rollback": f"System: Too many steps have been taken without progress. Maybe you got lost? Rolling back to the start of subtask '{subtask_key}'."})
                    self.instruction_obj.current_step_count = 0
                    self.rolling_back = True
                else:
                    pass
                    self.current_path = []
                    self._need_restore_heading = False
                    action_to_execute = PolarAction(r=0, theta=np.pi, type='turn')
                    self.instruction_obj.current_step_count = 0

        elif action_to_execute is None and self.action_sequence:
            action_to_execute = self.action_sequence.pop(0)

        elif action_to_execute is None:
            action_to_execute = self._get_action_for_next_waypoint()

        self.instruction_obj.current_step_count += 1

        if action_to_execute is None:
            action_to_execute = PolarAction.null
            self.current_path = []

        self.execute_action(action_to_execute)

        if self.prev_agent_position is not None:
            self.traveled_distance += np.linalg.norm(agent_state.position - self.prev_agent_position)

        self.prev_agent_position = agent_state.position

        top_down_map = create_top_down_map_centered(
            self.mapper,
            self.config['camera']['fov'],
            self.target_world_position,
            action_candidates=self.last_actions,
            subtasks=self.instruction_obj.get_subtasks_key_coord(),
            progress=self.instruction_obj.get_progress(self.total_step_budget),
            debug_overlay=True,   # 这一路只进录像帧，可以放调试信息
            rejected_waypoints=getattr(self, 'rejected_waypoints', None),
            scored_frontiers=getattr(self, '_last_scored_frontiers', None),
        )

        if pil_labeled_img is None:
            current_image = self.curr_obs['color_sensor'][:, :, :3].copy()
            if action_to_execute is not None:
                action_text = self._action_to_text(action_to_execute)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)
                padding = 10
                (text_width, text_height), baseline = cv2.getTextSize(
                    action_text, font, font_scale, font_thickness
                )
                overlay = current_image.copy()
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (text_width + 2 * padding, text_height + 2 * padding + baseline),
                    bg_color,
                    -1
                )
                cv2.addWeighted(overlay, 0.6, current_image, 0.4, 0, current_image)
                cv2.putText(
                    current_image,
                    action_text,
                    (padding, text_height + padding),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )

            pil_labeled_img = Image.fromarray(current_image, 'RGB')
            top_down_map_rgb = cv2.cvtColor(top_down_map, cv2.COLOR_BGR2RGB)
            pil_map_image = Image.fromarray(top_down_map_rgb)
            self.img_buffer.append(pil_labeled_img)
            # self.img_buffer.append(pil_map_image)

        return self.curr_obs, top_down_map, self.last_vlm_response, pil_labeled_img

    def _action_to_text(self, action: PolarAction) -> str:
        """
        Convert PolarAction to English description.
        :param action:
        :return: English description of the action.
        """
        if action.type == 'stop':
            return "STOP"
        elif action.type == 'pause':
            return "PAUSE"
        elif action.type == 'null':
            return ""

        if abs(action.r) < 0.01:  # essentially no forward movement, pure rotation
            angle_deg = np.rad2deg(action.theta)
            if angle_deg > 0:
                return f"Turn Left {abs(angle_deg):.1f} degrees"
            elif angle_deg < 0:
                return f"Turn Right {abs(angle_deg):.1f} degrees"
            else:
                return "NO ROTATION"

        text = f"Move Forward {action.r:.2f}m"
        if abs(action.theta) > 0.01:
            angle_deg = np.rad2deg(action.theta)
            if angle_deg > 0:
                text += f" + Turn Left {abs(angle_deg):.1f} degrees"
            else:
                text += f" + Turn Right {abs(angle_deg):.1f} degrees"

        return text

    def _get_action_for_next_waypoint(self) -> Optional[PolarAction]:
        """
        Compute and return an action based on current position and the next path point.
        If a waypoint is reached, remove it from the path.
        The function prioritizes heading alignment before forward motion.
        """
        if not hasattr(self, 'current_path') or not self.current_path:
            return None

        agent_pos = self.mapper.current_position
        agent_rot_matrix = self.mapper.current_rotation
        next_waypoint = self.current_path[0]

        vec_to_waypoint_2d = np.array(next_waypoint[:2]) - agent_pos[:2]
        dist_to_waypoint = np.linalg.norm(vec_to_waypoint_2d)

        arrival_distance = 0.3  # arrival distance threshold (meters)
        if dist_to_waypoint < arrival_distance:
            self.current_path.pop(0)

            if not self.current_path:
                return None

            return self._get_action_for_next_waypoint()

        forward_vec_3d = -agent_rot_matrix[:, 2]
        forward_vec_2d = forward_vec_3d[:2]

        agent_angle = np.arctan2(forward_vec_2d[1], forward_vec_2d[0])
        waypoint_angle = np.arctan2(vec_to_waypoint_2d[1], vec_to_waypoint_2d[0])

        angle_diff = waypoint_angle - agent_angle
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        angle_threshold_rad = np.deg2rad(5)  # 5 deg tolerance

        if abs(angle_diff) > angle_threshold_rad:

            return PolarAction(r=0, theta=-angle_diff, type='turn')
        else:

            max_forward_step = 1  # max forward distance per step (meters)
            forward_dist = min(dist_to_waypoint, max_forward_step)
            return PolarAction(r=forward_dist, theta=0, type='move_forward')

    def execute_action(self, action: PolarAction):
        """
        Execute the given action in the simulator and update the map.

        :param action:
        """
        obs = self.sim_wrapper.step(action)
        instances = self.update_map(obs)
        self.curr_obs = obs
        self.instances = instances
        self.last_action = action

    def create_action(self, world_point_1, world_point_2):
        """
        Create a PolarAction from world_point_1 to world_point_2.
        :param world_point_1: starting world coordinate.
        :param world_point_2: target world coordinate.
        :return: a PolarAction object.
        """

        delta = np.array(world_point_2) - np.array(world_point_1)
        r = np.linalg.norm([delta[0], delta[2]])
        theta = np.arctan2(delta[0], -delta[2])
        return PolarAction(r, theta)
