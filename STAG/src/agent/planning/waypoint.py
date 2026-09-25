"""
规划 · 候选点生成与决策调度。

_preprocessing_module 产出可选 waypoint（A* 验证过可达），decide_waypoint
把它们连同前沿一起交给 VLM 并解析回复。

一条必须守住的顺序：**先校验 action_key 合法，再改任何状态**。反过来会让
一个非法动作（"F"、超出候选数的 18）先把子任务推进一次，重试时再推进一次。
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

from .prompt import _SkipGuidance  # noqa: F401  (decide_waypoint 的异常路径需要)
from ..common import robust_json_parse
from ..perception.topdown_map import create_top_down_map_centered


class WaypointMixin:
    def _world_to_pixel_coords(self, waypoints_world, agent_position, agent_rotation, camera_intrinsic):
        """
        Convert world-coordinate waypoints to image pixel coordinates.
        :param waypoints_world:
        :return:
        """
        if waypoints_world.shape[0] == 0:
            return np.array([]).reshape(0, 2), np.array([], dtype=int), np.array([], dtype=float)

        relative_points = waypoints_world - agent_position
        if hasattr(agent_rotation, 'as_rotation_matrix'):
            rotation_matrix = agent_rotation.as_rotation_matrix()
        elif hasattr(agent_rotation, 'shape') and len(agent_rotation.shape) == 1 and agent_rotation.shape[0] == 4:
            rotation_matrix = R.from_quat(agent_rotation).as_matrix()
        else:
            rotation_matrix = agent_rotation
        rotation_matrix_inv = rotation_matrix.T
        local_points = np.dot(relative_points, rotation_matrix_inv.T)

        camera_points = np.copy(local_points)
        camera_points[:, 1] = -local_points[:, 1]
        camera_points[:, 2] = -local_points[:, 2]

        original_indices = np.arange(waypoints_world.shape[0])
        valid_mask = camera_points[:, 2] > 0.1
        if not np.any(valid_mask):
            return np.array([]).reshape(0, 2), np.array([], dtype=int), np.array([], dtype=float)

        valid_points = camera_points[valid_mask]
        valid_original_indices = original_indices[valid_mask]
        camera_depths = valid_points[:, 2].astype(float)

        fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
        cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]
        pixel_x = (valid_points[:, 0] * fx / valid_points[:, 2]) + cx
        pixel_y = (valid_points[:, 1] * fy / valid_points[:, 2]) + cy
        pixel_coords = np.column_stack([pixel_x, pixel_y])

        return pixel_coords, valid_original_indices, camera_depths

    def _preprocessing_module(self, obs, use_far_waypoints: bool = False):
        """
        Preprocessing: generate waypoints in current view, project onto image, including turn-left/right actions.
        Also mark visited candidates (<1m from trajectory) in red using trajectory_position history.
        """
        import numpy as np
        import cv2
        from scipy.spatial import cKDTree

        image = obs['color_sensor'].copy()
        height, width, _ = image.shape

        depth_img = obs.get('depth_sensor', None)
        depth_h = depth_w = None
        sx = sy = 1.0
        if depth_img is not None:
            if depth_img.ndim == 3 and depth_img.shape[2] == 1:
                depth_img = depth_img[:, :, 0]
            depth_h, depth_w = depth_img.shape[:2]
            sx = depth_w / float(width)
            sy = depth_h / float(height)

        actions = {}
        action_idx = 1

        waypoints_world_near, waypoints_relative_near = self.mapper.get_current_view_candidate_waypoints(
            waypoint_grid_resolution=1.0, min_distance=0.3, max_distance=1.5, merge_distance=0.4
        )

        if use_far_waypoints:
            waypoints_world_far, waypoints_relative_far = self.mapper.get_current_view_candidate_waypoints(
                waypoint_grid_resolution=2.0, min_distance=0.3, max_distance=1.5, merge_distance=1.5
            )
            agent_pos = self.mapper.current_position
            if waypoints_world_far.shape[0] > 0:
                distances_far = np.linalg.norm(waypoints_world_far - agent_pos, axis=1)
                far_mask = (distances_far > self.max_move_distance) & (distances_far < 8.0)
                waypoints_world_far = waypoints_world_far[far_mask]
                waypoints_relative_far = waypoints_relative_far[far_mask]
            else:
                waypoints_world_far = np.empty((0, 3))
                waypoints_relative_far = np.empty((0, 3))
        else:
            waypoints_world_far = np.empty((0, 3))
            waypoints_relative_far = np.empty((0, 3))

        if waypoints_world_far.shape[0] > 0:
            waypoints_world = np.vstack([waypoints_world_near, waypoints_world_far])
            waypoints_relative = np.vstack([waypoints_relative_near, waypoints_relative_far])
        else:
            waypoints_world = waypoints_world_near
            waypoints_relative = waypoints_relative_near

        visited_radius = 0.8
        if waypoints_world.shape[0] > 0:
            traj = np.array(getattr(self.mapper, 'trajectory_position', []), dtype=float)[:-1] - np.array([0, 0, 1.2])
            if traj.size > 0:

                traj_xz = traj[:, [0, 1]]
                wpts_xz = waypoints_world[:, [0, 1]]
                tree = cKDTree(traj_xz)

                dists, _ = tree.query(wpts_xz, k=1)
                visited_mask_all = dists < visited_radius
            else:
                visited_mask_all = np.zeros((waypoints_world.shape[0],), dtype=bool)
        else:
            visited_mask_all = np.zeros((0,), dtype=bool)

        pixel_coords, original_indices, cam_depths = self._world_to_pixel_coords(
            waypoints_world,
            self.mapper.current_position,
            self.mapper.current_rotation,
            self.mapper.camera_intrinsic
        )

        far_count = int(waypoints_world_far.shape[0]) if use_far_waypoints else 0


        # Build the A* graph buffer (mapper.waypoints) from the global navigable
        # map so plan_path_to_target() below has nodes to route through. Without
        # this the buffer stays empty and every path query returns [] (no_path),
        # leaving the agent with no selectable waypoint.
        self.mapper.get_candidate_waypoints(
            waypoint_grid_resolution=0.5, min_distance=0.3, max_distance=2.5
        )

        skip_by_edge = 0
        skip_by_distance = 0
        skip_by_path = 0
        skip_by_occlusion = 0
        print(f"waypoints: {waypoints_world.shape[0]}")

        # 记录每个被淘汰的候选点及原因，画在调试帧上。
        #
        # 之前只有 "skipped waypoints: edge 3, distance 16, no_path 0, occlusion 9"
        # 这一行数字，每次排查"为什么 agent 不往那边走"都只能靠猜。把点按原因
        # 画出来，一眼就能看出是过滤器太狠、还是那个方向真的不可达。
        self.rejected_waypoints = []
        projected = set(int(i) for i in original_indices)
        for gi in range(waypoints_world.shape[0]):
            if gi not in projected:
                # 压根没投影出来：在相机后方或侧方（camera z <= 0.1）
                self.rejected_waypoints.append(
                    {'world': waypoints_world[gi], 'reason': 'behind'})

        if pixel_coords.shape[0] > 0:
            for i in range(pixel_coords.shape[0]):
                x_pixel, y_pixel = int(pixel_coords[i, 0]), int(pixel_coords[i, 1])
                original_idx = int(original_indices[i])

                edge_margin_x = width * self.image_edge_threshold
                edge_margin_y = height * self.image_edge_threshold
                if not (edge_margin_x < x_pixel < width - edge_margin_x and
                        edge_margin_y < y_pixel < height - edge_margin_y):
                    skip_by_edge += 1
                    self.rejected_waypoints.append(
                        {'world': waypoints_world[original_idx], 'reason': 'edge'})
                    continue

                if depth_img is not None and cam_depths.shape[0] > i:
                    dx = int(np.clip(round(x_pixel * sx), 0, (depth_w - 1)))
                    dy = int(np.clip(round(y_pixel * sy), 0, (depth_h - 1)))
                    observed = float(depth_img[dy, dx])
                    if np.isfinite(observed) and observed > 0:
                        tol = max(0.12, 0.03 * observed)
                        if cam_depths[i] > observed + tol:
                            skip_by_occlusion += 1
                            self.rejected_waypoints.append(
                                {'world': waypoints_world[original_idx], 'reason': 'occlusion'})
                            continue

                wp_world = waypoints_world[original_idx]
                wp_relative = waypoints_relative[original_idx] + [0, 1.2, 0]

                wp_world_for_planning = np.copy(wp_world)
                distance = np.linalg.norm(wp_relative)

                if distance <= self.max_move_distance:
                    if distance < self.min_move_distance:
                        skip_by_distance += 1
                        self.rejected_waypoints.append(
                            {'world': wp_world, 'reason': 'distance'})
                        continue
                else:
                    if not use_far_waypoints or distance >= 10.0:
                        skip_by_distance += 1
                        self.rejected_waypoints.append(
                            {'world': wp_world, 'reason': 'distance'})
                        continue

                path = self.mapper.plan_path_to_target(wp_world_for_planning)
                if not path or len(path) == 0:
                    skip_by_path += 1
                    self.rejected_waypoints.append(
                        {'world': wp_world, 'reason': 'no_path'})
                    continue

                is_visited = bool(visited_mask_all[original_idx])

                # 落在 avoid 区域内的候选点：不从列表里剔除（指令说"避开"不等于
                # "此路不通"，硬删可能把唯一通路删掉），但标红并在给VLM的说明里
                # 点名，同时A*的边权已经对这片区域加了惩罚。
                in_avoid = False
                try:
                    for zc, zr in self.mapper.get_avoid_zones():
                        if np.linalg.norm(np.asarray(wp_world[:2], dtype=float) - zc) < zr:
                            in_avoid = True
                            break
                except Exception:
                    in_avoid = False

                min_radius, max_radius = 6, 35
                min_font, max_font = 0.4, 1.1
                norm_dist = np.clip(distance / 3, 0, 1)
                font_scale = max_font - (max_font - min_font) * norm_dist

                text = str(action_idx)
                (ref_text_width, ref_text_height), _ = cv2.getTextSize("99", cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_diag = int(np.sqrt(ref_text_width ** 2 + ref_text_height ** 2))
                radius = max(int(text_diag / 2 + 2), min_radius)

                overlay = image.copy()

                circle_color = (255, 200, 100) if distance > self.max_move_distance else (255, 255, 255)
                if is_visited:
                    circle_color = (255, 100, 100)

                cv2.circle(overlay, (x_pixel, y_pixel), radius, circle_color, -1)
                # avoid 区域内的候选点加一圈粗红边，和"已访问"的浅红填充区分开
                cv2.circle(overlay, (x_pixel, y_pixel), radius,
                           (0, 0, 255) if in_avoid else (0, 0, 0), 4 if in_avoid else 2)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                cv2.putText(image, text, (x_pixel - text_width // 2, y_pixel + text_height // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

                actions[text] = {
                    'type': 'waypoint',
                    'target_point_world': waypoints_world[original_idx],
                    'target_point_relative': waypoints_relative[original_idx],
                    'visited': is_visited,
                    'in_avoid': in_avoid,
                }
                action_idx += 1

        print(
            f"skipped waypoints: edge {skip_by_edge},  distance {skip_by_distance},  no_path {skip_by_path},  occlusion {skip_by_occlusion}")

        if self.first_stop:
            try:
                num_points = 50
                radius = 1.0
                thetas = np.linspace(0, 2 * np.pi, num_points)
                agent_pos = self.mapper.current_position
                floor_z = agent_pos[2] - 1.2

                circle_x = agent_pos[0] + radius * np.cos(thetas)
                circle_y = agent_pos[1] + radius * np.sin(thetas)
                circle_z = np.full_like(circle_x, floor_z)
                circle_world_points = np.vstack([circle_x, circle_y, circle_z]).T

                pixel_coords_circle, _, _ = self._world_to_pixel_coords(
                    circle_world_points,
                    self.mapper.current_position,
                    self.mapper.current_rotation,
                    self.mapper.camera_intrinsic
                )

                if pixel_coords_circle.shape[0] > 1:
                    pts = pixel_coords_circle.astype(np.int32).reshape((-1, 1, 2))
                    overlay = image.copy()
                    cv2.polylines(
                        overlay,
                        [pts],
                        isClosed=True,
                        color=(0, 255, 255),
                        thickness=2,
                        lineType=cv2.LINE_AA,
                    )
                    alpha = 0.6
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                    scale_factor = image.shape[0] / 1080.0
                    top_idx = np.argmin(pixel_coords_circle[:, 1])
                    top_point = pixel_coords_circle[top_idx]
                    text = "1m"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 1.2 * scale_factor
                    thickness = max(2, int(2 * scale_factor))
                    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

                    tx = int(top_point[0] - tw / 2)
                    ty = int(top_point[1] - 2 * scale_factor - th)

                    tx = max(2, min(tx, image.shape[1] - tw - 2))
                    ty = max(th + 2, min(ty, image.shape[0] - 2))

                    cv2.putText(image, text, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                    cv2.putText(image, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            except Exception as e:
                pass

        action_keys = ['L', 'R', 'B']
        positions = [(50, height // 2), (width - 50, height // 2), (width // 2, 50)]
        angles = [self.turn_angle_rad, -self.turn_angle_rad, np.pi]
        for key, pos, angle in zip(action_keys, positions, angles):
            overlay = image.copy()
            cv2.circle(overlay, pos, 30, (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
            cv2.putText(image, key, (pos[0] - 15, pos[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
            actions[key] = {'type': 'turn', 'angle': angle}

        subtask_points = self.instruction_obj.get_subtasks_key_coord()
        try:
            if subtask_points:
                keys, pts = [], []
                for k, coord in subtask_points.items():
                    if coord is None or len(coord) < 3:
                        continue
                    pts.append([float(coord[0]), float(coord[1]), float(coord[2])])
                    keys.append(k)

                if len(pts) > 0:
                    pts_np = np.array(pts, dtype=float)
                    pixel_coords_st, valid_idx_st, _ = self._world_to_pixel_coords(
                        pts_np,
                        self.mapper.current_position,
                        self.mapper.current_rotation,
                        self.mapper.camera_intrinsic
                    )

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale_factor = image.shape[0] / 1080.0
                    for i, px in enumerate(pixel_coords_st):
                        key = keys[int(valid_idx_st[i])]
                        x, y = int(px[0]), int(px[1])
                        if x < 0 or x >= width or y < 0 or y >= height:
                            continue

                        half = max(6, int(8 * scale_factor))
                        tl = (max(0, x - half), max(0, y - half))
                        br = (min(width - 1, x + half), min(height - 1, y + half))
                        cv2.rectangle(image, tl, br, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                        cv2.rectangle(image, tl, br, (0, 0, 0), 1, lineType=cv2.LINE_AA)

                        label = key
                        font_scale = max(0.5, 0.8 * scale_factor)
                        thickness = max(1, int(2 * scale_factor))
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        tx = br[0] + int(6 * scale_factor)
                        ty = tl[1] - int(4 * scale_factor)

                        bx1 = max(0, tx - 3)
                        by1 = max(0, ty - th - 3)
                        bx2 = min(width - 1, tx + tw + 3)
                        by2 = min(height - 1, ty + 3)
                        if bx2 > bx1 and by2 > by1:
                            cv2.rectangle(image, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
                            cv2.rectangle(image, (bx1, by1), (bx2, by2), (0, 0, 0), 1)

                        text_org = (min(max(0, tx), width - 1 - tw), min(max(th, ty), height - 1))
                        cv2.putText(image, label, text_org, font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        except Exception as e:
            pass

        return image, actions

    def decide_waypoint(self, is_stuck=False, is_first_step=False):
        """
        Decide the current target waypoint or action.
        This function analyzes the current state (map + 4 views) via VLM, selects a waypoint or action,
        then returns the action details for subsequent planning and execution.
        :return:
                 or {'type': 'turn', 'angle': ...}. If VLM decides to stop, returns {'type': 'stop'}.
                 Returns None if all attempts fail.
        """
        obs = self.curr_obs

        labeled_image_np, actions = self._preprocessing_module(obs)
        self.last_actions = actions

        img = labeled_image_np.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 3
        label = 'Front View'
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

        text_x = img.shape[1] - text_size[0] - 20
        text_y = 50

        overlay = img.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (255, 255, 255), -1)
        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        alpha = 0.7
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        pil_labeled_rgb_image = Image.fromarray(img[:, :, :3], 'RGB')

        view_images = []
        for direction, label in [('left', 'Left View'), ('right', 'Right View'), ('back', 'Back View')]:
            if direction in obs and 'color_sensor' in obs[direction]:
                img = obs[direction]['color_sensor'].copy()

                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = img.shape[1] - text_size[0] - 20
                text_y = 50

                overlay = img.copy()
                cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                              (text_x + text_size[0] + 10, text_y + 10),
                              (255, 255, 255), -1)
                cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

                cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

                view_images.append(Image.fromarray(img[:, :, :3], 'RGB'))

        # 前沿打分要在建图之前算好：地图上要画 F1/F2/F3，而打分是在
        # generate_prompt 里做的——建图在它之前，第一步时属性还不存在。
        try:
            self._last_scored_frontiers = self._score_frontiers()
        except Exception as e:
            print(f"[frontier] scoring skipped: {e}")
            self._last_scored_frontiers = None

        # ---- 把前沿变成可选动作 ----
        #
        # 之前 F1/F2/F3 只是 prompt 里的一段描述，VLM 认同了也执行不了：
        # 候选 waypoint 全部落在 0.4~3m 的圆环内（max_move_distance=3），
        # 而推荐前沿的距离中位数是 5m，67% 超出可选范围。系统一边说"往那边
        # 走"，一边只给 3 米内的点选——这是实测里 40% 候选点因距离被丢弃、
        # agent 在房间里打转的直接原因。
        # 现在 F1 和 waypoint 1/2/3 在同一个选择空间里竞争，选中之后由 A*
        # 规划多步路径过去，一次决策可以走很远。
        if self.enable_frontier_action:
            floor_z = float(self.mapper.current_position[2]) - 1.2
            for i, f in enumerate(self._last_scored_frontiers or []):
                p = np.asarray(f.get('point', f['center']), dtype=float)
                actions[f"F{i + 1}"] = {
                    'type': 'frontier',
                    'target_point_world': np.array([p[0], p[1], floor_z]),
                    'frontier': f,
                }
            self.last_actions = actions

        top_down_map_np = create_top_down_map_centered(self.mapper, self.config['camera']['fov'],
                                                       self.target_world_position, action_candidates=self.last_actions, subtasks=self.instruction_obj.get_subtasks_key_coord(),
                                                       progress=self.instruction_obj.get_progress(self.total_step_budget),
                                                       scored_frontiers=self._last_scored_frontiers)
        top_down_map_rgb = cv2.cvtColor(top_down_map_np, cv2.COLOR_BGR2RGB)
        pil_map_image = Image.fromarray(top_down_map_rgb)

        active_instr = self.instruction
        prompt = self.generate_prompt(active_instr, is_stuck, is_first_step)

        if len(actions) == 3:
            prompt += "\nNo available waypoints detected in current view. Please choose to turn left (L), turn right (R), or turn around (B) to explore more."

        # avoid 区域内的候选点在图上是粗红边圈，这里再用文字点名一次。
        # 只是不鼓励，不禁止——万一唯一通路就在里面，还是得让VLM能选。
        avoid_keys = [k for k, v in actions.items()
                      if v.get('type') == 'waypoint' and v.get('in_avoid')]
        if avoid_keys:
            avoid_names = ", ".join(
                g['name'] for g in self.mapper.grounded_landmarks
                if g.get('role') == 'avoid_marker') or "an area to avoid"
            prompt += (f"\nWaypoints {', '.join(sorted(avoid_keys))} (thick red outline) fall inside "
                       f"[{avoid_names}], which the instruction tells you to avoid. Prefer any other "
                       f"waypoint. Only choose one of them if there is genuinely no other way through.")

        max_retry = 3
        for attempt in range(max_retry):
            try:
                if attempt == 1:
                    prompt += "\nNote: Please carefully choose a valid action from the provided options and ensure your output is a valid JSON."
                elif attempt >= 2:
                    # 最后一次退化成极简格式：前两次失败说明模型在长schema上翻车了，
                    # 再重复同样的要求没意义。只要一个动作，其余字段全部放弃。
                    valid = ", ".join(sorted(actions.keys()))
                    prompt = (f"Reply with ONLY this JSON object and nothing else:\n"
                              f'{{"action": "<one of: {valid}>"}}\n'
                              f"No explanation, no markdown fence, no other keys. "
                              f"Pick the option that best continues: {active_instr}")

                vlm_response_text = self._get_vlm_response_multiview(
                    pil_labeled_rgb_image, view_images[0] if len(view_images) > 0 else None,
                    view_images[1] if len(view_images) > 1 else None,
                    view_images[2] if len(view_images) > 2 else None,
                    pil_map_image, prompt, self.img_buffer
                )

                parsed_json = robust_json_parse(vlm_response_text)

                if parsed_json and isinstance(parsed_json, dict) and 'action' in parsed_json:

                    # 决策调用顺带带回的 front view 检测框，直接并进地图。
                    # 放在解析成功后、返回决策之前：此刻 mapper 的
                    # current_depth/位姿 仍对应送出去的那张 front view，
                    # execute_action() 还没跑，位姿没变。
                    self._apply_vlm_detections(parsed_json, obs)

                    if 'movement' in parsed_json:
                        self.instruction_obj.add_record_to_current_subtask({'movement': parsed_json['movement']})
                    if 'plan' in parsed_json:
                        self.instruction_obj.update_plan_for_current_subtask(parsed_json['plan'])

                    action_key = str(parsed_json['action']).upper()

                    # action == -1 是"停下并结束子任务"，子任务推进由 step() 的
                    # stop 分支负责，这里不能重复推进，否则会一次跳两个子任务。
                    if action_key == '-1':
                        if self._challenge_out_of_order():
                            # 时序约束：途经点还没去过就宣告到达
                            return {'type': 'challenge_stop'}, vlm_response_text, pil_labeled_rgb_image
                        if self._is_premature_completion():
                            # 步数远低于预算就宣告到达，先质询一轮再说。
                            # 复用了终点stop那套"连续两次才算数"的双重确认思路。
                            return {'type': 'challenge_stop'}, vlm_response_text, pil_labeled_rgb_image
                        if self._challenge_receded_stop():
                            # "你之前离目标更近过"——这一条专治走到了又走开
                            return {'type': 'challenge_stop'}, vlm_response_text, pil_labeled_rgb_image
                        return {'type': 'stop'}, vlm_response_text, pil_labeled_rgb_image

                    # 动作合法性校验必须在任何状态修改之前。
                    #
                    # 之前把 subtask_done 的处理放在了校验之前，结果动作非法时
                    # (实测出现过 "action": "F" 和只有10个候选点却选 "action": 18)
                    # 代码 continue 重试，但子任务已经推进了；重试的回复如果又设
                    # true 就再推进一次——一次决策吃掉两个子任务。26次推进里有3次
                    # 踩到这个。现在非法动作直接重试，不留下任何副作用。
                    if action_key not in actions:
                        continue

                    # ---- 子任务完成与移动解耦 ----
                    # 之前 action 字段身兼两职（选路点 / 宣告完成），两者互斥。
                    # 实测VLM想同时做这两件事时会选择移动、把"子任务已完成"写进
                    # thought 的散文里——代码不读散文，于是状态机停在旧子任务上，
                    # 直到VLM某一步终于肯放弃移动、单独吐一个 -1 为止。一次实跑里
                    # 这个滞后有25步，期间当前子任务的landmark距离一路单调变远。
                    #
                    # 现在 subtask_done 是独立布尔，可以和 action 同时给：既前进
                    # 到选中的路点，又把当前子任务标记完成。
                    done_flag = parsed_json.get('subtask_done',
                                                parsed_json.get('completed', False))
                    # 顺序约束和预算约束都适用于 subtask_done 这条路径：
                    # 不管用哪种方式宣告完成，跳过途经点都是错的。
                    if done_flag is True and self._challenge_out_of_order():
                        done_flag = False
                    if done_flag is True and self._is_premature_completion():
                        # 提前宣告：这一轮先不认，下一轮prompt里质询。动作照常执行，
                        # 所以不像 -1 那样要浪费一步。
                        done_flag = False
                    if done_flag is True and not self.instruction_obj.is_last_subtask():
                        # 完成坐标优先用选中路点而非当前站位：子任务的语义是
                        # "走到X就算完成"，宣告时人还没走到，用目标点更贴近真实边界。
                        done_coord = None
                        if action_key in actions and actions[action_key].get('type') == 'waypoint':
                            done_coord = np.asarray(
                                actions[action_key]['target_point_world'], dtype=float)
                        if done_coord is None:
                            done_coord = self.mapper.current_position - np.array([0, 0, 1.2])

                        finished_key = self.instruction_obj.get_current_subtask_key()
                        self.instruction_obj.mark_current_subtask_completed(
                            done_coord, self.mapper.current_rotation)
                        print(f"[subtask] {finished_key} done (declared alongside action "
                              f"{action_key}) -> now {self.instruction_obj.get_current_subtask_key()}")
                        self._reset_landmark_progress()

                    self._decision_failures = 0
                    return actions[action_key], vlm_response_text, pil_labeled_rgb_image
                else:
                    # 解析失败时把模型到底返回了什么打出来。之前这里完全静默，
                    # 排查 finish_status=error 的 episode 时无从下手。
                    preview = (vlm_response_text or "")[:220].replace("\n", " ")
                    print(f"[decide] attempt {attempt + 1}/{max_retry} unparsable: {preview!r}")
                    continue

            except Exception as e:
                print(f"[decide] attempt {attempt + 1}/{max_retry} raised: {e}")

        # ---- 三次都失败：不要直接放弃 ----
        #
        # 原来这里返回 None，step() 收到 None 就 self.error=True 并 stop，
        # 整个 episode 判为 error。实测 22 个 episode 里有 4 个(18%)是这么没的，
        # 其中一个只走了 0.00m。解析失败是模型的临时抽风，不是"无路可走"，
        # 直接终止是过度反应。
        #
        # 改成退化行为：原地右转继续探索，连续失败 3 次(不是同一步的3次重试，
        # 是跨步的3次彻底失败)才真的放弃。
        self._decision_failures = getattr(self, '_decision_failures', 0) + 1
        if self._decision_failures < 3:
            print(f"[decide] all retries failed ({self._decision_failures}/3) — "
                  f"falling back to turn-and-look instead of aborting the episode")
            return ({'type': 'turn', 'angle': -self.turn_angle_rad},
                    f"(Auto)VLM decision unparsable {self._decision_failures} time(s); turning to re-observe.",
                    pil_labeled_rgb_image)

        print("[decide] giving up after 3 consecutive decision failures")
        return None, "None", pil_labeled_rgb_image

    def plan_rollback_path(self, subtask_key: str = None):
        """
        Plan a rollback path to a previous subtask start point (default: current subtask). Returns the rotation.
        Returns the planned waypoint sequence and rotation.
        :param subtask_key:
        :return:
        """
        if subtask_key is None:
            subtask_key = self.instruction_obj.get_current_subtask_key()

        if not subtask_key:
            pass
            return None, None

        try:
            target_idx = self.instruction_obj.sub_instruction_keys.index(subtask_key.upper())
        except ValueError:
            pass
            return None, None

        if target_idx == 0:

            target_pos = self.init_pos
            target_rot = self.init_rot
        else:

            prev_subtask_key = self.instruction_obj.sub_instruction_keys[target_idx - 1]
            target_pos, target_rot = self.instruction_obj.get_subtask_pos_rot_by_key(prev_subtask_key)

        if target_pos is None or target_rot is None:
            pass
            return None, None

        path = self.mapper.plan_path_to_target(target_pos)

        if path is None or len(path) == 0:
            pass
            return None, None

        return path, target_rot
