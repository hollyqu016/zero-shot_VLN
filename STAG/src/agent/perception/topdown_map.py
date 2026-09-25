"""
俯视图渲染。

这里全是纯函数——不读 agent 状态，只接收显式参数并返回图像。所以它不是
mixin，调用方 import 函数即可。把它单独拆出来的理由很简单：三个绘图函数
加起来 1275 行，占了原 agent.py 的三成，而它们跟导航决策毫无关系。
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


def draw_direction_markers(img: np.ndarray) -> np.ndarray:
    """
    Draw red direction triangles (top/bottom/left/right) with F/B/L/R labels (auto-scaled).
    """
    h, w = img.shape[:2]
    if h < 20 or w < 20:
        return img

    s = max(6, int(min(h, w) * 0.025))  # triangle half-size (shrunk)
    m = max(2, int(s * 0.3))  # margin
    border_th = max(1, s // 8)  # stroke thickness
    font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_triangle_with_label(bg, center, direction, label):
        cx, cy = center
        if direction == 'up':
            pts = np.array([[cx, cy - s], [cx - s, cy + s], [cx + s, cy + s]], dtype=np.int32)
        elif direction == 'down':
            pts = np.array([[cx, cy + s], [cx - s, cy - s], [cx + s, cy - s]], dtype=np.int32)
        elif direction == 'left':
            pts = np.array([[cx - s, cy], [cx + s, cy - s], [cx + s, cy + s]], dtype=np.int32)
        else:  # right
            pts = np.array([[cx + s, cy], [cx - s, cy - s], [cx - s, cy + s]], dtype=np.int32)

        cv2.fillPoly(bg, [pts], (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.polylines(bg, [pts], isClosed=True, color=(0, 0, 0), thickness=border_th, lineType=cv2.LINE_AA)

        centroid = pts.mean(axis=0).astype(int)
        x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
        x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
        allow_w = max(8, int((x_max - x_min) * 0.55))
        allow_h = max(8, int((y_max - y_min) * 0.45))

        (tw1, th1), _ = cv2.getTextSize(label, font, 0.7, 3)
        if tw1 == 0 or th1 == 0:
            return
        scale = min(allow_w / tw1, allow_h / th1)
        scale = max(0.4, min(2.0, scale))  # clamp range
        thickness = max(1, int(scale * 1.2))

        (tw, th), bl = cv2.getTextSize(label, font, scale, thickness)

        text_org = (int(centroid[0] - tw / 2), int(centroid[1] + th / 2) - 1)

        cv2.putText(bg, label, text_org, font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)

    draw_triangle_with_label(img, (w // 2, m + s), 'up', 'F')

    draw_triangle_with_label(img, (w // 2, h - m - s), 'down', 'B')

    draw_triangle_with_label(img, (m + s, h // 2), 'left', 'L')

    draw_triangle_with_label(img, (w - m - s, h // 2), 'right', 'R')

    return img
def create_top_down_map_centered(
    mapper: Instruct_Mapper,
    fov_angle_deg: float,
    target_point=None,
    map_scale=0.025,
    map_size_px=1024,
    fov_range=1.0,
    show_all_labels=False,
    label_volume_threshold=100,
    crop_size_m=16.0,
    action_candidates=None,
    subtasks=None,
    progress=None,
    debug_overlay=False,
    rejected_waypoints=None,
    scored_frontiers=None,
):
    """
    action_candidates: Dict[str, Dict], each entry example:
      "waypoint 1": {
        "type": "waypoint",
        "target_point_world": [x, y, z]
      }
    subtasks: Dict[str, Sequence[float]] subtask name -> world coordinate ([x,y,z]).
    progress: Instruction.get_progress() 的返回值，用于在图底部画子任务进度条。
    debug_overlay: 画通行余量圆、四条测距射线、以及被淘汰的候选 waypoint。
        这些是判据的**原始测量值**，用来分辨"判据写错了"还是"测量就不对"，
        只应该开在录像帧上——给VLM看只会增加噪音。
    rejected_waypoints: [{'world': xyz, 'reason': str}]，被过滤掉的候选点。
    """
    t_start = time.time()
    timings = {}

    t = time.time()
    agent_pos = mapper.current_position
    agent_rot = mapper.current_rotation

    forward_vec_3d = agent_rot @ np.array([0, 0, -1])
    forward_vec_2d = forward_vec_3d[:2]
    if np.linalg.norm(forward_vec_2d) < 1e-6:
        forward_vec_2d = np.array([0, -1])
    forward_vec_2d /= np.linalg.norm(forward_vec_2d)

    agent_angle_rad = np.arctan2(forward_vec_2d[1], forward_vec_2d[0])
    rotation_angle = -agent_angle_rad - np.pi / 2

    c, s = np.cos(rotation_angle), np.sin(rotation_angle)
    rotation_matrix = np.array([[c, -s], [s, c]])

    def transform_and_to_pixel(world_coords):
        if world_coords.ndim == 1:
            world_coords = world_coords.reshape(1, -1)

        translated_coords = world_coords[:, :2] - agent_pos[:2]

        rotated_coords = translated_coords @ rotation_matrix.T

        pixel_coords = (rotated_coords / map_scale) + np.array([map_size_px / 2, map_size_px / 2])
        return pixel_coords.astype(int)

    timings['calc_transform'] = time.time() - t

    top_down_map = np.full((map_size_px, map_size_px, 3), 255, dtype=np.uint8)
    map_h, map_w = top_down_map.shape[:2]

    pcd_resolution = 0.025
    dilation_radius = max(1, int(pcd_resolution / map_scale / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    t = time.time()
    if not mapper.navigable_pcd.is_empty() and mapper.navigable_pcd.point.positions.shape[0] > 0:
        nav_pixels = transform_and_to_pixel(mapper.navigable_pcd.point.positions.cpu().numpy())
        valid_mask = (nav_pixels[:, 0] >= 0) & (nav_pixels[:, 0] < map_w) & (nav_pixels[:, 1] >= 0) & (
                    nav_pixels[:, 1] < map_h)
        if np.any(valid_mask):
            nav_mask = np.zeros((map_h, map_w), dtype=np.uint8)
            nav_mask[nav_pixels[valid_mask, 1], nav_pixels[valid_mask, 0]] = 255
            dilated_nav_mask = cv2.dilate(nav_mask, kernel)
            top_down_map[dilated_nav_mask == 255] = (200, 200, 200)
    timings['draw_navigable'] = time.time() - t

    # ---------------- 走廊染色 ----------------
    # 走廊/门口这类空间没有中心也没有边界，画成圆圈等于凭空捏造一个范围。
    # 它天然的形状是"我走过的那条带子"，所以直接给地板上色——而且只对
    # 走过的格子上色，没探索的地方保持原灰，视觉上不会撒谎。
    # 地板底色(200,200,200)是地图上唯一还空着的通道，不和 frontier(黄)、
    # 障碍(黑)、轨迹(红)、landmark pin、推断房间(灰圈)冲突。
    t = time.time()
    space_labels = getattr(mapper, 'space_labels', None) or {}
    if space_labels:
        cell = getattr(mapper, 'SPACE_CELL', 0.25)
        pts, cols = [], []
        # BGR。比地板底色(200,200,200)明显偏冷，但仍读作"地板"而不是新物体。
        SPACE_TINT = {'corridor': (228, 206, 178)}
        for (ix, iy), st in space_labels.items():
            c = SPACE_TINT.get(st)
            if c is None:
                continue
            pts.append([ix * cell, iy * cell, 0.0])
            cols.append(c)
        if pts:
            sp_px = transform_and_to_pixel(np.array(pts, dtype=float))
            half_px = max(1, int(cell / map_scale / 2))
            for (px, py), c in zip(sp_px, cols):
                if 0 <= px < map_w and 0 <= py < map_h:
                    x1s, x2s = max(0, px - half_px), min(map_w, px + half_px + 1)
                    y1s, y2s = max(0, py - half_px), min(map_h, py + half_px + 1)
                    region = top_down_map[y1s:y2s, x1s:x2s]
                    # 只染已知的地板，不覆盖障碍和未探索的白底
                    m = np.all(region == (200, 200, 200), axis=-1)
                    region[m] = c
    timings['draw_space'] = time.time() - t

    t = time.time()
    obstacle_mask_bool = np.zeros((map_h, map_w), dtype=bool)
    if not mapper.obstacle_pcd.is_empty() and mapper.obstacle_pcd.point.positions.shape[0] > 0:
        obs_pixels = transform_and_to_pixel(mapper.obstacle_pcd.point.positions.cpu().numpy())
        valid_mask = (obs_pixels[:, 0] >= 0) & (obs_pixels[:, 0] < map_w) & (obs_pixels[:, 1] >= 0) & (
                    obs_pixels[:, 1] < map_h)
        if np.any(valid_mask):
            obs_mask = np.zeros((map_h, map_w), dtype=np.uint8)
            obs_mask[obs_pixels[valid_mask, 1], obs_pixels[valid_mask, 0]] = 255
            dilated_obs_mask = cv2.dilate(obs_mask, kernel)
            top_down_map[dilated_obs_mask == 255] = (50, 50, 50)
            obstacle_mask_bool = (dilated_obs_mask == 255)

            contours, _ = cv2.findContours(dilated_obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_down_map, contours, -1, (0, 0, 0), thickness=5)
    timings['draw_obstacle'] = time.time() - t

    t = time.time()
    if hasattr(mapper, 'frontier_pcd') and not mapper.frontier_pcd.is_empty():
        frontier_pixels = transform_and_to_pixel(mapper.frontier_pcd.point.positions.cpu().numpy())
        valid_mask = (frontier_pixels[:, 0] >= 0) & (frontier_pixels[:, 0] < map_w) & \
                     (frontier_pixels[:, 1] >= 0) & (frontier_pixels[:, 1] < map_h)
        valid_frontier_pixels = frontier_pixels[valid_mask]
        for pixel in valid_frontier_pixels:
            cv2.circle(top_down_map, tuple(pixel), 1, (0, 255, 255), -1)  # yellow dot
    timings['draw_frontier'] = time.time() - t

    t = time.time()
    if len(mapper.object_entities) > 0:
        for entity in mapper.object_entities:
            if entity['pcd'].point.positions.shape[0] > 0:
                obj_pixels = transform_and_to_pixel(entity['pcd'].point.positions.cpu().numpy())
                valid_pixels_mask = (obj_pixels[:, 0] >= 0) & (obj_pixels[:, 0] < map_w) & (obj_pixels[:, 1] >= 0) & (
                            obj_pixels[:, 1] < map_h)
                if not np.any(valid_pixels_mask): continue

                valid_obj_pixels = obj_pixels[valid_pixels_mask]
                intersection_mask = obstacle_mask_bool[valid_obj_pixels[:, 1], valid_obj_pixels[:, 0]]

                if np.any(intersection_mask):
                    final_pixels = valid_obj_pixels[intersection_mask]
                    if len(final_pixels) == 0: continue

                    obj_colors_rgb = entity['pcd'].point.colors.cpu().numpy()[valid_pixels_mask][intersection_mask]
                    object_color = (obj_colors_rgb[0] * 255)[::-1].astype(np.uint8)

                    object_mask_img = np.zeros((map_h, map_w), dtype=np.uint8)
                    object_mask_img[final_pixels[:, 1], final_pixels[:, 0]] = 255
                    fill_kernel = np.ones((5, 5), np.uint8)
                    filled_mask = cv2.morphologyEx(object_mask_img, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
                    top_down_map[filled_mask == 255] = object_color
    timings['draw_objects'] = time.time() - t

    label_upscale_factor = 2
    if label_upscale_factor > 1:
        new_h, new_w = map_h * label_upscale_factor, map_w * label_upscale_factor
        top_down_map = cv2.resize(top_down_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        new_h, new_w = map_h, map_w

    t = time.time()
    drawn_labels = []
    label_merge_distance_px = 50

    # 当前子任务关心的类别：这些类别的标签强制显示，不受体积阈值限制。
    # 否则像 clock / mirror 这种体积小但指令明确点名的地标会被
    # label_volume_threshold 静默过滤掉，正是最不该丢的那批标签。
    landmark_classes = {
        str(g['class_name']).lower()
        for g in getattr(mapper, 'grounded_landmarks', [])
    }

    for entity in sorted(mapper.object_entities, key=lambda e: e['class']):
        is_landmark_class = str(entity['class_name']).lower() in landmark_classes
        if (not show_all_labels and not is_landmark_class
                and entity['pcd'].point.positions.shape[0] < label_volume_threshold): continue
        if entity['pcd'].point.positions.shape[0] == 0: continue

        center_pixel = transform_and_to_pixel(entity['center'])[0]
        class_name = entity['class_name']

        should_merge = False
        for i, label_info in enumerate(drawn_labels):
            if label_info['class'] == class_name and np.linalg.norm(
                    center_pixel - label_info['center_pixel']) < label_merge_distance_px:
                drawn_labels[i]['center_pixel'] = (label_info['center_pixel'] + center_pixel) / 2
                should_merge = True
                break
        if not should_merge:
            drawn_labels.append({'class': class_name, 'center_pixel': center_pixel})

    for label_info in drawn_labels:
        center_pixel_scaled = (label_info['center_pixel'] * label_upscale_factor).astype(int)
        if not (0 <= center_pixel_scaled[0] < new_w and 0 <= center_pixel_scaled[1] < new_h): continue

        class_name = label_info['class']
        font_scale, font_thickness = 0.7, 1
        (tw, th), bl = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        box_tl = (center_pixel_scaled[0] - tw // 2 - 3, center_pixel_scaled[1] - th - 3)
        box_br = (center_pixel_scaled[0] + tw // 2 + 3, center_pixel_scaled[1] + bl)

        alpha = 0.8
        overlay = top_down_map.copy()
        cv2.rectangle(overlay, box_tl, box_br, (255, 255, 255), -1)
        cv2.addWeighted(overlay, alpha, top_down_map, 1 - alpha, 0, top_down_map)
        cv2.rectangle(top_down_map, box_tl, box_br, (0, 0, 0), 1)

        text_org = (box_tl[0] + 3, box_br[1] - bl)
        cv2.putText(top_down_map, class_name, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), font_thickness, cv2.LINE_AA)
    timings['draw_labels'] = time.time() - t

    t = time.time()
    if len(mapper.trajectory_position) > 1:
        traj_pixels = transform_and_to_pixel(np.array(mapper.trajectory_position[-150:]))
        valid_traj_pixels = (traj_pixels * label_upscale_factor).astype(int)
        overlay = top_down_map.copy()
        cv2.polylines(overlay, [valid_traj_pixels], isClosed=False, color=(0, 0, 255),
                      thickness=2 * label_upscale_factor)
        alpha = 0.5  # trajectory alpha, adjustable
        top_down_map = cv2.addWeighted(overlay, alpha, top_down_map, 1 - alpha, 0)
    timings['draw_trajectory'] = time.time() - t

    # waypoints = mapper.get_candidate_waypoints(min_distance=0.2, max_distance=1.0, waypoint_grid_resolution=1.0)

    # if waypoints.shape[0] > 0:
    #     waypoint_pixels = transform_and_to_pixel(waypoints)
    #     waypoint_pixels_scaled = (waypoint_pixels * label_upscale_factor).astype(int)
    #     for pixel in waypoint_pixels_scaled:
    #         if 0 <= pixel[0] < new_w and 0 <= pixel[1] < new_h:

    #
    t = time.time()
    if target_point is not None:
        target_pixel = transform_and_to_pixel(np.array(target_point))[0]
        target_pixel_scaled = (target_pixel * label_upscale_factor).astype(int)

        if (0 <= target_pixel_scaled[0] < new_w) and (0 <= target_pixel_scaled[1] < new_h):

            cv2.circle(top_down_map, tuple(target_pixel_scaled), 8, (0, 255, 0), -1)  # green filled circle
            cv2.circle(top_down_map, tuple(target_pixel_scaled), 8, (0, 0, 0), 2)  # black border
    timings['draw_target'] = time.time() - t

    new_h, new_w = top_down_map.shape[:2]  # use resized dims if already resized

    if action_candidates:

        overlay = top_down_map.copy()

        for text, cand in action_candidates.items():
            if cand.get('type') != 'waypoint':
                continue  # only handle waypoint type for now

            try:
                idx = int(text.split()[-1])
            except (ValueError, IndexError):
                continue  # skip if text format mismatches

            coord = cand.get('target_point_world')
            if coord is None:
                continue

            pix = transform_and_to_pixel(np.array(coord, dtype=float))[0]
            pix = (pix * label_upscale_factor).astype(int)

            if not (0 <= pix[0] < new_w and 0 <= pix[1] < new_h):
                continue

            radius = max(10, int(7 * label_upscale_factor))
            circle_color = (0, 128, 255)  # orange border (BGR)
            fill_color = (255, 255, 255)  # white background
            border_thickness = max(2, label_upscale_factor)

            cv2.circle(overlay, tuple(pix), radius, fill_color, -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, tuple(pix), radius, circle_color, border_thickness, lineType=cv2.LINE_AA)

            text_idx = str(idx)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * label_upscale_factor
            font_thickness = max(1, label_upscale_factor)
            (tw, th), bl = cv2.getTextSize(text_idx, font, font_scale, font_thickness)
            text_org = (int(pix[0] - tw / 2), int(pix[1] + th / 2))
            cv2.putText(overlay, text_idx, text_org, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)

    t = time.time()
    agent_pixel_scaled = (np.array([map_size_px / 2, map_size_px / 2]) * label_upscale_factor).astype(int)

    fov_rad = np.deg2rad(fov_angle_deg)
    angles = np.linspace(-np.pi / 2 - fov_rad / 2, -np.pi / 2 + fov_rad / 2, 30)
    fov_range_px = fov_range / map_scale * label_upscale_factor
    sector_points = [(agent_pixel_scaled + (np.array([np.cos(a), np.sin(a)]) * fov_range_px)).astype(int) for a in
                     angles]
    sector_pts = np.array([agent_pixel_scaled] + sector_points, np.int32)

    overlay = top_down_map.copy()
    cv2.fillPoly(overlay, [sector_pts], (255, 255, 0), lineType=cv2.LINE_AA)
    top_down_map = cv2.addWeighted(overlay, 0.2, top_down_map, 0.8, 0)

    if subtasks:
        overlay = top_down_map.copy()

        new_h, new_w = overlay.shape[:2]

        for name, coord in subtasks.items():
            if coord is None:
                continue

            try:
                pix = transform_and_to_pixel(np.array(coord, dtype=float))[0]
            except Exception:
                continue
            pix = (pix * label_upscale_factor).astype(int)

            if not (0 <= pix[0] < new_w and 0 <= pix[1] < new_h):
                continue

            half = max(6, int(6 * label_upscale_factor))
            tl = (int(pix[0] - half), int(pix[1] - half))
            br = (int(pix[0] + half), int(pix[1] + half))

            tl = (max(0, tl[0]), max(0, tl[1]))
            br = (min(new_w - 1, br[0]), min(new_h - 1, br[1]))

            cv2.rectangle(overlay, tl, br, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.rectangle(overlay, tl, br, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            text = str(name)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 * label_upscale_factor
            font_thickness = max(1, label_upscale_factor)
            (tw, th), bl = cv2.getTextSize(text, font, font_scale, font_thickness)

            tx = br[0] + int(6 * label_upscale_factor)
            ty = tl[1] - int(4 * label_upscale_factor)

            bx1 = max(0, tx - 3)
            by1 = max(0, ty - th - 3)
            bx2 = min(new_w - 1, tx + tw + 3)
            by2 = min(new_h - 1, ty + 3)

            if bx2 > bx1 and by2 > by1:
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)

            text_org = (min(max(0, tx), new_w - 1 - tw), min(max(th, ty), new_h - 1))
            cv2.putText(overlay, text, text_org, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

        top_down_map = cv2.addWeighted(overlay, 0.95, top_down_map, 0.05, 0)

    # ---------------- 前沿打分 F1/F2/F3 ----------------
    # 原来 frontier 只是一片统一的黄点（而且因为计算被注释掉，其实一个点都没有）。
    # 现在画成带编号的圆环，大小随前沿宽度、颜色随得分——VLM 需要的是"往哪个
    # 方向探"，不是"哪些像素是边界"。
    t = time.time()
    for rank, f in enumerate(scored_frontiers or []):
        try:
            c = np.asarray(f['center'], dtype=float)
            p = transform_and_to_pixel(np.array([c[0], c[1], 0.0]))[0]
            p = (p * label_upscale_factor).astype(int)
            if not (0 <= p[0] < new_w and 0 <= p[1] < new_h):
                continue
            r_px = int(np.clip(f.get('width_m', 1.0) / 2.0 / map_scale, 8, 60) * label_upscale_factor)
            s = float(f.get('score', 0.0))
            # 得分越高越暖：低分青灰，高分明黄
            col = (int(80 + 40 * (1 - s)), int(180 + 60 * s), int(255 * s))
            cv2.circle(top_down_map, tuple(p), r_px, col, 3, lineType=cv2.LINE_AA)
            lab = f"F{rank + 1}"
            (tw_f, th_f), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            org = (int(p[0] - tw_f / 2), int(p[1] + th_f / 2))
            cv2.putText(top_down_map, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 5, cv2.LINE_AA)
            cv2.putText(top_down_map, lab, org, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        col, 2, cv2.LINE_AA)
        except Exception:
            continue
    timings['draw_frontier_rank'] = time.time() - t

    # ---------------- grounded landmark 图层 ----------------
    # 画在 agent 箭头之前，这样箭头永远压在最上层不会被 pin 挡住。
    t = time.time()
    agent_pixel_scaled_for_lm = (np.array([map_size_px / 2, map_size_px / 2]) * label_upscale_factor).astype(int)
    grounded = list(getattr(mapper, 'grounded_landmarks', []) or [])

    # BGR。目的地绿、途经点蓝、避让点红，未知角色灰。
    ROLE_STYLE = {
        'destination_marker': {'color': (60, 200, 60), 'tag': 'DEST'},
        'waypoint_marker':    {'color': (230, 150, 40), 'tag': 'VIA'},
        'avoid_marker':       {'color': (50, 50, 220), 'tag': 'AVOID'},
        'unknown':            {'color': (140, 140, 140), 'tag': '?'},
    }

    placed_label_boxes = []  # 已放置的标签框，用于简单避让，防止多个pin的标签叠在一起

    # 第一层：所有推断出的房间，不管指令有没有点名。
    #
    # 之前只画"当前子任务点名了的"区域，结果地图上明明躺着一张 bed、
    # bedroom 早就推断出来了，却因为指令里没出现 "bedroom" 这个词而不显示——
    # 同一帧 VLM 还在猜 "this appears to be a dressing area or vestibule corner"。
    # 房间语义是通用上下文，和当前子任务是否用得上无关，应该一直可见。
    #
    # 用中性灰 + 圆括号 (bedroom)，和被点名的区域（角色配色 + 方括号 [bedroom]）
    # 在视觉上分开，避免VLM把"顺带告诉你这是卧室"误读成"你要去卧室"。
    named_area_centers = [(g.get('class_name'), np.asarray(g['center'][:2], dtype=float))
                          for g in grounded if g.get('kind') == 'area']

    # 背景房间层已经整个去掉了，连调试帧也不画。
    #
    # 演进过程值得记一下：先是无条件画所有推断出的房间，实跑发现五六个 4m
    # 半径的灰圈互相叠压、标签糊成一团还盖住障碍结构；于是改成只在调试帧画，
    # 但调试帧恰恰是人唯一的观察窗口，留成那样等于没法看。真正的问题不在
    # 画不画，而在一个卫生间场景能推断出 garage 和两个 office——这种精度下
    # 它在哪儿都是噪音。
    #
    # 现在：房间信息只以两种形式出现——被指令点名的画成 [方括号] 色块(下面
    # 那个循环)，以及 agent 标签里的 "in <room>"(只报当前所在的那一个)。
    # 完整列表改成打日志，需要排查时看 [rooms] 那行，不占地图。
    for a in []:
        try:
            ac = np.asarray(a['center'][:2], dtype=float)
            # 已经作为子任务landmark画过的房间不重复画
            if any(t == a['area_type'] and np.linalg.norm(ac - c) < 1.0
                   for t, c in named_area_centers):
                continue

            pix = transform_and_to_pixel(np.array([ac[0], ac[1], 0.0]))[0]
            pix = (pix * label_upscale_factor).astype(int)
            r_px = int(float(a.get('radius', 2.0)) / map_scale * label_upscale_factor)
            if not (-r_px < pix[0] < new_w + r_px and -r_px < pix[1] < new_h + r_px):
                continue

            grey = (150, 150, 150)
            panel = top_down_map.copy()
            cv2.circle(panel, tuple(pix), r_px, grey, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(panel, 0.10, top_down_map, 0.90, 0, top_down_map)
            cv2.circle(top_down_map, tuple(pix), r_px, grey, 2, lineType=cv2.LINE_AA)

            txt = f"({a['area_type']})"
            a_scale = float(np.clip(r_px / 140.0, 0.7, 1.8))
            a_th = max(2, int(a_scale * 1.4))
            (tw_a, th_a), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, a_scale, a_th)
            org = (int(pix[0] - tw_a / 2), int(pix[1] + th_a / 2))
            if 0 <= org[0] < new_w - tw_a and th_a < org[1] < new_h:
                cv2.putText(top_down_map, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                            a_scale, (255, 255, 255), a_th + 3, cv2.LINE_AA)
                cv2.putText(top_down_map, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                            a_scale, (110, 110, 110), a_th, cv2.LINE_AA)
        except Exception:
            continue

    # 第二层：被当前子任务点名的区域（大面积色块），物体pin后画，避免色块盖住pin
    for g in grounded:
        if g.get('kind') != 'area':
            continue
        try:
            style = ROLE_STYLE.get(g.get('role'), ROLE_STYLE['unknown'])
            color = style['color']
            pix = transform_and_to_pixel(np.asarray(g['center'], dtype=float))[0]
            pix = (pix * label_upscale_factor).astype(int)
            r_px = int(float(g.get('radius', 2.0)) / map_scale * label_upscale_factor)

            # 有几何分区的轮廓就画真实形状，没有再退回圆。圆是房间形状的
            # 很差的近似——四米半径的圆会盖住走廊和隔壁房间，这也是之前
            # 地图被灰圈糊住的直接原因。
            poly = None
            cw = g.get('contour')
            if cw is not None and len(cw) >= 3:
                cw = np.asarray(cw, dtype=float)
                z = np.full((cw.shape[0], 1), float(g['center'][2]) if len(g['center']) > 2 else 0.0)
                poly = (transform_and_to_pixel(np.hstack([cw, z]))
                        * label_upscale_factor).astype(np.int32)

            panel = top_down_map.copy()
            if poly is not None:
                cv2.fillPoly(panel, [poly], color, lineType=cv2.LINE_AA)
            else:
                cv2.circle(panel, tuple(pix), r_px, color, -1, lineType=cv2.LINE_AA)
            cv2.addWeighted(panel, 0.16, top_down_map, 0.84, 0, top_down_map)
            if poly is not None:
                cv2.polylines(top_down_map, [poly], True, color, 3, lineType=cv2.LINE_AA)
            else:
                cv2.circle(top_down_map, tuple(pix), r_px, color, 3, lineType=cv2.LINE_AA)

            # 区域名画在圆心，字号随半径放大，和物体标签形成明显的层级差异
            area_label = f"[{g.get('name', '')}]"
            a_scale = float(np.clip(r_px / 120.0, 0.8, 2.4))
            a_th = max(2, int(a_scale * 1.6))
            (aw, ah), _ = cv2.getTextSize(area_label, cv2.FONT_HERSHEY_SIMPLEX, a_scale, a_th)
            org = (int(pix[0] - aw / 2), int(pix[1] + ah / 2))
            if 0 <= org[0] < new_w - aw and ah < org[1] < new_h:
                cv2.putText(top_down_map, area_label, org, cv2.FONT_HERSHEY_SIMPLEX,
                            a_scale, (255, 255, 255), a_th + 3, cv2.LINE_AA)
                cv2.putText(top_down_map, area_label, org, cv2.FONT_HERSHEY_SIMPLEX,
                            a_scale, color, a_th, cv2.LINE_AA)
        except Exception:
            continue

    for g in grounded:
        if g.get('kind') == 'area':
            continue
        try:
            style = ROLE_STYLE.get(g.get('role'), ROLE_STYLE['unknown'])
            color = style['color']
            pix = transform_and_to_pixel(np.asarray(g['center'], dtype=float))[0]
            pix = (pix * label_upscale_factor).astype(int)
            if not (0 <= pix[0] < new_w and 0 <= pix[1] < new_h):
                continue

            # 目的地额外画一条到agent的虚线，方便一眼看出"还差多远、在哪个方位"
            if g.get('role') == 'destination_marker':
                p0 = tuple(agent_pixel_scaled_for_lm)
                p1 = (int(pix[0]), int(pix[1]))
                total = int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                dash, gap = 14, 10
                drawn = 0
                while drawn < total:
                    a = drawn / max(total, 1)
                    b = min(drawn + dash, total) / max(total, 1)
                    pa = (int(p0[0] + (p1[0] - p0[0]) * a), int(p0[1] + (p1[1] - p0[1]) * a))
                    pb = (int(p0[0] + (p1[0] - p0[0]) * b), int(p0[1] + (p1[1] - p0[1]) * b))
                    cv2.line(top_down_map, pa, pb, color, 2, lineType=cv2.LINE_AA)
                    drawn += dash + gap

            r_out = max(14, int(11 * label_upscale_factor))

            if g.get('role') == 'avoid_marker':
                # 红色禁行圆 + 叉。半径按规划器实际使用的 AVOID_RADIUS 画，
                # 这样图上看到的范围就是 A* 里真正加了惩罚的范围，不会误导。
                zone_r_m = float(g.get('radius', getattr(mapper, 'AVOID_RADIUS', 1.5))) \
                    if g.get('kind') == 'area' else getattr(mapper, 'AVOID_RADIUS', 1.5)
                zone_r_px = max(r_out * 2, int(zone_r_m / map_scale * label_upscale_factor))
                overlay_av = top_down_map.copy()
                cv2.circle(overlay_av, tuple(pix), zone_r_px, color, -1, lineType=cv2.LINE_AA)
                cv2.addWeighted(overlay_av, 0.3, top_down_map, 0.7, 0, top_down_map)
                cv2.circle(top_down_map, tuple(pix), zone_r_px, color, 2, lineType=cv2.LINE_AA)
                d = int(r_out * 0.7)
                cv2.line(top_down_map, (pix[0] - d, pix[1] - d), (pix[0] + d, pix[1] + d), color, 3, cv2.LINE_AA)
                cv2.line(top_down_map, (pix[0] - d, pix[1] + d), (pix[0] + d, pix[1] - d), color, 3, cv2.LINE_AA)
                cv2.circle(top_down_map, tuple(pix), r_out, color, 3, lineType=cv2.LINE_AA)
            elif g.get('role') == 'destination_marker':
                cv2.circle(top_down_map, tuple(pix), r_out, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(top_down_map, tuple(pix), r_out, (255, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(top_down_map, tuple(pix), r_out + 3, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            else:  # waypoint_marker / unknown：空心圈，视觉权重低于目的地
                cv2.circle(top_down_map, tuple(pix), r_out, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(top_down_map, tuple(pix), r_out, color, 4, lineType=cv2.LINE_AA)

            # 标签：角色 + 指令原文里的名字 + 距离。距离让VLM不用自己从图上估。
            # 同类多实例时补一个 (1/N)，提示"这是N个同类里挑出来的"，
            # 因为消歧可能挑错，这个提示能让人在录像里一眼看出该怀疑哪些pin。
            label = f"{style['tag']}:{g.get('name', '')}"
            if g.get('n_candidates', 1) > 1:
                label += f" (1/{g['n_candidates']})"
            label += f" {g.get('distance', 0.0):.1f}m"

            f_scale, f_th = 0.62, 2
            (tw_l, th_l), bl_l = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_th)
            tx_l = int(pix[0] + r_out + 6)
            ty_l = int(pix[1] + th_l // 2)
            tx_l = max(2, min(tx_l, new_w - tw_l - 4))
            ty_l = max(th_l + 4, min(ty_l, new_h - 4))

            # 标签避让：和已放置的标签框相交就整体下移一行，最多试6次。
            # 试满还冲突就照原位画——盖住一点也比整条标签消失强。
            box_h = th_l + bl_l + 6
            for _ in range(6):
                cand = (tx_l - 4, ty_l - th_l - 4, tx_l + tw_l + 4, ty_l + bl_l + 2)
                hit = any(not (cand[2] < b[0] or cand[0] > b[2] or cand[3] < b[1] or cand[1] > b[3])
                          for b in placed_label_boxes)
                if not hit:
                    break
                ty_l += box_h + 2
                if ty_l > new_h - 4:
                    ty_l = max(th_l + 4, int(pix[1] + th_l // 2) - box_h)
                    break
            placed_label_boxes.append((tx_l - 4, ty_l - th_l - 4, tx_l + tw_l + 4, ty_l + bl_l + 2))

            cv2.rectangle(top_down_map, (tx_l - 4, ty_l - th_l - 4), (tx_l + tw_l + 4, ty_l + bl_l + 2),
                          (255, 255, 255), -1)
            cv2.rectangle(top_down_map, (tx_l - 4, ty_l - th_l - 4), (tx_l + tw_l + 4, ty_l + bl_l + 2),
                          color, 2)
            # pin 移位后画一小段引线连回本体，避免标签认错主人
            if abs(ty_l - int(pix[1] + th_l // 2)) > 4:
                cv2.line(top_down_map, (int(pix[0]), int(pix[1])), (tx_l - 4, ty_l - th_l // 2),
                         color, 1, cv2.LINE_AA)
            cv2.putText(top_down_map, label, (tx_l, ty_l), cv2.FONT_HERSHEY_SIMPLEX, f_scale,
                        (0, 0, 0), f_th, cv2.LINE_AA)
        except Exception:
            continue
    timings['draw_landmarks'] = time.time() - t

    arrow_length = int(12 * label_upscale_factor * 1.2)  # shorter arrow, less sharp
    base_half = max(12, int(arrow_length * 0.6))  # wider base for blunter triangle

    triangle_pts = np.array([
        agent_pixel_scaled + np.array([0, -arrow_length]),  # apex
        agent_pixel_scaled + np.array([-base_half, int(arrow_length * 0.6)]),  # bottom-left
        agent_pixel_scaled + np.array([base_half, int(arrow_length * 0.6)])  # bottom-right
    ], dtype=np.int32)

    cv2.fillPoly(top_down_map, [triangle_pts], (0, 0, 255), lineType=cv2.LINE_AA)

    # ---------------- 局部空间状态：标签 + 调试射线 ----------------
    ls = getattr(mapper, 'local_space', None) or {}
    if ls.get('state') and ls.get('state') != 'unknown':
        ap = agent_pixel_scaled

        # 调试层（只进录像）：通行余量圆 + 四条测距射线。
        # 染色是历史、标签是此刻，而这一层是产生两者的原始测量。
        if debug_overlay and ls.get('clearance') is not None:
            cr = int(float(ls['clearance']) / map_scale * label_upscale_factor)
            cv2.circle(top_down_map, tuple(ap), max(2, cr), (0, 200, 255), 1, cv2.LINE_AA)
            rays = ls.get('rays') or {}
            # 地图是ego-centric、前向朝上，所以四个方向在图上就是上下左右
            for key, (dx, dy) in (('f', (0, -1)), ('b', (0, 1)),
                                  ('l', (-1, 0)), ('r', (1, 0))):
                dist_m = rays.get(key)
                if dist_m is None:
                    continue
                L = int(float(dist_m) / map_scale * label_upscale_factor)
                end = (int(ap[0] + dx * L), int(ap[1] + dy * L))
                cv2.line(top_down_map, tuple(ap), end, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.circle(top_down_map, end, 4, (0, 200, 255), -1, cv2.LINE_AA)

        badge = ls['state']
        if ls.get('clearance') is not None:
            badge += f" {float(ls['clearance']):.1f}m"
        # "我在哪个房间" —— 取代了原来那一堆灰圈，一个答案而不是一张房间图
        try:
            room = mapper.current_room()
            if room:
                badge += f"  in {room['area_type']}"
        except Exception:
            pass
        bs, bt = 0.6, 2
        (bw, bh), bbl = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, bs, bt)
        bx = int(ap[0] - bw / 2)
        by = int(ap[1] + arrow_length + bh + 10)
        if 0 <= bx < new_w - bw and bh < by < new_h:
            cv2.rectangle(top_down_map, (bx - 5, by - bh - 5), (bx + bw + 5, by + bbl + 2),
                          (255, 255, 255), -1)
            cv2.rectangle(top_down_map, (bx - 5, by - bh - 5), (bx + bw + 5, by + bbl + 2),
                          (90, 90, 90), 1)
            cv2.putText(top_down_map, badge, (bx, by), cv2.FONT_HERSHEY_SIMPLEX,
                        bs, (40, 40, 40), bt, cv2.LINE_AA)

    # ---------------- 被淘汰的候选 waypoint（只进调试帧）----------------
    # 每次排查"为什么 agent 不往那边走"，以前只有一行数字可看。把点按原因
    # 画出来，能立刻分辨是过滤器太狠还是那个方向真的不可达。
    rejected_legend = []
    if debug_overlay and rejected_waypoints:
        REJ = {                     # BGR
            'edge':      ((0, 165, 255), 'edge'),        # 橙：投影落在画面边缘
            'occlusion': ((200, 0, 200), 'occl'),        # 紫：被前景遮挡
            'distance':  ((150, 150, 150), 'dist'),      # 灰：太近或太远
            'no_path':   ((0, 0, 255), 'nopath'),        # 红：A* 找不到路
            'behind':    ((255, 200, 100), 'behind'),    # 浅蓝：在相机后方
        }
        counts = {}
        for r in rejected_waypoints:
            try:
                col, _ = REJ.get(r.get('reason'), ((120, 120, 120), '?'))
                counts[r.get('reason')] = counts.get(r.get('reason'), 0) + 1
                p = transform_and_to_pixel(np.asarray(r['world'], dtype=float))[0]
                p = (p * label_upscale_factor).astype(int)
                if not (0 <= p[0] < new_w and 0 <= p[1] < new_h):
                    continue
                d = max(4, int(4 * label_upscale_factor))
                cv2.line(top_down_map, (p[0] - d, p[1] - d), (p[0] + d, p[1] + d), col, 2, cv2.LINE_AA)
                cv2.line(top_down_map, (p[0] - d, p[1] + d), (p[0] + d, p[1] - d), col, 2, cv2.LINE_AA)
            except Exception:
                continue

        # 图例留到裁剪之后再画：这里的画布是完整的 map_size_px，而最终输出
        # 只是以 agent 为中心的 16m 裁剪窗口，贴在画布边缘的东西会被整个切掉。
        # (清单最初就踩过这个坑)
        rejected_legend = [
            (REJ.get(k, ((120, 120, 120), '?'))[1], v, REJ.get(k, ((120, 120, 120), '?'))[0])
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]

    outline_thickness = max(3, 2 * label_upscale_factor)
    cv2.polylines(top_down_map, [triangle_pts], isClosed=True, color=(100, 100, 100),
                  thickness=outline_thickness, lineType=cv2.LINE_AA)

    timings['draw_agent'] = time.time() - t

    try:
        start_world = np.array([0.0, 0.0, 0.0])
        start_pixel = transform_and_to_pixel(start_world)[0]  # returns (x, y)
        start_pixel_scaled = (start_pixel * label_upscale_factor).astype(int)

        if 0 <= start_pixel_scaled[0] < new_w and 0 <= start_pixel_scaled[1] < new_h:

            cv2.circle(top_down_map, tuple(start_pixel_scaled), 7, (255, 0, 0), -1)  # solid blue
            cv2.circle(top_down_map, tuple(start_pixel_scaled), 9, (0, 0, 0), 2)  # black border

            text = "START"
            font_scale = 0.7
            thickness = 1
            (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            text_org = (start_pixel_scaled[0] + 10, start_pixel_scaled[1] + th // 2)

            bx1 = text_org[0] - 3
            by1 = text_org[1] - th - 3
            bx2 = text_org[0] + tw + 3
            by2 = text_org[1] + 3

            bx1 = max(bx1, 0);
            by1 = max(by1, 0);
            bx2 = min(bx2, new_w - 1);
            by2 = min(by2, new_h - 1)
            overlay = top_down_map.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.8, top_down_map, 0.2, 0, top_down_map)

            cv2.putText(top_down_map, text, text_org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
                        cv2.LINE_AA)
    except Exception:

        pass

    crop_size_px = int(crop_size_m / map_scale * label_upscale_factor)
    agent_pixel_scaled = (np.array([map_size_px / 2, map_size_px / 2]) * label_upscale_factor).astype(int)
    half_crop = crop_size_px // 2
    y1 = max(agent_pixel_scaled[1] - half_crop, 0)
    y2 = min(agent_pixel_scaled[1] + half_crop, top_down_map.shape[0])
    x1 = max(agent_pixel_scaled[0] - half_crop, 0)
    x2 = min(agent_pixel_scaled[0] + half_crop, top_down_map.shape[1])
    top_down_map_cropped = top_down_map[y1:y2, x1:x2]

    top_down_map_cropped = draw_direction_markers(top_down_map_cropped)

    # ---------------- landmark 清单（画在裁剪之后，否则会被裁掉）----------------
    # 左上角列出当前子任务的所有landmark及其状态：
    #   ● 已接地(带距离)   ○ 尚未在图上检测到   ✕ 不在检测器词表里
    # "不在词表里"和"还没看到"是完全不同的两种失败，必须分开显示——前者
    # 再怎么探索也不可能接地(比如 window/hallway 根本不在那207类里)。
    try:
        _grounded = list(getattr(mapper, 'grounded_landmarks', []) or [])
        _pending = list(getattr(mapper, 'unmatched_landmarks', []) or [])
        if _grounded or _pending:
            rows = []
            # 已接地的 landmark 按时序状态区分符号，而不是一律 '*'。
            # "已经走过去了"和"看到了还没去"对下一步该做什么的含义完全不同。
            STATE_MARK = {'seen': '*', 'near': '>', 'passed': 'V'}
            for g in _grounded:
                tag = ROLE_STYLE.get(g.get('role'), ROLE_STYLE['unknown'])
                st = g.get('state', 'seen')
                mark = "#" if g.get('kind') == 'area' else STATE_MARK.get(st, '*')
                txt = f"{mark} {g.get('name','')}  {g.get('distance',0.0):.1f}m"
                if st == 'passed':
                    txt += f" (was {g.get('min_dist', 0.0):.1f}m, passed)"
                elif st == 'near':
                    txt += " (here)"
                rows.append((txt, tag['color']))
            REASON_TEXT = {
                'not_in_vocab': ('x', 'not in vocab'),
                'not_yet_detected': ('o', 'not seen yet'),
                'area_pending': ('o', 'area, not identified yet'),
                'area_no_proxy': ('x', 'area, no visual cue'),
            }
            for u in _pending:
                sym, why = REASON_TEXT.get(u.get('reason'), ('o', str(u.get('reason'))))
                col = (120, 120, 120) if sym == 'x' else (90, 90, 90)
                rows.append((f"{sym} {u.get('name','')}  ({why})", col))

            ch, cw = top_down_map_cropped.shape[:2]
            f_scale, f_th = 0.55, 1
            line_h = 24
            pad = 8
            box_w = max(
                cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, f_scale, f_th)[0][0] for txt, _ in rows
            ) + 2 * pad
            box_w = min(box_w, cw - 8)
            box_h = line_h * len(rows) + 2 * pad
            # y 从 48 起而不是 4：run_experiments 的可视化面板会在图左上角画
            # "Top-Down Map" 标题，压在 y<40 的区域，清单贴着顶边会被盖住看不清。
            x0, y0 = 4, 48

            if y0 + box_h < ch - 8:
                panel = top_down_map_cropped.copy()
                cv2.rectangle(panel, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
                cv2.addWeighted(panel, 0.82, top_down_map_cropped, 0.18, 0, top_down_map_cropped)
                cv2.rectangle(top_down_map_cropped, (x0, y0), (x0 + box_w, y0 + box_h), (0, 0, 0), 1)
                for i, (txt, col) in enumerate(rows):
                    org = (x0 + pad, y0 + pad + line_h * i + int(line_h * 0.7))
                    cv2.putText(top_down_map_cropped, txt, org, cv2.FONT_HERSHEY_SIMPLEX,
                                f_scale, col, f_th, cv2.LINE_AA)
    except Exception:
        pass

    # ---------------- 被淘汰 waypoint 的图例（裁剪之后，右上角）----------------
    try:
        if rejected_legend:
            ch, cw = top_down_map_cropped.shape[:2]
            lh, pad, bw = 22, 6, 150
            bh = lh * len(rejected_legend) + 2 * pad
            x0r, y0r = cw - bw - 6, 6
            if x0r > 0 and y0r + bh < ch:
                panel = top_down_map_cropped.copy()
                cv2.rectangle(panel, (x0r, y0r), (x0r + bw, y0r + bh), (255, 255, 255), -1)
                cv2.addWeighted(panel, 0.82, top_down_map_cropped, 0.18, 0, top_down_map_cropped)
                cv2.rectangle(top_down_map_cropped, (x0r, y0r), (x0r + bw, y0r + bh), (0, 0, 0), 1)
                for i, (nm, v, col) in enumerate(rejected_legend):
                    yy = y0r + pad + lh * i + int(lh * 0.7)
                    cv2.line(top_down_map_cropped, (x0r + 8, yy - 5), (x0r + 18, yy + 5),
                             col, 2, cv2.LINE_AA)
                    cv2.line(top_down_map_cropped, (x0r + 8, yy + 5), (x0r + 18, yy - 5),
                             col, 2, cv2.LINE_AA)
                    cv2.putText(top_down_map_cropped, f"{nm} x{v}", (x0r + 26, yy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    except Exception:
        pass

    # ---------------- 子任务进度条（裁剪之后，贴在图底部）----------------
    # 每段宽度 ∝ relative_duration，所以一眼能看出"这个子任务本来就该走很久"
    # 还是"它超支了"。已完成=绿，当前=按 used/expected 比例填充(超预算转红)，
    # 未开始=浅灰。
    try:
        if progress:
            ch, cw = top_down_map_cropped.shape[:2]
            bar_h = 26
            pad = 4
            y0 = ch - bar_h - pad
            x0, x1_ = pad, cw - pad
            total_w = x1_ - x0
            rel_sum = sum(max(r.get('rel', 0.0), 1e-6) for r in progress) or 1.0

            panel = top_down_map_cropped.copy()
            cv2.rectangle(panel, (x0, y0), (x1_, y0 + bar_h), (255, 255, 255), -1)
            cv2.addWeighted(panel, 0.85, top_down_map_cropped, 0.15, 0, top_down_map_cropped)

            cx = x0
            for r in progress:
                seg_w = max(2, int(total_w * max(r.get('rel', 0.0), 1e-6) / rel_sum))
                seg_x2 = min(cx + seg_w, x1_)
                status = r.get('status')
                ratio = float(r.get('ratio', 0.0))

                if status == 'done':
                    fill_col, fill_w = (80, 190, 80), seg_x2 - cx
                elif status == 'current':
                    # 超预算转红；填充长度截断在本段内，超出部分靠颜色表达
                    fill_col = (60, 90, 235) if ratio > 1.0 else (40, 170, 245)
                    fill_w = int((seg_x2 - cx) * min(ratio, 1.0))
                else:
                    fill_col, fill_w = (225, 225, 225), 0

                cv2.rectangle(top_down_map_cropped, (cx, y0), (seg_x2, y0 + bar_h),
                              (228, 228, 228), -1)
                if fill_w > 0:
                    cv2.rectangle(top_down_map_cropped, (cx, y0), (cx + fill_w, y0 + bar_h),
                                  fill_col, -1)
                cv2.rectangle(top_down_map_cropped, (cx, y0), (seg_x2, y0 + bar_h),
                              (90, 90, 90), 1)

                # 段内标号：只画序号，宽度不够就不画
                num = r.get('key', '').replace('SUBTASK_', '')
                if status == 'current':
                    num += f" {int(round(ratio * 100))}%"
                (tw_p, th_p), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                if tw_p + 6 < seg_x2 - cx:
                    org = (cx + (seg_x2 - cx - tw_p) // 2, y0 + (bar_h + th_p) // 2)
                    cv2.putText(top_down_map_cropped, num, org, cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (20, 20, 20), 1, cv2.LINE_AA)
                cx = seg_x2
                if cx >= x1_:
                    break
    except Exception:
        pass

    t_end = time.time()

    return top_down_map_cropped
def create_top_down_map_global(
    mapper: Instruct_Mapper,
    map_scale: float = 0.025,
    padding_px: int = 60,
    fov_angle_deg: float = 60.0,
    fov_range: float = 2.5,
    target_point=None,
    action_candidates=None,
    subtasks=None,
    enable_semantic: bool = True,
    enable_navigation: bool = True,
    show_all_labels: bool = False,
    label_volume_threshold: int = 100,
):
    """
    Global top-down view (not agent-centered).:
    - Auto-compute scene bounds and expand canvas.
    - enable_semantic: whether to render object point clouds and semantic labels.
    - enable_navigation: whether to render target / candidate waypoints / subtask nodes.
    """
    t_start = time.time()

    point_sets = []

    if hasattr(mapper, "navigable_pcd") and not mapper.navigable_pcd.is_empty():
        point_sets.append(mapper.navigable_pcd.point.positions.cpu().numpy()[:, :2])
    if hasattr(mapper, "obstacle_pcd") and not mapper.obstacle_pcd.is_empty():
        point_sets.append(mapper.obstacle_pcd.point.positions.cpu().numpy()[:, :2])
    if enable_semantic and hasattr(mapper, "object_entities"):
        for ent in mapper.object_entities:
            if ent['pcd'].point.positions.shape[0] > 0:
                point_sets.append(ent['pcd'].point.positions.cpu().numpy()[:, :2])
    if hasattr(mapper, "frontier_pcd") and not mapper.frontier_pcd.is_empty():
        point_sets.append(mapper.frontier_pcd.point.positions.cpu().numpy()[:, :2])
    if mapper.trajectory_position:
        traj_arr = np.array(mapper.trajectory_position)[:, :2]
        point_sets.append(traj_arr)
    if enable_navigation:
        if target_point is not None:
            point_sets.append(np.array(target_point)[None, :2])
        if action_candidates:
            for cand in action_candidates.values():
                coord = cand.get("target_point_world")
                if coord is not None:
                    point_sets.append(np.array(coord)[None, :2])
        if subtasks:
            for coord in subtasks.values():
                if coord is not None:
                    point_sets.append(np.array(coord)[None, :2])

    if not point_sets:

        map_size_px = 1024
        top_down_map = np.full((map_size_px, map_size_px, 3), 255, dtype=np.uint8)
        return top_down_map

    all_pts = np.vstack(point_sets)
    min_xy = all_pts.min(axis=0)
    max_xy = all_pts.max(axis=0)

    span = max(max_xy[0] - min_xy[0], max_xy[1] - min_xy[1])

    base_size_px = int(span / map_scale) + 2 * padding_px

    base_size_px = max(base_size_px, 1024)

    width_px = int((max_xy[0] - min_xy[0]) / map_scale) + 2 * padding_px
    height_px = int((max_xy[1] - min_xy[1]) / map_scale) + 2 * padding_px
    width_px = max(width_px, 1024)
    height_px = max(height_px, 1024)

    top_down_map = np.full((height_px, width_px, 3), 255, dtype=np.uint8)
    map_h, map_w = top_down_map.shape[:2]
    offset = min_xy - padding_px * map_scale  # world-coord bottom-left offset

    def world_to_pixel(world_coords):
        arr = np.asarray(world_coords, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        px = (arr[:, :2] - offset) / map_scale
        return px.astype(int)

    pcd_resolution = 0.025
    dilation_radius = max(1, int(pcd_resolution / map_scale / 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_radius * 2 + 1, dilation_radius * 2 + 1))

    if hasattr(mapper, "navigable_pcd") and not mapper.navigable_pcd.is_empty():
        nav_pts_world = mapper.navigable_pcd.point.positions.cpu().numpy()
        nav_px = world_to_pixel(nav_pts_world)
        valid = (nav_px[:, 0] >= 0) & (nav_px[:, 0] < map_w) & (nav_px[:, 1] >= 0) & (nav_px[:, 1] < map_h)
        if np.any(valid):
            mask = np.zeros((map_h, map_w), dtype=np.uint8)
            mask[nav_px[valid, 1], nav_px[valid, 0]] = 255
            mask = cv2.dilate(mask, kernel)
            top_down_map[mask == 255] = (200, 200, 200)

    if enable_navigation:
        try:
            wps = mapper.get_candidate_waypoints(min_distance=0.3, max_distance=1.5, waypoint_grid_resolution=0.5)
            if wps is not None:

                if hasattr(wps, "detach"):
                    wps = wps.detach().cpu().numpy()
                else:
                    wps = np.asarray(wps)
                if wps.size > 0:
                    if wps.ndim == 1:
                        wps = wps.reshape(1, -1)

                    wp_xy = wps[:, :2]
                    wp_px = world_to_pixel(wp_xy)
                    valid = (wp_px[:, 0] >= 0) & (wp_px[:, 0] < map_w) & (wp_px[:, 1] >= 0) & (wp_px[:, 1] < map_h)
                    if np.any(valid):
                        overlay = top_down_map.copy()
                        for p in wp_px[valid]:

                            cv2.circle(overlay, tuple(p), 2, (0, 165, 255), -1, cv2.LINE_AA)
                        top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)
        except Exception:

            pass

    obstacle_mask_bool = np.zeros((map_h, map_w), dtype=bool)
    if hasattr(mapper, "obstacle_pcd") and not mapper.obstacle_pcd.is_empty():
        obs_pts_world = mapper.obstacle_pcd.point.positions.cpu().numpy()
        obs_px = world_to_pixel(obs_pts_world)
        valid = (obs_px[:, 0] >= 0) & (obs_px[:, 0] < map_w) & (obs_px[:, 1] >= 0) & (obs_px[:, 1] < map_h)
        if np.any(valid):
            mask = np.zeros((map_h, map_w), dtype=np.uint8)
            mask[obs_px[valid, 1], obs_px[valid, 0]] = 255
            mask = cv2.dilate(mask, kernel)
            top_down_map[mask == 255] = (50, 50, 50)
            obstacle_mask_bool = (mask == 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_down_map, contours, -1, (0, 0, 0), thickness=4)

    if hasattr(mapper, "frontier_pcd") and not mapper.frontier_pcd.is_empty():
        fr_px = world_to_pixel(mapper.frontier_pcd.point.positions.cpu().numpy())
        valid = (fr_px[:, 0] >= 0) & (fr_px[:, 0] < map_w) & (fr_px[:, 1] >= 0) & (fr_px[:, 1] < map_h)
        for p in fr_px[valid]:
            cv2.circle(top_down_map, tuple(p), 1, (0, 255, 255), -1)

    if enable_semantic and hasattr(mapper, "object_entities"):
        for ent in mapper.object_entities:
            if ent['pcd'].point.positions.shape[0] == 0:
                continue
            obj_px = world_to_pixel(ent['pcd'].point.positions.cpu().numpy())
            valid = (obj_px[:, 0] >= 0) & (obj_px[:, 0] < map_w) & (obj_px[:, 1] >= 0) & (obj_px[:, 1] < map_h)
            if not np.any(valid):
                continue
            obj_valid_px = obj_px[valid]

            inter_mask = obstacle_mask_bool[obj_valid_px[:, 1], obj_valid_px[:, 0]]
            if np.any(inter_mask):
                final_px = obj_valid_px[inter_mask]
                color_arr = ent['pcd'].point.colors.cpu().numpy()[valid][inter_mask]
            else:
                final_px = obj_valid_px
                color_arr = ent['pcd'].point.colors.cpu().numpy()[valid]
            if final_px.shape[0] == 0:
                continue

            obj_color = (color_arr[0] * 255)[::-1].astype(np.uint8)
            mask_img = np.zeros((map_h, map_w), dtype=np.uint8)
            mask_img[final_px[:, 1], final_px[:, 0]] = 255
            fill_kernel = np.ones((5, 5), np.uint8)
            filled = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, fill_kernel, iterations=2)
            top_down_map[filled == 255] = obj_color

    if False and enable_semantic and hasattr(mapper, "object_entities"):
        drawn_labels = []
        merge_dist_px = 40
        for ent in sorted(mapper.object_entities, key=lambda e: e['class']):
            if ent['pcd'].point.positions.shape[0] == 0:
                continue
            if (not show_all_labels) and ent['pcd'].point.positions.shape[0] < label_volume_threshold:
                continue
            center_px = world_to_pixel(ent['center'])[0]
            if not (0 <= center_px[0] < map_w and 0 <= center_px[1] < map_h):
                continue
            cname = ent['class_name']
            merged = False
            for i, info in enumerate(drawn_labels):
                if info['class'] == cname and np.linalg.norm(center_px - info['center']) < merge_dist_px:
                    drawn_labels[i]['center'] = (info['center'] + center_px) / 2
                    merged = True
                    break
            if not merged:
                drawn_labels.append({'class': cname, 'center': center_px})

        for info in drawn_labels:
            cp = info['center'].astype(int)
            font_scale = 0.6
            thickness = 1
            (tw, th), bl = cv2.getTextSize(info['class'], cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            tl = (cp[0] - tw // 2 - 3, cp[1] - th - 4)
            br = (cp[0] + tw // 2 + 3, cp[1] + bl + 2)
            tl = (max(0, tl[0]), max(0, tl[1]))
            br = (min(map_w - 1, br[0]), min(map_h - 1, br[1]))
            overlay = top_down_map.copy()
            cv2.rectangle(overlay, tl, br, (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.85, top_down_map, 0.15, 0, top_down_map)
            cv2.rectangle(top_down_map, tl, br, (0, 0, 0), 1)
            text_org = (tl[0] + 3, br[1] - bl)
            cv2.putText(top_down_map, info['class'], text_org, cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    if enable_navigation and mapper.trajectory_position and len(mapper.trajectory_position) > 1:
        traj_px = world_to_pixel(np.array(mapper.trajectory_position[-200:])[:, :2]).astype(int)
        valid = (traj_px[:, 0] >= 0) & (traj_px[:, 0] < map_w) & (traj_px[:, 1] >= 0) & (traj_px[:, 1] < map_h)
        traj_px = traj_px[valid]
        if traj_px.shape[0] > 1:
            overlay = top_down_map.copy()
            cv2.polylines(overlay, [traj_px], False, (0, 0, 255), 2)
            top_down_map = cv2.addWeighted(overlay, 0.6, top_down_map, 0.4, 0)

    if enable_navigation:

        # if target_point is not None:
        #     tp = world_to_pixel(np.array(target_point))[0]
        #     if 0 <= tp[0] < map_w and 0 <= tp[1] < map_h:
        #         cv2.circle(top_down_map, tuple(tp), 9, (0, 255, 0), -1)
        #         cv2.circle(top_down_map, tuple(tp), 9, (0, 0, 0), 2)

        if action_candidates:
            overlay = top_down_map.copy()
            for text, cand in action_candidates.items():
                if cand.get("type") != "waypoint":
                    continue
                coord = cand.get("target_point_world")
                if coord is None:
                    continue
                pix = world_to_pixel(np.array(coord))[0]
                if not (0 <= pix[0] < map_w and 0 <= pix[1] < map_h):
                    continue

                try:
                    idx = int(text.split()[-1])
                except Exception:
                    idx = None
                radius = 11
                cv2.circle(overlay, tuple(pix), radius, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, tuple(pix), radius, (0, 128, 255), 2, cv2.LINE_AA)
                if idx is not None:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), bl = cv2.getTextSize(str(idx), font, 0.6, 2)
                    org = (int(pix[0] - tw / 2), int(pix[1] + th / 2))
                    cv2.putText(overlay, str(idx), org, font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            top_down_map = cv2.addWeighted(overlay, 0.9, top_down_map, 0.1, 0)

        if subtasks:
            overlay = top_down_map.copy()
            for name, coord in subtasks.items():
                if coord is None:
                    continue
                pix = world_to_pixel(np.array(coord))[0]
                if not (0 <= pix[0] < map_w and 0 <= pix[1] < map_h):
                    continue

                half = 10
                tl = (pix[0] - half, pix[1] - half)
                br = (pix[0] + half, pix[1] + half)
                tl = (max(0, tl[0]), max(0, tl[1]))
                br = (min(map_w - 1, br[0]), min(map_h - 1, br[1]))

                cv2.rectangle(overlay, tl, br, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.rectangle(overlay, tl, br, (0, 0, 0), 1, cv2.LINE_AA)

                text = str(name)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.45  # fit 16x16 block
                thickness = 1
                (tw, th), bl = cv2.getTextSize(text, font, font_scale, thickness)

                cx = (tl[0] + br[0]) // 2
                cy = (tl[1] + br[1]) // 2

                text_org = (int(cx - tw / 2), int(cy + th / 2) - 1)

                cv2.putText(overlay, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            top_down_map = cv2.addWeighted(overlay, 0.95, top_down_map, 0.05, 0)

    agent_pos = getattr(mapper, "current_position", np.array([0.0, 0.0, 0.0]))
    agent_rot = getattr(mapper, "current_rotation", np.eye(3))
    agent_px = world_to_pixel(agent_pos)[0]
    if 0 <= agent_px[0] < map_w and 0 <= agent_px[1] < map_h:
        forward_vec_3d = agent_rot @ np.array([0, 0, -1])
        forward_2d = forward_vec_3d[:2]
        if np.linalg.norm(forward_2d) < 1e-6:
            forward_2d = np.array([1.0, 0.0])
        forward_2d /= np.linalg.norm(forward_2d)
        angle = np.arctan2(forward_2d[1], forward_2d[0])

        # fov_rad = np.deg2rad(fov_angle_deg)
        # num_pts = 40
        # angles = np.linspace(angle - fov_rad / 2, angle + fov_rad / 2, num_pts)
        # fov_range_px = fov_range / map_scale
        # sector_pts = [agent_px]
        # for a in angles:
        #     end_world = agent_pos[:2] + np.array([np.cos(a), np.sin(a)]) * fov_range
        #     sector_pts.append(world_to_pixel(end_world)[0])
        # overlay = top_down_map.copy()
        # cv2.fillPoly(overlay, [np.array(sector_pts, dtype=np.int32)], (255, 255, 0))
        # top_down_map = cv2.addWeighted(overlay, 0.2, top_down_map, 0.8, 0)

        length = 30
        base_half = 10
        tip = agent_px + (forward_2d * length).astype(int)
        left_dir = np.array([[0, -1], [1, 0]]) @ forward_2d
        right_dir = -left_dir
        p_left = agent_px + (forward_2d * (length * 0.2) + left_dir * base_half).astype(int)
        p_right = agent_px + (forward_2d * (length * 0.2) + right_dir * base_half).astype(int)
        tri = np.array([tip, p_left, p_right], dtype=np.int32)
        cv2.fillPoly(top_down_map, [tri], (0, 0, 255))
        cv2.polylines(top_down_map, [tri], True, (50, 50, 50), 2)

    start_px = world_to_pixel(np.array([0.0, 0.0]))[0]
    if enable_navigation and 0 <= start_px[0] < map_w and 0 <= start_px[1] < map_h:
        cv2.circle(top_down_map, tuple(start_px), 7, (255, 0, 0), -1)
        cv2.circle(top_down_map, tuple(start_px), 9, (0, 0, 0), 2)
        (tw, th), bl = cv2.getTextSize("START", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        org = (start_px[0] + 10, start_px[1] + th // 2)
        bx1, by1 = org[0] - 3, org[1] - th - 3
        bx2, by2 = org[0] + tw + 3, org[1] + 3
        bx1 = max(bx1, 0); by1 = max(by1, 0)
        bx2 = min(bx2, map_w - 1); by2 = min(by2, map_h - 1)
        overlay = top_down_map.copy()
        # cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.8, top_down_map, 0.2, 0, top_down_map)
        # cv2.putText(top_down_map, "START", org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    return cv2.cvtColor(top_down_map, cv2.COLOR_BGR2RGB)
