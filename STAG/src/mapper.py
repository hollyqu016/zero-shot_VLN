import math

from mapping_utils.geometry import *
from mapping_utils.preprocess import *
from mapping_utils.projection import *
from mapping_utils.transform import *
from mapping_utils.path_planning import *

from matplotlib import colormaps
from habitat_sim.utils.common import d3_40_colors_rgb

import open3d as o3d
from config_utils import hyper
from segmentation.instance_segmentation import instance_segmentation
from segmentation.object_list import (categories, normalize_category_name, resolve_category,
                                      resolve_area, AREA_PROXY_OBJECTS)
from segmentation.instance_segmentation import get_class_color
import heapq

import time
import cv2
import numpy as np
import gc
from typing import Any

class Instruct_Mapper:
    def __init__(self,
                 camera_intrinsic,
                 pcd_resolution=0.05,
                 grid_resolution=0.1,
                 grid_size=5,
                 floor_height=-1.2,
                 ceiling_height=0.6,
                 translation_func=habitat_translation,
                 rotation_func=habitat_rotation,
                 rotate_axis=[0, 1, 0],
                 resolution=(480, 640),
                 device='cuda:0',
                 config=None):
        self.device = device
        # 超参一律从 config['hyper'] 取，取不到就用下面各 class 常量的同值默认。
        # 见 _apply_hyper()：它把类常量覆盖成实例属性，所有 self.XXX 的读法不变。
        self._config = config or {}
        # 必须在 __init__ 的最前面：后面的初始化与 _switch_to_floor 都可能读到
        # 这些阈值。之前误放进 reset()，导致超参要等第一次 reset 才生效，
        # 而且每 reset 一次就重打一遍日志（一个 episode 打两遍）。
        self._apply_hyper(self._config)
        self.camera_intrinsic = camera_intrinsic
        self.pcd_resolution = pcd_resolution
        self.grid_resolution = grid_resolution
        self.grid_size = grid_size
        self.initial_floor_height = floor_height
        self.initial_ceiling_height = ceiling_height
        self.floor_height = floor_height
        self.last_floor_height = self.initial_floor_height  # height at last floor update
        self.ceiling_height = ceiling_height
        self.last_ceiling_height = self.initial_ceiling_height  # height at last floor update
        self.translation_func = translation_func
        self.rotation_func = rotation_func
        self.rotate_axis = np.array(rotate_axis)
        self.pcd_device = o3d.core.Device(device.upper())
        self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)  # new: added this line
        self.stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self.floor_level = 0
        self.floor_height_difference = None  # floor height
        self.last_stable_z = None  # last stable height
        self.last_z = None  # last changed height
        self.height_change_timestamp = None  # timestamp when height started changing
        self.is_changing_floor = False  # whether floor change is being detected
        self.stable_update_counter = 0
        self.resolution = resolution

        self.pcds_per_floor = {}
        self._switch_to_floor(0, 0)

        self.local_stair_radius = 2.5
        self.local_stair_min_points = 250
        self.enable_stair_detection = True
        self.trajectory_position = []

        self.waypoints = np.array([])

        # 当前子任务的landmark接地结果，由 ground_landmarks() 每步重算。
        # 结构见 ground_landmarks 的docstring。渲染层(create_top_down_map_centered)
        # 直接读这个字段，所以即使grounding从未被调用过也必须存在且为list。
        self.grounded_landmarks = []
        self.unmatched_landmarks = []
        self.inferred_areas = []
        self._avoid_zones_cache = []
        # 局部空间形态：当前判定 + 按栅格累积的走廊/门口标记
        self.local_space = {'state': 'unknown'}
        self._space_hist = []
        self.space_labels = {}
        # 自由空间几何分区
        self.room_labels = None
        self.room_origin = None
        self.room_regions = {}
        self._room_calls = 0
        self.frontiers = []
        self._frontier_calls = 0

        # 主视图缓存，供 apply_external_instances() 复用同一份反投影数据
        self.current_depth = None
        self._primary_view_pcd = None
        self._primary_view_intrinsic = self.camera_intrinsic

    def _switch_to_floor(self, floor_level, height_diff, last_stable_z=0):
        """Switch to the specified floor, loading or initializing its point cloud and instance data."""

        if self.floor_level in self.pcds_per_floor:
            print(f"Saving data for floor level: {self.floor_level}")
            self.pcds_per_floor[self.floor_level]['scene_pcd'] = self.scene_pcd
            self.pcds_per_floor[self.floor_level]['navigable_pcd'] = self.navigable_pcd
            self.pcds_per_floor[self.floor_level]['obstacle_pcd'] = self.obstacle_pcd
            self.pcds_per_floor[self.floor_level]['trajectory_position'] = self.trajectory_position
            self.pcds_per_floor[self.floor_level]['object_entities'] = self.object_entities

        if floor_level not in self.pcds_per_floor:

            self.pcds_per_floor[floor_level] = {
                'scene_pcd': o3d.t.geometry.PointCloud(self.pcd_device),
                'navigable_pcd': o3d.t.geometry.PointCloud(self.pcd_device),
                'obstacle_pcd': o3d.t.geometry.PointCloud(self.pcd_device),
                'trajectory_position': [],
                'object_entities': []
            }
            print(f"Initialized data for new floor level: {floor_level}")

        print("Switching pcds to floor level:", floor_level)
        current_floor_data = self.pcds_per_floor[floor_level]
        self.scene_pcd = current_floor_data['scene_pcd']
        self.navigable_pcd = current_floor_data['navigable_pcd']
        self.obstacle_pcd = current_floor_data['obstacle_pcd']
        self.trajectory_position = current_floor_data['trajectory_position']
        self.object_entities = current_floor_data['object_entities']
        self.floor_level = floor_level

        if self.floor_height_difference is not None:
            self.floor_height = last_stable_z + height_diff + self.initial_floor_height
            self.ceiling_height = last_stable_z + self.initial_ceiling_height + height_diff
            self.last_floor_height = self.floor_height
            self.last_ceiling_height = self.ceiling_height

    def _calculate_floor_height_difference(self):
        """
        Compute floor height from ceiling-floor height difference in point cloud.
        """
        if self.scene_pcd.is_empty():
            return None
        points_z = self.scene_pcd.point.positions[:, 2].cpu().numpy()

        floor_z = np.percentile(points_z, 5)
        ceiling_z = np.percentile(points_z, 95)

        height_diff = ceiling_z - floor_z

        if 2.0 < height_diff < 5.0:
            self.floor_height_difference = height_diff
            print(f"Calculated floor height difference: {self.floor_height_difference:.2f}m")
        return self.floor_height_difference

    def _detect_and_update_floor(self):
        """
        Detect floor transitions and switch active floor accordingly.
        """
        current_z = self.current_position[2]

        if self.last_stable_z is None: self.last_stable_z = current_z
        if self.last_z is None: self.last_z = current_z

        if self.floor_height_difference is None:
            self._calculate_floor_height_difference()
            return

        if not self.is_changing_floor and abs(current_z - self.last_z) > 0.05:
            self.is_changing_floor = True
            self.height_change_timestamp = time.time()
            self.last_z = current_z

        if self.is_changing_floor:
            if abs(current_z - self.last_z) > 0.05:
                self.height_change_timestamp = time.time()
                self.last_z = current_z
                self.stable_update_counter = 0
            elif self.stable_update_counter >= 0:
                # height_diff = current_z - self.last_stable_z
                height_diff = current_z - (self.last_floor_height + 1.2)
                flag, metrics = self.is_on_platform(self.scene_pcd)
                print(f"--- Height change stabilized. Height diff: {height_diff:.2f}, On platform: {flag}, Metrics: {metrics}")

                if flag and abs(current_z - (self.last_floor_height + 1.2)) >= min(self.floor_height_difference * 0.75, 0.5):

                    floor_change = math.ceil(abs(height_diff) / self.floor_height_difference) * (height_diff // abs(height_diff))
                    new_floor_level = int(self.floor_level + floor_change)

                    print(f"---!Floor change detected. From level {self.floor_level} to {new_floor_level}.")
                    # self._switch_to_floor(new_floor_level, height_diff, self.last_stable_z)
                    self._switch_to_floor(new_floor_level, 0, metrics['plane_height_mean'] + 1.4)

                    print(f"!Switched to floor level: {self.floor_level}")
                    print(f"Updated floor height: {self.floor_height:.2f}, ceiling height: {self.ceiling_height:.2f}, height diff: {height_diff:.2f}")

                    self.last_stable_z = current_z
                else:

                    print(f"--- No significant floor change. Height diff: {height_diff:.2f}, On platform: {flag}, Metrics: {metrics}")
                    self.last_stable_z = current_z + 0.2

                    self.floor_height = self.last_stable_z + self.initial_floor_height
                    self.ceiling_height = self.last_stable_z + self.initial_ceiling_height
                    print(f"Updated floor height: {self.floor_height:.2f}, ceiling height: {self.ceiling_height:.2f}")

                self.last_z = current_z
                self.is_changing_floor = False
                self.stable_update_counter = 0
            else:
                self.stable_update_counter += 1

    def reset(self, position, rotation):
        self.update_iterations = 0
        self.initial_position = self.translation_func(position)
        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)

        self.pcds_per_floor.clear()
        self.floor_level = 0
        self.floor_height = self.initial_floor_height
        self.ceiling_height = self.initial_ceiling_height
        self.last_stable_z = None
        self.last_z = None
        self.is_changing_floor = False
        self.stable_update_counter = 0
        self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)  # new: added this line
        self.stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        self.grounded_landmarks = []
        self.unmatched_landmarks = []
        self.inferred_areas = []
        self._avoid_zones_cache = []
        # 局部空间形态：当前判定 + 按栅格累积的走廊/门口标记
        self.local_space = {'state': 'unknown'}
        self._space_hist = []
        self.space_labels = {}
        # 自由空间几何分区
        self.room_labels = None
        self.room_origin = None
        self.room_regions = {}
        self._room_calls = 0
        self.frontiers = []
        self._frontier_calls = 0

        # 主视图缓存，供 apply_external_instances() 复用同一份反投影数据
        self.current_depth = None
        self._primary_view_pcd = None
        self._primary_view_intrinsic = self.camera_intrinsic

        self._switch_to_floor(0, 0)

    def _apply_hyper(self, config):
        """
        用 config['hyper'] 覆盖下面那些 class 常量。

        为什么保留 class 常量而不是删掉：它们同时是文档和默认值。配置缺项、
        旧 yaml、离线自测脚本直接构造 mapper —— 这些情况都要能跑，否则一个
        漏填的字段会让整批实验在第一个 episode 挂掉。
        """
        room = hyper(config, 'room')
        self.ROOM_GRID_RES = room('grid_res', self.ROOM_GRID_RES)
        self.ROOM_CLOSE_M = room('close_m', self.ROOM_CLOSE_M)
        self.ROOM_SEED_CLEAR = room('seed_clear', self.ROOM_SEED_CLEAR)
        self.ROOM_MIN_AREA = room('min_area', self.ROOM_MIN_AREA)
        self.ROOM_REBUILD_EVERY = room('rebuild_every', self.ROOM_REBUILD_EVERY)
        self.ROOM_MAX_EXPAND_M = room('max_expand_m', self.ROOM_MAX_EXPAND_M)
        self.ROOM_DOOR_CUT = room('door_cut', self.ROOM_DOOR_CUT)
        self.ROOM_MIN_EVIDENCE = room('min_evidence', self.ROOM_MIN_EVIDENCE)
        self.ROOM_MIN_SCORE = room('min_score', self.ROOM_MIN_SCORE)

        space = hyper(config, 'space')
        self.SPACE_RAY_MAX = space('ray_max', self.SPACE_RAY_MAX)
        self.SPACE_SLAB_HALF = space('slab_half', self.SPACE_SLAB_HALF)
        self.SPACE_CORRIDOR_CLEAR = space('corridor_clear', self.SPACE_CORRIDOR_CLEAR)
        self.SPACE_ROOM_CLEAR = space('room_clear', self.SPACE_ROOM_CLEAR)
        self.SPACE_CORRIDOR_ELONG = space('corridor_elong', self.SPACE_CORRIDOR_ELONG)
        self.SPACE_VOTE_WINDOW = space('vote_window', self.SPACE_VOTE_WINDOW)
        self.SPACE_CELL = space('cell', self.SPACE_CELL)

        fr = hyper(config, 'frontier')
        self.FRONTIER_MIN_CELLS = fr('min_cells', self.FRONTIER_MIN_CELLS)
        self.FRONTIER_REBUILD_EVERY = fr('rebuild_every', self.FRONTIER_REBUILD_EVERY)

        av = hyper(config, 'avoid')
        self.AVOID_RADIUS = av('radius', self.AVOID_RADIUS)
        self.AVOID_WEIGHT = av('weight', self.AVOID_WEIGHT)

        for line in (room.describe(), space.describe(), fr.describe(), av.describe()):
            if line:
                print(f"[mapper] {line}")

    # ---- 局部空间形态判据的参数 ----
    SPACE_RAY_MAX = 6.0        # 最远探测距离(米)，探不到障碍就按这个值算"敞开"
    SPACE_SLAB_HALF = 0.25     # 测距平板的半宽(米)，约等于机身宽度
    SPACE_CORRIDOR_CLEAR = 1.2  # 走廊的通行余量上限
    SPACE_ROOM_CLEAR = 1.5      # 房间的通行余量下限
    SPACE_CORRIDOR_ELONG = 2.5  # 走廊的前后/左右长度比下限
    SPACE_VOTE_WINDOW = 5       # 滑窗投票长度
    SPACE_CELL = 0.25           # 地板染色的栅格边长(米)

    # ---- 自由空间分区的参数 ----
    ROOM_GRID_RES = 0.05      # 栅格分辨率(米)
    ROOM_CLOSE_M = 0.35       # 闭运算核半径：填掉家具占地和采样空洞
    ROOM_SEED_CLEAR = 1.0     # 房间种子的通行余量下限(米)
    ROOM_MIN_AREA = 1.5       # 小于这个面积(m²)的区域丢弃
    ROOM_REBUILD_EVERY = 8    # 每隔多少次调用重建一次分区
    ROOM_MAX_EXPAND_M = 3.0   # 每个种子最多向外扩张多远(米)
    ROOM_DOOR_CUT = 0.55      # 通行余量低于此值的格子不允许扩散通过(米)
    ROOM_MIN_EVIDENCE = 2     # 确认房间名至少需要几类互相印证的证据。
                              # 1 会让卧室里一个误检的冰箱把房间判成厨房——实测过
    ROOM_MIN_SCORE = 1.0      # 房间命名的最低得分

    def _build_free_grid(self):
        """
        把 navigable 点云投影成世界系二值栅格。返回 (grid, origin_xy) 或 None。

        用 navigable 而不是 obstacle：未探索区域在 navigable 里是"缺失"，
        而在 obstacle 里和墙无法区分。前者只是信息不足，后者是信息错误——
        这是不做墙检测、直接分割自由空间的核心理由。
        """
        if self.navigable_pcd.is_empty():
            return None
        try:
            pts = self.navigable_pcd.point.positions.cpu().numpy()
        except Exception:
            return None
        if pts.shape[0] < 50:
            return None

        res = self.ROOM_GRID_RES
        xy = pts[:, :2]
        lo = xy.min(axis=0) - 1.0
        hi = xy.max(axis=0) + 1.0
        w = int((hi[0] - lo[0]) / res) + 1
        h = int((hi[1] - lo[1]) / res) + 1
        if w < 8 or h < 8 or w * h > 4_000_000:   # 防止异常点云撑爆内存
            return None

        grid = np.zeros((h, w), dtype=np.uint8)
        ix = np.clip(((xy[:, 0] - lo[0]) / res).astype(int), 0, w - 1)
        iy = np.clip(((xy[:, 1] - lo[1]) / res).astype(int), 0, h - 1)
        grid[iy, ix] = 255

        # 闭运算：桌子底下走不过去，但那仍然是同一个房间。不填的话家具会把
        # 房间切碎，后面的连通域分析全乱。
        k = max(3, int(self.ROOM_CLOSE_M / res) | 1)
        grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        return grid, lo

    def segment_rooms(self, force=False):
        """
        把可通行空间分割成房间区域。

        做法：通行余量图上取 >1m 的连通域当房间种子，再沿自由空间做多源
        测地扩散。门口(余量≈0.42m)和走廊(0.6~1.0m)天然低于房间，自动成为
        分界，所以不需要检测墙、也不需要检测门。

        结果写入 self.room_labels / self.room_origin / self.room_regions。
        """
        self._room_calls = getattr(self, '_room_calls', 0) + 1
        if not force and self.room_labels is not None \
                and self._room_calls % self.ROOM_REBUILD_EVERY != 1:
            return self.room_regions

        built = self._build_free_grid()
        if built is None:
            return self.room_regions
        grid, lo = built
        res = self.ROOM_GRID_RES
        free = grid > 0

        dist = cv2.distanceTransform(grid, cv2.DIST_L2, 5) * res
        seeds = (dist > self.ROOM_SEED_CLEAR).astype(np.uint8)
        n_lbl, seed_lbl = cv2.connectedComponents(seeds)
        if n_lbl <= 1:
            return self.room_regions

        # 多源测地扩散：每轮把标签向外膨胀一格，只填自由且尚未着色的像素。
        # 用测地而不是欧氏 Voronoi，是因为后者会穿墙——隔壁房间的种子如果
        # 直线距离更近，就会把这边的地板抢过去。
        #
        # 两道闸门，缺一不可（实测教训：只有测地约束时，一个厨房种子把整层楼
        # 连走廊带隔壁全吃掉了，地图上一个 [kitchen] 多边形里同时有 desk、
        # 四个 workbench、mirror、carpet）：
        #
        #   1. 门口截断：余量低于 ROOM_DOOR_CUT 的格子不许扩散通过。
        #      我原本以为"门口余量低会自然成为分水岭"，这是错的——分水岭要求
        #      两侧都有种子同时涨水，而走廊(余量 0.6~1.0m)够不上种子门槛，
        #      于是房间的标签会一路穿过门口把走廊连同隔壁一起吞并。
        #   2. 距离上限：每个种子最多向外长 ROOM_MAX_EXPAND_M。房间是有限大的，
        #      一个种子不该主张三米开外的地盘。
        #
        # 两道闸门之外的格子保持 label 0 —— 那是"不属于任何已识别房间"，
        # 是个诚实的答案，交给局部走廊判据去回答"这里是什么"。
        passable = free & (dist >= self.ROOM_DOOR_CUT)
        labels = seed_lbl.astype(np.int32)
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(max(1, int(self.ROOM_MAX_EXPAND_M / res))):
            grown = cv2.dilate(labels.astype(np.uint16), kernel).astype(np.int32)
            fill = (labels == 0) & passable & (grown > 0)
            if not np.any(fill):
                break
            labels[fill] = grown[fill]
        labels[~free] = 0

        regions = {}
        min_cells = int(self.ROOM_MIN_AREA / (res * res))
        for k in range(1, n_lbl):
            mask = (labels == k)
            cnt = int(mask.sum())
            if cnt < min_cells:
                labels[mask] = 0
                continue
            ys, xs = np.nonzero(mask)
            cx = lo[0] + (xs.mean() + 0.5) * res
            cy = lo[1] + (ys.mean() + 0.5) * res
            area = cnt * res * res
            # 等效半径：给还在用 center+radius 的下游代码(停止判据等)用
            radius = float(np.sqrt(area / np.pi))

            contour = None
            try:
                cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
                if cs:
                    c = max(cs, key=cv2.contourArea)
                    c = cv2.approxPolyDP(c, 0.15 / res, True).reshape(-1, 2)
                    contour = np.stack([lo[0] + (c[:, 0] + 0.5) * res,
                                        lo[1] + (c[:, 1] + 0.5) * res], axis=1)
            except Exception:
                contour = None

            regions[k] = {'label': k, 'center': np.array([cx, cy]), 'area': area,
                          'radius': radius, 'contour': contour}

        self.room_labels = labels
        self.room_origin = lo
        self.room_regions = regions
        return regions

    FRONTIER_MIN_CELLS = 12     # 小于这么多格的碎片不算一个前沿
    FRONTIER_REBUILD_EVERY = 4

    def compute_frontiers(self, force=False):
        """
        算出"已知自由空间"和"未观测区域"的边界，并聚成若干离散前沿。

        注意：项目里原本就有 frontier_pcd 这个字段，渲染层也画黄点、prompt 图例
        还写着 "Yellow Boundary: the frontier between explored and unexplored areas"，
        但计算那一整段在 mapper 里是**注释掉的**——frontier_pcd 永远是空点云。
        也就是说 VLM 一直被告知有这么个东西，实际从来没见过。这里补上。

        复用 segment_rooms 的栅格：
            free     = 可通行(已观测)
            obstacle = 障碍(已观测)
            unknown  = 两者都不是 -> 没看过
            frontier = free 且紧邻 unknown 且不是障碍
        写入 self.frontiers（离散候选）和 self.frontier_pcd（兼容既有渲染）。
        """
        self._frontier_calls = getattr(self, '_frontier_calls', 0) + 1
        if not force and self.frontiers and self._frontier_calls % self.FRONTIER_REBUILD_EVERY != 1:
            return self.frontiers

        built = self._build_free_grid()
        if built is None:
            return self.frontiers
        grid, lo = built
        res = self.ROOM_GRID_RES
        h, w = grid.shape
        free = grid > 0

        # 障碍栅格：不打掉墙的话，已探索区域的整圈外沿都会被当成前沿
        obst = np.zeros((h, w), dtype=np.uint8)
        if not self.obstacle_pcd.is_empty():
            try:
                op = self.obstacle_pcd.point.positions.cpu().numpy()
                z0 = float(self.current_position[2]) - 1.2
                op = op[(op[:, 2] > z0 + 0.15) & (op[:, 2] < z0 + 1.8)]
                ix = np.clip(((op[:, 0] - lo[0]) / res).astype(int), 0, w - 1)
                iy = np.clip(((op[:, 1] - lo[1]) / res).astype(int), 0, h - 1)
                obst[iy, ix] = 255
                # 注意核尺寸是直径：要 0.25m 的膨胀半径就得 2*r/res+1
                k = max(3, int(2 * 0.25 / res) | 1)
                obst = cv2.dilate(obst, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            except Exception:
                pass

        unknown = ((~free) & (obst == 0)).astype(np.uint8)
        # 开运算去掉"细缝型未知"。
        #
        # 地板点云和墙面点云是两个不同表面采样出来的，中间必然留一条几十厘米
        # 的空隙，那条缝既不是 free 也不是 obstacle，会被当成未知区域——结果
        # 一个四面封闭的房间，整圈外沿都被判成前沿（实测 368 个格子，质心正好
        # 在房间中心，因为那是个环）。真正的未探索区域是以米计的，开运算能
        # 干净地把两者分开，比去调墙的膨胀半径稳得多。
        ko = max(3, int(2 * 0.4 / res) | 1)
        unknown = cv2.morphologyEx(unknown, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko)))
        unk_d = cv2.dilate(unknown, np.ones((3, 3), np.uint8))
        front = (free & (unk_d > 0) & (obst == 0)).astype(np.uint8)
        if front.sum() == 0:
            self.frontiers = []
            return self.frontiers

        n, lbl = cv2.connectedComponents(front)
        agent_xy = np.asarray(self.current_position[:2], dtype=float)
        out, pts = [], []
        for k in range(1, n):
            m = (lbl == k)
            cnt = int(m.sum())
            if cnt < self.FRONTIER_MIN_CELLS:
                continue
            ys, xs = np.nonzero(m)
            cx = lo[0] + (xs.mean() + 0.5) * res
            cy = lo[1] + (ys.mean() + 0.5) * res
            # 质心不一定落在前沿上（弧形前沿的质心在弧的凹侧，可能是墙里）。
            # 前沿要作为可导航目标，必须给一个**保证在自由空间上**的代表点，
            # 取离质心最近的那个前沿格。
            k_near = int(np.argmin((xs - xs.mean()) ** 2 + (ys - ys.mean()) ** 2))
            px = lo[0] + (xs[k_near] + 0.5) * res
            py = lo[1] + (ys[k_near] + 0.5) * res
            out.append({
                'center': np.array([cx, cy]),
                'point': np.array([px, py]),      # 可导航的代表点
                'n_cells': cnt,
                'width_m': cnt * res,          # 前沿的展开长度，越宽越可能通向大空间
                'distance': float(np.linalg.norm(np.array([cx, cy]) - agent_xy)),
            })
            step = max(1, cnt // 60)           # 抽稀，只为渲染用
            for i in range(0, len(xs), step):
                pts.append([lo[0] + (xs[i] + 0.5) * res, lo[1] + (ys[i] + 0.5) * res,
                            float(self.current_position[2]) - 1.2])

        out.sort(key=lambda f: f['distance'])
        self.frontiers = out
        try:
            if pts:
                arr = np.asarray(pts, dtype=np.float64)
                self.frontier_pcd = gpu_pointcloud_from_array(
                    arr, np.tile(np.array([[0.0, 1.0, 1.0]]), (arr.shape[0], 1)), self.pcd_device)
            else:
                self.frontier_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        except Exception:
            pass
        return self.frontiers

    def _region_at(self, xy):
        """世界坐标落在哪个区域标签上；不在任何区域返回 0。"""
        if self.room_labels is None:
            return 0
        res = self.ROOM_GRID_RES
        ix = int((float(xy[0]) - self.room_origin[0]) / res)
        iy = int((float(xy[1]) - self.room_origin[1]) / res)
        h, w = self.room_labels.shape
        if not (0 <= ix < w and 0 <= iy < h):
            return 0
        return int(self.room_labels[iy, ix])

    def area_hypotheses(self, min_weight=0.25):
        """
        房间假设，**包括还不够格被确认的弱假设**。

        为什么需要这个：`infer_area_regions` 有个 min_score=1.0 的硬门槛，
        证据不够就整个丢掉。用来"宣布你在哪个房间"这是对的，但用来给探索
        指方向就致命了——实跑里前沿引导的空间项 139 次全部落空、0 次命中，
        因为它要求目标房间**已经被确认**。而一旦确认了，你也就不需要被引导
        过去了。先验只在不再需要它的时候才可用。

        弱假设正好补这个洞：透过门缝瞥见一个水槽，不足以断言"那是厨房"，
        但完全足够作为"厨房可能在那个方向"的线索。返回：
            {'area_type','center','score','confirmed','evidence'}
        confirmed=True 的来自 inferred_areas（强证据），False 的来自单个
        特征物体（弱证据，score 就是它的 proxy 权重）。
        """
        out = []
        for a in (getattr(self, 'inferred_areas', []) or []):
            out.append({'area_type': a['area_type'],
                        'center': np.asarray(a['center'][:2], dtype=float),
                        'score': float(a.get('score', 1.0)),
                        'confirmed': True,
                        'evidence': [c for c, _ in a.get('evidence', [])]})

        for e in (self.object_entities or []):
            try:
                if e['pcd'].point.positions.shape[0] == 0:
                    continue
                cls = normalize_category_name(e['class_name'])
                c = np.asarray(e['center'][:2], dtype=float)
            except Exception:
                continue
            for room_type, proxies in AREA_PROXY_OBJECTS.items():
                w = None
                for p, pw in proxies.items():
                    if normalize_category_name(p) == cls:
                        w = pw
                        break
                if w is None or w < min_weight:
                    continue
                # 已经落在同类型的确认房间里，就不重复出弱假设
                if any(h['confirmed'] and h['area_type'] == room_type
                       and float(np.linalg.norm(c - h['center'])) <= 4.0 for h in out):
                    continue
                out.append({'area_type': room_type, 'center': c, 'score': float(w),
                            'confirmed': False, 'evidence': [cls]})
        return out

    def current_room(self, min_score=None):  # 与 infer_area_regions 同阈值
        """
        agent 此刻所在的房间（只返回一个，或 None）。

        为什么不把所有推断出的房间都画在地图上：实跑里同时会有五六个房间被
        推断出来，4m 半径的圈叠在一起把地图糊成一片，标签互相压字，反而盖住
        了障碍结构。VLM 真正要回答的是"我在哪个房间"，一个答案就够。

        和 infer_area_regions 用同一个 min_score——精度问题已经在权重定标里
        解决了（单个 desk 根本不会产生 office，不需要在这里再设第二道门槛）。
        这里真正额外的约束是**包含性**：agent 必须落在房间圆内。
        选分数最高的；同分取半径小的（定位更确切）。
        """
        min_score = self.ROOM_MIN_SCORE if min_score is None else min_score
        agent_xy = np.asarray(self.current_position[:2], dtype=float)
        # 有几何分区就用精确的包含判定：agent 落在哪个区域标签上
        lab = self._region_at(agent_xy)
        if lab > 0:
            for a in (getattr(self, 'inferred_areas', []) or []):
                if a.get('label') == lab and a.get('score', 0.0) >= min_score:
                    return a
            return None

        # 分区还没建好（episode 最初几步）时退化成圆形包含判定
        best = None
        for a in (getattr(self, 'inferred_areas', []) or []):
            if a.get('score', 0.0) < min_score:
                continue
            d = float(np.linalg.norm(np.asarray(a['center'][:2], dtype=float) - agent_xy))
            if d > float(a.get('radius', 0.0)):
                continue
            key = (a.get('score', 0.0), -float(a.get('radius', 0.0)))
            if best is None or key > best[0]:
                best = (key, a)
        return best[1] if best else None

    def classify_local_space(self):
        """
        判断 agent 此刻脚下是走廊还是房间。

        为什么需要这个：走廊、room、corridor 这类地标没有任何特征家具，
        用物体反推永远接不了地——实测一批 episode 里 landmark 接地失败的原因
        有 201 次是 area_no_proxy，几乎全是 hallway/room。但它们的定义本来就
        不是"里面有什么"，而是"什么形状"：走廊窄且长，房间宽且方。

        只看 agent 周围几米，所以不受地图不完整影响（走过的地方早就扫过了），
        也不需要把整层楼分割成房间。

        返回 dict，同时写入 self.local_space；走廊/门口的判定会按 SPACE_CELL
        栅格累积进 self.space_labels 供地图染色。
        """
        out = {'state': 'unknown', 'raw_state': 'unknown', 'clearance': None,
               'along': None, 'across': None, 'elongation': None}
        self.local_space = out
        if not hasattr(self, 'space_labels'):
            self.space_labels = {}

        if self.obstacle_pcd.is_empty():
            return out
        try:
            obs = self.obstacle_pcd.point.positions.cpu().numpy()
        except Exception:
            return out
        if obs.shape[0] == 0:
            return out

        agent_xy = np.asarray(self.current_position[:2], dtype=float)
        floor_z = float(self.current_position[2]) - 1.2

        # 只取齐腰高度带的障碍点：地面残点和天花板都会污染通行宽度
        zmask = (obs[:, 2] > floor_z + 0.15) & (obs[:, 2] < floor_z + 1.6)
        p = obs[zmask][:, :2] - agent_xy
        if p.shape[0] == 0:
            return out
        d = np.linalg.norm(p, axis=1)
        near = d < self.SPACE_RAY_MAX
        p, d = p[near], d[near]
        if p.shape[0] == 0:
            return out

        rot = np.asarray(self.current_rotation, dtype=float)
        fwd = -rot[:, 2][:2]
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([0.0, -1.0])
        right = rot[:, 0][:2]
        n = np.linalg.norm(right)
        right = right / n if n > 1e-6 else np.array([1.0, 0.0])

        v = p @ fwd      # 纵向（前正后负）
        u = p @ right    # 横向（右正左负）

        # 用"平板"而不是角度扇区测距。
        #
        # 扇区在窄走廊里会立刻打到侧墙：±25°的锥形在 0.8/tan(25°)≈1.7m 处就
        # 撞上 0.8m 外的侧墙，于是一条 20m 长的走廊被测成"前方只有1.9m"，
        # 细长比算出来 2.4，判不成走廊。平板只问"正前方这条机身宽的带子里
        # 最近的障碍有多远"，语义正好是"我能直着走多远"。
        half = self.SPACE_SLAB_HALF

        def slab(axis_val, lateral_val):
            m = (axis_val > 0) & (np.abs(lateral_val) < half)
            return float(axis_val[m].min()) if np.any(m) else float(self.SPACE_RAY_MAX)

        f_d, b_d = slab(v, u), slab(-v, u)
        r_d, l_d = slab(u, v), slab(-u, v)

        clearance = float(d.min())
        along = f_d + b_d
        across = r_d + l_d
        elong = along / max(across, 1e-6)

        # 只区分 corridor / room。原本还想单独判 doorway，但门口和窄走廊在
        # 单帧几何上几乎无法可靠区分（门口 clearance≈0.4、走廊≈0.6~1.0，重叠严重），
        # 而且对上层没有价值——"我在不在走廊里"这个问题上两者是同一回事，
        # doorway 本身又已经是检测器词表里的一个类别。少一个阈值少一处调参。
        if clearance < self.SPACE_CORRIDOR_CLEAR and elong > self.SPACE_CORRIDOR_ELONG:
            raw = 'corridor'
        elif clearance >= self.SPACE_ROOM_CLEAR or elong < 1.5:
            raw = 'room'
        else:
            raw = 'unknown'

        # 滑窗多数投票：agent 偶尔卡在沙发和茶几之间时，单帧会误判成走廊，
        # 而染色是累积的、错了会永久留在地图上，所以宁可晚两三步再落色。
        if not hasattr(self, '_space_hist'):
            self._space_hist = []
        self._space_hist.append(raw)
        if len(self._space_hist) > self.SPACE_VOTE_WINDOW:
            self._space_hist.pop(0)
        votes = {}
        for s in self._space_hist:
            votes[s] = votes.get(s, 0) + 1
        state = max(votes.items(), key=lambda kv: kv[1])[0]

        out.update({'state': state, 'raw_state': raw, 'clearance': clearance,
                    'along': along, 'across': across, 'elongation': elong,
                    'rays': {'f': f_d, 'b': b_d, 'l': l_d, 'r': r_d}})
        self.local_space = out

        # 只对走廊/门口落色；房间保持地板原色，未走过的地方什么都不写。
        # 染色半径取通行余量（封顶1.2m），走廊会被自身宽度填满。
        if state == 'corridor':
            if not hasattr(self, 'space_labels'):
                self.space_labels = {}
            r = min(max(clearance, 0.3), 1.2)
            steps = int(r / self.SPACE_CELL)
            for ix in range(-steps, steps + 1):
                for iy in range(-steps, steps + 1):
                    if ix * ix + iy * iy > steps * steps:
                        continue
                    cx = agent_xy[0] + ix * self.SPACE_CELL
                    cy = agent_xy[1] + iy * self.SPACE_CELL
                    key = (int(round(cx / self.SPACE_CELL)), int(round(cy / self.SPACE_CELL)))
                    self.space_labels[key] = state
        return out

    def fuse_instances(self, instances):
        """
        把一批检测实例反投影成物体点云并合并进 object_entities。

        从 update_multiview() 里抽出来的公共路径，这样本地检测器(YOLOE)和
        外部检测器(决策VLM顺带输出的框)可以共用同一套反投影+关联逻辑，
        不会出现两条通路把同一个物体放到地图上两个不同位置的情况。

        instances: [{'class_id','class_name','mask', 可选 'relevance'/'confidence'}, ...]
                   mask 是与 current_rgb 同尺寸的 bool 或 0/255 数组。
        """
        if not instances:
            return self.object_entities
        if getattr(self, '_primary_view_pcd', None) is None:
            # 还没跑过一次 update_multiview，没有可用的主视图点云
            return self.object_entities

        classes = [inst['class_id'] for inst in instances]
        class_names = [inst['class_name'] for inst in instances]
        masks = [inst['mask'] for inst in instances]
        # 本地YOLOE通路历史上一直传1.0(那个封装不返回置信度)，这里保持兼容：
        # 只有检测结果确实带了分数才用真实值。
        confidences = [float(inst.get('relevance', inst.get('confidence', 1.0)))
                       for inst in instances]

        saved_pcd = self.current_pcd
        self.current_pcd = self._primary_view_pcd
        try:
            current_entities = self.get_object_entities(
                self.current_depth,
                classes,
                class_names,
                masks,
                confidences,
                camera_intrinsic=self._primary_view_intrinsic,
            )
        finally:
            self.current_pcd = saved_pcd

        self.object_entities = self.associate_object_entities(self.object_entities, current_entities)
        return self.object_entities

    def apply_external_instances(self, detections, image_shape=None, coord_scale=1000.0):
        """
        接收决策VLM顺带输出的检测框，转成 instances 后并进地图。

        这样检测就不用再单独发一次VLM请求——决策调用本来就已经把 front view
        发过去了，让它在同一次回复里额外吐一组框，边际成本只有几十个output token。

        detections: [{'c': class_name, 'b': [x1,y1,x2,y2]}, ...]
                    坐标是 0~coord_scale 的归一化整数(和 instance_segmentation_qwen
                    里的约定一致)，(0,0)为左上角。
        返回实际并入的实例数。

        注意调用时机：必须在 update_map() 之后、执行动作之前调用，此时
        mapper 的 current_depth / current_position / current_rotation 才和
        送给VLM的那张 front view 对应同一个位姿。
        """
        if not detections:
            return 0
        if getattr(self, 'current_depth', None) is None:
            return 0

        h, w = (image_shape[:2] if image_shape is not None else self.current_depth.shape[:2])

        instances = []
        for i, det in enumerate(detections):
            try:
                raw_name = det.get('c') if isinstance(det, dict) else None
                box = det.get('b') if isinstance(det, dict) else None
                if not raw_name or not box or len(box) != 4:
                    continue

                # 类别名对齐到封闭词表。对不上就丢弃——与其塞一个检测器/指令
                # 两边都不认识的类别名进地图，不如不要，否则 landmark 匹配时
                # 只会制造噪音。
                canonical, _ = resolve_category(str(raw_name))
                if canonical is None:
                    continue

                x1, y1, x2, y2 = [float(v) for v in box]
                x1 = int(np.clip(x1 / coord_scale * w, 0, w))
                x2 = int(np.clip(x2 / coord_scale * w, 0, w))
                y1 = int(np.clip(y1 / coord_scale * h, 0, h))
                y2 = int(np.clip(y2 / coord_scale * h, 0, h))
                if x1 >= x2 or y1 >= y2:
                    continue

                mask = np.zeros((h, w), dtype=np.uint8)
                mask[y1:y2, x1:x2] = 255

                instances.append({
                    'class_id': i,
                    'class_name': canonical,
                    'mask': mask,
                    # VLM给的是矩形框而不是精确mask，反投影出来的点云必然混进
                    # 一些背景点，所以置信度打个折，让本地检测器(精确mask)的
                    # 结果在 associate 时优先。
                    'confidence': float(det.get('s', 0.6)) * 0.8,
                })
            except Exception:
                continue

        if not instances:
            return 0

        self.fuse_instances(instances)
        return len(instances)

    # avoid 区域的默认作用半径(米)和最大惩罚(米，与A*的欧氏边权同量纲)。
    # 物体类地标用固定半径；区域类地标(整个房间要避开)用它自己的 radius。
    AVOID_RADIUS = 1.5
    AVOID_WEIGHT = 6.0

    def get_avoid_zones(self):
        """当前子任务里所有 avoid_marker 的 (中心xy, 半径)。"""
        zones = []
        for g in getattr(self, 'grounded_landmarks', []) or []:
            if g.get('role') != 'avoid_marker':
                continue
            try:
                c = np.asarray(g['center'][:2], dtype=float)
            except Exception:
                continue
            r = float(g.get('radius', self.AVOID_RADIUS)) if g.get('kind') == 'area' \
                else self.AVOID_RADIUS
            zones.append((c, r))
        return zones

    def _avoid_penalty(self, point):
        """
        点落在 avoid 区域内时的额外代价，随距中心线性衰减到0。

        量纲和A*边权(米)一致，AVOID_WEIGHT=6.0 意味着"穿过禁区中心"约等于
        多走6米——足以让规划器优先选绕路，但绕路超过6米时仍然会直穿，
        这正是软惩罚想要的行为。
        """
        zones = self._avoid_zones_cache
        if not zones:
            return 0.0
        p = np.asarray(point[:2], dtype=float)
        pen = 0.0
        for c, r in zones:
            d = float(np.linalg.norm(p - c))
            if d < r:
                pen += self.AVOID_WEIGHT * (1.0 - d / r)
        return pen

    def infer_area_regions(self, min_score=None):
        """
        给几何分区出来的每个区域起个名字。

        两步：segment_rooms() 先按自由空间的形状把地图切成区域（不检测墙、
        不检测门，靠通行余量的分水岭），然后这里用落在每个区域里的物体给
        它命名。

        为什么不是按距离聚类物体：那样"哪些物体共处一室"只是距离猜测。实测
        一张 bed 和 4m 外的 desk 会被切成两簇，推出 bedroom + office 两个
        重叠的圈；一个卫生间场景甚至同时冒出 garage、两个 office、两个
        stairwell。有了区域之后这变成几何事实——同一个连通域里的 bed 和
        desk 就是同一个卧室里的床和书桌。而且一个区域只判一种类型，
        重叠圈从根上没有了。

        min_score=1.0：只有"房间定义级"的物体(bed/toilet/bathtub/stove/
        refrigerator/washing_machine/stairs，权重1.0)能单独成立，其余需要
        多个证据叠加。

        写入 self.inferred_areas，每项：
            {'area_type','center','radius','area','label','contour','score','evidence'}

        局限：开放式布局(厨房和客厅连通、中间无墙无门)在几何上就是一个区域，
        分不开——这类场景任何几何方法都无解。hallway/room/closet 的
        AREA_PROXY_OBJECTS 是空的，走廊靠 classify_local_space() 那条通路。
        """
        min_score = self.ROOM_MIN_SCORE if min_score is None else min_score
        self.inferred_areas = []
        if not self.object_entities:
            return self.inferred_areas

        regions = self.segment_rooms()
        if not regions:
            return self.inferred_areas

        # 把每个物体归到它所在的区域。这是几何分区带来的关键变化：
        # "哪些物体共处一室"从**距离猜测**变成**几何事实**。
        # 之前按 3m 链式聚类，一张 bed 和 4m 外的 desk 会被切成两簇 ->
        # bedroom + office；现在它们落在同一个连通域里 -> 一个卧室，里面有书桌。
        per_region = {}
        for e in self.object_entities:
            try:
                if e['pcd'].point.positions.shape[0] == 0:
                    continue
                lab = self._region_at(np.asarray(e['center'][:2], dtype=float))
                if lab <= 0 or lab not in regions:
                    continue
                per_region.setdefault(lab, {})
                cls = normalize_category_name(e['class_name'])
                # 同类重复出现不叠加（一个厨房里有3把椅子不代表它更像厨房）
                per_region[lab][cls] = True
            except Exception:
                continue

        for lab, classes in per_region.items():
            reg = regions[lab]
            # 一个区域只判一种房间类型——取得分最高的那个。
            # 这本身就消灭了"同一片地方既是 bedroom 又是 office"的重叠圈。
            best_type, best_score, best_ev = None, 0.0, []
            for area_type, proxies in AREA_PROXY_OBJECTS.items():
                if not proxies:
                    continue
                ev = [(cls_name, w) for cls_name, w in proxies.items()
                      if normalize_category_name(cls_name) in classes]
                score = sum(w for _, w in ev)
                if score > best_score:
                    best_type, best_score, best_ev = area_type, score, ev

            # 确认一个房间需要**至少两类互相印证的证据**。
            #
            # 之前只看总分：refrigerator / bed / toilet / stove 这些"房间定义级"
            # 物体权重 1.0，单个就能确认。那是 conf=0.6 时的设定；后来为提升
            # 覆盖把检测阈值降到 0.45，误检显著增加，而单物体确认对误检零容错。
            # 实测 ep34：agent 明明在卧室里（VLM 描述了床、地毯、衣帽间），
            # 一个把白衣柜误检成 refrigerator 的框就确认出一个"厨房"，接着
            # current_room() 谎报、拦截给出 "you are in the kitchen, not the
            # bedroom" 的错误理由。
            #
            # 一个检测可能是错的，两个不同类别同时指向同一个房间则要可靠得多。
            # 单个强证据不会浪费——它仍然通过 area_hypotheses() 作为**弱假设**
            # 参与探索引导，只是不再有资格宣称"你在这个房间里"。
            if (best_type is None or best_score < min_score
                    or len(best_ev) < self.ROOM_MIN_EVIDENCE):
                continue

            self.inferred_areas.append({
                'area_type': best_type,
                'center': reg['center'],
                'radius': reg['radius'],
                'area': reg['area'],
                'label': lab,
                'contour': reg.get('contour'),
                'score': round(float(best_score), 2),
                'evidence': sorted(best_ev, key=lambda kv: -kv[1]),
            })

        # 房间列表不再画到地图上（太占地方且误报显眼），改成打一行日志。
        # 排查"为什么把卫生间认成车库"的时候看这行，带上各自的证据物体。
        if self.inferred_areas:
            desc = "  ".join(
                f"{a['area_type']}({a['score']}:{'+'.join(c for c, _ in a['evidence'])})"
                for a in sorted(self.inferred_areas, key=lambda x: -x['score']))
            if desc != getattr(self, '_last_rooms_desc', None):
                print(f"[rooms] {desc}")
                self._last_rooms_desc = desc

        return self.inferred_areas

    def ground_landmarks(self, landmarks, subtask_key=None):
        """
        把当前子任务的landmark接地到已建图的object_entities上。

        landmarks: 来自 SpatioTemporalInstructionDecomposer 的
            [{'name','role','relative_position','class_name',...}, ...]
        subtask_key: 仅用于打日志/在图上标注属于哪个子任务。

        为什么每步全量重算而不做增量：object_entities一般只有几十个，landmark
        更是个位数，一次全匹配的开销远小于一次点云voxel_down_sample，没必要为
        增量更新引入"上次匹配到的entity这次被associate合并掉了"这类状态同步问题。

        写入 self.grounded_landmarks，每项：
            {'name','role','relative_position','class_name','match_how',
             'center': np.ndarray(3,), 'confidence': float,
             'distance': float, 'n_candidates': int}
        未接地的landmark写入 self.unmatched_landmarks（只含文本字段），
        渲染层用它在地图角落列"待找清单"。
        """
        self.grounded_landmarks = []
        self.unmatched_landmarks = []

        # 房间推断放在早退之前：即使当前子任务没有任何landmark（比如"Turn left"），
        # 推断出的房间也要照常更新，渲染层会把它们作为背景语义层画出来。
        self.infer_area_regions()

        if not landmarks:
            return self.grounded_landmarks

        # 检测器输出的class_name带下划线，指令侧是自然语言，两边都归一化后再比
        entities_by_class = {}
        for e in self.object_entities:
            if e['pcd'].point.positions.shape[0] == 0:
                continue
            key = normalize_category_name(e['class_name'])
            entities_by_class.setdefault(key, []).append(e)

        agent_xy = np.asarray(self.current_position[:2], dtype=float)
        # current_rotation 的第三列取负得到前向向量，和 _get_action_for_next_waypoint
        # 里的 forward_vec_3d = -agent_rot_matrix[:, 2] 保持一致
        rot = np.asarray(self.current_rotation, dtype=float)
        fwd = -rot[:, 2][:2]
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([0.0, -1.0])
        # 右向量直接取旋转矩阵第一列，而不是把fwd转90度算出来。
        #
        # 原因：habitat_rotation() 做的是 T @ R，其中 T=[[1,0,0],[0,0,1],[0,1,0]]
        # 的行列式是 -1，而且只左乘没做相似变换，所以 current_rotation 的
        # 行列式恒为 -1 —— 这意味着整个world系是左手系。在左手系里
        # "叉积 fwd × v > 0 即目标在左侧" 这条右手系直觉是反的，照搬会把
        # 左右消歧整个搞反。直接读第一列(相机右向量)则不依赖手性假设，
        # 无论上游怎么改坐标约定都不会错。
        right = rot[:, 0][:2]
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0])

        areas_by_type = {}
        for a in self.inferred_areas:
            areas_by_type.setdefault(a['area_type'], []).append(a)

        for lm in landmarks:
            name = lm.get('name', '')

            # 先判断是不是区域类地标(bedroom/hallway/...)。这类东西不可能出现在
            # 目标检测器输出里，走的是"特征物体反推"这条独立通路，不能和物体
            # 走同一套 resolve_category，否则会被误判成 not_in_vocab。
            area_type, area_how = resolve_area(name)
            if area_type is not None:
                cands = areas_by_type.get(area_type, [])

                # 走廊类地标没有特征家具，物体反推这条路走不通，改用局部空间
                # 形态：agent 此刻站在一条窄而长的通道里，就认为"已经在走廊里"。
                # 距离记为 0——它不是一个要走过去的点，而是一个已经身处其中的状态。
                if not cands and area_type in ('hallway', 'stairwell') \
                        and getattr(self, 'local_space', {}).get('state') == 'corridor':
                    ls = self.local_space
                    self.grounded_landmarks.append({
                        'name': name, 'role': lm.get('role', 'unknown'),
                        'relative_position': lm.get('relative_position', 'unknown'),
                        'kind': 'space', 'class_name': area_type,
                        'match_how': f"local_{ls['state']}",
                        'center': np.array([agent_xy[0], agent_xy[1],
                                            float(self.current_position[2]) - 1.2]),
                        'radius': float(min(max(ls.get('clearance') or 1.0, 0.5), 1.5)),
                        'confidence': 0.7, 'distance': 0.0, 'n_candidates': 1,
                        'clearance': ls.get('clearance'), 'elongation': ls.get('elongation'),
                        'subtask_key': subtask_key,
                    })
                    continue

                if not cands:
                    has_proxy = bool(AREA_PROXY_OBJECTS.get(area_type))
                    self.unmatched_landmarks.append({
                        'name': name,
                        'role': lm.get('role', 'unknown'),
                        'kind': 'area',
                        'area_type': area_type,
                        # 区分两种失败：走廊这类压根没有特征物体可推(需要几何分区)，
                        # 和卧室这类有判据但还没看到床(继续探索就可能接上)。
                        'reason': 'area_pending' if has_proxy else 'area_no_proxy',
                    })
                    continue

                best = max(cands, key=lambda a: (a['score'], -float(
                    np.linalg.norm(a['center'] - agent_xy))))
                self.grounded_landmarks.append({
                    'name': name,
                    'role': lm.get('role', 'unknown'),
                    'relative_position': lm.get('relative_position', 'unknown'),
                    'kind': 'area',
                    'class_name': area_type,
                    'match_how': f"area_{area_how}",
                    'center': np.array([best['center'][0], best['center'][1],
                                        float(self.current_position[2]) - 1.2]),
                    'radius': best['radius'],
                    'contour': best.get('contour'),   # 渲染层用它画真实形状而不是圆
                    'confidence': min(1.0, best['score']),
                    'distance': float(np.linalg.norm(best['center'] - agent_xy)),
                    'n_candidates': len(cands),
                    'evidence': best['evidence'],
                    'subtask_key': subtask_key,
                })
                continue

            # decomposer可能已经给出了class_name(词表可用时)，但它可能是未对齐的
            # 原始文本，所以这里统一再过一遍resolve_category，以自身name优先。
            canonical, how = resolve_category(name)
            if canonical is None and lm.get('class_name'):
                canonical, how = resolve_category(lm['class_name'])
                if canonical is not None:
                    how = f"via_llm_{how}"

            if canonical is None:
                # 房间名(kitchen/hallway)或词表里没有的物体(window)会走到这里。
                # 不是bug，如实记录原因即可。
                self.unmatched_landmarks.append({
                    'name': name,
                    'role': lm.get('role', 'unknown'),
                    'reason': 'not_in_vocab',
                })
                continue

            hits = entities_by_class.get(normalize_category_name(canonical), [])
            if not hits:
                self.unmatched_landmarks.append({
                    'name': name,
                    'role': lm.get('role', 'unknown'),
                    'class_name': canonical,
                    'reason': 'not_yet_detected',
                })
                continue

            n_candidates = len(hits)

            # 同类多实例时用指令里的方位词消歧("the window on your left")。
            # 把候选点投影到右向量上：投影>0在右侧，<0在左侧。过滤后如果一个都
            # 不剩(方位词和实际不符，很常见)，就退回全部候选——宁可接地到错误的
            # 一侧，也好过完全不接地导致地图上什么都不显示。
            rel_pos = lm.get('relative_position', 'unknown')
            if rel_pos in ('left', 'right') and n_candidates > 1:
                side_filtered = []
                for e in hits:
                    v = np.asarray(e['center'][:2], dtype=float) - agent_xy
                    side = float(np.dot(v, right))  # >0 右, <0 左
                    if (side > 0) == (rel_pos == 'right'):
                        side_filtered.append(e)
                if side_filtered:
                    hits = side_filtered

            # 在剩余候选里挑：距离近的优先，置信度作为次要项。
            # 用 dist - 2.0*conf 这种线性打分而不是纯按置信度，是因为VLN里
            # "指令说的那个X"几乎总是离你最近的那个X，距离比检测置信度更有信息量。
            def _score(e):
                d = float(np.linalg.norm(np.asarray(e['center'][:2], dtype=float) - agent_xy))
                return d - 2.0 * float(e.get('confidence', 0.0))

            best = min(hits, key=_score)
            dist = float(np.linalg.norm(np.asarray(best['center'][:2], dtype=float) - agent_xy))

            self.grounded_landmarks.append({
                'name': name,
                'role': lm.get('role', 'unknown'),
                'relative_position': rel_pos,
                'kind': 'object',
                'class_name': canonical,
                'match_how': how,
                'center': np.asarray(best['center'], dtype=float),
                'confidence': float(best.get('confidence', 0.0)),
                'distance': dist,
                'n_candidates': n_candidates,
                'subtask_key': subtask_key,
            })

        if self.grounded_landmarks or self.unmatched_landmarks:
            g = ", ".join(
                f"{d['name']}->{d['class_name']}[{d.get('kind','object')}]"
                f"({d['match_how']},{d['distance']:.1f}m)"
                for d in self.grounded_landmarks) or "-"
            u = ", ".join(f"{d['name']}({d['reason']})" for d in self.unmatched_landmarks) or "-"
            print(f"[grounding] {subtask_key or ''} matched: {g} | pending: {u}")

        return self.grounded_landmarks

    def get_candidate_waypoints(self,
                                waypoint_grid_resolution=1.0,
                                min_distance=0.0,
                                max_distance=5.0,
                                cylinder_radius=0.2,
                                cylinder_height=1.0,
                                use_distance_filter=False):
        """
        Generate candidate waypoints using cylinder-occupancy rules:
        1. voxel downsample to candidates
        2. for each candidate, build a cylinder (base center at point, radius=cylinder_radius, height=cylinder_height)
        3. if no obstacle in cylinder => valid
           if obstacle exists but stairs present in cylinder => still valid
           otherwise discard
        4. optionally apply 2D distance filter against obstacles
        Returns: (N,3) numpy.ndarray
        """
        if self.navigable_pcd.is_empty():
            self.waypoints = np.array([])
            return np.array([])

        if self.obstacle_pcd.is_empty():
            try:
                cand = self.navigable_pcd.voxel_down_sample(waypoint_grid_resolution)
                self.waypoints = cand.point.positions.cpu().numpy()
                return self.waypoints
            except:
                self.waypoints = np.array([])
                return np.array([])

        try:
            candidate_waypoints_pcd = self.navigable_pcd.voxel_down_sample(waypoint_grid_resolution)
            if candidate_waypoints_pcd.is_empty():
                self.waypoints = np.array([])
                return np.array([])

            cand_pts = candidate_waypoints_pcd.point.positions.cpu().numpy()

            obs_legacy = self.obstacle_pcd.to_legacy()
            obs_kdtree = o3d.geometry.KDTreeFlann(obs_legacy)

            stair_kdtree = None
            has_stair = hasattr(self, 'stair_pcd') and not self.stair_pcd.is_empty()
            if has_stair:
                stair_legacy = self.stair_pcd.to_legacy()
                stair_kdtree = o3d.geometry.KDTreeFlann(stair_legacy)

            obstacle_pts = self.obstacle_pcd.point.positions.cpu().numpy()
            stair_pts = self.stair_pcd.point.positions.cpu().numpy() if has_stair else None

            valid_flags = np.zeros(cand_pts.shape[0], dtype=bool)

            for i, p in enumerate(cand_pts):
                base_z = p[2] + 0.2
                top_z = base_z + cylinder_height
                p2 = p + [0, 0, 0.8]
                p = p + [0, 0, 0.5]

                k_obs, idx_obs, _ = obs_kdtree.search_radius_vector_3d(o3d.utility.Vector3dVector([p]).__getitem__(0),
                                                                       cylinder_radius)
                has_obs = False
                if k_obs > 0:
                    sel = obstacle_pts[idx_obs, :]

                    has_obs = np.any((sel[:, 2] >= base_z) & (sel[:, 2] <= top_z))
                else:
                    k_obs2, idx_obs2, _ = obs_kdtree.search_radius_vector_3d(
                        o3d.utility.Vector3dVector([p2]).__getitem__(0),
                        cylinder_radius)
                    if k_obs2 > 0:
                        sel2 = obstacle_pts[idx_obs2, :]
                        has_obs = np.any((sel2[:, 2] >= base_z) & (sel2[:, 2] <= top_z))

                if not has_obs:
                    valid_flags[i] = True
                    continue

                if stair_kdtree is not None:
                    k_stair, idx_stair, _ = stair_kdtree.search_radius_vector_3d(
                        o3d.utility.Vector3dVector([p]).__getitem__(0),
                        cylinder_radius)
                    if k_stair > 0:
                        sel_st = stair_pts[idx_stair, :]
                        has_stair_in_cyl = np.any((sel_st[:, 2] >= base_z) & (sel_st[:, 2] <= top_z))
                        if has_stair_in_cyl:
                            valid_flags[i] = True

            if not np.any(valid_flags):
                self.waypoints = np.array([])
                return np.array([])

            if use_distance_filter:
                distance_to_obstacle = pointcloud_2d_distance(candidate_waypoints_pcd, self.obstacle_pcd)
                dist_np = distance_to_obstacle.cpu().numpy()
                dist_mask = (dist_np >= min_distance) & (dist_np < max_distance)
                valid_flags = valid_flags & dist_mask
                if not np.any(valid_flags):
                    self.waypoints = np.array([])
                    return np.array([])

            indices_np = np.where(valid_flags)[0]
            indices_tensor = o3d.core.Tensor(indices_np, device=self.pcd_device)
            final_pcd = candidate_waypoints_pcd.select_by_index(indices_tensor)
            self.waypoints = final_pcd.point.positions.cpu().numpy()
            return self.waypoints

        except Exception as e:
            print(f"[get_candidate_waypoints] Error: {e}")
            self.waypoints = np.array([])
            return np.array([])

    def _denoise_pcd(self, pcd: o3d.t.geometry.PointCloud,
                     nb_neighbors: int = 20,
                     std_ratio: float = 1.0,
                     radius: float = 0.30,
                     min_points: int = 10) -> o3d.t.geometry.PointCloud:
        """
        Two-stage denoising:
        1) statistical outlier removal (mean distance > mean + std_ratio * std)
        2) radius outlier removal (fewer than min_points neighbors)
        compatible with tensor/legacy conversion.
        """
        try:
            if pcd.is_empty() or pcd.point.positions.shape[0] < (min_points * 2):
                return pcd

            legacy = pcd.to_legacy()
            legacy_sor, _ = legacy.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
            legacy_rad, _ = legacy_sor.remove_radius_outlier(nb_points=min_points, radius=radius)
            return o3d.t.geometry.PointCloud.from_legacy(legacy_rad, self.pcd_device)
        except Exception:
            return pcd

    def get_current_view_candidate_waypoints(self,
                                             waypoint_grid_resolution=1.0,
                                             min_distance=0.1,
                                             max_distance=1.0,
                                             merge_distance=0.5,
                                             cylinder_radius=0.2,
                                             cylinder_height=1.2,
                                             use_distance_filter=False,
                                             max_view_angle_deg=75,
                                             head_clearance=0.6,
                                             head_check_radius=0.30):
        if self.current_view_navigable_pcd.is_empty():
            return np.array([]), np.array([])

        try:
            src = self._denoise_pcd(self.current_view_navigable_pcd,
                                    nb_neighbors=20, std_ratio=1.0,
                                    radius=0.35, min_points=15)

            candidate_waypoints_pcd = src.voxel_down_sample(waypoint_grid_resolution)
            if candidate_waypoints_pcd.is_empty():
                return np.array([]), np.array([])

            cand_pts = candidate_waypoints_pcd.point.positions.cpu().numpy()

            has_obstacle = not self.obstacle_pcd.is_empty()
            if has_obstacle:
                obs_legacy = self.obstacle_pcd.to_legacy()
                obs_kdtree = o3d.geometry.KDTreeFlann(obs_legacy)
                obstacle_pts = self.obstacle_pcd.point.positions.cpu().numpy()
            else:
                obs_kdtree = None
                obstacle_pts = None

            has_stair = hasattr(self, 'stair_pcd') and (not self.stair_pcd.is_empty())
            if has_stair:
                stair_legacy = self.stair_pcd.to_legacy()
                stair_kdtree = o3d.geometry.KDTreeFlann(stair_legacy)
                stair_pts = self.stair_pcd.point.positions.cpu().numpy()
            else:
                stair_kdtree = None
                stair_pts = None

            valid_flags = np.zeros(cand_pts.shape[0], dtype=bool)

            for i, p in enumerate(cand_pts):
                base_z = p[2] + 0.1
                top_z = base_z + cylinder_height
                p2 = p + [0, 0, 0.8]
                p = p + [0, 0, 0.5]

                has_obs_in_cyl = False
                if has_obstacle:
                    k_obs, idx_obs, _ = obs_kdtree.search_radius_vector_3d(p, cylinder_radius)
                    if k_obs > 0:
                        sel = obstacle_pts[idx_obs]
                        has_obs_in_cyl = np.any((sel[:, 2] >= base_z) & (sel[:, 2] <= top_z))
                    else:
                        k_obs2, idx_obs2, _ = obs_kdtree.search_radius_vector_3d(p2, cylinder_radius)
                        if k_obs2 > 0:
                            sel2 = obstacle_pts[idx_obs2]
                            has_obs_in_cyl = np.any((sel2[:, 2] >= base_z) & (sel2[:, 2] <= top_z))

                # A candidate is valid only when its cylinder is obstacle-free.
                if not has_obs_in_cyl:
                    valid_flags[i] = True
                    continue

                if has_stair:
                    k_stair, idx_stair, _ = stair_kdtree.search_radius_vector_3d(p, cylinder_radius)
                    if k_stair > 5:
                        sel_st = stair_pts[idx_stair]
                        has_stair_in_cyl = np.any((sel_st[:, 2] >= base_z) & (sel_st[:, 2] <= top_z))
                        if has_stair_in_cyl:
                            valid_flags[i] = True

            if not np.any(valid_flags):
                return np.array([]), np.array([])

            if use_distance_filter and has_obstacle:
                distance_to_obstacle = pointcloud_2d_distance(candidate_waypoints_pcd, self.obstacle_pcd).cpu().numpy()
                dist_mask = (distance_to_obstacle >= min_distance) & (distance_to_obstacle < max_distance)
                valid_flags = valid_flags & dist_mask
                if not np.any(valid_flags):
                    return np.array([]), np.array([])

            indices_np = np.where(valid_flags)[0]
            indices_tensor = o3d.core.Tensor(indices_np, device=self.pcd_device)
            final_waypoints_pcd = candidate_waypoints_pcd.select_by_index(indices_tensor)
            waypoints_world = final_waypoints_pcd.point.positions.cpu().numpy()

            if merge_distance and merge_distance > 0 and waypoints_world.shape[0] > 1:
                cell_size = float(merge_distance)
                voxel_idx = np.floor(waypoints_world / cell_size + 0.5).astype(np.int32)
                keys = [tuple(v) for v in voxel_idx]
                accum = {}
                for k, pp in zip(keys, waypoints_world):
                    if k in accum:
                        accum[k][0] += pp
                        accum[k][1] += 1
                    else:
                        accum[k] = [pp.copy(), 1]
                waypoints_world = np.vstack([v[0] / v[1] for v in accum.values()]).astype(np.float32)

            if waypoints_world.shape[0] > 0 and max_view_angle_deg is not None:
                R = self.current_rotation
                if R.shape == (4, 4):
                    R = R[:3, :3]

                local_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
                forward_world = (R @ local_forward).astype(np.float32)

                forward2d = forward_world[:2]
                n = np.linalg.norm(forward2d)
                if n < 1e-6:
                    forward_world = R @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    forward2d = forward_world[:2]
                    n = np.linalg.norm(forward2d)

                if n > 1e-6:
                    forward2d /= n
                    vecs2d = waypoints_world[:, :2] - self.current_position[:2]
                    vnorm = np.linalg.norm(vecs2d, axis=1, keepdims=True) + 1e-8
                    dir2d = vecs2d / vnorm

                    dist2d = np.linalg.norm(vecs2d, axis=1)
                    close_keep_radius = 1.0
                    close_mask = dist2d <= float(close_keep_radius)

                    cosv = np.clip(np.sum(dir2d * forward2d[None, :], axis=1), -1.0, 1.0)
                    angles_deg = np.degrees(np.arccos(cosv))
                    ang_mask = angles_deg <= float(max_view_angle_deg)

                    combined_mask = close_mask | ang_mask
                    if not np.any(combined_mask):
                        return np.array([]), np.array([])

                    waypoints_world = waypoints_world[ang_mask]

            if waypoints_world.shape[0] == 0:
                return np.array([]), np.array([])

            max_allowed_z = self.current_position[2] + 0.2
            min_allowed_z = self.current_position[2] - 2.2
            hmask = (waypoints_world[:, 2] <= max_allowed_z) & (waypoints_world[:, 2] >= min_allowed_z)
            if not np.any(hmask):
                return np.array([]), np.array([])

            waypoints_world = waypoints_world[hmask]
            waypoints_relative = waypoints_world - self.current_position
            return waypoints_world, waypoints_relative[:, [0, 2, 1]]

        except Exception as e:
            print(f"[get_current_view_candidate_waypoints] Error: {e}")
            return np.array([]), np.array([])

    def _is_path_clear(self, p1, p2, navigable_kdtree, obstacle_kdtree, stair_kdtree=None, check_radius=0.1,
                       obstacle_radius=0.1,
                       step_size=0.1):
        """2D distance-based path reachability check (obstacle-free), considering stairs."""
        vec_2d = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        dist_2d = np.linalg.norm(vec_2d)
        if dist_2d < step_size:
            return True
        num_steps = int(dist_2d / step_size)
        if num_steps == 0:
            return True
        dir_vec_2d = vec_2d / dist_2d
        ground_z = self.floor_height

        for i in range(1, num_steps):
            inter_point_2d = np.array([p1[0], p1[1]]) + dir_vec_2d * (i * step_size)
            check_point = np.array([inter_point_2d[0], inter_point_2d[1], ground_z])
            check_point_2 = np.array([inter_point_2d[0], inter_point_2d[1], ground_z + 0.4])
            check_point_3 = np.array([inter_point_2d[0], inter_point_2d[1], ground_z + 0.8])

            k_nav, _, _ = navigable_kdtree.search_radius_vector_3d(check_point, check_radius)
            if k_nav < 10:
                return False

            k_obs, _, _ = obstacle_kdtree.search_radius_vector_3d(check_point_2, obstacle_radius)
            k_obs_2, _, _ = obstacle_kdtree.search_radius_vector_3d(check_point_3, obstacle_radius)

            if k_obs + k_obs_2 > 0:

                if stair_kdtree is not None:
                    k_stair, _, _ = stair_kdtree.search_radius_vector_3d(check_point_2, 0.3)
                    if k_stair > 5:
                        continue
                return False

        return True

    def plan_path_to_target(self, target_point, max_distance_to_neighbor=1.0, max_iterations=100, use_waypoint_buffer=True):
        """
        A* 2D path planning from current position to target (optimized):
        - precompute adjacencies to avoid O(N^2) full scan
        - standard heap update strategy (duplicate entries allowed, stale ones skipped on pop)
        - vectorized heuristic distance via numpy
        """
        t1 = time.time()

        if abs(target_point[2] - (self.current_position[2] - 1.2)) > 1.0:
            pass
            return []

        # 每次规划只算一次 avoid 区域，_avoid_penalty 在A*内层循环里被高频调用，
        # 不能每次都重新遍历 grounded_landmarks
        self._avoid_zones_cache = self.get_avoid_zones()

        if not use_waypoint_buffer:
            self.waypoints = self.get_candidate_waypoints(
                min_distance=0.3, max_distance=2.5, waypoint_grid_resolution=0.5
            )

        if self.waypoints.shape[0] == 0:
            pass
            return []

        navigable_kdtree = None if self.navigable_pcd.is_empty() else o3d.geometry.KDTreeFlann(
            self.navigable_pcd.to_legacy())
        obstacle_kdtree = None if self.obstacle_pcd.is_empty() else o3d.geometry.KDTreeFlann(
            self.obstacle_pcd.to_legacy())
        stair_kdtree = None if self.stair_pcd.is_empty() else o3d.geometry.KDTreeFlann(self.stair_pcd.to_legacy())

        start_point = (self.current_position - np.array([0, 0, 1.2], dtype=np.float32))[:3]
        goal_point = np.asarray(target_point[:3], dtype=np.float32)

        nodes = np.vstack([self.waypoints[:, :3], start_point[None, :], goal_point[None, :]]).astype(np.float32)

        nodes_unique, unique_idx = np.unique(nodes, axis=0, return_index=True)
        nodes = nodes_unique[np.argsort(unique_idx)]

        def _find_idx(arr, p):
            m = np.where(np.all(arr == p, axis=1))[0]
            if m.size > 0:
                return int(m[0])

            d2 = np.sum((arr - p[None, :]) ** 2, axis=1)
            return int(np.argmin(d2))

        start_node_idx = _find_idx(nodes, start_point)
        goal_node_idx = _find_idx(nodes, goal_point)

        n_nodes = nodes.shape[0]
        if n_nodes == 0:
            return []

        node_pcd = o3d.geometry.PointCloud()
        node_pcd.points = o3d.utility.Vector3dVector(nodes.astype(np.float64))
        node_kdtree = o3d.geometry.KDTreeFlann(node_pcd)

        neighbors = [[] for _ in range(n_nodes)]
        for i in range(n_nodes):
            p = nodes[i].astype(np.float64)
            k, idxs, _ = node_kdtree.search_radius_vector_3d(p, float(max_distance_to_neighbor) + 1e-6)
            if k <= 1:
                continue
            pz = nodes[i, 2]

            cand = [j for j in idxs if j != i and abs(float(pz - nodes[j, 2])) <= 0.3]
            neighbors[i] = cand

        goal_xyz = nodes[goal_node_idx]

        def heuristic_idx(i: int) -> float:
            d = nodes[i] - goal_xyz
            return float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))

        open_heap = []
        came_from = np.full((n_nodes,), -1, dtype=np.int32)

        g_score = np.full((n_nodes,), np.inf, dtype=np.float64)
        f_score = np.full((n_nodes,), np.inf, dtype=np.float64)

        g_score[start_node_idx] = 0.0
        f0 = heuristic_idx(start_node_idx)
        f_score[start_node_idx] = f0
        heapq.heappush(open_heap, (f0, start_node_idx))

        iterations = 0

        while open_heap:
            if iterations >= max_iterations:
                pass

                return []
            iterations += 1

            current_f, current_idx = heapq.heappop(open_heap)

            if current_f > f_score[current_idx]:
                continue

            if current_idx == goal_node_idx:

                path_idx = []
                cur = current_idx
                while cur != -1:
                    path_idx.append(cur)
                    cur = came_from[cur]
                path_idx.reverse()
                path = [tuple(nodes[i].tolist()) for i in path_idx]

                return path

            p_cur = nodes[current_idx]

            for neighbor_idx in neighbors[current_idx]:
                p_nei = nodes[neighbor_idx]

                if navigable_kdtree is not None:
                    if not self._is_path_clear(
                            p_cur, p_nei,
                            navigable_kdtree, obstacle_kdtree, stair_kdtree,
                            check_radius=0.3, obstacle_radius=0.2, step_size=0.2
                    ):
                        continue

                d = p_cur - p_nei
                edge_cost = float(np.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]))
                # 指令里明确要求避开的地标("avoid the wet floor")在这里变成额外
                # 代价。用软惩罚而不是硬禁行：如果绕不开(唯一通路正好穿过禁区)，
                # 宁可付出代价通过，也好过规划失败让agent原地转圈。
                edge_cost += self._avoid_penalty(p_nei)
                tentative_g = g_score[current_idx] + edge_cost

                if tentative_g < g_score[neighbor_idx]:
                    came_from[neighbor_idx] = current_idx
                    g_score[neighbor_idx] = tentative_g
                    new_f = tentative_g + heuristic_idx(neighbor_idx)
                    f_score[neighbor_idx] = new_f
                    heapq.heappush(open_heap, (new_f, neighbor_idx))

        return []

    def plan_path_to_target_naive(self, target_point, step_length=1.0, clearance_step=0.2):
        """
        Ablation: straight-line strategy, no global search.
        Sample uniformly along a 2D line from current position to target, check clearance segment-wise.
        If entire path is clear, return waypoint list; otherwise return [].
        Parameters:
            target_point: (x, y, z) target world coords (same reference frame as get_candidate_waypoints)
            step_length:  sampling interval (meters)
            clearance_step: _is_path_clear internal step (meters)
        Returns:
            list[(x, y, z)] straight-line waypoints (including start/end); returns [] if unreachable
        """
        try:
            tp = np.asarray(target_point, dtype=np.float32)

            if abs(tp[2] - (self.current_position[2] - 1.2)) > 1.0:
                return []

            start_point = (self.current_position - np.array([0, 0, 1.2], dtype=np.float32)).astype(np.float32)
            vec2d = tp[:2] - start_point[:2]
            dist2d = float(np.linalg.norm(vec2d))
            if dist2d < 1e-6:
                return [tuple(start_point.tolist()), tuple(tp.tolist())]

            dir2d = vec2d / dist2d
            n_steps = int(dist2d // step_length)

            if self.navigable_pcd.is_empty():
                navigable_kdtree = None
            else:
                navigable_kdtree = o3d.geometry.KDTreeFlann(self.navigable_pcd.to_legacy())

            if self.obstacle_pcd.is_empty():
                obstacle_kdtree = None
            else:
                obstacle_kdtree = o3d.geometry.KDTreeFlann(self.obstacle_pcd.to_legacy())

            if self.stair_pcd.is_empty():
                stair_kdtree = None
            else:
                stair_kdtree = o3d.geometry.KDTreeFlann(self.stair_pcd.to_legacy())

            if navigable_kdtree is None or obstacle_kdtree is None:
                path = [tuple(start_point.tolist())]
                for i in range(1, n_steps + 1):
                    seg_len = min(step_length * i, dist2d)
                    p2 = start_point.copy()
                    p2[0] += dir2d[0] * seg_len
                    p2[1] += dir2d[1] * seg_len

                    p2[2] = start_point[2] + (tp[2] - start_point[2]) * (seg_len / (dist2d + 1e-6))
                    path.append(tuple(p2.tolist()))
                path.append(tuple(tp.tolist()))
                return path

            clear = self._is_path_clear(
                start_point, tp,
                navigable_kdtree, obstacle_kdtree,
                stair_kdtree=stair_kdtree,
                check_radius=0.3,
                obstacle_radius=0.2,
                step_size=clearance_step
            )
            if not clear:
                return []

            path = [tuple(start_point.tolist())]
            for i in range(1, n_steps + 1):
                seg_len = min(step_length * i, dist2d)
                p2 = start_point.copy()
                p2[0] += dir2d[0] * seg_len
                p2[1] += dir2d[1] * seg_len
                p2[2] = start_point[2] + (tp[2] - start_point[2]) * (seg_len / (dist2d + 1e-6))
                path.append(tuple(p2.tolist()))

            if path[-1] != tuple(tp.tolist()):
                path.append(tuple(tp.tolist()))
            return path
        except Exception as e:
            print(f"[plan_path_to_target_naive] Error: {e}")
            return []

    def _identify_and_process_stairs(self,
                                     scene_pcd,
                                     normal_radius=0.2,
                                     normal_max_nn=30,
                                     stair_min_abs_z=0.2,
                                     stair_max_abs_z=0.7,
                                     stair_cluster_eps=0.4,
                                     stair_cluster_min_points=500,
                                     stair_min_height_span=0.5,
                                     platform_abs_z_thresh=0.9,
                                     platform_z_delta=0.1,
                                     platform_cluster_eps=0.35,
                                     platform_cluster_min_points=35,
                                     platform_min_points_total=120,
                                     bbox_xy_expand=0.3,
                                     stair_min_xy_extent=0.45,
                                     stair_min_height_bins=4,
                                     stair_min_axis_z_corr=0.3,
                                     enable_tread_detection=False,
                                     tread_abs_z_thresh=0.85,
                                     tread_z_delta=0.08,
                                     tread_axis_expand=0.20,
                                     tread_side_expand=0.12,
                                     tread_cluster_min_points=50,
                                     _profile=True):
        """
        Detect stairs and their top/bottom platforms; return stair point cloud.

        Key optimizations:
        1. use voxel connectivity instead of per-frame DBSCAN to reduce clustering time.
        2. geometry consistency check on candidate clusters, filtering slanted walls, handrails, stray surfaces.
        3. platform search: coarse 2D KDTree filter, then precise rectangle + height filtering.
        """
        if scene_pcd.is_empty() or len(scene_pcd.point.positions) < 100:
            return o3d.t.geometry.PointCloud(self.pcd_device)

        t0 = time.perf_counter()

        def _voxel_key(vox):
            return tuple(int(v) for v in vox)

        def _voxel_cc_cluster(points_np, voxel, min_points, neighbor_range=1):
            if points_np is None or points_np.shape[0] == 0:
                return []

            voxel = max(float(voxel), float(self.pcd_resolution), 1e-3)
            vox = np.floor(points_np / voxel).astype(np.int32)
            vox_view = np.ascontiguousarray(vox).view(
                [("x", np.int32), ("y", np.int32), ("z", np.int32)]
            ).reshape(-1)
            uniq_view, inv = np.unique(vox_view, return_inverse=True)
            uniq_vox = uniq_view.view(np.int32).reshape(-1, 3)

            buckets = [[] for _ in range(uniq_vox.shape[0])]
            for point_idx, voxel_idx in enumerate(inv):
                buckets[int(voxel_idx)].append(point_idx)

            lut = {_voxel_key(v): i for i, v in enumerate(uniq_vox)}
            visited = np.zeros(uniq_vox.shape[0], dtype=bool)
            clusters = []

            offsets = [
                (dx, dy, dz)
                for dx in range(-neighbor_range, neighbor_range + 1)
                for dy in range(-neighbor_range, neighbor_range + 1)
                for dz in range(-neighbor_range, neighbor_range + 1)
                if not (dx == 0 and dy == 0 and dz == 0)
            ]

            for seed in range(uniq_vox.shape[0]):
                if visited[seed]:
                    continue

                queue = [seed]
                visited[seed] = True
                voxel_ids = []
                point_count = 0

                while queue:
                    cur = queue.pop()
                    voxel_ids.append(cur)
                    point_count += len(buckets[cur])
                    base = uniq_vox[cur]

                    for offset in offsets:
                        nb_key = (
                            int(base[0] + offset[0]),
                            int(base[1] + offset[1]),
                            int(base[2] + offset[2]),
                        )
                        nb = lut.get(nb_key)
                        if nb is None or visited[nb]:
                            continue
                        visited[nb] = True
                        queue.append(nb)

                if point_count < int(min_points):
                    continue

                point_ids = []
                for voxel_id in voxel_ids:
                    point_ids.extend(buckets[voxel_id])
                clusters.append(np.asarray(point_ids, dtype=np.int64))

            return clusters

        def _looks_like_stairs(points_np):
            if points_np.shape[0] < int(stair_cluster_min_points):
                return False

            xy = points_np[:, :2]
            z = points_np[:, 2]
            h_span = float(z.max() - z.min())
            if h_span < float(stair_min_height_span):
                return False

            xy_extent = np.ptp(xy, axis=0)
            if float(np.max(xy_extent)) < float(stair_min_xy_extent):
                return False

            centered_xy = xy - xy.mean(axis=0, keepdims=True)
            try:
                _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
                axis = vh[0]
            except np.linalg.LinAlgError:
                return False

            along = centered_xy @ axis
            along_span = float(along.max() - along.min())
            if along_span < float(stair_min_xy_extent):
                return False

            corr = np.corrcoef(along, z)[0, 1] if points_np.shape[0] > 2 else 0.0
            if not np.isfinite(corr) or abs(float(corr)) < float(stair_min_axis_z_corr):
                return False

            bin_size = max(float(self.pcd_resolution) * 2.0, 0.08)
            height_bins = np.unique(np.floor((z - z.min()) / bin_size).astype(np.int32))
            if height_bins.size < int(stair_min_height_bins):
                return False

            return True

        try:
            t_norm0 = time.perf_counter()
            scene_pcd.estimate_normals(radius=normal_radius, max_nn=normal_max_nn)
            t_norm1 = time.perf_counter()

            t_cpu0 = time.perf_counter()
            normals = scene_pcd.point.normals.cpu().numpy()
            abs_z = np.abs(normals[:, 2])
            t_cpu1 = time.perf_counter()

            stair_mask = (abs_z > stair_min_abs_z) & (abs_z < stair_max_abs_z)
            if not np.any(stair_mask):
                self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                return o3d.t.geometry.PointCloud(self.pcd_device)

            stair_indices = np.where(stair_mask)[0]
            stair_indices_tensor = o3d.core.Tensor(stair_indices, dtype=o3d.core.Dtype.Int64, device=self.pcd_device)
            stair_candidates = scene_pcd.select_by_index(stair_indices_tensor)
            if stair_candidates.is_empty():
                self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                return o3d.t.geometry.PointCloud(self.pcd_device)

            t_cl0 = time.perf_counter()
            stair_points_np = stair_candidates.point.positions.cpu().numpy()
            stair_clusters = _voxel_cc_cluster(
                stair_points_np,
                voxel=max(float(stair_cluster_eps) * 0.5, 0.05),
                min_points=stair_cluster_min_points,
                neighbor_range=1,
            )
            t_cl1 = time.perf_counter()
            if not stair_clusters:
                self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                return o3d.t.geometry.PointCloud(self.pcd_device)

            platform_mask = abs_z > platform_abs_z_thresh
            platform_indices = np.where(platform_mask)[0]
            platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            if platform_indices.size > 0:
                platform_indices_tensor = o3d.core.Tensor(platform_indices, dtype=o3d.core.Dtype.Int64,
                                                          device=self.pcd_device)
                platform_pcd = scene_pcd.select_by_index(platform_indices_tensor)

            tread_pcd_all = o3d.t.geometry.PointCloud(self.pcd_device)
            if enable_tread_detection:
                tread_mask = abs_z > tread_abs_z_thresh
                tread_indices = np.where(tread_mask)[0]
                if tread_indices.size > 0:
                    tread_indices_tensor = o3d.core.Tensor(tread_indices, dtype=o3d.core.Dtype.Int64,
                                                           device=self.pcd_device)
                    tread_pcd_all = scene_pcd.select_by_index(tread_indices_tensor)

            plat_pts = None
            plat_kdtree = None
            if not platform_pcd.is_empty():
                plat_pts = platform_pcd.point.positions.cpu().numpy()
                plat_pts_2d = plat_pts.copy()
                plat_pts_2d[:, 2] = 0.0
                plat_legacy_2d = o3d.geometry.PointCloud()
                plat_legacy_2d.points = o3d.utility.Vector3dVector(plat_pts_2d.astype(np.float64, copy=False))
                plat_kdtree = o3d.geometry.KDTreeFlann(plat_legacy_2d)

            tread_pts = None
            tread_kdtree = None
            if enable_tread_detection and not tread_pcd_all.is_empty():
                tread_pts = tread_pcd_all.point.positions.cpu().numpy()
                tread_pts_2d = tread_pts.copy()
                tread_pts_2d[:, 2] = 0.0
                tread_legacy_2d = o3d.geometry.PointCloud()
                tread_legacy_2d.points = o3d.utility.Vector3dVector(tread_pts_2d.astype(np.float64, copy=False))
                tread_kdtree = o3d.geometry.KDTreeFlann(tread_legacy_2d)

            stair_clusters_to_merge = []
            tread_clusters_to_merge = []
            platform_clusters_to_merge = []

            t_loop0 = time.perf_counter()
            for cluster_indices_np in stair_clusters:
                cluster_indices = o3d.core.Tensor(cluster_indices_np, dtype=o3d.core.Dtype.Int64,
                                                  device=self.pcd_device)
                cluster = stair_candidates.select_by_index(cluster_indices)
                if cluster.point.positions.shape[0] < stair_cluster_min_points:
                    continue

                cluster_pts = cluster.point.positions.cpu().numpy()
                if not _looks_like_stairs(cluster_pts):
                    continue

                stair_clusters_to_merge.append(cluster)

                if enable_tread_detection and tread_pts is not None and tread_kdtree is not None:
                    stair_xy = cluster_pts[:, :2]
                    z_min = float(cluster_pts[:, 2].min())
                    z_max = float(cluster_pts[:, 2].max())
                    center_xy = stair_xy.mean(axis=0)
                    centered_xy = stair_xy - center_xy[None, :]
                    try:
                        _, _, vh = np.linalg.svd(centered_xy, full_matrices=False)
                        axis = vh[0].astype(np.float64, copy=False)
                    except np.linalg.LinAlgError:
                        axis = None

                    if axis is not None:
                        axis_norm = float(np.linalg.norm(axis))
                        if axis_norm > 1e-6:
                            axis = axis / axis_norm
                            side_axis = np.array([-axis[1], axis[0]], dtype=np.float64)

                            stair_along = centered_xy @ axis
                            stair_side = centered_xy @ side_axis
                            along_min = float(stair_along.min()) - float(tread_axis_expand)
                            along_max = float(stair_along.max()) + float(tread_axis_expand)
                            side_min = float(stair_side.min()) - float(tread_side_expand)
                            side_max = float(stair_side.max()) + float(tread_side_expand)

                            radius = float(np.sqrt(max(abs(along_min), abs(along_max)) ** 2 +
                                                   max(abs(side_min), abs(side_max)) ** 2) + 1e-6)
                            query = np.array([center_xy[0], center_xy[1], 0.0], dtype=np.float64)
                            k_tread, candidate_tread_indices, _ = tread_kdtree.search_radius_vector_3d(query, radius)
                            if k_tread > 0:
                                candidate_tread_indices = np.asarray(candidate_tread_indices, dtype=np.int64)
                                candidate_tread_pts = tread_pts[candidate_tread_indices]
                                rel_xy = candidate_tread_pts[:, :2] - center_xy[None, :]
                                cand_along = rel_xy @ axis
                                cand_side = rel_xy @ side_axis
                                footprint_cond = (
                                    (cand_along >= along_min) & (cand_along <= along_max) &
                                    (cand_side >= side_min) & (cand_side <= side_max)
                                )
                                z_cond = (
                                    (candidate_tread_pts[:, 2] >= z_min - float(tread_z_delta)) &
                                    (candidate_tread_pts[:, 2] <= z_max + float(tread_z_delta))
                                )
                                tread_sel = candidate_tread_indices[footprint_cond & z_cond]
                                if tread_sel.size >= int(tread_cluster_min_points):
                                    tread_sel_tensor = o3d.core.Tensor(tread_sel, dtype=o3d.core.Dtype.Int64,
                                                                       device=self.pcd_device)
                                    candidate_treads = tread_pcd_all.select_by_index(tread_sel_tensor)
                                    candidate_tread_pts = candidate_treads.point.positions.cpu().numpy()
                                    tread_clusters = _voxel_cc_cluster(
                                        candidate_tread_pts,
                                        voxel=max(float(platform_cluster_eps) * 0.5, 0.05),
                                        min_points=tread_cluster_min_points,
                                        neighbor_range=1,
                                    )
                                    for tread_indices_np in tread_clusters:
                                        tread_indices_tensor = o3d.core.Tensor(
                                            tread_indices_np,
                                            dtype=o3d.core.Dtype.Int64,
                                            device=self.pcd_device
                                        )
                                        tread_sub_pcd = candidate_treads.select_by_index(tread_indices_tensor)
                                        if tread_sub_pcd.point.positions.shape[0] >= tread_cluster_min_points:
                                            tread_clusters_to_merge.append(tread_sub_pcd)

                if plat_pts is None or plat_kdtree is None:
                    continue

                min_x, max_x = cluster_pts[:, 0].min(), cluster_pts[:, 0].max()
                min_y, max_y = cluster_pts[:, 1].min(), cluster_pts[:, 1].max()
                min_z, max_z = cluster_pts[:, 2].min(), cluster_pts[:, 2].max()

                cx = 0.5 * (min_x + max_x)
                cy = 0.5 * (min_y + max_y)
                hx = 0.5 * (max_x - min_x) + float(bbox_xy_expand)
                hy = 0.5 * (max_y - min_y) + float(bbox_xy_expand)
                query_radius = float(np.sqrt(hx * hx + hy * hy) + 1e-6)
                query = np.array([cx, cy, 0.0], dtype=np.float64)
                k, candidate_indices, _ = plat_kdtree.search_radius_vector_3d(query, query_radius)
                if k <= 0:
                    continue

                candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
                candidate_plat_pts = plat_pts[candidate_indices]

                xy_cond = (candidate_plat_pts[:, 0] >= min_x - bbox_xy_expand) & \
                          (candidate_plat_pts[:, 0] <= max_x + bbox_xy_expand) & \
                          (candidate_plat_pts[:, 1] >= min_y - bbox_xy_expand) & \
                          (candidate_plat_pts[:, 1] <= max_y + bbox_xy_expand)
                if not np.any(xy_cond):
                    continue

                sel_xy = candidate_indices[xy_cond]
                sel_xy_pts = plat_pts[sel_xy]

                z_cond = (np.abs(sel_xy_pts[:, 2] - min_z) < platform_z_delta) | \
                         (np.abs(sel_xy_pts[:, 2] - max_z) < platform_z_delta)

                sel = sel_xy[z_cond]
                if sel.size == 0:
                    continue

                sel_tensor = o3d.core.Tensor(sel, dtype=o3d.core.Dtype.Int64, device=self.pcd_device)
                candidate_platform = platform_pcd.select_by_index(sel_tensor)

                if candidate_platform.point.positions.shape[0] >= platform_min_points_total:
                    candidate_platform_pts = candidate_platform.point.positions.cpu().numpy()
                    platform_clusters = _voxel_cc_cluster(
                        candidate_platform_pts,
                        voxel=max(float(platform_cluster_eps) * 0.5, 0.05),
                        min_points=platform_cluster_min_points,
                        neighbor_range=1,
                    )
                    for sub_indices_np in platform_clusters:
                        sub_indices = o3d.core.Tensor(sub_indices_np, dtype=o3d.core.Dtype.Int64,
                                                      device=self.pcd_device)
                        sub_pcd = candidate_platform.select_by_index(sub_indices)
                        if sub_pcd.point.positions.shape[0] >= platform_cluster_min_points:
                            platform_clusters_to_merge.append(sub_pcd)
            t_loop1 = time.perf_counter()

            stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            for cluster in stair_clusters_to_merge:
                stair_pcd = gpu_merge_pointcloud(stair_pcd, cluster)

            tread_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            for tread in tread_clusters_to_merge:
                tread_pcd = gpu_merge_pointcloud(tread_pcd, tread)

            collected_platform = o3d.t.geometry.PointCloud(self.pcd_device)
            for platform in platform_clusters_to_merge:
                collected_platform = gpu_merge_pointcloud(collected_platform, platform)

            self.platform_pcd = collected_platform.voxel_down_sample(self.pcd_resolution) \
                if not collected_platform.is_empty() else o3d.t.geometry.PointCloud(self.pcd_device)

            if stair_pcd.is_empty() and self.platform_pcd.is_empty():
                return o3d.t.geometry.PointCloud(self.pcd_device)

            merged = gpu_merge_pointcloud(stair_pcd, tread_pcd) if enable_tread_detection else stair_pcd
            # merged = gpu_merge_pointcloud(merged, self.platform_pcd).voxel_down_sample(self.pcd_resolution)

            if _profile:
                t1 = time.perf_counter()
                print(
                    "[StairDetect] "
                    f"normals={t_norm1 - t_norm0:.4f}s, "
                    f"cpu={t_cpu1 - t_cpu0:.4f}s, "
                    f"cluster={t_cl1 - t_cl0:.4f}s, "
                    f"loop={t_loop1 - t_loop0:.4f}s, "
                    f"total={t1 - t0:.4f}s"
                )

            return merged

        except Exception as e:
            print(f"[StairDetect] Error: {e}")
            self.platform_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            return o3d.t.geometry.PointCloud(self.pcd_device)

    def is_on_platform(self,
                       scene_pcd,
                       search_radius=1.5,
                       plane_distance=0.05,
                       min_area=0.5,
                       min_support_ratio=0.35,
                       min_radius=0.35,
                       max_height_diff=0.4,
                       bottom_height_tolerance=0.1,
                       area_grid_resolution=0.12):
        """
        Check for sufficiently large horizontal platform near current position.
        use local height bucketing and 2D grid occupancy instead of RANSAC + convex hull for speed.
        Returns: (flag, metrics)
        """
        metrics = {}
        try:
            if scene_pcd.is_empty() or scene_pcd.point.positions.shape[0] < 50:
                return False, metrics

            pts_np = scene_pcd.point.positions.cpu().numpy()
            agent_xy = self.current_position[:2]
            dx = pts_np[:, 0] - agent_xy[0]
            dy = pts_np[:, 1] - agent_xy[1]
            dist2 = dx * dx + dy * dy
            local_mask = dist2 <= (search_radius ** 2)
            if not np.any(local_mask):
                return False, metrics

            local_pts = pts_np[local_mask]
            local_count = local_pts.shape[0]
            if local_count < 30:
                return False, metrics

            agent_z = float(self.current_position[2] - 1.2)
            local_z = local_pts[:, 2]
            finite_mask = np.isfinite(local_z)
            if not np.any(finite_mask):
                return False, metrics
            local_pts = local_pts[finite_mask]
            local_z = local_z[finite_mask]
            local_count = local_pts.shape[0]
            if local_count < 30:
                return False, metrics

            z_min = float(local_z.min())

            bin_size = max(float(plane_distance), 1e-3)
            z_bins = np.floor((local_z - z_min) / bin_size).astype(np.int32)
            bin_ids, counts = np.unique(z_bins, return_counts=True)
            if bin_ids.size == 0:
                return False, metrics

            bin_heights = z_min + (bin_ids.astype(np.float64) + 0.5) * bin_size
            close_bins = np.where(
                (np.abs(bin_heights - agent_z) < float(bottom_height_tolerance)) &
                (counts >= 20)
            )[0]
            if close_bins.size > 0:
                candidate_bins = close_bins
            else:
                near_bottom = np.abs(bin_heights - agent_z) <= max(
                    float(max_height_diff), float(bottom_height_tolerance)
                )
                candidate_bins = np.where(near_bottom)[0]
            if candidate_bins.size == 0:
                candidate_bins = np.arange(bin_ids.size)

            if close_bins.size > 0:
                best_rel = candidate_bins[np.argmin(np.abs(bin_heights[candidate_bins] - agent_z))]
            else:
                best_rel = candidate_bins[np.argmax(counts[candidate_bins])]
            plane_z_seed = bin_heights[best_rel]
            height_band = max(float(plane_distance) * 1.5, 0.06)
            inlier_mask = np.abs(local_z - plane_z_seed) <= height_band
            inlier_count = int(np.count_nonzero(inlier_mask))
            if inlier_count < 20:
                return False, metrics

            inlier_pts = local_pts[inlier_mask]
            plane_z_mean = float(np.median(inlier_pts[:, 2]))
            z_std = float(inlier_pts[:, 2].std())
            support_ratio = inlier_count / (local_count + 1e-6)

            pts_xy = inlier_pts[:, :2]
            grid_resolution = max(float(area_grid_resolution), float(self.pcd_resolution), 1e-3)
            grid_xy = np.floor(pts_xy / grid_resolution).astype(np.int32)
            grid_view = np.ascontiguousarray(grid_xy).view(
                [("x", np.int32), ("y", np.int32)]
            ).reshape(-1)
            occupied_cells = int(np.unique(grid_view).shape[0])
            area_xy = occupied_cells * grid_resolution * grid_resolution

            radius_est = math.sqrt(area_xy / math.pi) if area_xy > 0 else 0.0
            vertical_diff = abs(agent_z - plane_z_mean)
            vertical_score = 1.0 if z_std < 0.08 else max(0.0, 1.0 - z_std / 0.4)

            close_to_agent_bottom = vertical_diff < float(bottom_height_tolerance)

            flag = (
                close_to_agent_bottom or (
                    vertical_score >= 0.95 and
                    support_ratio >= min_support_ratio and
                    area_xy >= min_area and
                    radius_est >= min_radius and
                    vertical_diff <= max_height_diff and
                    z_std < 0.08
                )
            )

            if not flag and vertical_score >= 0.95 and area_xy >= (min_area * 0.8) and vertical_diff <= 0.5:
                flag = True

            metrics = {
                "inlier_count": int(inlier_count),
                "local_count": int(local_count),
                "support_ratio": float(support_ratio),
                "vertical_score": float(vertical_score),
                "plane_height_mean": float(plane_z_mean),
                "agent_z": float(agent_z),
                "vertical_diff": float(vertical_diff),
                "area_xy": float(area_xy),
                "radius_est": float(radius_est),
                "z_std": float(z_std),
                "occupied_cells": int(occupied_cells),
                "bottom_height_tolerance": float(bottom_height_tolerance)
            }
            return bool(flag), metrics
        except Exception as e:
            print(f"[is_on_platform] error: {e}")
            return False, metrics

    def _safe_clear_tpcd(self, pcd):
        """Safely clear an o3d.t point cloud and return empty on same device."""
        try:
            if isinstance(pcd, o3d.t.geometry.PointCloud):
                pcd.clear()
        except Exception:
            pass
        return o3d.t.geometry.PointCloud(self.pcd_device)

    def release_vram(self):
        """Attempt to release GPU memory cache and trigger Python GC."""
        try:

            if hasattr(o3d.core, "cuda") and hasattr(o3d.core.cuda, "release_cache"):
                o3d.core.cuda.release_cache()
        except Exception:
            pass
        gc.collect()

    def translate_2d_coordinate_to_3d_world(
            self,
            x,
            y,
            depth: np.ndarray,
            camera_intrinsic=None,
            agent_position=None,
            agent_rotation=None):
        """
        Convert 2D image coords + depth to 3D world coordinates.
        Coordinate system consistent with get_pointcloud_from_depth / translate_to_world:
        - camera coords: [pixel_x, pixel_z, -pixel_y]
        - world coords: world = R @ camera_point + t
        accept camera_intrinsic/agent_position/agent_rotation for non-primary camera views,
        e.g., a single panorama panel.
        """

        if depth is None:
            return None

        if len(depth.shape) == 3:
            depth = depth[:, :, 0]

        h, w = depth.shape[:2]
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or xi >= w or yi < 0 or yi >= h:
            return None

        d = float(depth[yi, xi])
        if not np.isfinite(d) or d <= 0:
            return None

        intrinsic = self.camera_intrinsic if camera_intrinsic is None else camera_intrinsic
        fx = float(intrinsic[0][0])
        fy = float(intrinsic[1][1])
        cx = float(intrinsic[0][2])
        cy = float(intrinsic[1][2])

        pixel_z = (h - 1 - yi - cy) * d / fy
        pixel_x = (xi - cx) * d / fx
        pixel_y = d
        camera_point = np.array([pixel_x, pixel_z, -pixel_y], dtype=np.float32)

        position = self.current_position if agent_position is None else agent_position
        rotation = self.current_rotation if agent_rotation is None else agent_rotation
        if hasattr(rotation, 'as_rotation_matrix'):
            rotation = rotation.as_rotation_matrix()
        elif hasattr(rotation, 'shape') and rotation.shape == (4, 4):
            rotation = rotation[:3, :3]

        world_point = rotation @ camera_point + position
        return world_point

    def update_multiview(self, views, primary_index=None, do_instance_segmentation=True,
                         segment_all_views=False):
        """
        Batch fuse multiple synchronized views.

        Instead of calling update multiple times in the agent, only perform depth backprojection per view,
        then update global scene/navigable/obstacle point clouds once; instance segmentation runs only on primary view.
        views: [{'rgb': ..., 'depth': ..., 'position': ..., 'rotation': ..., 'camera_intrinsic': optional}, ...]

        segment_all_views: 对所有视角都跑检测，而不是只跑主视图(front)。
            按需开启——当前子任务还有 landmark 没接地时才值得多付这个开销。
            指令里的 "the window on your left" 这类侧方地标，只跑front是
            永远进不了地图的。
        """
        t0 = time.time()
        if not views:
            return []

        if primary_index is None:
            primary_index = len(views) - 1
        primary_index = max(0, min(primary_index, len(views) - 1))
        primary_view = views[primary_index]

        self.current_position = self.translation_func(primary_view['position']) - self.initial_position
        self.current_rotation = self.rotation_func(primary_view['rotation'])
        curr_agent_height = self.current_position[2] + self.initial_floor_height
        self._detect_and_update_floor()

        if len(self.trajectory_position) == 0 or not np.allclose(self.trajectory_position[-1], self.current_position,
                                                                 atol=1e-1):
            if isinstance(self.current_position, np.ndarray):
                self.trajectory_position.append(self.current_position.copy())
            else:
                self.trajectory_position.append(np.array(self.current_position))

        merged_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        primary_processed = None
        secondary_processed = []

        for idx, view in enumerate(views):
            current_depth = preprocess_depth(view['depth'])
            current_rgb = preprocess_image(view['rgb'])[:, :, :3]
            if np.sum(current_depth) <= 0:
                if idx == primary_index:
                    primary_processed = {
                        'rgb': current_rgb,
                        'depth': current_depth,
                        'position': self.translation_func(view['position']) - self.initial_position,
                        'rotation': self.rotation_func(view['rotation']),
                        'camera_intrinsic': view.get('camera_intrinsic', self.camera_intrinsic),
                        'pcd': o3d.t.geometry.PointCloud(self.pcd_device),
                    }
                continue

            view_position = self.translation_func(view['position']) - self.initial_position
            view_rotation = self.rotation_func(view['rotation'])
            view_intrinsic = view.get('camera_intrinsic', self.camera_intrinsic)
            camera_points, camera_colors = get_pointcloud_from_depth(current_rgb, current_depth, view_intrinsic)
            world_points = translate_to_world(camera_points, view_position, view_rotation)
            view_pcd = gpu_pointcloud_from_array(world_points, camera_colors,
                                                 self.pcd_device).voxel_down_sample(self.pcd_resolution)
            processed = {
                'rgb': current_rgb,
                'depth': current_depth,
                'position': view_position,
                'rotation': view_rotation,
                'camera_intrinsic': view_intrinsic,
                'pcd': view_pcd,
            }
            if idx == primary_index:
                primary_processed = processed
            else:
                secondary_processed.append(processed)
            merged_pcd = gpu_merge_pointcloud(merged_pcd, view_pcd)

        if primary_processed is None:
            primary_processed = {
                'rgb': preprocess_image(primary_view['rgb'])[:, :, :3],
                'depth': preprocess_depth(primary_view['depth']),
                'position': self.current_position,
                'rotation': self.current_rotation,
                'camera_intrinsic': primary_view.get('camera_intrinsic', self.camera_intrinsic),
                'pcd': o3d.t.geometry.PointCloud(self.pcd_device),
            }

        self.current_rgb = primary_processed['rgb']
        self.current_depth = primary_processed['depth']
        self.current_position = primary_processed['position']
        self.current_rotation = primary_processed['rotation']

        if merged_pcd.is_empty():
            self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            pass
            return []

        self.current_pcd = merged_pcd.voxel_down_sample(self.pcd_resolution)

        # 缓存主视图的点云和内参：apply_external_instances() 需要用完全相同的
        # 一份数据来回填外部(VLM)给出的检测框，否则两条检测通路落进地图的
        # 三维位置会有系统性偏差。
        self._primary_view_pcd = primary_processed['pcd']
        self._primary_view_intrinsic = primary_processed.get('camera_intrinsic', self.camera_intrinsic)

        if do_instance_segmentation:
            instances = instance_segmentation(self.current_rgb)
        else:
            instances = []

        if instances:
            self.fuse_instances(instances)

        # 按需对侧方视角也跑检测。每个视角必须用它自己的 depth/点云/内参 做反投影，
        # 否则物体会被放到主视图的方向上去，所以这里逐个换掉 _primary_view_* 缓存
        # 再调 fuse_instances，结束后恢复。
        n_side = 0
        n_side_views = 0
        if do_instance_segmentation and segment_all_views and secondary_processed:
            # 位姿也必须一起换！
            #
            # get_object_entities() 内部是用 self.current_position /
            # self.current_rotation 做反投影的。第一版只换了 depth/点云/内参，
            # 位姿仍是主视图的，结果左视图里 3m 外的床被当成"正前方 3m"放进
            # 地图——方向完全错，物体标签成片落在 agent 从没看过的白色区域里。
            saved = (self.current_rgb, self.current_depth,
                     self._primary_view_pcd, self._primary_view_intrinsic,
                     self.current_position, self.current_rotation)
            try:
                for proc in secondary_processed:
                    if proc['pcd'].is_empty():
                        continue
                    self.current_rgb = proc['rgb']
                    self.current_depth = proc['depth']
                    self._primary_view_pcd = proc['pcd']
                    self._primary_view_intrinsic = proc['camera_intrinsic']
                    self.current_position = proc['position']
                    self.current_rotation = proc['rotation']
                    side_inst = instance_segmentation(self.current_rgb)
                    n_side_views += 1
                    if side_inst:
                        n_side += len(side_inst)
                        self.fuse_instances(side_inst)
            except Exception as e:
                print(f"[mapper] side-view segmentation skipped: {e}")
            finally:
                (self.current_rgb, self.current_depth,
                 self._primary_view_pcd, self._primary_view_intrinsic,
                 self.current_position, self.current_rotation) = saved

        # 检测覆盖的观测点。上一轮改完边缘过滤和侧视角之后，日志里完全看不出
        # 侧视角到底跑没跑、检测数变了没有，只能靠 grounding 结果间接猜。
        print(f"[detect] front={len(instances)} side={n_side}(from {n_side_views} views) "
              f"entities={len(self.object_entities)}")

        self.scene_pcd = gpu_merge_pointcloud(self.current_pcd, self.scene_pcd).voxel_down_sample(self.pcd_resolution)
        if not self.scene_pcd.is_empty():
            self.scene_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] > self.floor_height - 1.0).nonzero()[0])

        if not self.scene_pcd.is_empty():
            self.useful_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] < self.ceiling_height).nonzero()[0])
        else:
            self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if self.enable_stair_detection and (not self.useful_pcd.is_empty()):
            try:
                pts_np = self.useful_pcd.point.positions.cpu().numpy()
                agent_xy = self.current_position[:2]
                r = float(self.local_stair_radius + 1)
                dx = pts_np[:, 0] - agent_xy[0]
                dy = pts_np[:, 1] - agent_xy[1]
                local_mask = (dx * dx + dy * dy) <= (r * r)
                local_idx = np.where(local_mask)[0]

                if local_idx.size >= 100:
                    local_idx_tensor = o3d.core.Tensor(local_idx, dtype=o3d.core.Dtype.Int64,
                                                       device=self.useful_pcd.device)
                    local_pcd = self.useful_pcd.select_by_index(local_idx_tensor)
                    current_stair_pcd = self._identify_and_process_stairs(local_pcd)
                    local_pcd.clear()
                    del local_pcd
                else:
                    current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            except Exception as _e:
                current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        else:
            current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if not current_stair_pcd.is_empty():
            pts = current_stair_pcd.point.positions.cpu().numpy()
            agent_xy = self.current_position[:2]
            dx = pts[:, 0] - agent_xy[0]
            dy = pts[:, 1] - agent_xy[1]
            local_mask = (dx * dx + dy * dy) <= (self.local_stair_radius ** 2)
            local_count = int(np.count_nonzero(local_mask))
            if local_count >= self.local_stair_min_points:
                pass
                self.stair_pcd = gpu_merge_pointcloud(self.stair_pcd, current_stair_pcd).voxel_down_sample(
                    self.pcd_resolution)

        original_navigable_pcd = self.current_pcd.select_by_index(
            (self.current_pcd.point.positions[:, 2] < self.floor_height + 0.2).nonzero()[0])

        current_navigable_point = gpu_merge_pointcloud(original_navigable_pcd, current_stair_pcd).voxel_down_sample(
            self.grid_resolution)

        current_navigable_position = current_navigable_point.point.positions.cpu().numpy()

        if current_navigable_position.shape[0] == 0:
            self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        else:
            standing_position = np.array(
                [self.current_position[0], self.current_position[1], current_navigable_position[:, 2].mean()])
            interpolate_points = np.linspace(np.ones_like(current_navigable_position) * standing_position,
                                             current_navigable_position, 20).reshape(-1, 3)
            height_diff = 0.4
            interpolate_points = interpolate_points[
                (interpolate_points[:, 2] > self.floor_height - height_diff) & (
                        interpolate_points[:, 2] < self.floor_height + height_diff)]
            if interpolate_points.shape[0] > 0:
                interpolate_colors = np.ones_like(interpolate_points) * 100
                try:
                    self.current_view_navigable_pcd = gpu_pointcloud_from_array(interpolate_points, interpolate_colors,
                                                                                self.pcd_device).voxel_down_sample(
                        self.grid_resolution)
                    if not current_stair_pcd.is_empty():
                        self.current_view_navigable_pcd = gpu_merge_pointcloud(
                            self.current_view_navigable_pcd, current_stair_pcd
                        ).voxel_down_sample(self.grid_resolution)
                    self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,
                                                              self.current_view_navigable_pcd).voxel_down_sample(
                        self.pcd_resolution)
                except Exception as e:
                    print(f"Error in merging navigable pointcloud: {e}")
                    self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                    if not self.useful_pcd.is_empty():
                        self.navigable_pcd = self.useful_pcd.select_by_index(
                            (self.useful_pcd.point.positions[:, 2] < self.floor_height + 0.2).nonzero()[0]
                        )
            else:
                self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if not self.useful_pcd.is_empty():
            self.obstacle_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] > self.floor_height + 0.2).nonzero()[0])
        else:
            self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self.trajectory_pcd = gpu_pointcloud_from_array(np.array(self.trajectory_position),
                                                        np.zeros((len(self.trajectory_position), 3)), self.pcd_device)
        if self.navigable_pcd.is_empty() and not self.useful_pcd.is_empty():
            self.navigable_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] < curr_agent_height + 0.2).nonzero()[0])

        self.update_iterations += 1

        try:
            if 'original_navigable_pcd' in locals():
                original_navigable_pcd.clear(); del original_navigable_pcd
            if 'current_navigable_point' in locals():
                current_navigable_point.clear(); del current_navigable_point
            if 'current_stair_pcd' in locals():
                current_stair_pcd.clear(); del current_stair_pcd

            if isinstance(self.current_pcd, o3d.t.geometry.PointCloud):
                self.current_pcd.clear()
                self.current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

            if isinstance(self.useful_pcd, o3d.t.geometry.PointCloud):
                self.useful_pcd.clear()
                self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        except Exception:
            pass

        if (self.update_iterations % 5) == 0:
            self.release_vram()

        return instances

    def update(self, rgb, depth, position, rotation, do_instance_segmentation=True):

        t0 = time.time()

        self.current_position = self.translation_func(position) - self.initial_position
        self.current_rotation = self.rotation_func(rotation)
        curr_agent_height = self.current_position[2] + self.initial_floor_height
        self._detect_and_update_floor()
        t1 = time.time()

        self.current_depth = preprocess_depth(depth)
        self.current_rgb = preprocess_image(rgb)[:, :, :3]

        if len(self.trajectory_position) == 0 or not np.allclose(self.trajectory_position[-1], self.current_position,
                                                                 atol=1e-1):

            if isinstance(self.current_position, np.ndarray):
                self.trajectory_position.append(self.current_position.copy())
            else:
                self.trajectory_position.append(np.array(self.current_position))

        if np.sum(self.current_depth) > 0:
            camera_points, camera_colors = get_pointcloud_from_depth(self.current_rgb, self.current_depth,
                                                                     self.camera_intrinsic)
            world_points = translate_to_world(camera_points, self.current_position, self.current_rotation)
            self.current_pcd = gpu_pointcloud_from_array(world_points, camera_colors,
                                                         self.pcd_device).voxel_down_sample(self.pcd_resolution)
            # 单视图路径没有 primary/secondary 之分，当前点云就是主视图点云
            self._primary_view_pcd = self.current_pcd
            self._primary_view_intrinsic = self.camera_intrinsic
        else:

            self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            pass
            return
        t2 = time.time()

        if do_instance_segmentation:
            instances = instance_segmentation(self.current_rgb)
        else:
            instances = []
        t3 = time.time()

        if instances:
            classes = [inst['class_id'] for inst in instances]
            class_names = [inst['class_name'] for inst in instances]
            masks = [inst['mask'] for inst in instances]
            confidences = [1.0] * len(classes)
            t4 = time.time()
            current_entities = self.get_object_entities(self.current_depth, classes, class_names, masks, confidences)
            t5 = time.time()
            self.object_entities = self.associate_object_entities(self.object_entities, current_entities)
            t6 = time.time()
        else:
            t4 = t5 = t6 = time.time()

        self.scene_pcd = gpu_merge_pointcloud(self.current_pcd, self.scene_pcd).voxel_down_sample(self.pcd_resolution)
        if not self.scene_pcd.is_empty():
            self.scene_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] > self.floor_height - 1.0).nonzero()[0])

        if not self.scene_pcd.is_empty():
            self.useful_pcd = self.scene_pcd.select_by_index(
                (self.scene_pcd.point.positions[:, 2] < self.ceiling_height).nonzero()[0])
        else:
            self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            self.navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if self.enable_stair_detection and (not self.useful_pcd.is_empty()):
            try:
                pts_np = self.useful_pcd.point.positions.cpu().numpy()
                agent_xy = self.current_position[:2]
                r = float(self.local_stair_radius + 1)
                dx = pts_np[:, 0] - agent_xy[0]
                dy = pts_np[:, 1] - agent_xy[1]
                local_mask = (dx * dx + dy * dy) <= (r * r)
                local_idx = np.where(local_mask)[0]

                if local_idx.size >= 100:
                    local_idx_tensor = o3d.core.Tensor(local_idx, dtype=o3d.core.Dtype.Int64,
                                                       device=self.useful_pcd.device)
                    local_pcd = self.useful_pcd.select_by_index(local_idx_tensor)
                    current_stair_pcd = self._identify_and_process_stairs(local_pcd)

                    local_pcd.clear()
                    del local_pcd
                else:
                    current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
            except Exception as _e:
                current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        else:
            current_stair_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if not current_stair_pcd.is_empty():
            pts = current_stair_pcd.point.positions.cpu().numpy()
            agent_xy = self.current_position[:2]
            dx = pts[:, 0] - agent_xy[0]
            dy = pts[:, 1] - agent_xy[1]
            local_mask = (dx * dx + dy * dy) <= (self.local_stair_radius ** 2)
            local_count = int(np.count_nonzero(local_mask))
            if local_count >= self.local_stair_min_points:
                pass
                self.stair_pcd = gpu_merge_pointcloud(self.stair_pcd, current_stair_pcd).voxel_down_sample(
                    self.pcd_resolution)

        original_navigable_pcd = self.current_pcd.select_by_index(
            (self.current_pcd.point.positions[:, 2] < self.floor_height + 0.1).nonzero()[0])

        current_navigable_point = gpu_merge_pointcloud(original_navigable_pcd, current_stair_pcd).voxel_down_sample(self.pcd_resolution)

        current_navigable_position = current_navigable_point.point.positions.cpu().numpy()

        if current_navigable_position.shape[0] == 0:

            self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        else:
            standing_position = np.array(
                [self.current_position[0], self.current_position[1], current_navigable_position[:, 2].mean()])
            interpolate_points = np.linspace(np.ones_like(current_navigable_position) * standing_position,
                                             current_navigable_position, 50).reshape(-1, 3)
            height_diff = 0.4
            interpolate_points = interpolate_points[
                (interpolate_points[:, 2] > self.floor_height - height_diff) & (
                        interpolate_points[:, 2] < self.floor_height + height_diff)]
            if interpolate_points.shape[0] > 0:
                interpolate_colors = np.ones_like(interpolate_points) * 100
                try:

                    self.current_view_navigable_pcd = gpu_pointcloud_from_array(interpolate_points, interpolate_colors,
                                                                                self.pcd_device).voxel_down_sample(
                        self.grid_resolution)

                    if 'current_stair_pcd' in locals() and not current_stair_pcd.is_empty():
                        self.current_view_navigable_pcd = gpu_merge_pointcloud(
                            self.current_view_navigable_pcd, current_stair_pcd
                        ).voxel_down_sample(self.grid_resolution)

                    self.navigable_pcd = gpu_merge_pointcloud(self.navigable_pcd,
                                                              self.current_view_navigable_pcd).voxel_down_sample(
                        self.pcd_resolution)
                except Exception as e:
                    print(f"Error in merging navigable pointcloud: {e}")
                    self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
                    if not self.useful_pcd.is_empty():
                        self.navigable_pcd = self.useful_pcd.select_by_index(
                            (self.useful_pcd.point.positions[:, 2] < self.floor_height + 0.1).nonzero()[0]
                        )
            else:
                self.current_view_navigable_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        if not self.useful_pcd.is_empty():
            self.obstacle_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] > self.floor_height + 0.1).nonzero()[0])
        else:
            self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        # if not self.obstacle_pcd.is_empty() and not stair_pcd.is_empty():
        #     dist_to_stairs = pointcloud_distance(self.obstacle_pcd, stair_pcd)

        #     not_stair_mask = dist_to_stairs > self.pcd_resolution
        #     if not_stair_mask.sum() > 0:
        #         indices_tensor = o3d.core.Tensor(not_stair_mask.cpu().numpy(), device=self.pcd_device).nonzero()[0]
        #         self.obstacle_pcd = self.obstacle_pcd.select_by_index(indices_tensor)
        #     else:
        #         self.obstacle_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self.trajectory_pcd = gpu_pointcloud_from_array(np.array(self.trajectory_position),
                                                        np.zeros((len(self.trajectory_position), 3)), self.pcd_device)
        if self.navigable_pcd.is_empty() and not self.useful_pcd.is_empty():
            self.navigable_pcd = self.useful_pcd.select_by_index(
                (self.useful_pcd.point.positions[:, 2] < curr_agent_height + 0.1).nonzero()[0])

        # if not self.navigable_pcd.is_empty() and not self.obstacle_pcd.is_empty():
        #     self.frontier_pcd = project_frontier(self.obstacle_pcd, self.navigable_pcd, self.floor_height + 0.2,
        #                                          self.grid_resolution)
        #     if self.frontier_pcd.shape[0] > 0:
        #         self.frontier_pcd[:, 2] = self.navigable_pcd.point.positions.cpu().numpy()[:, 2].mean()
        #         self.frontier_pcd = gpu_pointcloud_from_array(self.frontier_pcd,
        #                                                       np.ones((self.frontier_pcd.shape[0], 3)) * np.array(
        #                                                           [[255, 0, 0]]), self.pcd_device)
        #     else:
        #         self.frontier_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        # else:
        #     self.frontier_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

        self.update_iterations += 1
        t7 = time.time()

        # ========================

        # ========================
        try:
            if 'original_navigable_pcd' in locals():
                original_navigable_pcd.clear(); del original_navigable_pcd
            if 'current_navigable_point' in locals():
                current_navigable_point.clear(); del current_navigable_point
            if 'current_stair_pcd' in locals():
                current_stair_pcd.clear(); del current_stair_pcd

            if isinstance(self.current_pcd, o3d.t.geometry.PointCloud):
                self.current_pcd.clear()
                self.current_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

            if isinstance(self.useful_pcd, o3d.t.geometry.PointCloud):
                self.useful_pcd.clear()
                self.useful_pcd = o3d.t.geometry.PointCloud(self.pcd_device)
        except Exception:
            pass

        if (self.update_iterations % 5) == 0:
            self.release_vram()

        # print(f"  get_object_entities: {t5 - t4:.3f}s")
        # print(f"  associate_object_entities: {t6 - t5:.3f}s")

        return instances

    def get_object_entities(self, depth, classes, class_names, masks, confidences, camera_intrinsic=None):
        entities = []

        object_height_threshold = self.ceiling_height - 0.2
        intrinsic = self.camera_intrinsic if camera_intrinsic is None else camera_intrinsic

        for cls, cls_name, mask, score in zip(classes, class_names, masks, confidences):
            if depth[mask > 0].min() < 1.0 and score < 0.5:
                continue

            camera_points = get_pointcloud_from_depth_mask(depth, mask, intrinsic)
            world_points = translate_to_world(camera_points, self.current_position, self.current_rotation)

            color = get_class_color(cls)
            point_colors = np.array([color] * world_points.shape[0])
            if world_points.shape[0] < 10:
                continue
            object_pcd = gpu_pointcloud_from_array(world_points, point_colors, self.pcd_device).voxel_down_sample(
                self.pcd_resolution * 2)
            object_pcd = gpu_cluster_filter(object_pcd)
            if object_pcd.point.positions.shape[0] < 10:
                continue

            center = object_pcd.point.positions.mean(dim=0).cpu().numpy()

            if center[2] > object_height_threshold:
                continue

            entity = {'class': cls, 'class_name': cls_name, 'pcd': object_pcd, 'confidence': score, 'center': center}
            entities.append(entity)
        return entities

    def associate_object_entities(self, ref_entities, eval_entities):
        if len(ref_entities) == 0:
            return eval_entities

        ref_centers = np.array([entity['center'] for entity in ref_entities])
        ref_indices_to_remove = []

        for entity in eval_entities:
            eval_center = entity['center']
            eval_pcd = entity['pcd']
            eval_class = entity['class']
            eval_pcd_size = eval_pcd.point.positions.shape[0]

            center_distances = np.linalg.norm(ref_centers - eval_center, axis=1)
            distance_threshold = 1.0
            candidate_indices = np.where(center_distances < distance_threshold)[0]
            candidate_indices_2 = np.where(center_distances < distance_threshold * 0.5)[0]

            if len(candidate_indices) == 0:
                ref_entities.append(entity)
                ref_centers = np.vstack([ref_centers, eval_center])
                continue

            same_class_candidates = []
            overlap_scores = []
            remaining_pcd = eval_pcd

            for idx in candidate_indices:
                if ref_entities[idx]['class'] == eval_class:
                    same_class_candidates.append(idx)
                    if remaining_pcd.is_empty(): break

                    cdist = pointcloud_distance(remaining_pcd, ref_entities[idx]['pcd'])
                    overlap_condition = (cdist < 0.1)
                    overlap_ratio = overlap_condition.sum().cpu().numpy() / (overlap_condition.shape[0] + 1e-6)
                    overlap_scores.append(overlap_ratio)

                    nonoverlap_condition = overlap_condition.logical_not()
                    if nonoverlap_condition.sum() > 0:
                        remaining_pcd = remaining_pcd.select_by_index(
                            o3d.core.Tensor(nonoverlap_condition.cpu().numpy(), device=self.pcd_device).nonzero()[0]
                        )
                    else:
                        remaining_pcd = o3d.t.geometry.PointCloud(self.pcd_device)

            merged_with_same_class = False
            if len(overlap_scores) > 0:
                max_overlap_score = np.max(overlap_scores)
                if max_overlap_score >= 0.25:
                    best_candidate_idx = same_class_candidates[np.argmax(overlap_scores)]
                    best_entity = ref_entities[best_candidate_idx]
                    best_entity['pcd'] = gpu_merge_pointcloud(best_entity['pcd'], remaining_pcd)
                    if not best_entity['pcd'].is_empty():
                        best_entity['center'] = best_entity['pcd'].point.positions.mean(dim=0).cpu().numpy()
                        ref_centers[best_candidate_idx] = best_entity['center']
                    ref_entities[best_candidate_idx] = best_entity
                    merged_with_same_class = True

            if merged_with_same_class:
                continue

            merged_with_different_class = False
            for idx in candidate_indices_2:
                if ref_entities[idx]['class'] != eval_class:
                    ref_entity = ref_entities[idx]
                    ref_pcd_size = ref_entity['pcd'].point.positions.shape[0]

                    if ref_pcd_size == 0 or eval_pcd_size == 0: continue
                    size_ratio = max(ref_pcd_size, eval_pcd_size) / min(ref_pcd_size, eval_pcd_size)
                    if not (0.6 < size_ratio < 1.4):
                        continue

                    cdist = pointcloud_distance(eval_pcd, ref_entity['pcd'])
                    overlap_ratio = (cdist < 0.1).sum().cpu().numpy() / (eval_pcd_size + 1e-6)

                    if overlap_ratio > 0.6:

                        ref_entity['pcd'] = gpu_merge_pointcloud(ref_entity['pcd'], eval_pcd)
                        ref_entity['class'] = eval_class
                        ref_entity['class_name'] = entity['class_name']
                        if not ref_entity['pcd'].is_empty():
                            ref_entity['center'] = ref_entity['pcd'].point.positions.mean(dim=0).cpu().numpy()
                            ref_centers[idx] = ref_entity['center']
                        ref_entities[idx] = ref_entity
                        merged_with_different_class = True
                        break

            if merged_with_different_class:
                continue

            if not remaining_pcd.is_empty() and remaining_pcd.point.positions.shape[0] >= 15:
                entity['pcd'] = remaining_pcd
                ref_entities.append(entity)
                ref_centers = np.vstack([ref_centers, entity['center']])

        return ref_entities

    def update_object_pcd(self):
        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        try:
            return gpu_pointcloud(object_pcd, self.pcd_device)
        except:
            return self.scene_pcd

    def get_view_pointcloud(self, rgb, depth, translation, rotation):
        current_position = self.translation_func(translation) - self.initial_position
        current_rotation = self.rotation_func(rotation)
        current_depth = preprocess_depth(depth)
        current_rgb = preprocess_image(rgb)
        camera_points, camera_colors = get_pointcloud_from_depth(current_rgb, current_depth, self.camera_intrinsic)
        world_points = translate_to_world(camera_points, current_position, current_rotation)
        current_pcd = gpu_pointcloud_from_array(world_points, camera_colors, self.pcd_device).voxel_down_sample(
            self.pcd_resolution)
        return current_pcd

    def get_obstacle_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd, self.obstacle_pcd)
            affordance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance < 0.25] = 0
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_trajectory_affordance(self):
        try:
            distance = pointcloud_distance(self.navigable_pcd, self.trajectory_pcd)
            affordance = (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_semantic_affordance(self, target_class, threshold=0.1):
        semantic_pointcloud = o3d.t.geometry.PointCloud()
        for entity in self.object_entities:
            if entity['class'] in target_class:
                semantic_pointcloud = gpu_merge_pointcloud(semantic_pointcloud, entity['pcd'])
        try:
            distance = pointcloud_2d_distance(self.navigable_pcd, semantic_pointcloud)
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > threshold] = 0
            affordance = affordance.cpu().numpy()
            return affordance
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_gpt4v_affordance(self, gpt4v_pcd):
        try:
            distance = pointcloud_distance(self.navigable_pcd, gpt4v_pcd)
            affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
            affordance[distance > 0.1] = 0
            return affordance.cpu().numpy()
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_action_affordance(self, action):
        try:
            if action == 'Explore':
                distance = pointcloud_2d_distance(self.navigable_pcd, self.frontier_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.2] = 0
                return affordance.cpu().numpy()
            elif action == 'Move_Forward':
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, self.current_rotation)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Around':
                R = np.array([np.pi, np.pi, np.pi]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Left':
                R = np.array([np.pi / 2, np.pi / 2, np.pi / 2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Turn_Right':
                R = np.array([-np.pi / 2, -np.pi / 2, -np.pi / 2]) * self.rotate_axis
                turn_extrinsic = np.matmul(self.current_rotation,
                                           quaternion.as_rotation_matrix(quaternion.from_euler_angles(R)))
                pixel_x, pixel_z, depth_values = project_to_camera(self.navigable_pcd, self.camera_intrinsic,
                                                                   self.current_position, turn_extrinsic)
                filter_condition = (pixel_x >= 0) & (pixel_x < self.camera_intrinsic[0][2] * 2) & (pixel_z >= 0) & (
                            pixel_z < self.camera_intrinsic[1][2] * 2) & (depth_values > 1.5) & (depth_values < 2.5)
                filter_pcd = self.navigable_pcd.select_by_index(
                    o3d.core.Tensor(np.where(filter_condition == 1)[0], device=self.navigable_pcd.device))
                distance = pointcloud_distance(self.navigable_pcd, filter_pcd)
                affordance = 1 - (distance - distance.min()) / (distance.max() - distance.min() + 1e-6)
                affordance[distance > 0.1] = 0
                return affordance.cpu().numpy()
            elif action == 'Enter':
                return self.get_semantic_affordance(['doorway', 'door', 'entrance', 'exit'])
            elif action == 'Exit':
                return self.get_semantic_affordance(['doorway', 'door', 'entrance', 'exit'])
            else:
                return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)
        except:
            return np.zeros((self.navigable_pcd.point.positions.shape[0],), dtype=np.float32)

    def get_objnav_affordance_map(self, action, target_class, gpt4v_pcd, complete_flag=False, failure_mode=False):
        if failure_mode:
            obstacle_affordance = self.get_obstacle_affordance()
            affordance = self.get_action_affordance('Explore')
            affordance = np.clip(affordance, 0.1, 1.0)
            affordance[obstacle_affordance == 0] = 0
            return affordance, self.visualize_affordance(affordance)
        elif complete_flag:
            affordance = self.get_semantic_affordance([target_class], threshold=0.1)
            return affordance, self.visualize_affordance(affordance)
        else:
            obstacle_affordance = self.get_obstacle_affordance()
            semantic_affordance = self.get_semantic_affordance([target_class], threshold=1.5)
            action_affordance = self.get_action_affordance(action)
            gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
            history_affordance = self.get_trajectory_affordance()
            affordance = 0.25 * semantic_affordance + 0.25 * action_affordance + 0.25 * gpt4v_affordance + 0.25 * history_affordance
            affordance = np.clip(affordance, 0.1, 1.0)
            affordance[obstacle_affordance == 0] = 0
            return affordance, self.visualize_affordance(affordance / (affordance.max() + 1e-6))

    def get_debug_affordance_map(self, action, target_class, gpt4v_pcd):
        obstacle_affordance = self.get_obstacle_affordance()
        semantic_affordance = self.get_semantic_affordance([target_class], threshold=1.5)
        action_affordance = self.get_action_affordance(action)
        gpt4v_affordance = self.get_gpt4v_affordance(gpt4v_pcd)
        history_affordance = self.get_trajectory_affordance()
        return self.visualize_affordance(semantic_affordance / (semantic_affordance.max() + 1e-6)), \
            self.visualize_affordance(history_affordance / (history_affordance.max() + 1e-6)), \
            self.visualize_affordance(action_affordance / (action_affordance.max() + 1e-6)), \
            self.visualize_affordance(gpt4v_affordance / (gpt4v_affordance.max() + 1e-6)), \
            self.visualize_affordance(obstacle_affordance / (obstacle_affordance.max() + 1e-6))

    def visualize_affordance(self, affordance):
        cmap = colormaps.get('jet')
        color_affordance = cmap(affordance)[:, 0:3]
        color_affordance = cpu_pointcloud_from_array(self.navigable_pcd.point.positions.cpu().numpy(), color_affordance)
        return color_affordance

    def get_appeared_objects(self):
        return [entity['class'] for entity in self.object_entities]

    def save_pointcloud_debug(self, path="./"):
        save_pcd = o3d.geometry.PointCloud()
        try:
            assert self.useful_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.useful_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.useful_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "scene.ply", save_pcd)
        except:
            pass
        try:
            assert self.navigable_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.navigable_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.navigable_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "navigable.ply", save_pcd)
        except:
            pass
        try:
            assert self.obstacle_pcd.point.positions.shape[0] > 0
            save_pcd.points = o3d.utility.Vector3dVector(self.obstacle_pcd.point.positions.cpu().numpy())
            save_pcd.colors = o3d.utility.Vector3dVector(self.obstacle_pcd.point.colors.cpu().numpy())
            o3d.io.write_point_cloud(path + "obstacle.ply", save_pcd)
        except:
            pass

        object_pcd = o3d.geometry.PointCloud()
        for entity in self.object_entities:
            points = entity['pcd'].point.positions.cpu().numpy()
            colors = entity['pcd'].point.colors.cpu().numpy()
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points)
            new_pcd.colors = o3d.utility.Vector3dVector(colors)
            object_pcd = object_pcd + new_pcd
        if len(object_pcd.points) > 0:
            o3d.io.write_point_cloud(path + "object.ply", object_pcd)

    def release_all_vram(self):
        """Release GPU memory and large object refs; safe to call repeatedly."""

        pcd_attr_names = [
            'scene_pcd', 'navigable_pcd', 'obstacle_pcd', 'trajectory_pcd',
            'current_pcd', 'useful_pcd', 'frontier_pcd', 'current_view_navigable_pcd',
            'stair_pcd', 'platform_pcd'
        ]
        for name in pcd_attr_names:
            self._safe_clear_tpcd(getattr(self, name, None))

        try:
            if hasattr(self, 'pcds_per_floor') and isinstance(self.pcds_per_floor, dict):
                for _, data in list(self.pcds_per_floor.items()):
                    for k in ['scene_pcd', 'navigable_pcd', 'obstacle_pcd']:
                        self._safe_clear_tpcd(data.get(k))

                    for ent in data.get('object_entities', []):
                        self._safe_clear_tpcd(ent.get('pcd'))
                    data['object_entities'] = []
                    data['trajectory_position'] = []
                self.pcds_per_floor.clear()
        except Exception:
            pass

        try:
            if hasattr(self, 'object_entities') and isinstance(self.object_entities, list):
                for ent in self.object_entities:
                    try:
                        self._safe_clear_tpcd(ent.get('pcd'))
                    except Exception:
                        pass
                self.object_entities.clear()
        except Exception:
            pass

        for name in ['current_depth', 'current_rgb', 'trajectory_position', 'frontier_pcd']:
            try:
                setattr(self, name, None)
            except Exception:
                pass

        try:
            import torch
            if hasattr(self, 'device') and isinstance(self.device, str) and self.device.lower().startswith('cuda'):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        try:
            gc.collect()
        except Exception:
            pass

    def __enter__(self):
        """Supports with statement."""
        return self

    def __exit__(self, exc_type, exc, tb):
        """Auto-release GPU memory on with exit."""
        try:
            self.release_vram()
        except Exception:
            pass

        return False

    def __del__(self):
        """Fallback GPU memory release on destruction."""
        try:
            self.release_vram()
        except Exception:
            pass
