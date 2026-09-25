#!/usr/bin/env python3
"""
同步自检：确认改动过的文件都完整落到本地了。

动机：同步过程已经出过两次问题，而且症状都很误导——
  1. object_list.py 多了一行不该有的 import，报成"循环导入"
  2. spatio_temporal_decomposer.py 里的类整个不见了，报成"cannot import name"
两次都不是代码逻辑问题，但都要花时间从报错反推。这个脚本直接检查每个文件
该有的顶层定义和关键标记，几秒钟给出结论。

它只读文件、不 import 任何重依赖（habitat / open3d / ultralytics 都不需要），
所以在任何机器上都能跑。

用法：
    python scripts/check_sync.py
"""

import os
import re
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

# 每个文件：必须存在的顶层定义 + 必须出现的关键字符串
#
# agent.py 已按 VLN 四阶段拆成子包，所以这里逐个模块检查——拆分最容易
# 出的错就是某个方法在搬移中掉了，而症状要等到运行时才暴露。
EXPECT = {
    "segmentation/object_list.py": {
        "defs": ["normalize_category_name", "resolve_category", "resolve_area",
                 "_singularize"],
        "marks": ["CATEGORY_NAMES", "category_names_str", "AREA_ALIASES",
                  "AREA_PROXY_OBJECTS", "GENERIC_AREA_TYPES",
                  '"rail": "handrail"', '"lampshade": "lamp"'],
        # 这个文件是叶子模块，被 mapper/agent/decomposer 依赖，
        # 一旦它 import 项目内其它模块就会成环
        "no_project_imports": True,
    },
    "segmentation/instance_segmentation.py": {
        "defs": ["instance_segmentation", "get_class_color", "set_model_path", "_get_model"],
        "marks": ["w * 0.05", "conf=0.45", "STZS_PATH_YOLOE_CKPT"],
    },
    "config_utils.py": {
        "defs": ["hyper", "paths", "resolve_path", "_cast_like"],
        "classes": ["Section"],
        "marks": ["ENV_PREFIX", "describe"],
        # 配置层必须是叶子：任何模块都能 import 它，它不能 import 任何模块
        "no_project_imports": True,
    },
    "agent/decomposer/spatio_temporal_decomposer.py": {
        "classes": ["SpatioTemporalInstructionDecomposer"],
        "marks": ["_STAGE_CONNECTIVES", "_NOUN_PHRASE", "_TRIM_TRAILING_STOPWORD",
                  "(?!the\\b|a\\b|an\\b)", "slightly", "boundary_consistent"],
    },
    "mapper.py": {
        "defs": ["ground_landmarks", "infer_area_regions", "current_room",
                 "classify_local_space", "segment_rooms", "compute_frontiers",
                 "fuse_instances", "apply_external_instances", "get_avoid_zones",
                 "_avoid_penalty", "_build_free_grid", "_region_at", "_apply_hyper"],
        "marks": ["MORPH_OPEN", "self.current_position = proc['position']",
                  "import cv2", "ROOM_MIN_EVIDENCE", "from config_utils import hyper"],
    },

    # ---- 指令分解 ----
    "agent/decomposer/instruction.py": {
        "classes": ["Instruction"],
        "defs": ["get_progress", "get_current_step_budget", "mark_current_subtask_completed",
                 "roll_back_to_specific_subtask"],
    },
    "agent/decomposer/decomposition.py": {
        "classes": ["DecompositionMixin"],
        "defs": ["decompose_instruction"],
    },

    # ---- 感知 ----
    "agent/perception/mapping.py": {
        "classes": ["MappingMixin"],
        "defs": ["update_map", "_needs_side_segmentation", "_apply_vlm_detections"],
    },
    "agent/perception/grounding.py": {
        "classes": ["GroundingMixin"],
        "defs": ["_ground_current_landmarks", "_reset_landmark_progress",
                 "_track_landmark_progress"],
        "marks": ["ever_passed", "NEAR_M", "PASSED_MIN_M", "PASSED_GAP_M"],
    },
    "agent/perception/topdown_map.py": {
        "defs": ["create_top_down_map_centered", "create_top_down_map_global",
                 "draw_direction_markers"],
        "marks": ["rejected_waypoints", "space_labels", "scored_frontiers"],
    },

    # ---- 规划 ----
    "agent/planning/waypoint.py": {
        "classes": ["WaypointMixin"],
        "defs": ["_preprocessing_module", "decide_waypoint", "_world_to_pixel_coords",
                 "plan_rollback_path"],
        "marks": ["subtask_done"],
    },
    "agent/planning/frontier.py": {
        "classes": ["FrontierMixin"],
        "defs": ["_score_frontiers"],
        "marks": ["FRONTIER_MIN_USEFUL_M"],
    },
    "agent/planning/verification.py": {
        "classes": ["VerificationMixin"],
        "defs": ["_verify_local_end", "_challenge_out_of_order", "_is_too_far_to_stop",
                 "_has_receded_from_destination", "_maybe_force_advance",
                 "_is_premature_completion", "_challenge_receded_stop",
                 "_current_budget_row"],
        "marks": ["LAST CHANCE", "NOT DONE YET"],
    },
    "agent/planning/prompt.py": {
        "classes": ["PromptMixin", "_SkipGuidance"],
        "defs": ["generate_prompt"],
        "marks": ["ARRIVED?", "scored_frontiers"],
    },

    # ---- 行动 ----
    "agent/action/executor.py": {
        "classes": ["ActionMixin"],
        "defs": ["step", "execute_action", "create_action", "_get_action_for_next_waypoint",
                 "_action_to_text", "_stop"],
        # 拦截时不清空路径：清空会让 agent 原地重规划，白花一次 VLM 调用
        "marks": ["challenge_stop", "self.current_path"],
    },

    # ---- VLM 接口 ----
    "agent/llm/client.py": {
        "classes": ["LLMMixin"],
        "defs": ["_get_vlm_response", "_get_vlm_response_multiview", "_get_llm_response"],
    },

    # ---- 装配 ----
    "agent/agent.py": {
        "classes": ["PathPlannerAgent", "GPTAgent"],
        "marks": ["DecompositionMixin", "MappingMixin", "GroundingMixin", "WaypointMixin",
                  "FrontierMixin", "VerificationMixin", "PromptMixin", "ActionMixin",
                  "LLMMixin", "budget_ratio", "_DETECTION_SIDE_TASK",
                  "from openai import AzureOpenAI, OpenAI"],
    },
    "agent/common.py": {
        "defs": ["robust_json_parse", "_setup_proxy"],
    },

    "run_experiments.py": {
        "marks": ["[episode] EXCEPTION", "sim_cfg.setdefault('max_steps'",
                  "from config_utils import paths as cfg_paths, resolve_path",
                  'sim_config["hyper"]', "resolve_path(self.paths"],
        # 绝对路径必须全部进配置文件
        "no_abs_paths": True,
    },
}


def main():
    bad = []
    for rel, spec in EXPECT.items():
        path = os.path.join(SRC, rel)
        if not os.path.exists(path):
            print(f"[MISSING] {rel}")
            bad.append(rel)
            continue
        src = open(path, encoding="utf-8").read()

        miss_def = [d for d in spec.get("defs", [])
                    if not re.search(rf"^\s*def {re.escape(d)}\s*\(", src, re.M)]
        miss_cls = [c for c in spec.get("classes", [])
                    if not re.search(rf"^\s*class {re.escape(c)}\b", src, re.M)]
        miss_mark = [m for m in spec.get("marks", []) if m not in src]

        abs_paths = []
        if spec.get("no_abs_paths"):
            # 允许两种绝对路径：注释里的，以及 resolve_path() 的同值兜底
            # （保留兜底是为了让旧配置文件照样能跑）。resolve_path 常跨行写，
            # 所以要往前看几行而不是只看当前行。
            src_lines = src.splitlines()
            for i, line in enumerate(src_lines):
                if "/home/" not in line or line.lstrip().startswith("#"):
                    continue
                window = "\n".join(src_lines[max(0, i - 3):i + 1])
                if "resolve_path" in window or ".get(" in window:
                    continue
                abs_paths.append(line.strip()[:70])

        stray = []
        if spec.get("no_project_imports"):
            for m in re.finditer(r"^\s*(?:from|import)\s+(\S+)", src, re.M):
                mod = m.group(1).split(".")[0]
                if mod in ("segmentation", "mapper", "agent", "utils",
                           "mapping_utils", "simWrapper"):
                    stray.append(m.group(0).strip())

        problems = []
        if miss_cls:
            problems.append(f"缺类 {miss_cls}")
        if miss_def:
            problems.append(f"缺函数 {miss_def}")
        if miss_mark:
            problems.append(f"缺标记 {miss_mark}")
        if stray:
            problems.append(f"不该有的项目内 import {stray}")
        if abs_paths:
            problems.append(f"仍有写死的绝对路径 {abs_paths[:3]}（应移入 config 的 paths: 段）")

        if problems:
            print(f"[BAD ] {rel}\n       " + "\n       ".join(problems))
            bad.append(rel)
        else:
            print(f"[ OK ] {rel}  ({len(src.splitlines())} 行)")

    print()
    if bad:
        print(f"{len(bad)} 个文件不完整，重新同步这些再跑：")
        for b in bad:
            print(f"  src/{b}")
        return 1
    print("全部文件完整。可以运行 python scripts/test_landmark_grounding.py 做逻辑自检。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
