"""
规划 · prompt 组装。

_SkipGuidance 定义在这里：引导段落在证据不足时整段跳过，用异常比层层
if 判断更干净——组装到一半发现前提不成立就直接放弃这一段。
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


class _SkipGuidance(Exception):
    """引导段落的证据不足，跳过整段而不是拼一段半成品出来。"""
    pass



class PromptMixin:
    def generate_prompt(self, instruction: str = None, is_stuck: bool = False, is_first_step=False) -> str:
        """
        Generate an English prompt describing agent state, map, and available actions.
        This prompt is optimized for PathPlannerAgent decision logic.
        """
        pos = self.mapper.current_position
        pos_str = f"Your current coordinate (z is height): (x: {pos[0]:.2f}, y: {pos[1]:.2f}, z: {pos[2]:.2f})"

        # detections 字段只在开关打开时才出现在 schema 里。关掉时连字段都不提，
        # 避免模型仍然输出一堆用不上的框（纯输出 token 浪费）。
        det_schema = ""
        if self.enable_vlm_detection:
            det_schema = ('        "detections": [  // Objects visible in the FRONT VIEW. '
                          'c=category from the fixed vocabulary, b=[xmin,ymin,xmax,ymax] in '
                          '0-1000 coords. Max 10, [] if none.\n'
                          '            {"c": "sofa", "b": [23, 366, 394, 552]}\n'
                          '        ]\n')

        after_stop_str = """**IMPORTANT: Blue arc indicates a 1-meter range around the agent. Check again if you are as close as possible to the target position(<1m). If not, continue to get closer. If you are close enough, return "action": -1 to stop.**"""

        # 步数预算自省：把"这个子任务预计花多少步、已经花了多少"直接告诉VLM。
        # relative_duration 是分解器给的相对时长（归一化求和为1），这里换算成
        # 步数。超预算时措辞加重，让它倾向于收敛而不是继续漫游。
        budget_str = ""
        over_budget = False
        try:
            rows = self.instruction_obj.get_progress(self.total_step_budget)
            cur = next((r for r in rows if r['status'] == 'current'), None)
            if cur:
                pct = int(round(cur['ratio'] * 100))
                budget_str = (f"\nBudget: {cur['key']} is expected to take about "
                              f"{cur['expected_steps']:.0f} steps; you have used "
                              f"{cur['used_steps']} ({pct}%).")
                over_budget = cur['ratio'] > 1.5
        except Exception:
            budget_str = ""

        # 状态机滞后兜底。两个独立信号，任一成立就提示：
        #
        #   a) 步数超预算    —— 永远可用，不依赖任何landmark
        #   b) landmark 变远 —— 更精确，但要求landmark能接地
        #
        # 早先只用 (b)，结果在实跑里完全没触发过：那个 episode 的 SUBTASK_1 是
        # "Go out of the room you're in"，唯一landmark是 room，属于 area_no_proxy
        # 永远接不了地，于是 grounded 列表为空、检测器直接 return。
        # 最需要兜底的场景（landmark 是房间/走廊）恰恰是兜底失灵的场景，所以
        # 现在把不依赖接地的 (a) 作为主信号，(b) 只是增强。
        # 提前宣告完成的质询文本，只插一轮，取走即清空
        challenge_str = getattr(self, 'stop_challenge', "") or ""
        self.stop_challenge = ""

        # 当前子任务的显式终止条件。分解器一直在产出这个字段，之前只存不读，
        # 完成与否全凭VLM自己解释指令——现在把判据摆出来。
        end_str = ""
        try:
            _k, _st = self.instruction_obj.get_current_subtask()
            _e = (_st or {}).get('local_end')
            if _e and _e != 'unknown':
                end_str = f"\nCompletion condition for {_k}: \"{_e}\""
        except Exception:
            end_str = ""

        # 探索引导：把前沿打分结果说成人话。
        #
        # 这是唯一一处代码给出**方向建议**而不是罗列事实的地方。此前选 frontier
        # 完全是VLM的活，代码一点先验都不提供——而实测 72% 的失败是"根本没到
        # 目标区域"。
        explore_str = ""
        try:
            if not self.enable_guidance:
                self._last_scored_frontiers = None
                raise _SkipGuidance
            # 目标已经找到就不要再劝人去探索。
            #
            # 实测教训：_score_frontiers 在所有目标都接地之后会退化成"哪个开口
            # 最大往哪走"，却仍然顶着"ranked by how likely they lead to what this
            # subtask needs"的标题送进 prompt。结果目标在 0.4m 外、系统还在推荐
            # 去探索未知区域——同一批 9 个 episode 里 4 个曾进到 1.5m 内，最后
            # 全停在刚超出 3m 阈值的地方，行程比上一版多了 39%。
            # 引导和到达是互斥的信号，不该同时出现。
            dest_found = any(
                g.get('role') == 'destination_marker'
                and float(g.get('distance', 99.0)) <= 4.0
                for g in (self.mapper.grounded_landmarks or []))

            sf = None if dest_found else getattr(self, '_last_scored_frontiers', None)
            if dest_found:
                self._last_scored_frontiers = None
            elif sf is None:
                sf = self._score_frontiers()
                self._last_scored_frontiers = sf
            # 没有信息时保持沉默。
            #
            # 实测教训：目标房间还没识别出来时（这是常态），打分退化成"哪个开口
            # 最大"，和指令毫无关系，却仍以"最可能通向目标的方向"的口吻送出。
            # VLM 很信它（一次运行里 F1 被引用 169 次），于是放弃自己的视觉判断
            # 去追最大的洞。同一批 22 个 episode，oracle SR 从 40.9% 掉到 31.8%
            # ——不是探索过头，是被引导到了错的地方。
            # 宁可不说，也不要给一个听起来很确定的默认值。
            if sf and not sf[0].get('informed'):
                print(f"[hint] EXPLORE suppressed (no room prior; would have been "
                      f"{sf[0]['bearing']} {sf[0]['distance']:.0f}m)")
                sf = None
                self._last_scored_frontiers = None

            if sf:
                lines = [f"  F{i+1} ({f['bearing']}, {f['distance']:.0f}m): {f['why']}"
                         for i, f in enumerate(sf)]
                if self.enable_frontier_action:
                    head = ("\nUnexplored openings, ranked by how likely they lead to what this "
                            "subtask needs (F1 is the system's best guess). **These are selectable "
                            "actions** — reply with \"action\": \"F1\" and a path will be planned all "
                            "the way there, which is usually far more efficient than nudging "
                            "forward one waypoint at a time when the thing you need is not in "
                            "sight yet:\n")
                else:
                    # 动作没开的时候不能说"可以选"，否则模型会输出一个不存在的动作，
                    # 触发重试、浪费一次调用
                    head = ("\nUnexplored openings, ranked by how likely they lead to what this "
                            "subtask needs (for reference — steer toward it using the numbered "
                            "waypoints):\n")
                explore_str = head + "\n".join(lines)
                print(f"[hint] EXPLORE top={sf[0]['bearing']} {sf[0]['distance']:.0f}m "
                      f"score={sf[0]['score']:.2f} ({sf[0]['why'][:50]})")
        except _SkipGuidance:
            explore_str = ""
        except Exception:
            self._last_scored_frontiers = None
            explore_str = ""

        # 主动到达提示。
        #
        # 现有的拦截全挂在"VLM 想停"这个事件上，从不停的 episode 一次都触发
        # 不了——实测有个 episode 曾在离目标 0.10m 处经过，最后停在 8.34m 外，
        # 状态是 max_steps，所有 guard 全程没机会运行。这条是反方向的：
        # 事实已经算出来了（终点已接地、就在旁边），只是从来没人送到决策者面前。
        arrive_str = ""
        try:
            near = [g for g in (self.mapper.grounded_landmarks or [])
                    if g.get('role') == 'destination_marker'
                    and float(g.get('distance', 99.0)) <= self.NEAR_M]
            if near:
                g = min(near, key=lambda x: float(x.get('distance', 99.0)))
                cur_key = self.instruction_obj.get_current_subtask_key()
                how = ("\"action\": -1" if self.instruction_obj.is_last_subtask()
                       else "\"subtask_done\": true")
                arrive_str = (
                    f"\n**ARRIVED?: '{g.get('name','')}' — the destination of {cur_key} — is "
                    f"{float(g['distance']):.1f}m away, i.e. you are already next to it. If this "
                    f"is what the instruction meant, set {how} NOW rather than exploring further "
                    "(do NOT pick an F option). Only keep going if this is clearly the wrong "
                    "object.**")
                # 这些提示只进 prompt，日志里看不见，导致完全无法判断它有没有触发过。
                # 同一个观测盲区在这个项目里已经出现第四次了，这次直接打出来。
                print(f"[hint] ARRIVED {cur_key}: '{g.get('name','')}' at "
                      f"{float(g['distance']):.1f}m")
        except Exception:
            arrive_str = ""

        receding = getattr(self, 'passed_landmark_hint', False)
        desync_str = ""
        if over_budget or receding:
            if receding and over_budget:
                why = ("you are over the step budget for this subtask AND you have been moving "
                       "AWAY from its landmarks for several steps")
            elif receding:
                why = ("you have been moving AWAY from the landmarks of the current subtask "
                       "for several steps")
            else:
                why = "you are well over the step budget for this subtask"
            desync_str = (
                f"\n**PROGRESS CHECK: {why}. Either this subtask is already finished — in which "
                "case set \"subtask_done\": true THIS TURN, since saying so in your reasoning text "
                "does NOT advance anything — or you are lost, in which case re-read the instruction "
                "and head back rather than keep exploring forward.**")

        prompt_objnav_1st = f"""
    You are an intelligent agent in a simulated indoor environment. Your high-level instruction is: {instruction}.
    {pos_str}
    {"Your position hasn't changed for a long time, possibly stuck or in a loop, so you need to reconsider your next move carefully." if is_stuck else ""}
    Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.

    **Image Inputs**:
    1.  **Top-Down Map**: This is your memory. It will be updated as you explore. It's oriented with you facing upwards.
        -   `Gray`: Navigable floor you have seen.
        -   `Black`: Obstacles or walls that cannot be passed.
        -   `Red Arrow`: Your current position and direction.
        -   `Red Trail`: Your recent trajectory.
        -   `Start Point`: Your starting position.
        -   `Text Labels`: Automatically detected objects (may be inaccurate).
        -   `Yellow Boundary`: The frontier between explored and unexplored areas.
        -   `Numbered Circles`: Candidate waypoints projected into your view.
    2.  **First-Person View**: This is what you see right now. It is divided into 3 perspectives: front, left, right.
        -   `Numbered Circles`: Candidate waypoints projected into your front view.
        -   `L, R, B Circles`: Turn actions (L: Left 90°, R: Right 90°, B: Turn Around 180°).
        -   `Action Text`: The last action you executed.   

    **Notes**:
    1. Labels on the map are sometimes wrong, so stick to the first-person view.
    2. You can't walk through or open a closed door. And the target will not be in these places.
    3. When you don't know where to go, prioritize exploring unexplored areas.
    4. Don't go up or down stairs.
    5. Red waypoints indicate locations that have been visited, and white ones indicate those that haven’t.

    **Your Task**:
    1.  **Analyze**: Briefly describe your current situation and environment, referencing both the map and your first-person view. Confirm your current progress on the instruction. Determine whether a previous decision or judgment was correct. 
    2.  **Strategize**: State your plan to make progress on the instruction. Which direction or area should you explore next? Pay attention to unexplored areas. When you're not sure where to go, you can turn your perspective and look around.
    3.  **Decide**: Choose the best action to execute your plan. Prioritize waypoints that lead towards the goal or into new, unexplored areas. Avoid choosing waypoints that require navigation through tight spaces or closed doors.

    **Output Format**:
    ```json
    {{
    "movement": "Describe the movement trajectory of the last step based on your last action and historical frames",
    "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
        "front view": "",
        "left view": "",
        "right view": "",
        "map": "",
    }},
    "thought": "",  // Check whether the current route and position are consistent with the instruction and previous plan. Judge if you are close to the target or need further exploration
    "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
    "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, imagine the approximate final position when the subtask is completed (near the target object). Give a short high-level route plan WITHOUT mentioning specific waypoint numbers.
    "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
    "subtask_done": true/false,  // Is subtask {self.instruction_obj.get_current_subtask_key()} finished as of this turn? This is INDEPENDENT of "action" — you can set it true AND still pick a waypoint to keep moving in the same turn. Declaring completion only in your reasoning text does NOTHING; this field is the only thing that advances the task. Do not leave it false while writing "I have completed this subtask" above.
    "action": "",  // Select a waypoint number on the image, or a turning action (L, R, B). For example: "action": 3 or "action": "L". Use -1 ONLY to stop where you are (final destination reached, or you must halt this turn).
{det_schema}    }}
    ```
    """.strip()

        prompt_objnav = f"""
        **Continue your task.**
        Your high-level instruction is: {instruction}.
        {pos_str}
        {"Your position hasn't changed for a long time, possibly stuck or in a loop, so you need to reconsider your next move carefully." if is_stuck else ""}
        Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.

        **Your Task**:
        1.  **Analyze**: Briefly describe your current situation and environment, referencing both the map and your first-person view. Confirm your current progress on the instruction. Determine whether a previous decision or judgment was correct. 
        2.  **Strategize**: State your plan to make progress on the instruction. Which direction or area should you explore next? Pay attention to unexplored areas. When you're not sure where to go, you can turn your perspective and look around.
        3.  **Decide**: Choose the best action to execute your plan. Prioritize waypoints that lead towards the goal or into new, unexplored areas. Avoid choosing waypoints that require navigation through tight spaces or closed doors.
        {after_stop_str if self.first_stop else ""}{end_str}{budget_str}{desync_str}{explore_str}{arrive_str}{challenge_str}
        **Output Format**:
        ```json
        {{
        "movement": "Describe the movement trajectory of the last step based on your last action and historical frames",
        "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
            "front view": "",
            "left view": "",
            "right view": "",
            "map": "",
        }},
        "thought": "",  // Check whether the current route and position are consistent with the instruction and previous plan. Judge if you are close to the target or need further exploration
        "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
        "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, imagine the approximate final position when the subtask is completed (near the target object). Give a short high-level route plan WITHOUT mentioning specific waypoint numbers.
        "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
        "subtask_done": true/false,  // Is subtask {self.instruction_obj.get_current_subtask_key()} finished as of this turn? This is INDEPENDENT of "action" — you can set it true AND still pick a waypoint to keep moving in the same turn. Declaring completion only in your reasoning text does NOTHING; this field is the only thing that advances the task. Do not leave it false while writing "I have completed this subtask" above.
        "action": "",  // Select a waypoint number on the image, or a turning action (L, R, B). For example: "action": 3 or "action": "L". Use -1 ONLY to stop where you are (final destination reached, or you must halt this turn).
{det_schema}        }}
        ```""".strip()

        prompt_vln_1st = f"""
        **Your Task**:
        {self.instruction_obj.get_all_subtasks_str()}

        ---
        The instruction describes a path from the starting position to the target position. Your task is to move from the starting position(0,0,0) to the final position. At the same time, make sure your route conforms to the instruction description.
        The "Go upstairs" in the instruction is only considered complete when you completely reach the top platform via the stairs. Make sure you have climbed all the steps and moved onto the platform.

        {pos_str}
        Before starting to execute the instruction, you have firstly turned around 2 times in place for a full 360 degrees to capture images of your surroundings.
        Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.
        Complete subtasks in order. Don't skip any action. Consider the route by taking into account the current and the next subtask.

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

        **Output Format**:
        ```json
        {{
        "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
            "front view": "",
            "left view": "",
            "right view": "",
            "map": "",
        }},
        "thought": "",  // Check whether the current route and position are in line with the instruction and previous plan and whether the expected destination of the subtask has been reached, and plan the next action(turn, move forward or complete).
        "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
        "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, based on the screen and instructions, envision the target position you expect to reach when the subtask is completed, and plan the route. Don't mention exact waypoint number here.
        "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
        "subtask_done": true/false,  // Is subtask {self.instruction_obj.get_current_subtask_key()} finished as of this turn? This is INDEPENDENT of "action" — you can set it true AND still pick a waypoint to keep moving in the same turn. Declaring completion only in your reasoning text does NOTHING; this field is the only thing that advances the task. Do not leave it false while writing "I have completed this subtask" above.
        "action": "",  // Select a waypoint number on the image, or a turning action (L, R, B). For example: "action": 3 or "action": "L". Use -1 ONLY to stop where you are (final destination reached, or you must halt this turn).
{det_schema}        }}
        ```""".strip()
        prompt_vln = f"""
        **Continue your task.**
        {self.instruction_obj.get_all_subtasks_str()}

        {"You can't move as expected, possibly stuck or blocked by something, try to find other ways to get out." if is_stuck else ""}

        ---
        The "Go upstairs" in the instruction is only considered complete when you completely reach the top platform via the stairs. Make sure you have climbed all the steps and moved onto the platform.

        {pos_str}
        {"Your position hasn't changed after multiple retries, possibly blocked by something, find out the reason why you are stuck and try to turn and look for other ways to get out." if is_stuck else ""}
        Your task is to choose a strategic waypoint or a turn action. Once you choose a waypoint, a low-level planner will automatically generate a path and navigate to it.
        Red waypoints indicate locations you have visited. Avoid going back to places you've already been unless necessary!!!
        Complete subtasks in order. Don't skip any action. Consider the route by taking into account the current and the next subtask.
        {after_stop_str if self.first_stop else ""}{end_str}{budget_str}{desync_str}{explore_str}{arrive_str}{challenge_str}
        **Output Format**:
        ```json
        {{
        "movement": "Describe the movement trajectory of the last step based on your last action and historical frames",
        "observation": {{   // According to the latest view and map, describe the scene you see and your position currently.
            "front view": "",
            "left view": "",
            "right view": "",
            "map": "",
        }},
        "thought": "",  // Check whether the current route and position are in line with the instruction and previous plan and whether the expected destination of the subtask has been reached, and plan the next action(turn, move forward or complete).
        "stuck": true/false,  // Check if you're blocked by obstacles or going in circles.
        "plan": "",  // For subtask {self.instruction_obj.get_current_subtask_key()}, based on the screen and instructions, envision the target position you expect to reach when the subtask is completed, and plan the route. Don't mention exact waypoint number here.
        "curr_step": "",  // Check if the subtask {self.instruction_obj.get_current_subtask_key()} is completed(reach the closest waypoint to the target). Analyze exact candidate waypoints with number and turning actions. Make a decision for the current step.
        "subtask_done": true/false,  // Is subtask {self.instruction_obj.get_current_subtask_key()} finished as of this turn? This is INDEPENDENT of "action" — you can set it true AND still pick a waypoint to keep moving in the same turn. Declaring completion only in your reasoning text does NOTHING; this field is the only thing that advances the task. Do not leave it false while writing "I have completed this subtask" above.
        "action": "",  // Select a waypoint number on the image, or a turning action (L, R, B). For example: "action": 3 or "action": "L". Use -1 ONLY to stop where you are (final destination reached, or you must halt this turn).
{det_schema}        }}
        ```""".strip()

        if self.mode == 'objnav':
            prompt = prompt_objnav if not is_first_step else prompt_objnav_1st
        else:
            prompt = prompt_vln if not is_first_step else prompt_vln_1st
        return prompt
