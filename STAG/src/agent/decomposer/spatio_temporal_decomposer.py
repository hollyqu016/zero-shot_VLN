"""
SpatioTemporalInstructionDecomposer (STID) —— 独立、不依赖 VLNPlanner 的时空指令分解器。

和 agent.py 里的版本相比，这里去掉了对 VLNPlanner 的继承，自带一份最小化的 LLM client
初始化和调用逻辑，方便脱离整个 habitat_sim / agent.py 的依赖链单独运行、单独测试。

用法（直接跑这个文件做功能性测试）：
    python spatio_temporal_decomposer.py

也可以在自己的测试脚本里单独 import：
    from spatio_temporal_decomposer import SpatioTemporalInstructionDecomposer
    stid = SpatioTemporalInstructionDecomposer()
    stages = stid.decompose("Go past the chair, then stop at the window on your left.")

环境变量（和 agent.py 里 VLNPlanner 用的一致，方便复用同一套本地Ollama/OpenAI兼容配置）：
    OPENAI_BASE_URL  默认 http://127.0.0.1:1143/v1
    OPENAI_API_KEY   默认 ollama
    OPENAI_MODEL     默认 qwen3.6:35b
"""

import os
import re
import sys
import json
from typing import Any, Dict, List

# segmentation/object_list.py 提供的封闭类别词表，用来把landmark的自由文本名字
# 对齐到检测器/VLM输出用的同一份词表（class_name），这样匹配可以退化成字符串比较，
# 不需要额外的embedding相似度模块。这里做了路径兜底和try/except，即使脱离整个项目
# 单独运行本文件（比如segmentation目录不在sys.path里），也不会因为这个可选功能导致
# 整个脚本崩溃——只是这种情况下landmark就没有canonical class_name，退化回自由文本。
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../src
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
try:
    from segmentation.object_list import category_names_str, normalize_category_name, CATEGORY_NAMES
    _COVERED_CATEGORY_SET = set(normalize_category_name(n) for n in CATEGORY_NAMES)
except Exception as _e:
    print(f"[STID] 无法加载 segmentation.object_list 的封闭词表，landmark将不会带canonical class_name: {_e}")
    category_names_str = ""
    _COVERED_CATEGORY_SET = set()
    def normalize_category_name(name: str) -> str:  # 兜底：至少做基础格式归一化
        return (name or "").strip().lower().replace("_", " ")


class SpatioTemporalInstructionDecomposer:
    """
    时空指令分解器：把"分段 + 空间锚点抽取"和"时间预算推断"拆成两条独立通道。

    通道①【显式语言学解析，纯规则，不调用LLM】：
        - 用话语连接词(then/after that/once you/until you等)和标点做分段，
          分段结果之后不会再被LLM修改（规划器的职责是"分段+打标签"，不应该由LLM决定
          "这段和上一段是不是同一个东西"这种身份判定——这正是VLN代码库里之前那个
          "LLM返回重复stage_index导致子任务被错误合并"的bug的根源，这里从设计上直接规避）。
        - 用介词/动词短语(past/through/at the/toward/avoid/don't enter等)给每个子句里的
          地标打角色标签：waypoint_marker(途经点) / destination_marker(终点) / avoid_marker(避让点)。
          角色判定是"位置感知"的（取landmark前面最近的一个线索词），而不是给整句只判一个角色，
          否则一句话里出现多个不同角色的地标会被错误地统一标成同一个角色。
        - 用方位词(on your left/on your right/behind)给地标打相对方位标签。
        规则覆盖不到的地方（local_start/local_end、部分地标的role）留空/标记unknown，交给通道②。

    通道②【隐式常识推断，依赖LLM】：
        只让LLM做规则做不到的两件事：
        1. 补全每个（已经固定不变的）子任务的 local_start / local_end 语义描述，以及
           规则未能判断角色的地标该归到哪一类；
        2. 给每个子任务估计一个"相对时长预算"relative_duration ∈ [0,1]（这个子任务大约
           占整条指令总移动量的比例），而不是SHORT/LONG这种没有量化锚点的二值标签。
           所有子任务的 relative_duration 会被强制归一化，使其严格求和为1。

    最后还会做一次【阶段连续性校验】：检查 stage_k.local_end 和 stage_(k+1).local_start
    是否语义上指向同一位置（轻量词级重叠度，不是embedding），结果记录在
    boundary_consistent 字段，只留痕不阻断流程。

    局限性（研究原型，不是生产级实现）：
    - 地标抽取用的是轻量正则启发式(the/a/an后面的名词短语)，没有接依存句法/NER，
      复杂句式容易漏抽或抽错，后续建议换成spaCy等真正的句法解析器。
    - 阶段连续性校验用的是词级Jaccard重叠度，不是语义embedding，判断力有限。
    """

    def __init__(self):
        # openai 包放在 __init__ 里延迟导入，而不是模块顶层 import：这样即使当前环境没装
        # openai（比如只想先测通道①的规则逻辑），import 这个模块本身也不会报错；
        # 只有真正实例化、要用到LLM时才需要 openai 可用。
        from openai import OpenAI
        base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1143/v1")
        api_key = os.getenv("OPENAI_API_KEY", "ollama")
        self.model_name = os.getenv("OPENAI_MODEL", "qwen3.6:35b")
        print(f"[*] Initializing SpatioTemporalInstructionDecomposer with Model: {self.model_name}")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _get_llm_response(self, system_prompt: str, user_instruction: str) -> str:
        """调用 LLM，发送 system prompt + user 消息，返回模型文字回复。"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_instruction}
            ],
            temperature=0.05
        )
        return response.choices[0].message.content

    # ---- 通道①用到的规则表：话语连接词、介词/动词短语->角色、方位词->相对位置 ----
    _STAGE_CONNECTIVES = re.compile(
        r"(?:,?\s*\b(?:and\s+then|then|after\s+that|once\s+you|until\s+you|after\s+which)\b\s*)"
        r"|(?:\s*;\s*)|(?:\.\s+)",
        flags=re.I
    )
    _AVOID_CUES = re.compile(r"\b(?:do not|don't|avoid|without entering|never enter)\b", re.I)
    _DESTINATION_CUES = re.compile(r"\b(?:stop at|stop near|reach|arrive at|to the|toward|towards|at the)\b", re.I)
    _WAYPOINT_CUES = re.compile(r"\b(?:past|through|via|by the|beside|next to)\b", re.I)
    _LEFT_CUE = re.compile(r"\b(?:on your left|to the left|left side)\b", re.I)
    _RIGHT_CUE = re.compile(r"\b(?:on your right|to the right|right side)\b", re.I)
    _BEHIND_CUE = re.compile(r"\bbehind\b", re.I)
    _CURVE_CUES = re.compile(r"\b(?:around|detour)\b", re.I)
    # 名词短语抽取：the/a/an 后面的1~3个词。之前只截断2~4个词时，遇到
    # "the table and go to the kitchen"这种连续两个"the X"短语会把中间的动词/连接词
    # (and go to)也一起抓进第一个短语——所以额外用_TRIM_TRAILING_STOPWORD在遇到介词/连接词/
    # 动词时提前截断，两者配合才能抓出干净的地标名。
    # 续接词里用负向先行断言排除冠词，否则1~3个词的贪婪捕获会把下一个名词短语的
    # 冠词一起吞掉，导致后面那个地标彻底消失。典型例子是R2R里高频的"X of the Y"：
    #   "the top of the stairs"  贪婪捕获 -> "top of the"，match直接吃到第二个the，
    #                            finditer从" stairs."继续，再也匹配不到冠词 -> stairs丢失
    #   "the end of the hall"    同理 -> hall丢失，整个子任务地标为空
    # 加上断言后变成 "the top of" + "the stairs" 两次匹配，前者被停用词规则丢弃，
    # 后者正确保留成stairs。
    _NOUN_PHRASE = re.compile(
        r"\b(?:the|a|an)\s+([a-zA-Z][a-zA-Z\-]*(?:\s+(?!the\b|a\b|an\b)[a-zA-Z][a-zA-Z\-]*){0,2})",
        flags=re.I
    )
    # 名词短语后面如果紧跟介词/连接词/动词(on/at/near/toward/and/then/go/stop/turn/avoid等)，
    # 说明抓多了——要么是修饰这个地标的方位信息，要么已经越界抓到下一个动作短语，都要切掉，
    # 只保留地标本身的名字。方位信息由下面的 _LEFT_CUE/_RIGHT_CUE/_BEHIND_CUE 单独判断。
    # 补充了关系从句的引导词/代词/系动词：VLN指令里"the room you are in"、
    # "the door that is open"这类定语从句非常常见，之前没收录这些词，"the room you're in"
    # 会被抓成地标"room you"——一个既不是物体也不是房间的字符串，注定永远匹配不上。
    _TRIM_TRAILING_STOPWORD = re.compile(
        r"\b(?:on|in|at|near|by|beside|behind|toward|towards|through|past|next|"
        r"and|but|then|to|go|goes|going|walk|walking|stop|stopping|turn|turning|"
        r"avoid|reach|arrive|of|into|from|with|"
        r"you|your|yours|i|we|they|he|she|it|"
        r"that|which|who|whose|where|when|"
        r"is|are|was|were|am|be|been|being|has|have|had|will|would|can|could|"
        # 动词/副词/小品词：实测 not_in_vocab 里过半是名词短语跨过这类词抓多了
        # ("kitchen passed" / "corner out" / "hallway slightly" / "hall take" /
        #  "room across" / "arched entry well")。这些词几乎不会出现在地标名里，
        # 但要小心别加 back/up/down —— "back door" 这类是合法的修饰词+中心语。
        r"pass|passed|passing|take|takes|taking|head|heads|heading|"
        r"follow|follows|following|enter|enters|entering|exit|exits|exiting|"
        r"continue|continues|continuing|proceed|leave|leaves|leaving|"
        r"wait|waits|waiting|keep|keeps|keeping|lead|leads|leading|"
        r"out|across|along|slightly|straight|ahead|further|again|once|well)\b.*$",
        flags=re.I
    )
    # 对1839条真实VLN指令做词频统计后发现两类噪音：
    # 1) "the/a/an + 名词短语"规则会误抓一批纯方向词/序数词/位置词当成地标
    #    （比如"turn right"里的"right"、"on the left"里的"left"），这些根本不是能在
    #    场景里检测到的实体，留着只会产生一堆永远匹配不上的候选。
    # 2) 之前_TRIM_TRAILING_STOPWORD没收录介词"of"/"into"/"from"/"with"，导致
    #    "at the end of the hallway"这种被越界抓成"end of the"、"turn right into the room"
    #    被抓成"right into the"——现在补上这几个介词，这类越界大部分能被切干净。
    # 这里做纯粹的停用词过滤，不影响"room/kitchen"这类真实地标（只是接地方式不同，
    # 不属于这次要处理的范围）。
    _LANDMARK_STOPWORDS = frozenset({
        'right', 'left', 'top', 'bottom', 'way', 'foot', 'feet', 'middle',
        'edge', 'side', 'end', 'set', 'front', 'back', 'area',
    })
    # 序数/指示类修饰词。单独出现不算地标，但和真实名词组合时(比如"front door"、
    # "end table")要保留整个短语——过滤逻辑是"这个短语里的每一个词是不是都属于
    # 停用词+序数词的并集"，而不是"含有其中任意一个词就丢弃"，否则会把"front door"
    # 这种合法的"修饰词+真实地标"也误杀掉。
    _ORDINAL_WORDS = frozenset({'first', 'second', 'third', 'fourth', 'next', 'other', 'another', 'last', 'far'})

    def _is_landmark_stopword(self, name: str) -> bool:
        """
        判断抽取出来的candidate是否是纯方向/序数/位置类噪音，而不是真实地标。
        单个词直接查停用词表；多个词时，只有当短语里所有词都落在"停用词∪序数词"
        范围内(比如"first right"、"next set")才判定为噪音——只要有一个词是真实
        名词(比如"door"、"table")，就保留整个短语，留给通道②的LLM去做"忽略修饰词"
        的规整。
        """
        tokens = name.split()
        if not tokens:
            return True
        if len(tokens) == 1:
            return tokens[0] in self._LANDMARK_STOPWORDS
        noise_words = self._LANDMARK_STOPWORDS | self._ORDINAL_WORDS
        return all(t in noise_words for t in tokens)

    def _split_into_clauses(self, instruction: str) -> List[str]:
        """通道①-第一步：按话语连接词/标点切分成子句，作为stage边界。切分结果之后不会再变。"""
        parts = self._STAGE_CONNECTIVES.split(instruction)
        clauses = [p.strip() for p in parts if p and p.strip()]
        merged: List[str] = []
        for c in clauses:
            if len(c) < 6 and merged:      # 太短的碎片(比如切出一个孤立介词)并入前一句
                merged[-1] = (merged[-1] + ' ' + c).strip()
            else:
                merged.append(c)
        return merged if merged else [instruction.strip()]

    def _nearest_preceding_role(self, clause: str, landmark_start_pos: int) -> str:
        """
        找出在landmark_start_pos之前、离它最近的角色线索(avoid/destination/waypoint cue)，
        用该线索的角色作为这个地标的role。用"最近的前置线索"而不是"整个clause只判一个role"，
        是因为一个未被拆分干净的子句里完全可能同时出现好几个不同角色的地标
        （比如"avoid the wet floor and stop at the door"里，wet floor该是avoid_marker，
        door该是destination_marker，不能整句只给一个角色）。
        """
        best_role, best_pos = 'unknown', -1
        for cue_re, role in ((self._AVOID_CUES, 'avoid_marker'),
                             (self._DESTINATION_CUES, 'destination_marker'),
                             (self._WAYPOINT_CUES, 'waypoint_marker')):
            for m in cue_re.finditer(clause):
                if m.start() <= landmark_start_pos and m.start() > best_pos:
                    best_pos, best_role = m.start(), role
        return best_role

    def _extract_landmarks_rule_based(self, clause: str) -> List[Dict[str, str]]:
        """通道①-第二步：从单个子句里用规则抽取地标，附带role和relative_position。
        role抽不出来时先留 'unknown'，交给通道②的LLM补；relative_position同理。"""
        landmarks = []
        seen_names = set()
        for m in self._NOUN_PHRASE.finditer(clause):
            raw_name = m.group(1)
            trimmed = self._TRIM_TRAILING_STOPWORD.sub('', raw_name).strip()
            # 注意：这里不能"trim完是空的话就退回用raw_name"——比如"next set of"，
            # "next"本身就在停用词表里、又刚好是整个短语的第一个词，trim会把整段都切空，
            # 这时候如果退回raw_name，反而会让切剩不下东西的噪音短语（本该被丢弃）
            # 原封不动地保留下来。trim结果为空就直接当成"这里没有有效地标"处理。
            name_clean = trimmed.strip().lower()
            if not name_clean or name_clean in seen_names:
                continue
            if self._is_landmark_stopword(name_clean):
                continue  # 纯方向/序数/位置词(right/left/top/...)，不是真实地标，直接丢弃
            seen_names.add(name_clean)

            role = self._nearest_preceding_role(clause, m.start(1))

            if self._LEFT_CUE.search(clause):
                position = 'left'
            elif self._RIGHT_CUE.search(clause):
                position = 'right'
            elif self._BEHIND_CUE.search(clause):
                position = 'behind'
            else:
                position = 'unknown'

            landmarks.append({'name': name_clean, 'role': role, 'relative_position': position})
        return landmarks

    def _build_stage_skeletons(self, instruction: str) -> List[Dict[str, Any]]:
        """
        通道①主流程：只做分段+地标抽取+角色/方位标注，完全不调用LLM。
        local_start/local_end 先占位成'unknown'——这两个字段的语义往往依赖上下文常识
        （比如"到窗边"到底指哪扇窗），规则很难可靠判断，交给通道②的LLM去推断。
        """
        clauses = self._split_into_clauses(instruction)
        stages = []
        for idx, clause in enumerate(clauses):
            landmarks = self._extract_landmarks_rule_based(clause)
            is_last = (idx == len(clauses) - 1)
            if self._CURVE_CUES.search(clause):
                geometry = 'CURVE'
            elif is_last and any(l['role'] in ('destination_marker', 'unknown') for l in landmarks):
                geometry = 'POINT'
            else:
                geometry = 'LINE'

            stages.append({
                'stage_index': idx + 1,   # 分段身份完全由通道①的枚举顺序决定，通道②的LLM不允许改动
                'raw_text': clause,
                'semantic_anchors': {
                    'local_start': 'unknown',
                    'action': 'move_forward',
                    'landmarks': landmarks,
                    'local_end': 'unknown',
                },
                'kinematic_tube': {
                    'spatial_geometry': geometry,
                    'temporal_duration': 'LONG',  # 占位，真正数值看relative_duration
                },
                'relative_duration': None,
                'parse_source': {
                    'landmarks': 'rule',
                    'local_start': 'pending',
                    'local_end': 'pending',
                    'relative_duration': 'pending',
                },
            })
        return stages

    def _llm_fill_gaps_and_temporal_budget(self, instruction: str, stages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        通道②：只问LLM两件事——① 每个（已固定不变的）子句的起止点位置描述，以及规则未能
        判断角色的地标该归到哪一类；② 所有子任务的相对时长预算(0~1，归一化后求和为1)。
        不要求LLM重新分段——分段结果完全来自通道①，绝不交给LLM重新判断。
        """
        skeleton_for_prompt = [
            {
                'stage_index': s['stage_index'],
                'raw_text': s['raw_text'],
                'all_landmarks': [l['name'] for l in s['semantic_anchors']['landmarks']],
                'landmarks_needing_role': [l['name'] for l in s['semantic_anchors']['landmarks'] if l['role'] == 'unknown'],
            }
            for s in stages
        ]

        # class_name词表约束段：只有真的拿到了封闭词表(category_names_str非空)才加进prompt，
        # 让LLM把每个landmark对齐到检测器/VLM用的同一份词表——这是对齐机制的核心，两边
        # (这里 + instance_segmentation()的检测prompt)被约束到同一个固定词表选词，
        # 之后landmark和object_entities的匹配就能退化成一次字符串比较，不需要embedding。
        #
        # vocab_section(说明文字)和landmark_classes_json_field(JSON schema里的字段声明)
        # 必须同生共死、整体开关：之前的bug是JSON返回格式那一行写死了要求"landmark_classes"
        # 字段，但"该从哪份词表里选"的说明却只在词表可用时才加入——一旦词表加载失败
        # (比如object_list.py没同步)，LLM仍然被要求返回landmark_classes，却完全没被告知
        # 有效取值范围，于是自己发挥编出"furniture"这类不在词表里的泛化词。现在词表不可用
        # 时干脆连这个字段都不请求，而不是"问了却不给约束"。
        vocab_section = ""
        landmark_classes_json_field = ""
        if category_names_str:
            vocab_section = (
                "\n5. `landmark_classes`: for EVERY name listed in `all_landmarks`, output the closest "
                "matching category chosen EXACTLY from this fixed vocabulary (do not invent new words):\n"
                f"[{category_names_str}]\n"
                "IMPORTANT: ignore descriptive modifiers/adjectives and map to the base/head-noun category. "
                "For example, 'dining table', 'coffee table', and 'pool table' all map to 'table'; "
                "'front door', 'open door', and 'double door' all map to 'door'; 'couch' maps to 'sofa'; "
                "'fridge' maps to 'refrigerator'. Do not treat modified phrases as missing from the vocabulary "
                "just because the exact phrase isn't listed — strip the modifier first.\n"
                "If a landmark truly does not fit any category above even after stripping modifiers "
                "(e.g. it's a room/area name like 'kitchen' rather than a discrete object), output `other`.\n"
            )
            landmark_classes_json_field = ', "landmark_classes": {"name": "category", ...}'

        system_prompt = f"""You are refining an already-segmented robot navigation instruction.
The stages (segmentation) are FIXED and must NOT be changed, reordered, merged, or split.
For each stage (by stage_index), infer:
1. `local_start`: best-guess starting location description (use context from previous stages if helpful).
2. `local_end`: strict termination condition/location for this stage.
3. `landmark_roles`: for each name listed in `landmarks_needing_role`, output one of
   ["waypoint_marker", "destination_marker", "avoid_marker"].
4. `relative_duration`: a float in [0,1] estimating what FRACTION of the whole instruction's total
   travel this stage represents. All stages' relative_duration values MUST sum to approximately 1.0.{vocab_section}
Return ONLY JSON: {{"stages": [{{"stage_index": int, "local_start": str, "local_end": str,
"landmark_roles": {{"name": "role", ...}}{landmark_classes_json_field},
"relative_duration": float}}, ...]}}"""

        user_prompt = (
            f"Full instruction: '{instruction}'\n\n"
            f"Fixed segmentation (do not change): {json.dumps(skeleton_for_prompt, ensure_ascii=False)}"
        )

        llm_stages: Dict[int, Dict[str, Any]] = {}
        try:
            raw = self._get_llm_response(system_prompt, user_prompt)
            m = re.search(r'\{.*\}', raw, flags=re.S)
            data = json.loads(m.group(0) if m else raw)
            for s in data.get('stages', []):
                if isinstance(s, dict) and 'stage_index' in s:
                    llm_stages[int(s['stage_index'])] = s
        except Exception as e:
            print(f"[STID] LLM gap-filling failed, falling back to uniform/default values: {e}")

        # ---- relative_duration 归一化：LLM给的数字不一定严格满足"求和为1"，这里强制校正 ----
        raw_durations = []
        for s in stages:
            info = llm_stages.get(s['stage_index'], {})
            d = info.get('relative_duration', None)
            try:
                d = float(d)
                if not (0.0 <= d <= 1.0):
                    d = None
            except (TypeError, ValueError):
                d = None
            raw_durations.append(d)

        known_sum = sum(d for d in raw_durations if d is not None)
        n_unknown = sum(1 for d in raw_durations if d is None)
        remaining = max(0.0, 1.0 - known_sum)
        fallback_each = (remaining / n_unknown) if n_unknown > 0 else 0.0
        filled_durations = [d if d is not None else fallback_each for d in raw_durations]
        total = sum(filled_durations) or 1.0
        normalized_durations = [d / total for d in filled_durations]  # 强制归一化，严格求和为1

        for s, norm_d, raw_d in zip(stages, normalized_durations, raw_durations):
            info = llm_stages.get(s['stage_index'], {})

            s['semantic_anchors']['local_start'] = info.get('local_start') or s['semantic_anchors']['local_start']
            s['semantic_anchors']['local_end'] = info.get('local_end') or s['semantic_anchors']['local_end']
            s['relative_duration'] = round(norm_d, 4)
            # 兼容旧schema：仍然派生一个SHORT/LONG供还没升级的下游代码使用(比如旧的可视化文案)
            s['kinematic_tube']['temporal_duration'] = 'SHORT' if norm_d < (1.0 / max(len(stages), 1)) else 'LONG'

            role_map = info.get('landmark_roles', {})
            if not isinstance(role_map, dict):
                role_map = {}
            class_map = info.get('landmark_classes', {})
            if not isinstance(class_map, dict):
                class_map = {}

            for landmark in s['semantic_anchors']['landmarks']:
                if landmark['role'] == 'unknown':
                    landmark['role'] = role_map.get(landmark['name'], 'waypoint_marker')  # 仍未知则兜底成途经点
                    landmark['role_source'] = 'llm' if landmark['name'] in role_map else 'fallback_default'
                else:
                    landmark['role_source'] = 'rule'

                # class_name：对齐到和instance_segmentation()检测输出同一份封闭词表，
                # 之后landmark<->object_entities的匹配直接比较这个字段就够了，不需要
                # embedding相似度。
                #
                # 优先级：如果landmark自己的名字本来就精确等于词表里的一个类别(比如
                # landmark叫"corner"，词表里也有"corner")，直接确认使用，不需要等LLM去
                # "发现"这件事——实测发现LLM在给了两百多个类别的情况下，偶尔会漏判本该
                # 精确命中的词，把它标成"other"(比如真实跑出来"corner"被判成了other，
                # 但corner明明就在词表里)。这种exact match属于可以100%确定的规则判断，
                # 完全没必要依赖LLM的"扫词表"能力，用规则兜底更可靠、也不增加任何调用开销。
                self_normalized = normalize_category_name(landmark['name'])
                if _COVERED_CATEGORY_SET and self_normalized in _COVERED_CATEGORY_SET:
                    landmark['class_name'] = self_normalized
                    landmark['class_name_source'] = 'exact_self_match'
                else:
                    raw_class = class_map.get(landmark['name'])
                    if raw_class:
                        landmark['class_name'] = normalize_category_name(raw_class)
                        landmark['class_name_source'] = 'llm'
                    else:
                        # LLM没给出对应类别时，退化成对landmark自身名字做同样的归一化，
                        # 至少保证格式一致，字符串比较不会因为大小写/下划线不同而误判。
                        landmark['class_name'] = self_normalized
                        landmark['class_name_source'] = 'fallback_self'

            s['parse_source']['local_start'] = 'llm' if info.get('local_start') else 'fallback_default'
            s['parse_source']['local_end'] = 'llm' if info.get('local_end') else 'fallback_default'
            s['parse_source']['relative_duration'] = 'llm' if raw_d is not None else 'fallback_uniform'
            s['parse_source']['landmark_classes'] = 'llm' if class_map else 'fallback_self'

        return stages

    @staticmethod
    def _text_overlap_score(a: str, b: str) -> float:
        """
        轻量词级Jaccard重叠度，用于阶段边界连续性校验。返回-1表示"信息不足、无法判断"
        （而不是"不一致"）。没有接embedding模型，判断力有限，后续可以换成语义相似度。
        """
        if not a or not b or a == 'unknown' or b == 'unknown':
            return -1.0
        wa = set(re.findall(r"[a-zA-Z']+", a.lower()))
        wb = set(re.findall(r"[a-zA-Z']+", b.lower()))
        if not wa or not wb:
            return -1.0
        return len(wa & wb) / len(wa | wb)

    def _check_boundary_consistency(self, stages: List[Dict[str, Any]], threshold: float = 0.2) -> None:
        """
        检查 stage_k.local_end 和 stage_(k+1).local_start 是否语义上指向同一位置
        （词级重叠度 >= threshold 视为一致）。只打标记、不阻断分解流程，方便调试/统计
        "有多少比例的相邻阶段衔接不一致"这类指标。
        """
        for i, s in enumerate(stages):
            if i == len(stages) - 1:
                s['boundary_consistent'] = None  # 最后一个stage没有下一段可比较
                continue
            score = self._text_overlap_score(
                s['semantic_anchors']['local_end'],
                stages[i + 1]['semantic_anchors']['local_start']
            )
            if score < 0:
                s['boundary_consistent'] = None
            else:
                s['boundary_consistent'] = bool(score >= threshold)
                s['boundary_overlap_score'] = round(score, 3)

    def decompose(self, instruction: str) -> List[Dict[str, Any]]:
        """
        分解主入口（双通道：规则做分段/地标角色标注，LLM只补空隙+估计相对时长）。
        通道①(_build_stage_skeletons)本身不依赖网络/LLM，任何时候都能跑；
        通道②(_llm_fill_gaps_and_temporal_budget)依赖LLM，如果调用失败，内部已经有
        try/except兜底成uniform/default值，不会导致整个decompose()崩溃。
        """
        stages = self._build_stage_skeletons(instruction)
        stages = self._llm_fill_gaps_and_temporal_budget(instruction, stages)
        self._check_boundary_consistency(stages)
        return stages

    def decompose_rule_only(self, instruction: str) -> List[Dict[str, Any]]:
        """
        只跑通道①（纯规则，不联网、不调用LLM）。用于在没有本地LLM服务可用时，
        单独验证分段和地标抽取/角色标注这部分规则逻辑是否work。
        """
        return self._build_stage_skeletons(instruction)


# ============================================================
# 功能性测试入口：直接运行本文件即可（python spatio_temporal_decomposer.py）
# ============================================================
if __name__ == "__main__":
    TEST_INSTRUCTIONS = [
        "Go past the chair, then stop at the window on your left.",
        "Walk through the hallway, avoid the wet floor, and stop at the door.",
        "Turn around the table and go to the kitchen.",
        "Exit the bedroom, walk down the hallway, and stop at the bathroom door.",
        "Do not enter the office. Instead, continue past the plant until you reach the elevator.",
    ]

    print("=" * 70)
    print("Step 1: 只测通道①（纯规则，不需要联网/本地LLM服务）")
    print("=" * 70)
    stid_offline = SpatioTemporalInstructionDecomposer.__new__(SpatioTemporalInstructionDecomposer)
    # 绕开 __init__ 里的 OpenAI client 初始化，因为 decompose_rule_only 完全用不到它，
    # 这样在没有本地LLM服务/没有网络的环境下也能先跑通道①做功能性测试。
    for instr in TEST_INSTRUCTIONS:
        print(f"\n指令: {instr}")
        stages = stid_offline.decompose_rule_only(instr)
        for s in stages:
            print(f"  [Stage {s['stage_index']}] raw_text={s['raw_text']!r}")
            print(f"    geometry={s['kinematic_tube']['spatial_geometry']}")
            for lm in s['semantic_anchors']['landmarks']:
                print(f"    landmark: {lm}")

    print("\n" + "=" * 70)
    print("Step 2: 完整双通道测试（需要能连上 OPENAI_BASE_URL 指向的本地LLM服务）")
    print("=" * 70)
    try:
        stid = SpatioTemporalInstructionDecomposer()
        instr = TEST_INSTRUCTIONS[0]
        print(f"\n指令: {instr}")
        stages = stid.decompose(instr)
        print(json.dumps(stages, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"[跳过] 完整测试需要本地LLM服务可达，当前调用失败: {e}")
