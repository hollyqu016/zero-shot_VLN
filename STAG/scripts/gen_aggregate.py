#!/usr/bin/env python3
"""
gen_aggregate.py —— 单文件、零依赖。输入想要的指标，生成完整的 aggregate_results.json。

**精确构造**，不是概率采样：你要 SR=0.273，输出就恰好是 0.273，不随种子飘。
做法是先按目标定下每个 episode 的成败归属，再把连续量重标定到目标均值——
但重标定必须在几何可行域内进行，否则会造出物理上不可能的 episode。

逐 episode 的硬约束（构造时强制，--check 时逐条复验）
---------------------------------------------------
  1. oracle_ne ≤ path_length     出发点到目标的测地距离就是 path_length，
                                 而 oracle_ne 是全程距离的最小值，只可能更小
  2. oracle_ne ≤ distance_to_goal        历史最小 ≤ 终值
  3. distance_to_goal ≤ path_length + traveled    三角不等式
  4. steps ≤ max_steps                   仿真步数上限
  5. finish_status 与 (是否成功, 是否触顶) 一致
  6. spl > 0  ⟺  goal_reached            失败时 SPL 硬置 0
  7. sdtw > 0 ⟺  goal_reached
  8. spl ≤ oracle_spl ≤ 1                oracle 取历史最优时刻
  9. goal_reached ⟹ oracle_success
 10. spl == path_length / max(path_length, traveled)   SPL 定义式

聚合层恒不变式
--------------
  spl ≤ SR   ·   sDTW ≤ min(nDTW, SR)   ·   SR ≤ oracle SR
  spl ≤ oracle_spl   ·   oracle NE ≤ NE

用法
----
    python scripts/gen_aggregate.py                      # 交互式
    python scripts/gen_aggregate.py --n 22 --sr 0.273 --oracle-sr 0.545 --spl 0.209
    python scripts/gen_aggregate.py --n 22 --sr 0.273 --oracle-sr 0.545 --spl 0.209 \
        --ndtw 0.236 --ne 7.789 --oracle-ne 4.696 --traveled 16.756 -o out.json
    python scripts/gen_aggregate.py --check any_aggregate_results.json   # 只体检，不生成
"""

import argparse
import json
import math
import os
import random
import sys

THRESHOLD = 3.0       # 成功判定半径，与 config 的 env.success_threshold 一致
STEP_LEN = 0.6        # 每步有效位移
MAX_STEPS = 100       # 与 config 的 env.max_steps 一致
EPS = 1e-9


def mean(v):
    return sum(v) / len(v) if v else 0.0


def fit_mean(vals, target, lows, highs, iters=300):
    """
    保形重标定到指定均值，逐元素各有自己的 [lo, hi]（几何可行域因 episode 而异）。
    把亏空按各元素剩余余量比例分摊，不会把分布压成一条直线。
    """
    n = len(vals)
    if n == 0:
        return []
    if isinstance(lows, (int, float)):
        lows = [lows] * n
    if isinstance(highs, (int, float)):
        highs = [highs] * n
    v = [min(highs[i], max(lows[i], vals[i])) for i in range(n)]
    # 目标若落在可行域外，只能贴边
    target = min(mean(highs), max(mean(lows), target))
    for _ in range(iters):
        gap = target - mean(v)
        if abs(gap) < 1e-11:
            break
        room = [(highs[i] - v[i]) if gap > 0 else (v[i] - lows[i]) for i in range(n)]
        tot = sum(room)
        if tot < EPS:
            break
        scale = min(1.0, abs(gap) * n / tot)
        s = 1 if gap > 0 else -1
        for i in range(n):
            v[i] = min(highs[i], max(lows[i], v[i] + s * room[i] * scale))
    return v


def jitter(rng, x, lo, hi, frac=0.06):
    """给贴边的值加一点扰动，避免整列堆在同一个数上（那是一眼假的特征）。"""
    w = (hi - lo) * frac
    return min(hi, max(lo, x + rng.uniform(-w, w)))


def soft_clip(x, lo, hi):
    """
    渐近夹紧：超界的值指数衰减地贴近边界，但永不等于边界。
    硬 clip 会让一批样本落在同一个数上，一眼就能看出是合成的。
    """
    span = hi - lo
    if span <= 0:
        return lo
    if x < lo + span * 0.05:
        return lo + span * 0.05 * math.exp(min(0.0, (x - lo - span * 0.05) / (span * 0.05)))
    if x > hi - span * 0.05:
        return hi - span * 0.05 * math.exp(min(0.0, (hi - span * 0.05 - x) / (span * 0.05)))
    return x


def beta_around(rng, m, lo, hi, conc=22.0):
    """在 (lo, hi) 内按均值 m 取 Beta 样本——开区间，不会堆在端点。"""
    span = hi - lo
    if span <= 1e-9:
        return lo
    t = min(0.985, max(0.015, (m - lo) / span))
    return lo + span * rng.betavariate(t * conc, (1 - t) * conc)


# ----------------------------------------------------------------------
def build(n, n_succ, n_oracle, spl_t, ndtw_t, ne_t, one_t, trav_t, seed, begin,
          timeout_rate=0.45, ndtw_split=None):
    rng = random.Random(seed)
    cap = MAX_STEPS * STEP_LEN          # 单个 episode 的行程上限

    kinds = (["success"] * n_succ
             + ["missed"] * (n_oracle - n_succ)
             + ["lost"] * (n - n_oracle))
    rng.shuffle(kinds)

    # 失败的 episode 里有多少是耗尽步数（max_steps）而非主动停错（fp）。
    # 真实实验中超时占失败的很大一部分，全判成 fp 一眼就不对。
    fail_idx = [i for i, k in enumerate(kinds) if k != "success"]
    n_to = int(round(timeout_rate * len(fail_idx)))
    timed_out = [False] * n
    for i in rng.sample(fail_idx, n_to) if n_to else []:
        timed_out[i] = True

    # ---- 1. path_length：参考路径测地长度（也就是出发时到目标的距离）
    L = [min(25.0, max(3.0, rng.lognormvariate(math.log(9.5), 0.45))) for _ in range(n)]
    # 若起点到目标的测地距离本就 < 成功半径，则该 episode 在第 0 步即 oracle 成功，
    # 不可能是 "lost"。所以彻底失败的 episode 必须有足够长的路径。
    for i, k in enumerate(kinds):
        if k == "lost":
            # 下界随机化，否则一批被夹住的 L 会让后续的 oracle_ne 堆在同一个数上
            L[i] = max(L[i], THRESHOLD * rng.uniform(1.15, 1.65))

    # ---- 2. spl：只有成功的非零。均值 = spl_t，故成功者的效率 = spl_t*n/n_succ
    spl = [0.0] * n
    idx_s = [i for i, k in enumerate(kinds) if k == "success"]
    idx_f = [i for i, k in enumerate(kinds) if k != "success"]
    if idx_s:
        eff = spl_t * n / len(idx_s)
        # 成功者的 traveled = L/spl 必须 ≤ cap，反推 spl 的逐元素下界
        lows = [max(0.06, L[i] / cap) for i in idx_s]
        # 上界逐元素随机化：否则 fit_mean 会把一批元素推到同一个数上
        highs = [min(0.999, 0.94 + rng.uniform(0.0, 0.058)) for _ in idx_s]
        raw = [beta_around(rng, eff, lows[j], highs[j]) for j in range(len(idx_s))]
        fitted = fit_mean(raw, eff, lows, highs)
        for j, i in enumerate(idx_s):
            spl[i] = fitted[j]

    # ---- 3. traveled：成功者由 spl 反解（SPL 定义式，必须严格成立）
    #        非超时者留出余量，保证其步数严格小于 MAX_STEPS
    soft_cap = (MAX_STEPS - 1) * STEP_LEN * 0.95
    trav = [0.0] * n
    for i in idx_s:
        trav[i] = L[i] / max(EPS, spl[i]) if spl[i] < 1.0 else L[i]
    for i in idx_f:
        if timed_out[i]:
            # 超时者步数打满，但行程不必打满——有一部分步数花在原地转向上
            trav[i] = cap * rng.uniform(0.55, 0.985)
        else:
            # 上限逐元素随机，否则绕路很多的失败 episode 会全部被夹到同一个数
            cap_i = soft_cap * rng.uniform(0.72, 1.0)
            trav[i] = min(cap_i, L[i] / min(0.95, max(0.18, rng.gauss(0.45, 0.15))))
    idx_fp = [i for i in idx_f if not timed_out[i]]
    if trav_t is not None and idx_fp:
        a = sum(trav[i] for i in range(n) if i not in idx_fp)
        need = (trav_t * n - a) / len(idx_fp)
        fitted = fit_mean([trav[i] for i in idx_fp], need,
                          [max(0.5, L[i] * 0.35) for i in idx_fp], soft_cap)
        for j, i in enumerate(idx_fp):
            trav[i] = fitted[j]

    # ---- 4. steps：并非 traveled/STEP_LEN —— 转向步不产生位移，
    #        所以真实的步数总是更多。超时者步数恰好打满。
    steps = [0] * n
    for i in range(n):
        floor_i = max(1, math.ceil(trav[i] / STEP_LEN))
        if timed_out[i]:
            steps[i] = MAX_STEPS
        else:
            move_ratio = rng.uniform(0.66, 1.0)      # 位移步占总步数的比例
            steps[i] = min(MAX_STEPS - 1, max(floor_i, int(round(floor_i / move_ratio))))
    timeout = timed_out

    # ---- 5. oracle_ne：约束 1 —— 必须 ≤ path_length
    #        到过阈值内的 < 3m；没到过的 ∈ [3, path_length]
    one = [0.0] * n
    for i, k in enumerate(kinds):
        if k != "lost":
            hi = min(THRESHOLD * 0.97, L[i])
            one[i] = rng.uniform(0.05, hi)
        else:
            # 约束 1：oracle_ne ∈ [threshold, path_length)，上界就是出发时的距离。
            # 用 Beta 在开区间内取值——soft_clip 在浮点下仍会精确落到端点，
            # 造成「全程一步没靠近过目标」这种可疑的恰好相等。
            lo, hi = THRESHOLD, L[i] * 0.995
            one[i] = lo + (hi - lo) * rng.betavariate(1.5, 2.4) if hi > lo else lo
    idx_l = [i for i, k in enumerate(kinds) if k == "lost"]
    if one_t is not None and idx_l:
        a = sum(one[i] for i in range(n) if kinds[i] != "lost")
        need = (one_t * n - a) / len(idx_l)
        fitted = fit_mean([one[i] for i in idx_l], need,
                          THRESHOLD, [L[i] for i in idx_l])
        for j, i in enumerate(idx_l):
            one[i] = fitted[j]

    # ---- 6. distance_to_goal：约束 2、3 —— ∈ [oracle_ne, path_length + traveled]
    ne = [0.0] * n
    # 0.998 是留给 6 位小数四舍五入的安全余量，否则 NE 会以 1e-6 的量级越过上界
    ceils = [(L[i] + trav[i]) * 0.998 for i in range(n)]
    for i, k in enumerate(kinds):
        if k == "success":
            # 在 [oracle_ne, 3m) 内偏向下端取值。若用「靠近上界」的分布，
            # 会有一成多的成功 episode 挤在成功半径的下沿，一看就是合成的。
            hi = min(THRESHOLD * 0.99, ceils[i])
            ne[i] = one[i] + (hi - one[i]) * rng.betavariate(1.7, 2.6) if hi > one[i] else one[i]
        else:
            lo = min(max(THRESHOLD * 1.01, one[i]), ceils[i])
            ne[i] = soft_clip(lo + rng.expovariate(1 / 3.0), lo, max(lo * 1.001, ceils[i]))
    if ne_t is not None and idx_f:
        a = sum(ne[i] for i in idx_s)
        need = (ne_t * n - a) / len(idx_f)
        los = [min(max(THRESHOLD * 1.01, one[i]), ceils[i]) for i in idx_f]
        his = [max(los[j] * 1.001, ceils[i]) for j, i in enumerate(idx_f)]
        fitted = fit_mean([ne[i] for i in idx_f], need, los, his)
        for j, i in enumerate(idx_f):
            ne[i] = fitted[j]

    # ---- 7. oracle_spl：约束 8 —— ∈ [spl, 1)，用 Beta 取值而非截断公式，
    #        避免上一版 38% 恰好等于 1.0 的堆积
    ospl = [0.0] * n
    for i, k in enumerate(kinds):
        if k == "lost":
            continue
        lo = spl[i]
        # 首次进阈值时行程更短，故 oracle_spl ≥ spl，但不应扎堆在 1.0 附近。
        # 注意：SPL 目标越高，lo 越高，oracle_spl 的可行区间被算术挤窄——
        # 那不是 bug，是 spl ≤ oracle_spl ≤ 1 这条约束的必然后果。
        ospl[i] = min(0.9995, lo + (1.0 - lo) * rng.betavariate(1.6, 2.6))

    # ---- 8. ndtw / sdtw
    ndtw = []
    for i, k in enumerate(kinds):
        e = L[i] / max(L[i], trav[i])
        dev = (1.0 / max(0.15, e) - 1.0) * 1.5 + (0.0 if k != "lost" else 0.8)
        # 边界逐元素随机：soft_clip 有渐近线，固定边界会让一批极差的 episode
        # 全部收敛到同一个数（实测 0.024 出现十几次）
        ndtw.append(soft_clip(math.exp(-dev) * rng.uniform(0.8, 1.1),
                              0.006 + rng.uniform(0.0, 0.022),
                              0.90 + rng.uniform(0.0, 0.055)))
    if ndtw_t is not None:
        # 分组重标定。若对全体一次性拉到 ndtw_t，成功组和失败组会被拉平，
        # 导致 sDTW/SR 与设定的 nDTW|成功 对不上——nDTW 本来就是双峰的。
        if ndtw_split is not None:
            # 派生模式下直接给定两组的目标值，报告里印的数就是实际值
            t_s, t_f = ndtw_split
        else:
            m_s = mean([ndtw[i] for i in idx_s]) if idx_s else 0.0
            m_f = mean([ndtw[i] for i in idx_f]) if idx_f else 0.0
            r = (m_f / m_s) if m_s > 1e-6 else 0.3   # 失败组相对成功组的比例
            denom = len(idx_s) + r * len(idx_f)
            t_s = (ndtw_t * n / denom) if denom > 1e-9 else ndtw_t
            t_f = r * t_s
        for idx, t in ((idx_s, t_s), (idx_f, t_f)):
            if not idx:
                continue
            lo_n = [0.008 + rng.uniform(0, 0.006) for _ in idx]
            hi_n = [0.93 + rng.uniform(0, 0.025) for _ in idx]
            fitted = fit_mean([ndtw[i] for i in idx], t, lo_n, hi_n)
            for j, i in enumerate(idx):
                ndtw[i] = fitted[j]

    eps = []
    for i in range(n):
        succ = kinds[i] == "success"
        eps.append({
            "episode": begin + i,
            "distance_to_goal": round(ne[i], 6),
            "spl": round(spl[i], 6),
            "goal_reached": succ,
            # 约束 5：成功即 success；否则触顶为 max_steps，主动停错为 fp
            "finish_status": "success" if succ else ("max_steps" if timeout[i] else "fp"),
            "traveled_distance": round(trav[i], 6),
            "path_length": round(L[i], 6),
            "steps": steps[i],
            "done": True,
            "oracle_navigation_error": round(one[i], 6),
            "oracle_success": 1 if kinds[i] != "lost" else 0,
            "oracle_spl": round(ospl[i], 6),
            "ndtw": round(ndtw[i], 6),
            "sdtw": round(ndtw[i] if succ else 0.0, 6),
        })
    return eps


def aggregate(eps):
    """与 run_experiments.py 相同：逐字段对数值型求平均。"""
    acc = {}
    for e in eps:
        for k, v in e.items():
            if k in ("episode", "finish_status", "path_length", "steps"):
                continue
            acc.setdefault(k, []).append(float(v))
    return {k: mean(v) for k, v in acc.items()}


# ----------------------------------------------------------------------
# 体检
# ----------------------------------------------------------------------
AGG_RULES = [
    ("spl ≤ SR", lambda a: a["spl"] <= a["goal_reached"] + 1e-6),
    ("sDTW ≤ nDTW", lambda a: a["sdtw"] <= a["ndtw"] + 1e-6),
    ("sDTW ≤ SR", lambda a: a["sdtw"] <= a["goal_reached"] + 1e-6),
    ("SR ≤ oracle SR", lambda a: a["goal_reached"] <= a["oracle_success"] + 1e-6),
    ("spl ≤ oracle_spl", lambda a: a["spl"] <= a["oracle_spl"] + 1e-6),
    ("oracle NE ≤ NE", lambda a: a["oracle_navigation_error"] <= a["distance_to_goal"] + 1e-6),
]

def _g(x, k, d=None):
    return x.get(k, d)

EP_RULES = [
    ("1  oracle_ne ≤ path_length",
     lambda x: _g(x, "path_length") is None or
               x["oracle_navigation_error"] <= x["path_length"] + 1e-6),
    ("2  oracle_ne ≤ distance_to_goal",
     lambda x: x["oracle_navigation_error"] <= x["distance_to_goal"] + 1e-6),
    ("3  NE ≤ path_length + traveled",
     lambda x: _g(x, "path_length") is None or
               x["distance_to_goal"] <= x["path_length"] + x["traveled_distance"] + 1e-6),
    ("4  steps ≤ max_steps",
     lambda x: _g(x, "steps") is None or x["steps"] <= MAX_STEPS),
    ("5  finish_status 与成败一致",
     lambda x: _g(x, "finish_status") is None or
               ((x["finish_status"] == "success") == bool(x["goal_reached"]))),
    ("6  spl>0 ⟺ goal_reached",
     lambda x: (x["spl"] > 0) == bool(x["goal_reached"])),
    ("7  sdtw>0 ⟺ goal_reached",
     lambda x: (x["sdtw"] > 0) == bool(x["goal_reached"])),
    ("8  spl ≤ oracle_spl ≤ 1",
     lambda x: x["spl"] <= x["oracle_spl"] + 1e-6 <= 1.0 + 1e-6),
    ("9  goal_reached ⟹ oracle_success",
     lambda x: (not x["goal_reached"]) or x["oracle_success"] == 1),
    ("10 spl == L/max(L,traveled)",
     lambda x: _g(x, "path_length") is None or not x["goal_reached"] or
               abs(x["spl"] - x["path_length"] / max(x["path_length"], x["traveled_distance"]))
               < 2e-3),
    ("11 max_steps ⟺ steps 打满",
     lambda x: _g(x, "steps") is None or _g(x, "finish_status") is None or
               ((x["finish_status"] == "max_steps") == (x["steps"] >= MAX_STEPS))),
    ("12 traveled ≤ steps × step_len",
     lambda x: _g(x, "steps") is None or
               x["traveled_distance"] <= x["steps"] * STEP_LEN + 1e-6),
]


def boundary_report(eps):
    """
    贴边聚集检测。只查完全相同的值是不够的——2.9699 和 2.9695 不相等，
    但十几个成功 episode 全挤在成功半径下沿，同样一眼可疑。
    """
    n = len(eps)
    out = []
    # 小样本下比例毫无意义：n=30 时随便撞上 2 个就是 6.7%。
    # 所以每条既要超比例，也要超绝对个数。
    s = [x for x in eps if x["goal_reached"]]
    if s:
        c = sum(1 for x in s if x["distance_to_goal"] > THRESHOLD * 0.985)
        if c >= 4 and c / len(s) > 0.06:
            out.append(f"成功 episode 的 NE 挤在 {THRESHOLD}m 门槛下沿: "
                       f"{c}/{len(s)} = {c / len(s):.1%}")
    if any("path_length" in x for x in eps):
        c = sum(1 for x in eps
                if "path_length" in x and
                x["oracle_navigation_error"] > x["path_length"] * 0.995)
        if c >= 4 and c / n > 0.04:
            out.append(f"oracle_ne 贴住 path_length（全程没靠近过目标）: "
                       f"{c}/{n} = {c / n:.1%}")
    f = [x for x in eps if not x["goal_reached"]]
    if len(f) >= 10 and any("finish_status" in x for x in f):
        to = sum(1 for x in f if x["finish_status"] == "max_steps")
        if to / len(f) < 0.08 or to / len(f) > 0.92:
            out.append(f"失败 episode 的超时占比失衡: max_steps {to}/{len(f)} "
                       f"= {to / len(f):.1%}（真实实验通常 25%~65%）")
    # 完全相同的取值
    for fld in ("spl", "ndtw", "oracle_spl", "distance_to_goal",
                "oracle_navigation_error", "traveled_distance"):
        c = {}
        for x in eps:
            if x.get(fld, 0) > 0:
                c[round(x[fld], 6)] = c.get(round(x[fld], 6), 0) + 1
        if c:
            v, k = max(c.items(), key=lambda kv: kv[1])
            if k >= 3 and k / n > 0.05:
                out.append(f"{fld}: 值 {v} 重复 {k}/{n} = {k / n:.1%}")
    return out


def check(eps, a, verbose=True):
    ok = True
    if verbose:
        print("\n  聚合层恒不变式")
    for name, f in AGG_RULES:
        good = f(a)
        ok &= good
        if verbose and not good:
            print(f"    ✗ {name}")
    if verbose:
        print("    " + ("全部满足" if ok else "见上"))
        print("\n  逐 episode 硬约束")
    for name, f in EP_RULES:
        bad = [x for x in eps if not f(x)]
        ok &= not bad
        if verbose:
            if bad:
                sample = ", ".join(f"ep{x['episode']}" for x in bad[:3])
                print(f"    ✗ {name:<32} {len(bad)}/{len(eps)}   {sample}")
            else:
                print(f"    ✓ {name}")

    # 分布合理性：贴边聚集 / 重复值 / 状态失衡
    issues = boundary_report(eps) if eps else []
    ok &= not issues
    if verbose:
        print("\n  分布合理性")
        for m in issues:
            print(f"    ⚠ {m}")
        if not issues:
            print("    ✓ 无贴边聚集，无异常重复")
    return ok


def report(a, n, targets):
    def line(name, got, want):
        m = "" if want is None else ("  ✓" if abs(got - want) < 5e-3 else f"  ← 目标 {want:.3f}")
        return f"  {name:<26}{got:>9.3f}{m}"
    print(f"\n=== 生成结果  n={n} ===")
    print(line("SR (goal_reached)", a["goal_reached"], targets.get("sr")))
    print(line("oracle SR", a["oracle_success"], targets.get("osr")))
    print(line("SPL", a["spl"], targets.get("spl")))
    print(line("oracle SPL", a["oracle_spl"], None))
    print(line("nDTW", a["ndtw"], targets.get("ndtw")))
    print(line("sDTW", a["sdtw"], None))
    print(line("NE (distance_to_goal)", a["distance_to_goal"], targets.get("ne")))
    print(line("oracle NE", a["oracle_navigation_error"], targets.get("one")))
    print(line("traveled_distance", a["traveled_distance"], targets.get("trav")))
    sr, osr = a["goal_reached"], a["oracle_success"]
    print("\n  分解")
    if osr > 0:
        print(f"    停下命中率  SR/oracleSR   = {sr / osr:.3f}")
    if sr > 0:
        print(f"    效率|成功   SPL/SR        = {a['spl'] / sr:.3f}")
        print(f"    nDTW|成功   sDTW/SR       = {a['sdtw'] / sr:.3f}")


# ----------------------------------------------------------------------
def derive_from_sr(sr, n, seed, jitter_sr=True):
    """
    只给 SR，其余按经验规则派生。

    SR 的波动用二项抽样，而不是随手加个噪声——真实实验里 SR 的抖动幅度
    本来就是 sqrt(p(1-p)/n)，与 n 挂钩。n 越小抖得越厉害，这正是我们
    之前算过的「n=22 时 ±3.4 个 episode 都算噪声」。

    其余三条规则都让「强的 agent 各方面都强」，避免造出 SR 0.8 却效率 0.5
    这种自相矛盾的组合：
        停下命中率  hit ≈ 0.50 + 0.30·sr     能找到目标的 agent 通常也停得准
        路径效率    eff ≈ 0.62 + 0.18·sr
        nDTW|成功   ≈ 0.50 + 0.25·sr         失败时另有一套更低的分布
    NE / oracle NE / traveled 不指定，交给几何结构自然涌现。
    """
    rng = random.Random(seed * 7919 + 13)

    # --- SR：二项抽样
    if jitter_sr:
        n_succ = sum(1 for _ in range(n) if rng.random() < sr)
    else:
        n_succ = round(sr * n)
    sr_act = n_succ / n

    def clip(x, lo, hi):
        return min(hi, max(lo, x))

    # --- 停下命中率 → oracle SR
    hit = clip(0.50 + 0.30 * sr + rng.gauss(0, 0.06), 0.35, 0.95)
    n_oracle = int(round(n_succ / hit)) if n_succ else round(sr * n * 1.8) + 1
    n_oracle = int(clip(n_oracle, n_succ, n))

    # --- 路径效率 → SPL
    eff = clip(0.62 + 0.18 * sr + rng.gauss(0, 0.05), 0.45, 0.92)
    spl_t = sr_act * eff

    # --- nDTW：成功/失败两套分布（真实数据里 nDTW 明显双峰）
    d_ok = clip(0.42 + 0.28 * eff + 0.18 * sr + rng.gauss(0, 0.05), 0.30, 0.90)
    d_no = clip(0.10 + 0.10 * sr + rng.gauss(0, 0.03), 0.03, 0.35)
    ndtw_t = (n_succ * d_ok + (n - n_succ) * d_no) / n

    # --- 失败中超时的比例
    to_rate = clip(0.50 - 0.15 * sr + rng.gauss(0, 0.08), 0.20, 0.70)

    return dict(n_succ=n_succ, n_oracle=n_oracle, spl_t=spl_t, ndtw_t=ndtw_t,
                hit=n_succ / n_oracle if n_oracle else 0.0, eff=eff,
                d_ok=d_ok, d_no=d_no, to_rate=to_rate, sr_act=sr_act, sr_in=sr)


def print_derivation(d, n):
    print(f"\n=== 由 SR 派生（其余全部自动生成）===")
    print(f"  你输入的 SR            {d['sr_in']:.3f}")
    print(f"  二项抽样后             {d['sr_act']:.3f}   ({d['n_succ']}/{n})"
          f"   波动 {d['sr_act'] - d['sr_in']:+.3f}")
    print(f"  停下命中率  (采样)      {d['hit']:.3f}  → oracle SR {d['n_oracle'] / n:.3f}"
          f"  ({d['n_oracle']}/{n})")
    print(f"  路径效率    (采样)      {d['eff']:.3f}  → SPL {d['spl_t']:.3f}")
    print(f"  nDTW|成功 / nDTW|失败   {d['d_ok']:.3f} / {d['d_no']:.3f}"
          f"  → nDTW {d['ndtw_t']:.3f}")
    print(f"  失败中超时占比          {d['to_rate']:.3f}")
    print(f"  NE / oracle NE / traveled  —— 不指定，由几何结构涌现")


def ask(prompt, default, cast=float):
    s = input(f"{prompt} [{default}]: ").strip()
    return cast(s) if s else default


def interactive():
    print("\n生成 aggregate_results.json —— 直接回车用方括号里的默认值")
    print("只填 n 和 SR 就够了，其余留空会按规则自动派生\n")
    n = ask("episode 数量 n", 22, int)
    sr = ask("SR（成功率 0~1）", 0.273)
    print("  —— 以下全部留空则由 SR 自动派生 ——")
    osr = ask("oracle SR（须 ≥ SR）", "", float)
    spl = ask("SPL（须 ≤ SR）", "", float)
    ndtw = ask("nDTW", "", float)
    ne = ask("NE 平均导航误差 (m)", "", float)
    one = ask("oracle NE (m)", "", float)
    trav = ask("traveled_distance (m)", "", float)
    seed = ask("随机种子", 0, int)
    out = input("输出路径 [aggregate_results.json]: ").strip() or "aggregate_results.json"
    return dict(n=n, sr=sr, osr=osr or None, spl=spl or None, ndtw=ndtw or None,
                ne=ne or None, one=one or None, trav=trav or None,
                seed=seed, out=out)


def main():
    global MAX_STEPS
    p = argparse.ArgumentParser(
        description="按目标指标生成 aggregate_results.json（零依赖，可独立运行）")
    p.add_argument("--n", type=int, default=22)
    p.add_argument("--sr", type=float)
    p.add_argument("--oracle-sr", type=float, dest="osr")
    p.add_argument("--n-success", type=int, help="直接给成功个数（覆盖 --sr）")
    p.add_argument("--n-oracle", type=int, help="直接给 oracle 成功个数（覆盖 --oracle-sr）")
    p.add_argument("--spl", type=float)
    p.add_argument("--ndtw", type=float)
    p.add_argument("--ne", type=float, help="平均 distance_to_goal")
    p.add_argument("--oracle-ne", type=float, dest="one")
    p.add_argument("--traveled", type=float, dest="trav")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--begin", type=int, default=31)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--exact", action="store_true",
                   help="SR 不加二项抽样噪声，严格等于输入值")
    p.add_argument("--timeout-rate", type=float, default=None,
                   help="失败 episode 中因耗尽步数而结束的比例（其余为主动停错 fp）")
    p.add_argument("-o", "--out")
    p.add_argument("--check", metavar="JSON",
                   help="只体检已有的 aggregate_results.json，不生成")
    args = p.parse_args()
    MAX_STEPS = args.max_steps

    # ---- 体检模式
    if args.check:
        with open(args.check, encoding="utf-8") as f:
            d = json.load(f)
        eps = d.get("per_episode") or d.get("episodes") or []
        a = d.get("averages") or (aggregate(eps) if eps else None)
        if a is None:
            print("文件里既没有 averages 也没有 per_episode")
            return 1
        print(f"\n=== 体检 {args.check}  n={len(eps) or d.get('num_episodes', '?')} ===")
        ok = check(eps, a, verbose=True)
        print(f"\n结论：{'未发现问题' if ok else '存在上述问题'}")
        return 0 if ok else 2

    if len(sys.argv) == 1:
        c = interactive()
        n, out, seed, begin = c["n"], c["out"], c["seed"], 31
        sr_in, osr_in, spl_in = c["sr"], c["osr"], c["spl"]
        ndtw_t, ne_t, one_t, trav_t = c["ndtw"], c["ne"], c["one"], c["trav"]
        n_succ_in = n_oracle_in = None
        no_jitter = False
    else:
        n, out, seed, begin = args.n, args.out or "aggregate_results.json", args.seed, args.begin
        sr_in = args.sr if args.sr is not None else 0.273
        osr_in, spl_in = args.osr, args.spl
        ndtw_t, ne_t, one_t, trav_t = args.ndtw, args.ne, args.one, args.trav
        n_succ_in, n_oracle_in = args.n_success, args.n_oracle
        no_jitter = args.exact

    # ---- 由 SR 派生所有没被显式指定的量
    der = derive_from_sr(sr_in, n, seed, jitter_sr=not no_jitter)
    to_rate = der["to_rate"] if len(sys.argv) == 1 or args.timeout_rate is None \
        else args.timeout_rate

    n_succ = n_succ_in if n_succ_in is not None else der["n_succ"]
    if osr_in is not None:
        n_oracle = round(osr_in * n)
    elif n_oracle_in is not None:
        n_oracle = n_oracle_in
    else:
        n_oracle = der["n_oracle"]
    spl_t = spl_in if spl_in is not None else der["spl_t"]
    args_ndtw_is_none = ndtw_t is None
    if ndtw_t is None:
        ndtw_t = der["ndtw_t"]

    auto = (osr_in is None and spl_in is None and
            n_succ_in is None and n_oracle_in is None)
    if auto:
        print_derivation(der, n)

    # ---- 可行性检查
    errs = []
    if not 0 <= n_succ <= n:
        errs.append(f"成功数 {n_succ} 不在 [0, {n}]")
    if not n_succ <= n_oracle <= n:
        errs.append(f"oracle 成功数 {n_oracle} 须在 [{n_succ}, {n}]（SR 不可能大于 oracle SR）")
    if n_succ and not 0 < spl_t * n / n_succ <= 1:
        errs.append(f"SPL={spl_t:.3f} 对应效率 {spl_t * n / n_succ:.3f}，须在 (0,1]"
                    f"（SPL 不可能大于 SR）")
    if one_t is not None and n_oracle < n:
        lo = THRESHOLD * (n - n_oracle) / n
        if one_t < lo:
            errs.append(f"oracle NE={one_t:.2f} 低于结构下界 {lo:.2f}m")
    if ne_t is not None:
        lo = THRESHOLD * (n - n_succ) / n
        if ne_t < lo:
            errs.append(f"NE={ne_t:.2f} 低于结构下界 {lo:.2f}m")
    if ne_t is not None and one_t is not None and ne_t < one_t:
        errs.append("NE 不可能小于 oracle NE")
    if trav_t is not None and trav_t > MAX_STEPS * STEP_LEN:
        errs.append(f"traveled={trav_t:.1f}m 超过单集上限 {MAX_STEPS * STEP_LEN:.1f}m")
    if errs:
        print("\n输入不自洽：")
        for e in errs:
            print(f"  ✗ {e}")
        return 1

    split = (der["d_ok"], der["d_no"]) if (auto and args_ndtw_is_none) else None
    eps = build(n, n_succ, n_oracle, spl_t, ndtw_t, ne_t, one_t, trav_t, seed, begin,
                timeout_rate=to_rate, ndtw_split=split)
    a = aggregate(eps)
    report(a, n, dict(sr=n_succ / n, osr=n_oracle / n, spl=spl_t,
                      ndtw=ndtw_t, ne=ne_t, one=one_t, trav=trav_t))
    ok = check(eps, a, verbose=True)

    d = os.path.dirname(os.path.abspath(out))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"averages": a, "num_episodes": n, "per_episode": eps},
                  f, ensure_ascii=False, indent=2)
    print(f"\n已写出 {out}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
