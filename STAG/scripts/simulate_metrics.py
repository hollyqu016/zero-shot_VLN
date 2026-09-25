#!/usr/bin/env python3
"""
合成生成 aggregate_results.json，用来观察各项指标之间的联动规律。

不是给论文造数据用的，是用来回答"为什么这几个数总是落在某个区间"。
做法：把导航能力拆成三个**正交**的旋钮，然后用与 run_experiments.py
完全相同的公式算出全部指标——所以恒不变式是由构造保证的，不是凑出来的。

    explore     能不能走到目标 3m 内         -> oracle_success
    hit         走到了肯不肯停               -> SR / oracle_success
    efficiency  成功时绕了多少路             -> SPL / SR

三个乘法分解（脚本会打印出来验证）：
    SR   = oracle_SR × hit
    SPL  = SR        × efficiency|success
    sDTW = SR        × nDTW|success

用法：
    # 生成一份，并打印分解与恒不变式检查
    python scripts/simulate_metrics.py --n 22 --explore 0.55 --hit 0.5 --efficiency 0.77

    # 扫描某个旋钮，看其余指标怎么跟着动（回答"规律是什么"）
    python scripts/simulate_metrics.py --sweep hit
    python scripts/simulate_metrics.py --sweep explore

    # 拟合一份真实结果：给定目标 SR/oracle_SR/SPL，反推需要的旋钮
    python scripts/simulate_metrics.py --match 0.273 0.545 0.209
"""

import argparse
import json
import math
import os
import random
import sys

SUCCESS_THRESHOLD = 3.0      # 与 config 中的 env.success_threshold 一致
STEP_LEN = 0.6               # 每步有效位移（含转向开销）


# ----------------------------------------------------------------------
# 单个 episode 的合成
# ----------------------------------------------------------------------
def make_episode(ep_id, rng, explore, hit, efficiency, max_steps, threshold):
    """
    生成一个 episode 的指标。刻意保持"物理上说得通"：
      · final_distance ≥ oracle_ne（后者是轨迹上的历史最小值，不可能比终值大）
      · traveled ≥ 0，且失败时通常更大（多绕）
      · oracle_spl 取首次成功时刻的 SPL，那时 traveled 更小，所以 ≥ spl
    这些约束就是恒不变式的来源。
    """
    # 参考路径长度：R2R 的量级，对数正态更贴近真实分布（长尾）
    L_geo = min(25.0, max(3.0, rng.lognormvariate(math.log(9.5), 0.45)))

    reached = rng.random() < explore

    if reached:
        # 曾经进入过阈值内，历史最近距离落在 [0.1, threshold)
        oracle_ne = rng.uniform(0.1, threshold * 0.95)
    else:
        # 没进去过：差多远？多数是"差一个房间"，少数彻底跑偏
        oracle_ne = threshold + rng.expovariate(1 / 4.0)

    # 路径效率：成功者更直，失败者更绕。eff ∈ (0,1]，traveled = L_geo / eff
    base_eff = efficiency if reached else efficiency * 0.75
    eff = min(0.98, max(0.15, rng.gauss(base_eff, 0.12)))
    traveled = L_geo / eff

    # 步数上限会截断行程
    step_cap = max_steps * STEP_LEN
    timed_out = traveled > step_cap
    traveled = min(traveled, step_cap)

    # 停不停：只有到过阈值内才谈得上"停对"
    stopped_well = reached and (not timed_out) and (rng.random() < hit)

    if stopped_well:
        # 停在阈值内，且不可能比历史最近还近
        final = min(threshold * 0.98, oracle_ne + rng.uniform(0.0, 0.6))
        status = "success"
    else:
        # 走过头 / 停错地方 / 超时。终值必然 ≥ 历史最近
        overshoot = rng.expovariate(1 / 3.2)
        final = max(oracle_ne + 0.2, oracle_ne + overshoot)
        if reached:
            final = max(final, threshold * 1.05)   # 到过但没停对 -> 必然在阈值外
        status = "max_steps" if timed_out else "fp"

    success = final < threshold
    spl = (L_geo / max(L_geo, traveled)) if success else 0.0

    # oracle_spl：首次进入阈值那一刻的 SPL。彼时行程更短，故 ≥ spl。
    # 这也是它系统性偏高的原因——"成功"只要求进 3m 球，却拿整条测地距离当分子。
    if reached:
        traveled_at_first = traveled * rng.uniform(0.45, 1.0)
        oracle_spl = L_geo / max(L_geo, traveled_at_first)
    else:
        oracle_spl = 0.0
    oracle_spl = max(oracle_spl, spl)   # 恒不变式：历次最大值不可能小于终值

    # nDTW：与路径效率正相关，但不等价——抄近路也可能完全不贴合参考路线
    dev = (1.0 / eff - 1.0) * 1.6 + (0.0 if reached else 0.9)
    ndtw = min(0.95, max(0.01, math.exp(-dev) * rng.uniform(0.75, 1.15)))
    sdtw = ndtw if success else 0.0

    return {
        "episode": ep_id,
        "distance_to_goal": final,
        "spl": spl,
        "goal_reached": success,
        "finish_status": status,
        "traveled_distance": traveled,
        "done": True,
        "oracle_navigation_error": oracle_ne,
        "oracle_success": 1 if reached else 0,
        "oracle_spl": oracle_spl,
        "ndtw": ndtw,
        "sdtw": sdtw,
    }


def aggregate(eps):
    """与 run_experiments.py 相同的聚合方式：逐字段累计数值型。"""
    acc = {}
    for e in eps:
        for k, v in e.items():
            if k == "episode":
                continue
            if isinstance(v, bool):
                v = 1.0 if v else 0.0
            if isinstance(v, (int, float)):
                acc.setdefault(k, []).append(float(v))
    return {k: sum(v) / len(v) for k, v in acc.items()}


# ----------------------------------------------------------------------
# 分析
# ----------------------------------------------------------------------
INVARIANTS = [
    ("spl ≤ SR", lambda a: a["spl"] <= a["goal_reached"] + 1e-9),
    ("sDTW ≤ nDTW", lambda a: a["sdtw"] <= a["ndtw"] + 1e-9),
    ("sDTW ≤ SR", lambda a: a["sdtw"] <= a["goal_reached"] + 1e-9),
    ("SR ≤ oracle SR", lambda a: a["goal_reached"] <= a["oracle_success"] + 1e-9),
    ("spl ≤ oracle_spl", lambda a: a["spl"] <= a["oracle_spl"] + 1e-9),
    ("oracle NE ≤ NE", lambda a: a["oracle_navigation_error"] <= a["distance_to_goal"] + 1e-9),
]


def report(a, n, title=""):
    if title:
        print(f"\n=== {title} ===")
    print(f"  n={n}  SR={a['goal_reached']:.3f}  oracleSR={a['oracle_success']:.3f}  "
          f"SPL={a['spl']:.3f}  oSPL={a['oracle_spl']:.3f}")
    print(f"  NE={a['distance_to_goal']:.2f}m  oracleNE={a['oracle_navigation_error']:.2f}m  "
          f"traveled={a['traveled_distance']:.2f}m  nDTW={a['ndtw']:.3f}  sDTW={a['sdtw']:.3f}")

    sr, osr = a["goal_reached"], a["oracle_success"]
    print("  分解：")
    print(f"    停下命中率 SR/oracleSR      = {sr / osr:.3f}" if osr > 0 else "    停下命中率 —")
    print(f"    效率|成功  SPL/SR           = {a['spl'] / sr:.3f}" if sr > 0 else "    效率|成功 —")
    print(f"    nDTW|成功  sDTW/SR          = {a['sdtw'] / sr:.3f}" if sr > 0 else "    nDTW|成功 —")
    bad = [name for name, f in INVARIANTS if not f(a)]
    print(f"  恒不变式：{'全部满足' if not bad else '违反 → ' + ', '.join(bad)}")


def run(n, explore, hit, efficiency, seed, max_steps, threshold, begin=31):
    rng = random.Random(seed)
    eps = [make_episode(begin + i, rng, explore, hit, efficiency, max_steps, threshold)
           for i in range(n)]
    return eps, aggregate(eps)


# ----------------------------------------------------------------------
def cmd_sweep(args):
    """扫一个旋钮，看其余指标怎么跟着动。"""
    knob = args.sweep
    print(f"\n扫描 {knob}（其余固定：explore={args.explore} hit={args.hit} "
          f"efficiency={args.efficiency}，每点 {args.n}×{args.repeat} episode）\n")
    print(f"{knob:>10}{'SR':>8}{'oSR':>8}{'命中率':>9}{'SPL':>8}{'oSPL':>8}"
          f"{'NE':>8}{'oNE':>8}{'nDTW':>8}{'sDTW':>8}")
    print("-" * 83)
    for v in [round(0.1 * i, 2) for i in range(1, 10)]:
        kw = dict(explore=args.explore, hit=args.hit, efficiency=args.efficiency)
        kw[knob] = v
        accs = []
        for r in range(args.repeat):
            _, a = run(args.n, seed=args.seed + r, max_steps=args.max_steps,
                       threshold=args.threshold, **kw)
            accs.append(a)
        a = {k: sum(x[k] for x in accs) / len(accs) for k in accs[0]}
        sr, osr = a["goal_reached"], a["oracle_success"]
        print(f"{v:>10.2f}{sr:>8.3f}{osr:>8.3f}{(sr/osr if osr else 0):>9.3f}"
              f"{a['spl']:>8.3f}{a['oracle_spl']:>8.3f}"
              f"{a['distance_to_goal']:>8.2f}{a['oracle_navigation_error']:>8.2f}"
              f"{a['ndtw']:>8.3f}{a['sdtw']:>8.3f}")
    print("\n观察要点：")
    print("  · 只调 hit 时 oracleSR 完全不动，SR 与 SPL 同比例变化")
    print("    —— 这正是真实实验里停止判定改进的特征（oracleSR 不变、SR 上升）")
    print("  · 只调 explore 时 oracleSR 与 SR 同向，命中率基本不变")
    print("  · SPL 恒 ≤ SR，且 SPL/SR 只由 efficiency 决定，与前两者无关")


def cmd_noise(args):
    """
    同一套能力参数、不同随机种子，看指标能抖多大。
    直接回答"n=22 时差几个 episode 才算信号"。
    """
    print(f"\n能力参数固定（explore={args.explore} hit={args.hit} "
          f"efficiency={args.efficiency}），只换随机种子，重复 {args.repeat} 次\n")
    print(f"{'n':>6}{'SR 均值':>10}{'SR 标准差':>11}{'SR 90%区间':>18}"
          f"{'≈几个episode':>13}{'SPL 均值':>10}{'SPL 标准差':>11}")
    print("-" * 79)
    for n in [12, 22, 50, 100, 200, 500]:
        srs, spls = [], []
        for r in range(args.repeat):
            _, a = run(n, args.explore, args.hit, args.efficiency,
                       args.seed + r * 977, args.max_steps, args.threshold)
            srs.append(a["goal_reached"])
            spls.append(a["spl"])
        mu = sum(srs) / len(srs)
        sd = (sum((x - mu) ** 2 for x in srs) / max(1, len(srs) - 1)) ** 0.5
        lo, hi = sorted(srs)[int(0.05 * len(srs))], sorted(srs)[int(0.95 * len(srs)) - 1]
        mu2 = sum(spls) / len(spls)
        sd2 = (sum((x - mu2) ** 2 for x in spls) / max(1, len(spls) - 1)) ** 0.5
        print(f"{n:>6}{mu:>10.3f}{sd:>11.3f}{f'[{lo:.3f}, {hi:.3f}]':>18}"
              f"{f'±{1.64 * sd * n:.1f}':>13}{mu2:>10.3f}{sd2:>11.3f}")
    print("\n『≈几个episode』= 90% 置信下 SR 的波动折算成成败翻转的 episode 数。")
    print("两组配置的差异若不超过这个数，就无法与随机性区分。")


def cmd_match(args):
    """给定目标 SR / oracleSR / SPL，反推需要什么样的三个旋钮。"""
    tgt_sr, tgt_osr, tgt_spl = args.match
    explore = tgt_osr
    hit = tgt_sr / tgt_osr if tgt_osr > 0 else 0.0
    eff = tgt_spl / tgt_sr if tgt_sr > 0 else 0.0
    print(f"\n目标 SR={tgt_sr:.3f} oracleSR={tgt_osr:.3f} SPL={tgt_spl:.3f}")
    print(f"→ 反推旋钮：explore={explore:.3f}  hit={hit:.3f}  efficiency={eff:.3f}")
    if not 0 <= hit <= 1:
        print("  ⚠ 命中率超出 [0,1]：SR 不可能大于 oracleSR，输入数据有问题")
    if not 0 <= eff <= 1:
        print("  ⚠ 效率超出 [0,1]：SPL 不可能大于 SR，输入数据有问题")
    eps, a = run(args.n, explore, hit, eff, args.seed, args.max_steps, args.threshold)
    report(a, len(eps), "用反推旋钮生成的结果")
    return eps, a


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=22, help="episode 数")
    p.add_argument("--explore", type=float, default=0.55, help="走到目标 3m 内的概率")
    p.add_argument("--hit", type=float, default=0.50, help="走到了之后停对的概率")
    p.add_argument("--efficiency", type=float, default=0.77, help="成功时的路径效率")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--threshold", type=float, default=SUCCESS_THRESHOLD)
    p.add_argument("--out", type=str, default=None, help="写出 aggregate_results.json 的路径")
    p.add_argument("--sweep", choices=["explore", "hit", "efficiency"], default=None)
    p.add_argument("--repeat", type=int, default=20, help="扫描时每个点重复多少次取平均")
    p.add_argument("--match", type=float, nargs=3, metavar=("SR", "ORACLE_SR", "SPL"),
                   default=None)
    p.add_argument("--noise", action="store_true",
                   help="同参数换种子，看小样本下指标的抖动幅度")
    args = p.parse_args()

    if args.sweep:
        cmd_sweep(args)
        return 0

    if args.noise:
        cmd_noise(args)
        return 0

    if args.match:
        eps, a = cmd_match(args)
    else:
        eps, a = run(args.n, args.explore, args.hit, args.efficiency,
                     args.seed, args.max_steps, args.threshold)
        report(a, len(eps), "生成结果")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"averages": a, "num_episodes": len(eps), "per_episode": eps},
                      f, ensure_ascii=False, indent=2)
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
