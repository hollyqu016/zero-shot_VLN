#!/usr/bin/env python3
"""
两次（或多次）实验结果的对比。

只统计**所有输入文件共有的 episode**，这一点很重要：之前有一次分析就是拿
不同的 episode 子集在比（基线 9/17 vs 新版 6/17），得出"引导伤害探索"的错误
结论，实际上在真正相同的 12 个 episode 上 oracle_success 三个配置全都一样。
子集不一致的比较毫无意义，所以这里强制取交集，并把被排除的 episode 打出来。

用法：
    python scripts/compare_runs.py 基线.json 引导开.json 引导关.json
    python scripts/compare_runs.py results/results_*/aggregate_results.json
"""

import json
import os
import sys
from collections import Counter

# 连续量比二值量敏感得多：oracle_success 是 "<3m 与否"，一个 episode 从 3.1m
# 改善到 2.9m 它才跳变，而 oracle_navigation_error 能直接反映靠近程度。
FIELDS = [
    ("goal_reached", "SR", "+"),
    ("oracle_success", "oracle SR", "+"),
    ("spl", "SPL", "+"),
    ("oracle_navigation_error", "oracle NE", "-"),
    ("distance_to_goal", "最终距离", "-"),
    ("traveled_distance", "行程", None),
    ("ndtw", "nDTW", "+"),
    ("sdtw", "sDTW", "+"),
]


def load(path):
    d = json.load(open(path, encoding="utf-8"))
    return {e["episode"]: e for e in d.get("per_episode", [])}


def mean(rows, k):
    v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return sum(v) / len(v) if v else float("nan")


def main(paths):
    runs = [(os.path.basename(os.path.dirname(p)) or os.path.basename(p), load(p))
            for p in paths]
    common = set.intersection(*[set(r[1]) for r in runs])
    if not common:
        print("没有共同的 episode，无法对比")
        return 1
    for name, r in runs:
        extra = sorted(set(r) - common)
        if extra:
            print(f"  [排除] {name}: 独有 episode {extra}")
    ids = sorted(common)
    print(f"\n共同 episode {len(ids)} 个: {ids}\n")

    names = [n[:14] for n, _ in runs]
    print(f"{'指标':<14}" + "".join(f"{n:>15}" for n in names))
    print("-" * (14 + 15 * len(runs)))
    for key, label, better in FIELDS:
        vals = [mean([r[i] for i in ids], key) for _, r in runs]
        best = None
        if better == "+":
            best = max(range(len(vals)), key=lambda i: vals[i])
        elif better == "-":
            best = min(range(len(vals)), key=lambda i: vals[i])
        cells = "".join(f"{v:>14.3f}{'*' if i == best else ' '}"
                        for i, v in enumerate(vals))
        print(f"{label:<14}{cells}")

    print()
    for name, r in runs:
        rows = [r[i] for i in ids]
        s = sum(x["goal_reached"] for x in rows)
        oc = sum(x["oracle_success"] for x in rows)
        st = dict(Counter(x["finish_status"] for x in rows))
        print(f"{name[:14]:<14} SR {s}/{len(ids)}  oracle {oc}/{len(ids)}  "
              f"命中率 {s / max(oc, 1) * 100:>3.0f}%  {st}")

    # n 很小的时候 SR 差几个 episode 根本说明不了问题，明确提示
    n = len(ids)
    print(f"\n注意：n={n}。SR 相差 {max(1, round(n * 0.18))} 个 episode 以内基本在噪声范围内，"
          "\n      判断方向请优先看 oracle NE / SPL 这类连续量是否同向。")

    print("\n--- 逐 episode（OK=成功，数字为 最终距离/历史最近） ---")
    print(f"{'ep':>4}" + "".join(f"{n:>16}" for n in names))
    for i in ids:
        cells = ""
        for _, r in runs:
            e = r[i]
            cells += f"{('OK' if e['goal_reached'] else '..'):>6} " \
                     f"{e['distance_to_goal']:5.1f}/{e['oracle_navigation_error']:4.1f}"
        print(f"{i:>4}{cells}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
