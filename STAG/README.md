# STAG: Spatio-Temporal Alignment and Grounding for Zero-Shot VLN

Instructions are decomposed into **spatio-temporal constraints**; observations are built into
**spatio-temporal facts**; the two are aligned in a reasoning layer. The spatial term and the
temporal term are multiplied at exactly one place — **frontier scoring** — which is the only
decision point where both participate.

Task: VLN-CE (R2R) on Habitat / Matterport3D, continuous action space, zero-shot — no training,
using an off-the-shelf VLM and detector.

> **Built on [HSGM](https://github.com/Teacher-Tom/HSGM_public)** (Li et al., CVPR 2026). The
> simulator wrapper, A\* planner, and the overall map-then-plan skeleton come from that codebase;
> see [Acknowledgement](#acknowledgement). STAG replaces the instruction handling, the map's
> semantic/region layers, and the entire verification-and-guidance stage.

---

## Method

### A · Spatio-temporal instruction decomposition (STID)

Two channels, where **segment identity is decided entirely by rules — the LLM may not change it.**

| Channel | Responsibility |
|---|---|
| ① Rules (no LLM) | Split on discourse cues; extract landmark noun phrases; assign roles by the nearest preceding cue — `waypoint_marker` / `destination_marker` / `avoid_marker`; detect `left`/`right`/`behind` |
| ② LLM gap-filling | `local_start` / `local_end` per segment; `relative_duration ∈ [0,1]` normalized so Σ=1; roles the rules left undecided; align landmark names to the shared closed vocabulary |

Roles are assigned **per position, not per sentence** — one clause can contain landmarks of several
roles. A boundary self-check compares `stage_k.local_end` against `stage_{k+1}.local_start` and
emits `boundary_consistent`.

*Rationale:* letting the LLM decide segment identity produces duplicate `stage_index` values, which
silently merges subtasks. Rule-based segmentation is stable and reproducible.

### B · Spatio-temporal map construction

- **Geometry** — RGB-D from four views (front / left / right / back), back-projected and fused into
  `navigable` / `obstacle` / `scene` point clouds. World frame is Z-up **left-handed** (`det = −1`).
- **Objects** — YOLOE open-vocabulary detection bound to a 207-class closed vocabulary shared with
  the instruction decomposer. Front view only by default; side views are segmented on demand when
  the current subtask still has an ungrounded object landmark.
- **Regions** — free-space geometric partition: distance transform → seeds with clearance > 1.0 m →
  multi-source **geodesic** expansion with a door cut (0.55 m) and an expansion cap (3.0 m). Cells
  outside both gates keep `label 0`, meaning "belongs to no identified room". A room is named only
  when scoring reaches ≥ 1.0 **and at least two distinct evidence classes corroborate it**.

### C · Spatio-temporal reasoning

|  | Verification — *did I do it?* | Guidance — *where next?* |
|---|---|---|
| **Spatial** | distance-to-goal gate; "was closer before" check | inverted room–object prior; strong + weak room hypotheses |
| **Temporal** | order gate (via ≺ destination); end-condition vs. map facts; sticky `seen`/`near`/`passed` | remaining step budget → reach radius |

The two guidance terms meet at

```
score(f) = spatial(f) × temporal(f)
```

A principle that recurs at three independent sites: **"cannot confirm" ≠ "confirmed false".**
When evidence is insufficient the module returns `unknown` and stays silent rather than voting
against. See `docs/PAPER_CONTEXT.md` §4.1.

### D · Decision

Action space is waypoints `1..n` (A\*-verified reachable), turns `L`/`R`/`B`, and frontiers
`F1..F3` (multi-step paths). The VLM sees multi-view images plus a top-down map and outputs an
action together with `subtask_done` — the two are **decoupled**, so advancing the subtask does not
force a movement and vice versa.

---

## Repository Structure

```
STAG/
├── config/
│   ├── vlnce_test.yaml         # R2R / RxR (VLN-CE, MP3D) — includes the `features:` ablation switches
│   └── objnav_test.yaml        # ObjectNav (HM3D)
├── docs/
│   └── PAPER_CONTEXT.md        # method, measured results, negative results, implementation pitfalls
├── figures/
│   └── framework.svg|pdf|png   # framework figure
├── scripts/
│   ├── batch_test.sh           # multi-episode evaluation
│   ├── check_sync.py           # verifies each source file has its expected top-level defs
│   ├── compare_runs.py         # forces episode intersection across runs before comparing
│   ├── gen_aggregate.py        # synthesize an aggregate_results.json; also `--check` any real one
│   ├── simulate_metrics.py     # explore how the metrics move together
│   └── test_landmark_grounding.py   # offline self-test (not imported by src/)
├── src/
│   ├── run_experiments.py      # main entry point
│   ├── simWrapper.py           # Habitat-Sim wrapper + PolarAction
│   ├── mapper.py               # online spatio-temporal mapping (Instruct_Mapper)
│   ├── agent/
│   │   ├── agent.py            # VLM planner + subtask FSM + verification/guidance
│   │   └── decomposer/         # spatio-temporal instruction decomposition
│   ├── segmentation/           # YOLOE instance segmentation + closed vocabulary
│   ├── mapping_utils/          # geometry / projection / transform / A* path planning
│   └── utils.py
├── requirements.txt
└── LICENSE                     # Apache-2.0
```

---

## Installation

Python 3.9+, CUDA GPU (`cuda:0` by default).

```bash
conda create -n stag python=3.9 -y
conda activate stag

# Habitat-Sim must come from conda
conda install -c conda-forge -c aihabitat habitat-sim withbullet -y
pip install habitat-lab
pip install -r requirements.txt
```

### YOLOE checkpoint

Place a YOLOE-seg checkpoint at `ckpt/yoloe-26l-seg.pt`.

---

## Datasets

The code reads datasets from `../data/` (relative to `src/`).

```
data/
├── scene_datasets/
│   ├── hm3d_v0.2/val/...                  # HM3D scenes (ObjectNav)
│   └── mp3d/<scene>/<scene>.glb           # MP3D scenes (R2R / RxR)
└── datasets/
    ├── objectnav_hm3d_v2/val/content/*.json.gz
    ├── r2r_vlnce/val_unseen.json
    └── RxR_VLNCE_v0/val_unseen/           # + *_guide_gt.json
```

See [VLN-CE](https://github.com/jacobkrantz/VLN-CE) for episode files and
[Habitat-Lab](https://github.com/facebookresearch/habitat-lab) for scene data. Update `scene_path`
in the YAML configs accordingly.

---

## VLM Configuration

An **OpenAI / Azure OpenAI compatible** chat-completion endpoint, configured via environment
variables.

```bash
# Standard OpenAI, or any compatible server such as a local vLLM
export OPENAI_API_KEY=sk-xxxxxxxx                  # or "EMPTY" for local vLLM
export OPENAI_BASE_URL=https://api.openai.com/v1   # or http://localhost:8000/v1
export OPENAI_MODEL=gpt-5

# Azure takes precedence when endpoint and key are both set
export AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
export AZURE_OPENAI_API_KEY=<your-azure-key>
export AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

The proxy at `127.0.0.1:7897` is auto-detected at import time, and only enabled when you have
**not** already set `http_proxy`/`https_proxy` **and** it is TCP-reachable.

---

## Quick Start

```bash
bash scripts/batch_test.sh r2r 0 10 Qwen/Qwen3.6-27B
```

Or directly:

```bash
cd src

python run_experiments.py --task r2r --config ../config/vlnce_test.yaml \
    --model_name Qwen/Qwen3.6-27B --begin_idx 0 --end_idx -1 --max_steps 100

python run_experiments.py --task rxr       --config ../config/vlnce_test.yaml
python run_experiments.py --task objectnav --config ../config/objnav_test.yaml
```

### CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--task` | `r2r` | `objectnav` / `r2r` / `rxr` |
| `--config` | `../config/vlnce_test.yaml` | YAML config path |
| `--comment` | `exp` | tag appended to run name |
| `--begin_idx` | `0` | start episode index (inclusive) |
| `--end_idx` | `-1` | end index (exclusive, `-1` for all) |
| `--max_steps` | `100` | max agent steps per episode |
| `--model_name` | `$OPENAI_MODEL` or `gpt-5` | VLM model name |
| `--show` | off | live visualization (requires display) |
| `--episode_ids` | — | `"1,2,3"` or path to a file |

---

## Ablation switches

Set in the `features:` block of `config/vlnce_test.yaml`, overridable per-run by environment
variables so you never have to edit the file mid-experiment:

| Env var | Config key | Effect when 0 |
|---|---|---|
| `STZS_GUIDANCE` | `enable_guidance` | no frontier guidance (explore hints + `F` rings on the map) |
| `STZS_INTERCEPTION` | `enable_interception` | no completion/stop interception (order gate, premature completion, receding, too-far) |
| `STZS_FRONTIER_ACTION` | `enable_frontier_action` | frontiers are not offered as selectable actions |
| `STZS_VLM_DETECTION` | `enable_vlm_detection` | detection uses local YOLOE only, never the VLM |

```bash
STZS_GUIDANCE=0 STZS_INTERCEPTION=1 python run_experiments.py \
    --task r2r --begin_idx 31 --end_idx 53 --comment g0_i1
```

> The env prefix is still `STZS_` from an earlier working name; it is kept so existing run scripts
> and logs stay valid. Rename together with the config keys when convenient.

---

## Outputs

- `results/results_<env.name>/episode_<id>_results.json` — per-episode metrics and VLM log
- `results/results_<env.name>/episode_<id>.mp4` — observation video
- `results/results_<env.name>/aggregate_results.json` — aggregated SR / SPL / nDTW / sDTW

Run `python scripts/gen_aggregate.py --check <path>` on any results file to verify the twelve
per-episode geometric invariants (e.g. `oracle_ne ≤ path_length`, `spl == L/max(L, traveled)`) —
a violation means a bug in metric computation, not a bad agent.

Use `python scripts/compare_runs.py` to compare runs; it forces the episode intersection first,
because comparing different episode subsets produces confident nonsense.

---

## Acknowledgement

This work is built on **HSGM** — *Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for
Vision Language Navigation*, Li et al., CVPR 2026
([code](https://github.com/Teacher-Tom/HSGM_public) ·
[paper](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Bridging_the_2D-3D_Gap_A_Hierarchical_Semantic-Geometric_Map_for_Vision_CVPR_2026_paper.html)),
released under Apache-2.0. The Habitat wrapper, A\* path planner, and map-then-plan structure derive
from that codebase.

```bibtex
@InProceedings{Li_2026_CVPR,
    author    = {Li, Kailing and Qian, Tianwen and Yang, Lijin and Fu, Yuqian and Gong, Jingyu and Wang, Xiaoling and He, Liang},
    title     = {Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {15243-15252}
}
```

---

## License

[Apache License 2.0](LICENSE) — inherited from the upstream project.
