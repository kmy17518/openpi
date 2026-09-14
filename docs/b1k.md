## Fine-tuning π₀.₅ on BEHAVIOR-1K

This tutorial walks through fine-tuning [π₀.₅](https://www.physicalintelligence.company/blog/pi05) on demonstration data from [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) using this repository.

**Last updated:** June 2026  
**OpenPi model:** π₀.₅ (`pi05`)  
**Robot:** R1Pro (dual-arm mobile manipulator)

> **Note:** Replace placeholders such as `<OPENPI_DIR>`, `<DATASET_ROOT>`, `<TASK_NAME>`, and `<REPO_ID>` with your own paths and identifiers throughout this guide.

---

### Overview

The BEHAVIOR-1K workflow in OpenPi follows four steps:

1. Prepare a LeRobot-format dataset from BEHAVIOR demonstrations
2. Register the robot and task (or reuse the built-in R1Pro config)
3. Compute normalization statistics and fine-tune π₀.₅
4. Deploy the checkpoint and run evaluation in BEHAVIOR-1K

OpenPi ships with a reference training config (`pi05_b1k`), B1K-specific training and serving scripts under `scripts/b1k/`, and a pre-registered R1Pro robot definition.

---

### Installation

OpenPi uses [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up, then install the environment:

```bash
cd <OPENPI_DIR>
GIT_LFS_SKIP_SMUDGE=1 uv sync
source .venv/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

`GIT_LFS_SKIP_SMUDGE=1` is required because LeRobot is pulled in as a dependency.

---

### 1. Prepare your dataset

Training expects demonstrations in [LeRobot](https://github.com/huggingface/lerobot) format on disk. Your dataset directory should look like:

```text
<DATASET_ROOT>/
├── meta/
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── observation.rgb.<camera_name>/
        └── chunk-000/
            └── file-000.mp4
```

**Collect or obtain data**

- If you collect data in BEHAVIOR-1K, convert it to LeRobot format before training. Follow the data-conversion instructions in the BEHAVIOR-1K repository for your simulator version.
- You can also start from an existing LeRobot dataset and point training at its local root.

**Set two identifiers**

| Placeholder | Description | Example |
|-------------|-------------|---------|
| `<REPO_ID>` | Dataset name used as the asset ID for norm stats (a HF hub id or short name, not a local path) | `behavior-1k/2026-challenge-demos` |
| `<DATASET_ROOT>` | Absolute path to the LeRobot dataset root (`data/`, `meta/`, `videos/`) | `/path/to/datasets/2026-challenge-demos` |

Normalization statistics are saved under `outputs/assets/<CONFIG_NAME>/<REPO_ID>/` (or, when training on a task subset, `outputs/assets/<CONFIG_NAME>/<REPO_ID>/task_subsets/<TASK_NAME>/`, see below). Both identifiers are passed on the command line as `--data.repo_id=<REPO_ID> --data.dataset-root=<DATASET_ROOT>` to every script below.

#### Which demos are on disk: one task or all 100

The 2026 challenge demos are one LeRobot v3.0 dataset on the Hub ([behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos)); each task is one chunk (`chunk-000` = task 0 = `turning_on_radio`, ...). `<DATASET_ROOT>` is either

- a **per-task partial download** — the challenge docs' `huggingface-cli download --include "data/$CHUNK/**" --include "meta/episodes/$CHUNK/**" --include "videos/*/$CHUNK/**" --include meta/info.json --include meta/stats.json --include meta/tasks.parquet --include meta/tasks.jsonl`, holding one task's chunk next to the dataset-wide metadata; or
- the **full 3.3 TB root** with all 100 tasks.

Training reads whatever is under the root (every task on disk by default). To train on one (or a few) task(s) pass `--data.task-names`; it works identically on both layouts:

```bash
--data.task-names <TASK_NAME>          # e.g. turning_on_radio; several names allowed
```

Only the selected tasks' episodes are loaded — on the full root this opens just those tasks' data files instead of all 955 — and their normalization statistics are computed over those episodes alone and stored under their own asset id, `<REPO_ID>/task_subsets/<TASK_NAME>` (several names are joined with `+`), so they never shadow the dataset-wide stats and different subsets do not clobber each other. On a partial download of that task the flag only changes the stats location; a partial download of *other* tasks fails fast (`No episodes of task ...`), as does a misspelled name (the error lists the available task names from `meta/tasks.parquet`).

---

### 2. Register robot and task

OpenPi uses Python config files to map dataset keys, observation streams, and action indices to the model.

#### Robot (R1Pro)

The R1Pro robot used in BEHAVIOR-1K is already registered in `src/openpi/configs/robots/b1k.py` as `b1k/R1Pro`. It defines:

- **Cameras:** head (`zed_link`), left wrist, and right wrist RGB streams
- **Action space (23-D):** base velocity, torso joints, dual arms, and grippers
- **Proprioception:** extracted from `observation.state` using the indices in the robot config

If you use a different robot or camera layout, copy this file and update `observations`, `action`, and `proprio` to match your dataset keys in `meta/info.json`.

#### Task prompts

`src/openpi/configs/tasks/b1k.py` maps every challenge task's snake_case name to its natural-language instruction (a verbatim copy of the demos' `meta/tasks.jsonl`, all 100 tasks). For your own tasks, add entries there:

```python
# src/openpi/configs/tasks/b1k.py
from . import TASK_REGISTRY

TASKS = {
    "<TASK_NAME>": "Natural-language instruction for your task.",
    # ...
}

TASK_REGISTRY["b1k"] = TASKS
```

At inference time, the task is referenced as `b1k/<TASK_NAME>` (bucket + task key). Which of the two texts — name or instruction — the policy is prompted with is chosen at training time, see [Language prompt](#language-prompt).

See `src/openpi/configs/robots/b1k.py` and `src/openpi/configs/tasks/b1k.py` for the full reference implementation.

---

### 3. Configure training

Training configs live in `src/openpi/training/config.py`. The reference config `pi05_b1k` fine-tunes π₀.₅ on B1K data:

```python
TrainConfig(
    name="pi05_b1k",
    model=pi0_config.Pi0Config(action_horizon=32, pi05=True),
    data=LeRobotB1KDataConfig(
        repo_id="<REPO_ID>",
        base_config=DataConfig(
            data_cls=_b1k_dataset.B1KLeRobotDataset,
            dataset_root="<DATASET_ROOT>",
            prompt_from_task=True,
            dataset_kwargs={"tolerance_s": 5e-4},
        ),
        robot_config_name="b1k/R1Pro",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    save_interval=10_000,
    num_train_steps=50_000,
    assets_base_dir="./outputs/assets",
    checkpoint_base_dir="./outputs/checkpoints",
)
```

**To train on your own task**, copy the `pi05_b1k` block and update:

| Field | What to change |
|-------|----------------|
| `name` | Unique config name, e.g. `pi05_b1k_<TASK_NAME>` |
| `repo_id` | Your `<REPO_ID>` |
| `dataset_root` | Your `<DATASET_ROOT>` |
| `robot_config_name` | Robot registry key (default: `b1k/R1Pro`) |

Key implementation details:

- **`LeRobotB1KDataConfig`** applies B1K-specific repacking, delta-action transforms for joint groups, and prompt loading from LeRobot task metadata. See `src/openpi/policies/b1k_policy.py`.
- **`dataset_root`** is required for B1K datasets; the generic `scripts/train.py` path does not set this automatically. `--data.dataset-root` / `--data.task-names` on the command line override the config's `dataset_root` / `task_names`.
- **`B1KLeRobotDataset`** (`src/openpi/training/b1k_dataset.py`) reads local roots only — never the Hub — and handles per-task partial downloads, whose episode indices do not start at 0.
- **`action_horizon=32`** matches the π₀.₅ B1K setup; keep training and inference horizons consistent.

You can override most fields from the command line when launching training (see below).

---

### 4. Compute normalization statistics

Before training, compute mean and standard deviation over state and actions in your dataset. The script takes the config name plus the same `--data.*` flags as training:

```bash
cd <OPENPI_DIR>
uv run scripts/compute_norm_stats.py <CONFIG_NAME> \
    --data.repo_id=<REPO_ID> \
    --data.dataset-root=<DATASET_ROOT> \
    --data.task-names <TASK_NAME>        # drop to compute over every task under <DATASET_ROOT>
```

Replace `<CONFIG_NAME>` with your training config name (e.g. `pi05_b1k`). This writes `norm_stats.json` to:

```text
outputs/assets/<CONFIG_NAME>/<REPO_ID>/task_subsets/<TASK_NAME>/   # with --data.task-names
outputs/assets/<CONFIG_NAME>/<REPO_ID>/                            # without
```

Training will fail with a missing-norm-stats error if this step is skipped or run with different `--data.*` flags. `--max-frames N` computes the stats over a random sample of `N` frames. For background on when to reload pre-training statistics instead, see [norm_stats.md](./norm_stats.md).

---

### 5. Fine-tune π₀.₅

Use the B1K training entry point `scripts/b1k/train_b1k.py`, which loads data via `create_b1k_data_loader`, logs camera views to Weights & Biases, and supports validation loss logging.

#### Single-node launch

The helper script `scripts/b1k/train_b1k.sh` wraps common defaults:

```bash
cd <OPENPI_DIR>
source .venv/bin/activate

# Default: pi05_b1k on 8 GPUs
./scripts/b1k/train_b1k.sh

# Custom config, GPU count, and device IDs
./scripts/b1k/train_b1k.sh <CONFIG_NAME> 4 0,1,2,3

# Resume an existing run
./scripts/b1k/train_b1k.sh <CONFIG_NAME> 4 0,1,2,3 --resume-run <EXP_NAME>
```

Or invoke the trainer directly:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/b1k/train_b1k.py <CONFIG_NAME> \
    --exp_name=<EXP_NAME> \
    --overwrite \
    --batch_size=64 \
    --num_train_steps=50000 \
    --data.repo_id=<REPO_ID> \
    --data.dataset-root=<DATASET_ROOT> \
    --data.task-names <TASK_NAME>
```

`--data.task-names <TASK_NAME>` restricts training to that task whether `<DATASET_ROOT>` is a per-task partial download or the full 100-task root (see [above](#which-demos-are-on-disk-one-task-or-all-100)); drop it to train on every task under the root. Use the same `--data.*` flags as for `compute_norm_stats.py` so training finds the matching statistics.

Checkpoints are saved under:

```text
outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP>/
```

**Common overrides**

| Flag | Purpose |
|------|---------|
| `--batch_size` | Per-step batch size (must divide evenly across GPUs) |
| `--num_train_steps` | Total optimization steps |
| `--data.repo_id` | Override dataset repo ID (asset id of the norm stats) |
| `--data.dataset-root` | Override local dataset path |
| `--data.task-names` | Train only on these tasks (default: every task under the root) |
| `--data.robot_config_name` | Override robot registry key |
| `--resume` / `--overwrite` | Resume from latest checkpoint or start fresh |
| `--val_log_interval` | Steps between validation loss evaluations |

Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use up to 90% of GPU memory.

#### Language prompt

The challenge demos carry two kinds of text per task (`meta/tasks.jsonl`). The policy is conditioned on one of them, selected with `--data.prompt-source`:

| `--data.prompt-source` | Text fed to the model (the PaliGemma tokenizer then replaces `_` with spaces) |
| ---------------------- | ------------------------------------------------------------------------------ |
| `task_name` (default)  | `turning_on_radio` — the snake_case task id, what LeRobot's `meta/tasks.parquet` holds |
| `task_description`     | `Turn on the radio receiver that's on the table in the living room.` — from the dataset's `meta/tasks.jsonl` (fallback: `src/openpi/configs/tasks/b1k.py`) |

The choice is recorded in the checkpoint (`assets/<asset_id>/prompt_source.json`, next to the norm stats), and `serve_b1k.py` prompts with the same kind of text automatically; `--prompt-source task_name|task_description` overrides it at serve time and `--text-prompt "..."` sets the text verbatim. Checkpoints trained before `--data.prompt-source` existed were trained on task names but this script served them with descriptions; they keep being served with descriptions unless you pass `--prompt-source task_name`.

π₀.₅ tokenizes the prompt together with the discretized state into `max_token_len` (200) tokens and truncates from the end. The longest instructions do not fit next to the 32-dim state, so training with `task_description` fails fast for such tasks and tells you the `--model.max-token-len` to pass (e.g. `--model.max-token-len 256`).

#### SLURM cluster

For cluster jobs, adapt `scripts/b1k/train_b1k.sbatch.sh` with your account, partition, and environment paths.

---

### 6. Evaluation

After fine-tuning, serve the policy and connect your BEHAVIOR-1K evaluation client over WebSocket.

#### Deploy the checkpoint

```bash
cd <OPENPI_DIR>
source .venv/bin/activate

uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro \
    --task b1k/<TASK_NAME> \
    policy:checkpoint \
    --policy.config <CONFIG_NAME> \
    --policy.dir <CHECKPOINT_DIR>
```

**Example** (replace paths with your run):

```bash
uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro \
    --task b1k/<TASK_NAME> \
    policy:checkpoint \
    --policy.config pi05_b1k \
    --policy.dir outputs/checkpoints/pi05_b1k/<EXP_NAME>/50000
```

This starts a WebSocket policy server on `0.0.0.0:8000`. The server:

1. Prompts with the kind of text the checkpoint was trained on: the task name `<TASK_NAME>` itself, or its instruction from `TASK_REGISTRY["b1k"]["<TASK_NAME>"]` (see [Language prompt](#language-prompt))
2. Wraps the policy with `B1KPolicyWrapper` for receding-horizon action execution
3. Accepts observations keyed by the R1Pro `obs_key` definitions in the robot config

**Optional serve flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--repo_id` | task bucket/name | Norm-stats asset ID if different from `--task` (use the training run's `<REPO_ID>`) |
| `--task-names` | none | Task subset the checkpoint was trained on (`--data.task-names` of training); selects that subset's norm stats in the checkpoint. If omitted and the checkpoint holds a single norm-stats file, that one is used with a warning |
| `--prompt-source` | recorded in checkpoint | `task_name` or `task_description`; overrides the prompt kind the checkpoint was trained with (see [Language prompt](#language-prompt)) |
| `--text-prompt` | none | Prompt the policy with exactly this text |
| `--control_mode` | `receding_horizon` | Action execution mode |
| `--action_horizon` | `16` | Steps to execute before replanning |
| `--port` | `8000` | Server port |
| `--record` | `false` | Record policy I/O for debugging |

Point your BEHAVIOR-1K robot client at the server host and port to stream observations and receive actions.

---

### Quick reference

| Item | Value |
|------|-------|
| Base checkpoint | `gs://openpi-assets/checkpoints/pi05_base/params` |
| Reference config | `pi05_b1k` in `src/openpi/training/config.py` |
| Robot registry key | `b1k/R1Pro` |
| Task registry format | `b1k/<TASK_NAME>` |
| Norm stats script | `scripts/compute_norm_stats.py <CONFIG_NAME> --data.repo_id=<REPO_ID> --data.dataset-root=<DATASET_ROOT> [--data.task-names <TASK_NAME>]` |
| Training script | `scripts/b1k/train_b1k.py` |
| Serving script | `scripts/b1k/serve_b1k.py` |
| Checkpoint directory | `outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP>/` |

---

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Missing norm stats error | Run `scripts/compute_norm_stats.py <CONFIG_NAME>` first, with the same `--data.*` flags (`--data.repo_id`, `--data.dataset-root`, `--data.task-names`) as training |
| `No episodes of task ...` | `<DATASET_ROOT>` is a partial download of other tasks; download the task's chunk or fix `--data.task-names` |
| `Unknown task(s) ...` | Task names must match the strings in `meta/tasks.parquet` (the error lists them) |
| Batch size not divisible by GPU count | Lower `--batch_size` or change the number of visible GPUs |
| Wrong camera or action keys | Verify `dataset_key` values in `src/openpi/configs/robots/b1k.py` match `meta/info.json` |
| Task prompt not found at serve time | Add `<TASK_NAME>` to `src/openpi/configs/tasks/b1k.py` under the `b1k` bucket, or serve with `--prompt-source task_name` / `--text-prompt` |
| `task prompt(s) exceed max_token_len` | The instruction plus the discretized state does not fit; pass the suggested `--model.max-token-len`, or train with `--data.prompt-source task_name` |
| Out of GPU memory | Reduce `--batch_size` or set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` |

For general fine-tuning concepts (LeRobot conversion, config structure, remote inference), see the [main README](../README.md) and [remote inference docs](./remote_inference.md).
