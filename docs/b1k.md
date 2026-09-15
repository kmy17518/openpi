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
| `<REPO_ID>` | Short dataset name used as the asset ID for norm stats and config | `turning_on_radio` |
| `<DATASET_ROOT>` | Absolute path to the LeRobot dataset root | `/path/to/datasets/b1k/<TASK_NAME>` |

The `<REPO_ID>` is typically the task or dataset folder name. Normalization statistics are saved under `outputs/assets/<CONFIG_NAME>/<REPO_ID>/`.

---

### 2. Register robot and task

OpenPi uses Python config files to map dataset keys, observation streams, and action indices to the model.

#### Robot (R1Pro)

The R1Pro robot used in BEHAVIOR-1K is already registered in `src/openpi/configs/robots/b1k.py` as `b1k/R1Pro`. It defines:

- **Cameras:** head (`zed_link`), left wrist, and right wrist RGB streams
- **Action space (23-D):** base velocity, torso joints, dual arms, and grippers
- **Proprioception:** extracted from `observation.state` using the indices in the robot config
- **Delta actions:** groups with `needs_delta_comp=True` (torso joints 1–3, both arms) are trained as offsets from the matching proprio state and turned back into absolute commands at serving time. Torso joint 4 (`action[6]`) is kept absolute: it is never commanded in the challenge demos (identically 0), while its measured position occasionally yields under load, so a delta target would be ≈0 with rare outliers far outside its q01/q99 normalization range. A group whose state slice cannot be found by size names it explicitly with `delta_state_indices` (indices into the extracted state vector).

If you use a different robot or camera layout, copy this file and update `observations`, `action`, and `proprio` to match your dataset keys in `meta/info.json`.

#### Task prompts

Register natural-language task instructions in `src/openpi/configs/tasks/b1k.py`:

```python
# src/openpi/configs/tasks/b1k.py
from . import TASK_REGISTRY

TASKS = {
    "<TASK_NAME>": "Natural-language instruction for your task.",
    # Add more tasks as needed.
}

TASK_REGISTRY["b1k"] = TASKS
```

At inference time, the task is referenced as `b1k/<TASK_NAME>` (bucket + task key).

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
            data_cls=_lerobot_compat.LeRobotDataset,
            dataset_root="<DATASET_ROOT>",
            prompt_from_task=True,
            dataset_kwargs={"tolerance_s": 8e-3},
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
- **`dataset_root`** is required for B1K datasets; the generic `scripts/train.py` path does not set this automatically.
- **`action_horizon=32`** matches the π₀.₅ B1K setup; keep training and inference horizons consistent.
- **`tolerance_s=8e-3`**: lerobot looks video frames up by their absolute timestamp inside the per-file mp4s and compares in float32. The challenge demos pack many episodes per file (the wrist-camera files run to ~28,000 s, where float32 resolves ~2 ms), so with the stock `5e-4` the trainer crashed mid-run with `FrameTimestampError` on frames past ~8,200 s. 8 ms is a quarter of a frame at 30 fps, so a frame that is genuinely off by one is still rejected.

You can override most fields from the command line when launching training (see below).

---

### 4. Compute normalization statistics

Before training, compute mean and standard deviation over state and actions in your dataset:

```bash
cd <OPENPI_DIR>
uv run scripts/compute_norm_stats.py --config-name <CONFIG_NAME>
```

Replace `<CONFIG_NAME>` with your training config name (e.g. `pi05_b1k`). This writes `norm_stats.json` to:

```text
outputs/assets/<CONFIG_NAME>/<REPO_ID>/
```

Training will fail with a missing-norm-stats error if this step is skipped. For background on when to reload pre-training statistics instead, see [norm_stats.md](./norm_stats.md).

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
    --num_train_steps=50000
```

Checkpoints are saved under:

```text
outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP>/
```

**Common overrides**

| Flag | Purpose |
|------|---------|
| `--batch_size` | Per-step batch size (must divide evenly across GPUs) |
| `--num_train_steps` | Total optimization steps |
| `--data.repo_id` | Override dataset repo ID |
| `--data.base_config.dataset_root` | Override local dataset path |
| `--data.robot_config_name` | Override robot registry key |
| `--resume` / `--overwrite` | Resume from latest checkpoint or start fresh |
| `--val_log_interval` | Steps between validation loss evaluations |

Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` to allow JAX to use up to 90% of GPU memory.

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

1. Loads the task prompt from `TASK_REGISTRY["b1k"]["<TASK_NAME>"]`
2. Wraps the policy with `B1KPolicyWrapper` for receding-horizon action execution
3. Accepts observations keyed by the R1Pro `obs_key` definitions in the robot config

**Optional serve flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--repo_id` | task bucket/name | Norm-stats asset ID if different from `--task` |
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
| Norm stats script | `scripts/compute_norm_stats.py --config-name <CONFIG_NAME>` |
| Training script | `scripts/b1k/train_b1k.py` |
| Serving script | `scripts/b1k/serve_b1k.py` |
| Checkpoint directory | `outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<STEP>/` |

---

### Troubleshooting

| Issue | Fix |
|-------|-----|
| Missing norm stats error | Run `scripts/compute_norm_stats.py` with your config name first |
| Batch size not divisible by GPU count | Lower `--batch_size` or change the number of visible GPUs |
| Wrong camera or action keys | Verify `dataset_key` values in `src/openpi/configs/robots/b1k.py` match `meta/info.json` |
| Task prompt not found at serve time | Add `<TASK_NAME>` to `src/openpi/configs/tasks/b1k.py` under the `b1k` bucket |
| Out of GPU memory | Reduce `--batch_size` or set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` |

For general fine-tuning concepts (LeRobot conversion, config structure, remote inference), see the [main README](../README.md) and [remote inference docs](./remote_inference.md).
