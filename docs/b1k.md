## Fine-tuning π₀.₅ on BEHAVIOR-1K

This tutorial walks through fine-tuning [π₀.₅](https://www.physicalintelligence.company/blog/pi05) on demonstration data from [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) using this repository.

**Last updated:** September 2026  
**OpenPi model:** π₀.₅ (`pi05`)  
**Robot:** R1Pro (dual-arm mobile manipulator)

> **This checkout (`my-clean`)** is the `my` branch as of commit `1bebd70` (2026-09-14, before its performance work), plus the correctness fixes from `my`'s 2026-09-16 audit that do not depend on that performance work (see [Fixes ported from the `my` audit](#fixes-ported-from-the-my-audit)), the dependency pin that makes `uv sync` work on ARM hosts, and a rewritten video [data loader](#data-loader) (independent of `my`'s; modelled on the diffusion-policy, ACT and GR00T baselines). Running it on a Blackwell Ultra (B300) or aarch64 host needs the environment settings in [Blackwell Ultra (B300) and ARM (aarch64) hosts](#blackwell-ultra-b300-and-arm-aarch64-hosts); the training commands below already include them.

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

**ARM Linux (aarch64, e.g. NVIDIA Grace / GB200 / GB300 hosts):** the same commands work. `torch==2.7.1` requires `triton==3.3.1`, for which PyPI only publishes x86-64 wheels, so a stock lock fails with `Distribution triton==3.3.1 @ registry+https://pypi.org/simple can't be installed because it doesn't have a source distribution or wheel for the current platform`. `pyproject.toml` therefore declares PyTorch's CUDA 12.8 index (`[[tool.uv.index]] pytorch-cu128`, `explicit = true`) and pins `torch` (Linux) and `triton` (Linux/aarch64) to it — `triton` is listed as a direct dependency behind the same marker only because uv ignores index pins for purely transitive packages — and `uv.lock` carries the matching aarch64 wheel entry. On x86-64 nothing changes: the same `torch==2.7.1+cu128` is installed and triton still comes from PyPI. (Taken from `my`'s commit `6ff9cf9`.) Note that the training step is JAX; torch and triton are only used by the LeRobot data loader.

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

Replace `<CONFIG_NAME>` with your training config name (e.g. `pi05_b1k`). This writes `norm_stats.json` and `b1k_metadata.json` (the robot/action convention the statistics were computed under, see [Action-representation compatibility](#action-representation-compatibility)) to:

```text
outputs/assets/<CONFIG_NAME>/<REPO_ID>/task_subsets/<TASK_NAME>/   # with --data.task-names
outputs/assets/<CONFIG_NAME>/<REPO_ID>/                            # without
```

Training will fail with a missing-norm-stats error if this step is skipped or run with different `--data.*` flags. `--max-frames N` computes the stats over a random sample of `N` frames. For background on when to reload pre-training statistics instead, see [norm_stats.md](./norm_stats.md). With `--data.task-names`, every requested task must have episodes on disk (also after an explicit `episodes` selection); nothing is written otherwise.

---

### 5. Fine-tune π₀.₅

Use the B1K training entry point `scripts/b1k/train_b1k.py`, which loads data via `create_b1k_data_loader`, logs camera views to Weights & Biases, and supports validation loss logging.

#### Single-node launch

Two environment variables are needed on the hosts described in [Blackwell Ultra (B300) and ARM (aarch64) hosts](#blackwell-ultra-b300-and-arm-aarch64-hosts); they are harmless elsewhere, so the commands below always set them:

```bash
# B300 / CUDA compute capability 10.3: jax 0.5.3's XLA aborts (exit code 134) in its Triton GEMM autotuner
# at the first matrix multiply; this routes matmuls through cuBLAS instead. No effect on GPUs XLA knows.
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false
# train_b1k.py at this commit writes the JAX compilation cache to ~/.cache/jax unconditionally; point HOME at a
# scratch directory if $HOME must stay untouched (e.g. `export HOME=<SCRATCH>; ln -s <JAX_CACHE> $HOME/.cache/jax`).
```

The helper script `scripts/b1k/train_b1k.sh` wraps common defaults. At this commit it `source`s a hard-coded developer `.venv` path (`/home/ubuntu/...`) and exits if that path does not exist — edit that line to `<OPENPI_DIR>/.venv/bin/activate` (or use the direct command below):

```bash
cd <OPENPI_DIR>
source .venv/bin/activate
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false   # B300 hosts, see above

# Default: pi05_b1k on 8 GPUs
./scripts/b1k/train_b1k.sh

# Custom config, GPU count, and device IDs
./scripts/b1k/train_b1k.sh <CONFIG_NAME> 4 0,1,2,3

# Resume an existing run
./scripts/b1k/train_b1k.sh <CONFIG_NAME> 4 0,1,2,3 --resume-run <EXP_NAME>
```

Or invoke the trainer directly:

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
uv run scripts/b1k/train_b1k.py <CONFIG_NAME> \
    --exp_name=<EXP_NAME> \
    --overwrite \
    --batch_size=64 \
    --num_train_steps=50000 \
    --data.repo_id=<REPO_ID> \
    --data.dataset-root=<DATASET_ROOT> \
    --data.task-names <TASK_NAME>
```

`--data.task-names <TASK_NAME>` restricts training to that task whether `<DATASET_ROOT>` is a per-task partial download or the full 100-task root (see [above](#which-demos-are-on-disk-one-task-or-all-100)); drop it to train on every task under the root. Use the same `--data.*` flags as for `compute_norm_stats.py` so training finds the matching statistics. The flag spelling here is `--data.dataset-root`; the challenge docs' `--data.base_config.dataset_root` is *not* accepted at this commit.

Single-GPU example with the settings of the `my` branch's `turning_on_radio` run (batch 576 fits in the 284 GB of a B300 with ~264 GiB in use; the token budget 112 is valid for `task_name` prompts, see [Language prompt](#language-prompt)):

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
uv run scripts/b1k/train_b1k.py pi05_b1k \
    --exp_name=<EXP_NAME> --project_name=<WANDB_PROJECT> \
    --batch_size=576 --num_workers=8 --num_train_steps=300000 \
    --save_interval=2500 --keep_period None --log_interval=10 \
    --model.max-token-len 112 \
    --data.repo_id=behavior-1k/2026-challenge-demos --data.dataset-root=<DATASET_ROOT> \
    --data.task-names turning_on_radio --data.prompt-source task_name
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
| `--data.repo_id` | Override dataset repo ID (asset id of the norm stats) |
| `--data.dataset-root` | Override local dataset path |
| `--data.task-names` | Train only on these tasks (default: every task under the root) |
| `--data.robot_config_name` | Override robot registry key |
| `--data.allow-legacy-assets` | Accept norm stats / a resume checkpoint without `b1k_metadata.json` (see [Action-representation compatibility](#action-representation-compatibility)) |
| `--num_workers` | Data-loader worker processes (default 8; ~22 ms of CPU per sample each, see [Data loader](#data-loader)) |
| `--model.max-token-len` | Prompt + state token budget (default 200; see [Prompt length](#prompt-length-and-max_token_len)) |
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

#### Prompt length and `max_token_len`

The "prompt" the language tower sees is not just the task text. For π₀.₅ it is

```text
Task: <task text>, State: <discretized state>;\nAction:
```

where the proprioceptive state is normalized, bucketed into 256 bins and written out as integers (`128 45 201 ...`). The state is tokenized **before** `PadStatesAndActions` pads it to the model's `action_dim` (32) — with the 23 values R1Pro actually has — and those digits are most of the prompt: `turning_on_radio` plus the state is 88–104 tokens on the challenge demos (106 worst case, every value a 3-digit bin), a long instruction plus the state up to ~200. The whole thing is padded or truncated to `max_token_len` (default 200 for π₀.₅; `--model.max-token-len` overrides it). Truncation cuts from the end, i.e. it silently drops state digits and the `Action:` marker, so `create_b1k_dataset` checks every selected task's prompt against the budget up front and fails fast with the `--model.max-token-len` needed.

Until the fix ported from `my`'s audit (finding 5, `b90f734`), that check sized its worst-case state with the padded `action_dim` = 32 instead of the extracted 23 dimensions and therefore over-estimated every prompt by ~36 tokens: with `--model.max-token-len 112` it rejected `turning_on_radio` at "142 tokens" although the real prompt fits (106). The check now receives the extracted state dimension from the robot config (`b1k_artifacts.state_dimension`), on the training side and at serving. Padding tokens are masked, so a smaller budget is purely an efficiency knob (fewer tokens per step); the checkpoint can be served with any budget its prompts fit in.

#### Checkpoint provenance: `b1k_metadata.json`

New checkpoints save `b1k_metadata.json` next to their norm stats (`assets/<asset_id>/`): the exact resolved task prompts, the inference-relevant model settings (`max_token_len`, `action_horizon`, `action_dim`, `pi05`, `discrete_state_input`, variants) and the robot/action representation (below). `serve_b1k.py` prompts with the recorded text by default — including dataset-specific descriptions and custom tasks absent from the registry — and restores the recorded model settings, so a checkpoint trained with `--model.max-token-len 112` is served with 112 without any flag; `--max-token-len` (serving) overrides it, e.g. for checkpoints written before this file existed. `--text-prompt` and `--prompt-source` remain explicit overrides. Resuming a run rejects a changed prompt, prompt source, model setting, action convention or normalization statistics instead of silently mixing training definitions.

#### Action-representation compatibility

`compute_norm_stats.py` writes `b1k_metadata.json` next to `norm_stats.json` too. Training, serving and resume compare the action convention recorded there — robot type, action/proprio groups, which groups are delta actions and how they map to state, `extra_delta_transform` — with the one the current robot config produces:

- A **recorded mismatch is always rejected** (`... use an incompatible action representation`).
- **Missing metadata** (statistics or checkpoints written before this change) fails closed: `... lack action-representation metadata`. Either recompute the statistics (a few seconds with the low-dim reader; the values are bit-identical, only the metadata file is added) or, after checking the convention yourself, pass `--data.allow-legacy-assets` (training / resume) or `--allow-legacy-assets` (serving). The permission only accepts *missing* metadata, never a recorded mismatch.

This checkout predicts all four R1Pro torso joints as deltas (the convention of the original challenge fork). The `my` branch later switched trunk joint 4 to an absolute target and registers the convention used here as `b1k/R1Pro-legacy-torso-delta`: the metadata written here is byte-for-byte what `my` expects for that robot, so `my`'s serving can consume checkpoints trained with this checkout with `--robot b1k/R1Pro-legacy-torso-delta`, and rejects them for its current `b1k/R1Pro` (verified against `my`'s `b1k_artifacts`).

#### SLURM cluster

For cluster jobs, adapt `scripts/b1k/train_b1k.sbatch.sh` with your account, partition, and environment paths.

---

### 6. Evaluation

After fine-tuning, serve the policy and connect your BEHAVIOR-1K evaluation client over WebSocket.

#### Deploy the checkpoint

```bash
cd <OPENPI_DIR>
source .venv/bin/activate
export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false   # B300 hosts, see the section below

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

1. Prompts with the exact text recorded in the checkpoint's `b1k_metadata.json` when present (see [Checkpoint provenance](#checkpoint-provenance-b1k_metadatajson)); otherwise with the kind of text the checkpoint was trained on — the task name `<TASK_NAME>` itself, or its instruction from `TASK_REGISTRY["b1k"]["<TASK_NAME>"]` (see [Language prompt](#language-prompt)) — and restores the recorded `max_token_len` and other model settings
2. Predicts `m = config.model.action_horizon` actions **once per observation request** and returns only the first `n = --action_horizon` of them (`B1KPolicyWrapper.act_chunk`); the response carries `action_chunk` of shape `(n, D)` or `(B, n, D)` and `action = action_chunk[..., 0, :]`. The client's `__action_chunk_size__` must equal `n` (the BEHAVIOR evaluator's `--replay-action-chunk-size`); mismatches are rejected before inference
3. Accepts observations keyed by the R1Pro `obs_key` definitions in the robot config; a `{"reset": true}` message clears that connection's state and receives no reply

**Optional serve flags**

| Flag | Default | Description |
|------|---------|-------------|
| `--repo_id` | task bucket/name | Norm-stats asset ID if different from `--task` (use the training run's `<REPO_ID>`) |
| `--task-names` | none | Task subset the checkpoint was trained on (`--data.task-names` of training); selects that subset's norm stats in the checkpoint. If omitted and the checkpoint holds a single norm-stats file, that one is used with a warning |
| `--prompt-source` | recorded in checkpoint | `task_name` or `task_description`; overrides the prompt kind the checkpoint was trained with (see [Language prompt](#language-prompt)) |
| `--text-prompt` | none | Prompt the policy with exactly this text |
| `--max-token-len` | recorded in checkpoint | Token budget for checkpoints without `b1k_metadata.json`; the selected prompt is checked against it with the extracted state dimension before the policy is loaded |
| `--allow-legacy-assets` | `false` | Accept a checkpoint without `b1k_metadata.json` after verifying its action convention matches `--robot` (a recorded mismatch is still rejected) |
| `--control_mode` | `receding_horizon` | Required; other modes are rejected because a chunk is one prediction from one observation |
| `--action_horizon` | `16` | Chunk length `n` returned per request, `1 <= n <= m` |
| `--port` | `8000` | Server port |
| `--record` | `false` | Record policy I/O for debugging |

Point your BEHAVIOR-1K robot client at the server host and port to stream observations and receive actions. Before the ported fix (audit finding 2), a chunk request was answered by calling `act()` repeatedly on the same observation, i.e. by draining the receding-horizon buffer: with a chunk shorter than `--action_horizon`, the next request was still served from the plan made for the *previous* observation. Now every request is one fresh prediction and the chunk size must equal the execution horizon.

---

### Blackwell Ultra (B300) and ARM (aarch64) hosts

The pinned toolchain — `jax[cuda12]==0.5.3`, `torch==2.7.1+cu128`, `torchcodec 0.11.1` — predates NVIDIA's Blackwell Ultra GPUs (B300, CUDA compute capability 10.3) and has gaps on ARM Linux (aarch64, e.g. NVIDIA Grace CPUs). Unlike the `my` branch, this checkout does not work around these in code; the table lists what is needed from the environment. Everything here was verified on a 4× B300 aarch64 host (Sep 2026).

| Symptom | Cause | What to do at this commit |
|---------|-------|---------------------------|
| `uv sync`: `Distribution triton==3.3.1 @ registry+https://pypi.org/simple can't be installed ...` | PyPI ships only x86-64 wheels for the triton that `torch==2.7.1` requires | Nothing — `pyproject.toml` / `uv.lock` pin triton to PyTorch's cu128 index on aarch64, see [Installation](#installation) |
| Training or serving dies with exit code 134 at the first jitted matmul; log: `Unknown compute capability 10.3. Defaulting to telling LLVM that we're compiling for sm_101`, then `F ... gemm_fusion_autotuner.cc ... ptxas exited with non-zero error code ... Instruction 'tcgen05.alloc' not supported on .target 'sm_101'` | XLA in jax 0.5.3 does not know compute capability 10.3 and falls back to the `sm_101` target, but its Triton GEMM emitter (XLA-internal, unrelated to the pip package `triton`) still generates Blackwell instructions for it | `export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` before `train_b1k.py` / `serve_b1k.py` / `compute_norm_stats.py`; matrix multiplies then run through cuBLAS. (`my` sets this automatically in `openpi.shared.xla_gpu_compat`.) The `Unknown compute capability 10.3` warning keeps being printed and is harmless |
| Log: `torchcodec is installed but cannot be loaded (Could not load libtorchcodec ...); decoding videos with pyav` | torchcodec's wheel does not load against this torch on aarch64 | Nothing — RGB streams are decoded by `VideoFrameReader` (PyAV) anyway, see [Data loader](#data-loader) |
| (fixed) The data loader was the bottleneck: with `--batch_size=576 --num_workers=8` the first batch took ~16 min, then bursts of steps at the GPU rate followed by 14–19 min of idle GPU (~139 s/step, 4 samples/s) | lerobot's PyAV path re-opened every packed mp4 per sample, for all six camera streams in six threads, while `torchvision.io` had installed PyAV's Python log callback: each `av.open` took ~2 s of GIL contention instead of ~2 ms | Fixed in this checkout by the [data loader](#data-loader) rewrite (bit-identical samples, ~1,200 samples/s on 30 cores) |
| The JAX compilation cache lands in `~/.cache/jax` even with `JAX_COMPILATION_CACHE_DIR` set | `train_b1k.py` sets `jax_compilation_cache_dir` to `~/.cache/jax` unconditionally (fixed on `my` in `5687c79`) | Point `HOME` at a scratch directory whose `.cache/jax` links to the cache you want |
| `./scripts/b1k/train_b1k.sh`: `source: /home/ubuntu/.../.venv/bin/activate: No such file or directory` | hard-coded developer path (fixed on `my` in `5687c79`) | Edit the `source` line or use the direct `uv run` command |
| `Unrecognized options: --data.base_config.dataset_root=...` | the challenge docs' spelling is only accepted on `my` | Use `--data.dataset-root` (and `--data.task-names`) |
| `wandb.init` fails with `CommError: returned error 401` | the host's `WANDB_BASE_URL` points at a server the `WANDB_API_KEY` is not valid for | Set `WANDB_BASE_URL=https://api.wandb.ai` (and `WANDB_ENTITY`) explicitly, or `WANDB_MODE=offline` / `--no-wandb-enabled` |

Measured with the single-GPU `turning_on_radio` command of [Fine-tune π₀.₅](#5-fine-tune-π₀₅) (`max_token_len` 112, PyAV, trainer + workers confined to 30 CPU cores) on one B300:

| Quantity | Before the loader rewrite (batch 576, 8 workers) | After (batch 512, 16 workers) | `my` (perf branch, batch 576, its own run doc) |
|----------|------------------:|------------------:|--------------------------------------------------:|
| GPU step time | 10.8 s | 9.1 s | 9.83 s |
| Peak GPU memory | 263.6 GiB | 263.6 GiB | 263.65 GiB |
| Data loader ready after launch | 16.5 min | 87 s | — |
| Steady-state throughput | ~139 s/step ≈ 4.1 samples/s (GPU idle 92 %) | 9.1 s/step = 56 samples/s, GPU 100 % busy | 58.6 samples/s |

The GPU-side numbers are close (the remat policy is the same `nothing_saveable`, hard-coded in `gemma.py` / `siglip.py`); the old throughput gap was entirely the data loader.

---

### Data loader

`B1KLeRobotDataset` (`src/openpi/training/b1k_dataset.py`) reads the LeRobot v3 root with lerobot's `DatasetReader` except for the video frames, which go through its own `VideoFrameReader`. The design takes the ideas the diffusion-policy, ACT and GR00T BEHAVIOR baselines use for the same data (their `VideoReader` / `_decode` / `VideoReaderPool`, RGB-only camera selection, uint8 frames), and does **not** follow the `my` branch's loader changes. Samples are bit-identical to the lerobot path (verified on real data across the full transform pipeline, and by `b1k_dataset_test.py` on synthetic H.264 clips).

Why the stock path was slow on the challenge demos (profile of one sample, single process, 30 cores available): lerobot's PyAV decoder **re-opens the packed 200 MB mp4 for every access** and does so for **all six camera streams at once in six threads**. `torchvision.io` (imported by lerobot) installs PyAV's Python-side FFmpeg log callback at import time; parsing an mp4 index emits tens of thousands of log lines, each taking the GIL, so six concurrent `av.open` calls took **1.9–2.3 s** instead of 3 ms. Everything else was cheap: seek + decode of a random frame costs 7 ms (720²) / 3 ms (480²), and the transforms ~16 ms. Result: 2.45 s per sample, 4 samples/s from 8 workers, 92 % GPU idle.

What `VideoFrameReader` does instead:

| Change | Idea from | Effect |
|--------|-----------|--------|
| `av.logging.set_level(None)` once per process before decoding | ACT's `_decode` | removes the GIL storm: 6 opens 1.93 s → 3 ms |
| One long-lived container per file and worker (LRU, `max_open_videos=32`), seek to the keyframe, decode forward | diffusion policy `VideoReader`, ACT `_decode`, GR00T `VideoReaderPool` | no per-sample index parse (~10 ms per open); GOP 8 → ~6 frames decoded per access |
| Only the robot config's camera streams (`video_keys`, set by `LeRobotB1KDataConfig.create`) — the three RGB cameras; the three depth streams are never decoded | GR00T RGB view, ACT `VIDEO_KEYS`, diffusion policy `cameras` | halves the decoding |
| Streams decoded sequentially with one FFmpeg thread (`decoder_threads=1`) instead of a 6-thread pool per sample | ACT `thread_count = 1`, GR00T `GR00T_FFMPEG_THREADS` | workers scale linearly under a CPU quota, no oversubscription |
| Only the requested frames are converted to RGB, returned as uint8 HWC | ACT / diffusion policy uint8 frames | lerobot converted every decoded frame and returned float32 CHW, which `B1KInputs` turned back into uint8 HWC (bit-identical: `(255 * (v / 255)).astype(uint8) == v` for all 256 values) |
| Data-loader workers pin JAX to the CPU, single-threaded (`_worker_init_fn`) | GR00T's CPU-only workers | `ResizeImages` (`jax.image.resize`) no longer creates a CUDA context per worker on the training GPU; XLA:CPU does not spawn a thread pool per worker |

Throughput of the loader alone at batch 512 (`scripts/b1k/benchmark_loader.py`, trainer process + workers pinned to 30 cores, steady state beyond the prefetch depth): **395 samples/s with 8 workers, ~1,150 with 16, ~1,200 with 24–28** (then limited by the main process's collate/IPC); 22 ms of CPU per sample in a worker (≈13 ms decode, ≈14 ms `ResizeImages`), first batch ~12 s after the workers spawn. The GPU consumes ~56 samples/s at batch 512, so `--num_workers=8` already keeps it busy; 16 leaves ample headroom and is what the measured run used.

Knobs (`dataset_kwargs` of the data config, i.e. `LeRobotB1KDataConfig.base_config.dataset_kwargs`): `video_keys` (streams to decode; default: the robot's cameras), `fast_video_reader` (default `True`; `False` = lerobot's decoder for everything), `decoder_threads` (default 1), `max_open_videos` (default 32). Depth streams, if requested, always use lerobot's decoder. `image_transforms` are only accepted with the lerobot path.

Not done (next steps if ever needed): a pixel-exact cache of pre-resized uint8 frames (diffusion policy / ACT `frame_cache`; ~150 KB × 3 per sample, ~190 GB for `turning_on_radio` at 224²) or GR00T's losslessly re-encoded pre-resized "RGB view" would remove decoding from the workers entirely; a faster `ResizeImages` (currently ~14 ms per sample on XLA:CPU) would matter before that.

---

### Fixes ported from the `my` audit

`my` audited its post-baseline changes on 2026-09-16 (`docs/audit_fixes_20260916.md` there, commit `b90f734`, 14 findings). The fixes below apply to code that already exists at `1bebd70` and do not depend on `my`'s performance work; they were ported here (same author, `kmy17518`), with the `my` regression tests where those apply (`b1k_artifacts_test.py`, additions to `b1k_dataset_test.py`, `websocket_b1k_server_test.py`, `policy_batch_test.py`).

| # | Finding | Ported change |
|---|---------|---------------|
| 2 | Stale-observation action chunk re-planning | `B1KPolicyWrapper.act_chunk`: one prediction per request, return `n` of `m`; the server requires `receding_horizon`, `__action_chunk_size__ == --action_horizon`, validates observation shapes and the reset message |
| 3 | Missing tasks silently omitted | `select_task_subset(..., episodes=)` requires episodes for **every** requested task, also after an explicit episode filter, before any dataset / stats reader is constructed (`data_loader`, `compute_norm_stats`, `B1KLeRobotDataset`) |
| 4 | Dataset-specific prompts lost at serving | `b1k_metadata.json` records the exact resolved task prompts; `serve_b1k.py` uses them by default (custom tasks included) |
| 5 | Token budget lost at serving; prompt check over-counted the state | Model settings (`max_token_len`, ...) saved with the checkpoint and restored at serving; `--max-token-len` override; `check_prompt_token_lengths` counts the extracted state dimension (23), not the padded `action_dim` (32) — the bug that made `--model.max-token-len 112` fail here (introduced with the check itself in `f2a718f`, 2026-09-14) |
| 12 | Configured noise ignored in batched inference | `Policy.infer_batch` honours `sample_kwargs["noise"]`, validates the noise shape and broadcasts a shared sample |
| 13 (in part) | Artifact provenance guard | Action-representation metadata on norm stats and checkpoints, validated at training, serving and resume; resume also compares the actual normalization arrays; `--data.allow-legacy-assets` / `--allow-legacy-assets` for unversioned artifacts. The torso migration itself (`b1k/R1Pro-legacy-torso-delta`, finding 13's trigger) is not part of this checkout, see [Action-representation compatibility](#action-representation-compatibility) |

Not ported, because the code they fix does not exist here or is the performance work itself: 1, 7, 8, 9 (Hugging Face checkpoint publisher and run launcher), 6 (explicit `delta_state_indices` mappings), 10 (camera-stream filtering), 11 (batch prefetch and persistent-worker cleanup), 14 (a test of `my`'s RNG code path). Pre-existing style debt and the two baseline test failures the audit lists (`data_loader_test.py::test_with_real_dataset`, FAST tokenizer) are unchanged.

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
| Task prompt not found at serve time | Add `<TASK_NAME>` to `src/openpi/configs/tasks/b1k.py` under the `b1k` bucket, or serve with `--prompt-source task_name` / `--text-prompt`; checkpoints with `b1k_metadata.json` serve their recorded prompts without this |
| `task prompt(s) exceed max_token_len` | The prompt plus the discretized 23-dim state does not fit; pass the suggested `--model.max-token-len`, or train with `--data.prompt-source task_name` (see [Prompt length](#prompt-length-and-max_token_len)) |
| `... lack action-representation metadata` | Norm stats or checkpoint written before `b1k_metadata.json` existed; recompute the stats, or pass `--data.allow-legacy-assets` (training) / `--allow-legacy-assets` (serving) after checking the convention |
| `... use an incompatible action representation` | The artifact was produced under another robot/action convention (e.g. `my`'s current `b1k/R1Pro`); select the matching robot config or recompute / retrain — never bypassable |
| Exit code 134, `Unknown compute capability 10.3` | B300 GPU with jax 0.5.3: `export XLA_FLAGS=--xla_gpu_enable_triton_gemm=false` (see [B300 / aarch64 hosts](#blackwell-ultra-b300-and-arm-aarch64-hosts)) |
| Out of GPU memory | Reduce `--batch_size` or set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` |

For general fine-tuning concepts (LeRobot conversion, config structure, remote inference), see the [main README](../README.md) and [remote inference docs](./remote_inference.md).
