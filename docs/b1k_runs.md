# BEHAVIOR-1K training runs: launch, monitor, upload, resume

How the π₀.₅ B1K runs of this fork are launched and kept alive unattended, with the exact commands. The procedure is
generic (one *run definition file* per run under `scripts/b1k/runs/`); the concrete example throughout is the run
**`single-task-turning-on-radio-bs512`**. The tutorial for the training code itself is [b1k.md](./b1k.md).

**Everything lives in this repo:**

| Path (inside openpi) | Purpose |
|---|---|
| `scripts/b1k/runs/single-task-turning-on-radio-bs512.env` | Run definition: every training flag, path, HF repo/folder and the upload schedule (shell `KEY=VALUE`). The one file to edit for a new run. |
| `scripts/b1k/train_b1k_run.sh` | Training launcher: waits for idle GPUs, runs `scripts/b1k/train_b1k.py` with the run's flags, tees the log, auto-`--resume`s after a crash. |
| `scripts/b1k/hf_checkpoint_uploader.py` | Monitor 1: eval-only copies of scheduled checkpoints → `<HF_REPO>/<exp>/checkpoint-<step>/`, keeps the repo READMEs. |
| `scripts/b1k/hf_latest_full_checkpoint_uploader.py` | Monitor 2: the newest full (resumable) checkpoint → `<HF_REPO>/<exp>/resume/checkpoint-<step>/`, removing superseded paths without rewriting history or deleting LFS objects. |
| `scripts/b1k/setup_jax_cuda13_venv.sh` | Builds the jax 0.10.2 + CUDA 13 venv the runs train in (`venv-openpi-jax010`). |
| `scripts/b1k/venv-openpi-jax010-freeze.txt` | Exact package set of that venv (`uv pip freeze`), used by the builder for bit-for-bit rebuilds. |
| `scripts/b1k/train_b1k.py`, `scripts/compute_norm_stats.py`, `scripts/b1k/serve_b1k.py` | The trainer, the norm-stats script and the policy server (see [b1k.md](./b1k.md)). |

The run `single-task-turning-on-radio-bs512` in one table:

| | |
|---|---|
| Config / model | `pi05_b1k`, π₀.₅ from `gs://openpi-assets/checkpoints/pi05_base/params`, `action_horizon=32` |
| Data | `behavior-1k/2026-challenge-demos` (full 100-task LeRobot v3 root on disk), task subset `turning_on_radio` only: 200 episodes, 429,928 frames |
| Global batch 512 | 4 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`), `--grad_accum_steps=2` (2 micro-batches of 256 = 64 samples/GPU), `--fsdp_devices=4`, `--model.remat-policy dots_with_no_batch_dims_saveable`, `--model.max-token-len 144`, `--num_workers=16` — the fastest measured 4-GPU configuration from [b1k.md](./b1k.md#gradient-accumulation-a-lighter-remat-policy-and-fsdp-measured); 2.0 s/step |
| Steps / checkpoints | `--num_train_steps=100000`, `--save_interval=2500 --max_to_keep=3 --keep_period None` (3 full checkpoints of ~42 GB on disk) |
| Optimizer (config defaults) | AdamW, `CosineDecaySchedule` (warmup 1,000, peak 2.5e-5 → 2.5e-6 over 30k steps, then constant), EMA 0.99 (saved `params/` are EMA weights), prompt = task name |
| Software | `venv-openpi-jax010`: Python 3.11, jax 0.10.2 + CUDA 13, flax 0.12.8, orbax 0.12.4 (see [Building the venv](#building-the-venv-venv-openpi-jax010)) |
| W&B | project `b1k-challenge-2026-pi`, run `single-task-turning-on-radio-bs512` (entity `kmy17518`, `WANDB_BASE_URL=https://api.wandb.ai`) |
| Hugging Face | [`kmy17518/b1k-challenge-2026-pi`](https://huggingface.co/kmy17518/b1k-challenge-2026-pi) → folder `single-task-turning-on-radio-bs512/` (layout below) |

---

## 1. Prerequisites

1. **This checkout** with the pinned venv synced (`GIT_LFS_SKIP_SMUDGE=1 uv sync`, see [b1k.md](./b1k.md#installation)).
2. **The dataset** — the LeRobot v3 root of `behavior-1k/2026-challenge-demos` (full, or a per-task partial
   download of chunk-000 = `turning_on_radio`; see [b1k.md](./b1k.md#which-demos-are-on-disk-one-task-or-all-100)).
   Set `DATASET_ROOT` in the run definition accordingly.
3. **Secrets in the environment**: `HF_TOKEN` (write access to `HF_REPO`) and `WANDB_API_KEY`. `WANDB_BASE_URL` must
   be the server the key is valid for (`https://api.wandb.ai` for a wandb.ai account). The launcher sources
   `$ENV_FILE` (default `/tmp/dev/env.sh`, which on this machine exports these and redirects all caches under
   `/tmp`) if it exists; on another machine export them yourself or point `ENV_FILE` at your own file.
4. **`tmux`** — the launcher and both monitors run in detached tmux sessions so they outlive the shell / agent
   session that started them.
5. **The training venv** — build it once with the script below.

### Building the venv (`venv-openpi-jax010`)

The pinned stack (jax 0.5.3) predates the GPUs of the training host (its XLA does not know their compute capability,
see [b1k.md](./b1k.md#blackwell-ultra-b300-and-arm-aarch64-hosts)) and is ~8 % slower there; the runs use a second
venv with **jax 0.10.2 + CUDA 13** built *next to* the repo (not by editing `uv.lock`). It is called
`venv-openpi-jax010` because 0.10.2 is the newest jax that runs on this repo's Python 3.11 (jax 0.11 requires
Python 3.12) — there is no jax 0.11 variant.

```bash
cd <OPENPI_DIR>
scripts/b1k/setup_jax_cuda13_venv.sh                       # -> ../venv-openpi-jax010, exact frozen package set
scripts/b1k/setup_jax_cuda13_venv.sh /path/to/venv          # custom location (then set VENV= in the run definition)
scripts/b1k/setup_jax_cuda13_venv.sh --from-recipe          # re-resolve per the docs/b1k.md recipe instead of the freeze
```

The script creates the venv with `uv venv --python 3.11`, installs `scripts/b1k/venv-openpi-jax010-freeze.txt`
(or, with `--from-recipe`, the pinned `.venv`'s non-JAX packages unpinned plus `jax[cuda13]==0.10.2`,
`flax==0.12.8`, `orbax-checkpoint==0.12.4`, `numpy>=2`, `torch==2.7.1`, `lerobot@c43f5811`, …), then openpi itself as
editable (`-e . -e packages/openpi-client`) and runs a CPU-only smoke test. All `uv pip` calls use `--no-config` so
that this repo's `[tool.uv] override-dependencies` (ml-dtypes 0.4.1, tensorstore 0.1.74) are not applied. Every
command in the venv needs `JAXTYPING_DISABLE=1` (flax 0.12 keeps `nnx.Variable`s in the optimizer state, which
`TrainState`'s jaxtyping annotation rejects) and its own compile cache `JAX_COMPILATION_CACHE_DIR=/tmp/.cache/jax-010`;
the launcher sets both. Two small compatibility changes in the repo make openpi run on it and are no-ops on the
pinned stack (`training/sharding.py` mesh `axis_types=Auto`, `models/model.py::restore_params` reading orbax ≥ 0.12
metadata). Measured on the 4-GPU training host, batch 512: 1.93 s/step vs 2.23 s/step for the pinned venv with the
same flags.

### Norm stats

Computed once per task subset, before the first launch (16 s over the parquet files; the trainer fails fast if they
are missing):

```bash
cd <OPENPI_DIR>
JAX_PLATFORMS=cpu JAXTYPING_DISABLE=1 ../venv-openpi-jax010/bin/python scripts/compute_norm_stats.py pi05_b1k \
    --data.repo_id=behavior-1k/2026-challenge-demos --data.dataset-root=<DATASET_ROOT> \
    --data.task-names turning_on_radio
# -> outputs/assets/pi05_b1k/behavior-1k/2026-challenge-demos/task_subsets/turning_on_radio/norm_stats.json
```

(`uv run scripts/compute_norm_stats.py …` in the pinned venv gives bit-identical stats.)

---

## 2. The run definition file

`scripts/b1k/runs/single-task-turning-on-radio-bs512.env` — shell `KEY=VALUE` lines, sourced by the launcher and
parsed by both monitors (which re-read it every cycle, so `HF_REPO`, `HF_EXP_FOLDER`, `UPLOAD_*` and
`NUM_TRAIN_STEPS` can be edited while they run; training flags take effect at the next launch).

| Group | Keys | Notes |
|---|---|---|
| identity | `EXP_NAME`, `CONFIG_NAME`, `WANDB_PROJECT`, `HF_REPO`, `HF_EXP_FOLDER` | `EXP_NAME` names the checkpoint dir `outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/`, the W&B run and the HF folder |
| data | `TASK_NAMES` (comma-separated), `DATASET_ROOT`, `REPO_ID` | `REPO_ID` is the norm-stats asset id, `DATASET_ROOT` the local LeRobot root |
| training | `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `FSDP_DEVICES`, `REMAT_POLICY`, `MAX_TOKEN_LEN`, `NUM_WORKERS`, `NUM_TRAIN_STEPS`, `SAVE_INTERVAL`, `MAX_TO_KEEP`, `CUDA_VISIBLE_DEVICES` | mapped 1:1 onto `train_b1k.py` flags, see below |
| software | `OPENPI_DIR`, `VENV`, `JAX_CACHE` | machine-specific |
| eval uploads | `UPLOAD_EVERY_UNTIL`, `UPLOAD_SWITCH_STEP`, `UPLOAD_EVERY_AFTER`, `UPLOAD_FINAL`, `STAGING_DIR`, `UPLOADER_POLL_SECONDS` | schedule: every `UPLOAD_EVERY_UNTIL` steps up to `UPLOAD_SWITCH_STEP`, then every `UPLOAD_EVERY_AFTER`, plus the final step `NUM_TRAIN_STEPS-1` |
| full mirror | `HF_RESUME_FOLDER`, `HF_FULL_REPO_PRIVATE`, `FULL_UPLOADER_POLL_SECONDS` | |

To start a **new run**, copy the file, change at least `EXP_NAME`, `HF_EXP_FOLDER`, `TASK_NAMES` and the
hyper-parameters, and use the new path in every command below.

---

## 3. Replicate the run from scratch

All commands from `<OPENPI_DIR>` (here `/tmp/dev/baselines/openpi`). `RUN=scripts/b1k/runs/single-task-turning-on-radio-bs512.env`.

```bash
cd <OPENPI_DIR>
RUN=scripts/b1k/runs/single-task-turning-on-radio-bs512.env

# 1. training (waits until the 4 GPUs are idle, then launches; auto-resumes after a crash)
tmux new-session -d -s train-radio-bs512 -c "$PWD" \
    "scripts/b1k/train_b1k_run.sh $RUN fresh; echo \"launcher exited with code \$?\"; exec bash"

# 2. monitor 1: eval-only checkpoints -> HF <HF_REPO>/<exp>/checkpoint-<step>/  (any python with huggingface_hub + hf_xet)
tmux new-session -d -s upload-eval-radio-bs512 -c "$PWD" \
    ". /tmp/dev/env.sh; /tmp/miniforge3/envs/hf/bin/python scripts/b1k/hf_checkpoint_uploader.py $RUN; echo \"uploader exited with code \$?\"; exec bash"

# 3. monitor 2: newest full checkpoint -> HF <HF_REPO>/<exp>/resume/checkpoint-<step>/
tmux new-session -d -s upload-full-radio-bs512 -c "$PWD" \
    ". /tmp/dev/env.sh; /tmp/miniforge3/envs/hf/bin/python scripts/b1k/hf_latest_full_checkpoint_uploader.py $RUN; echo \"full uploader exited with code \$?\"; exec bash"
```

`fresh` passes `--overwrite` (wipes an existing checkpoint directory of the same `EXP_NAME`); use `auto` (the
default: resume if the directory exists) or `resume` otherwise. The `. /tmp/dev/env.sh` part only provides
`HF_TOKEN` and the cache redirects on this machine; elsewhere export `HF_TOKEN` yourself. The monitors need no JAX:
any Python ≥ 3.10 with `huggingface_hub >= 1.0` and `hf_xet` works (here the `hf` conda env; the pinned `.venv`
also qualifies).

Each fresh launch records a generation UUID in the sibling `.EXP_NAME.generation.json` file beside the checkpoint
run directory. Upload state and staging are isolated under that generation; checkpoint identities also include
Orbax commit metadata, so a new checkpoint at an old numeric step is not mistaken for an already uploaded model.
Keep this marker with the run. Both monitors include generation/identity provenance remotely and preserve all
checkpoint assets, including `b1k_metadata.json`. Full checkpoints are staged as immutable copies before uploading:
allow disk space for those copies in addition to the trainer's retained checkpoints.

Cooperating launchers/uploaders must share `B1K_LOCK_DIR` (default `/tmp/openpi-b1k-locks`). Locks are nonblocking:
a duplicate experiment, an overlapping physical GPU allocation, or a fresh launch during active publication fails
clearly instead of waiting to overwrite later. These locks coordinate this host's processes, not independent hosts
or arbitrary external uploaders.

### The exact training command the launcher runs

`train_b1k_run.sh` assembles this from the run definition and logs it as a `launching:` line; for this run it is
(verbatim from the log):

```bash
cd /tmp/dev/baselines/openpi
JAXTYPING_DISABLE=1 PYTHONUNBUFFERED=1 JAX_COMPILATION_CACHE_DIR=/tmp/.cache/jax-010 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 CUDA_VISIBLE_DEVICES=0,1,2,3 WANDB_BASE_URL=https://api.wandb.ai \
/tmp/dev/baselines/venv-openpi-jax010/bin/python scripts/b1k/train_b1k.py pi05_b1k \
    --exp_name=single-task-turning-on-radio-bs512 --project_name=b1k-challenge-2026-pi \
    --batch_size=512 --grad_accum_steps=2 --fsdp_devices=4 \
    --model.remat-policy dots_with_no_batch_dims_saveable --model.max-token-len 144 \
    --num_workers=16 --num_train_steps=100000 \
    --save_interval=2500 --max_to_keep=3 --keep_period None \
    --data.repo_id=behavior-1k/2026-challenge-demos --data.dataset-root=/tmp/dev/datasets/2026-challenge-demos \
    --data.task-names turning_on_radio \
    [--overwrite | --resume]
```

Running it by hand (without the launcher) is equivalent, minus the GPU-idle wait, the log file and the auto-resume.
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` makes JAX preallocate 90 % of every GPU (what `nvidia-smi` shows); the actual
peak of this configuration is ~137 GiB per GPU.

### What the launcher does

`scripts/b1k/train_b1k_run.sh <run.env> [auto|fresh|resume]`

1. sources `$ENV_FILE` (default `/tmp/dev/env.sh`) if present, then the run definition; acquires exclusive per-run
   and physical-GPU locks before checking idle status (competing runs fail rather than queue a future overwrite);
2. waits until every GPU in `CUDA_VISIBLE_DEVICES` is idle (no compute process anywhere, < 4 GiB used) for 60 s —
   so it can be started while something else still runs on the GPUs and never fights it for memory;
3. runs the trainer with `stdout/stderr` tee'd to `$LOG_DIR/train-<EXP_NAME>.log` (`LOG_DIR` default `/tmp/dev/logs`);
4. on a non-zero exit waits 60 s and relaunches with `--resume`; gives up after 3 consecutive failures without a new
   checkpoint. (A trainer that dies before its first checkpoint is relaunched fresh by `--resume` — the trainer
   itself logs `Aborting resume` and starts a *new* W&B run in that case.)

### What the monitors do

**`hf_checkpoint_uploader.py`** (poll: `UPLOADER_POLL_SECONDS`, 60 s). For every completed checkpoint on the
schedule it copies `params/` (EMA weights, 12 GB), `assets/` (norm stats + `prompt_source.json`) and
`_CHECKPOINT_METADATA` to generation-and-checkpoint-identity-specific staging (verified by file count + bytes; `train_state/` is never copied),
adds `training_run.json` (provenance: run definition, git commit, train loss at that step, a serve command), uploads
it as `<HF_EXP_FOLDER>/checkpoint-<step>/`, checks `…/params/manifest.ocdbt` exists in the repo and refreshes the
experiment README (recipe + table `checkpoint | step | train loss | uploaded`) and the root README (experiment
index). A checkpoint counts as complete when its directory has its final numeric name and `_CHECKPOINT_METADATA`
carries `commit_timestamp_nsecs` (orbax writes to a temp dir and renames). The copy happens within a minute of the
save, long before `max_to_keep` deletes the checkpoint (3 × 2,500 steps ≈ 4 h later). Failures retry with backoff
(1 → 15 min); staged copies wait on disk; state in `STAGING_DIR/<generation>/state.json`. Exits after the final step of the current generation is uploaded.

**`hf_latest_full_checkpoint_uploader.py`** (poll: `FULL_UPLOADER_POLL_SECONDS`, 120 s). Whenever the newest
complete checkpoint differs from the one in `<HF_EXP_FOLDER>/resume/`, it uploads the whole checkpoint directory
(params + train_state + assets, ~42 GB, ~1 GB/s here) as `resume/checkpoint-<step>/`, verifies the remote tree
against the local one file by file (paths + sizes), writes `resume/LATEST.json`, `resume/wandb_id.txt`,
`resume/README.md`, then removes superseded `resume/checkpoint-*` paths using normal Hub commits. Metadata is
reconciled on retries even when the checkpoint content already exists; older checkpoints are retained until all
resume metadata is published successfully. The mirror **never permanently deletes LFS objects or rewrites history**:
other branches/tags and concurrent uploaders cannot be protected safely by a default-branch tree snapshot. Storage
quota is therefore not reclaimed by this mirror. Any permanent history cleanup is a separate administrative action.
State is kept under `STAGING_DIR/<generation>/full-state.json`. Publication also rechecks local/remote freshness
under the lock, so a delayed older upload cannot move `LATEST.json` backward or delete a newer checkpoint.
Higher-numbered folders from an older run generation may remain after a fresh restart; `LATEST.json` identifies
the active generation and checkpoint.

Resulting repo layout:

```text
kmy17518/b1k-challenge-2026-pi/
├── README.md                                        experiment index
└── single-task-turning-on-radio-bs512/
    ├── README.md                                    recipe + checkpoint table
    ├── checkpoint-10000/ … checkpoint-99999/        eval-only: params/ assets/ _CHECKPOINT_METADATA training_run.json
    └── resume/
        ├── checkpoint-<newest step>/                full: params/ train_state/ assets/ _CHECKPOINT_METADATA
        └── wandb_id.txt  LATEST.json  README.md
```

---

## 4. Monitoring commands

```bash
cd <OPENPI_DIR>
tmux ls                                                    # train-radio-bs512, upload-eval-radio-bs512, upload-full-radio-bs512
tmux attach -t train-radio-bs512                           # Ctrl-b d detaches; Ctrl-c inside would stop training
pgrep -af "train_b1k.py.*--exp_name=single-task-turning-on-radio-bs512"      # the trainer process

LOG=/tmp/dev/logs/train-single-task-turning-on-radio-bs512.log
tail -n 5 "$LOG"                                           # trainer + launcher output
grep -a -o -E "Step [0-9]+: grad_norm=[0-9.]+, loss=[0-9.]+" "$LOG" | tail -3   # loss every 100 steps (also on W&B)
grep -a "Progress on" "$LOG" | tail -1                      # tqdm: rate (s/it) and remaining time
grep -a "\[launcher\]" "$LOG"                              # every launch / relaunch, exit codes, GPU waits

tail -n 5 /tmp/dev/logs/upload-single-task-turning-on-radio-bs512.log        # eval uploader: heartbeat, staged, uploaded
tail -n 5 /tmp/dev/logs/upload-full-single-task-turning-on-radio-bs512.log   # full mirror: hand-overs, purge, Hub LFS bytes
ls outputs/checkpoints/pi05_b1k/single-task-turning-on-radio-bs512/          # completed steps on disk (+ wandb_id.txt)
cat /tmp/dev/hf-staging/single-task-turning-on-radio-bs512/state.json        # per-step staged/uploaded/errors
cat /tmp/dev/hf-staging/single-task-turning-on-radio-bs512/full-state.json   # step held in resume/, purge history
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv        # ~100 % util; memory is JAX's 90 % arena
python -c "from huggingface_hub import HfApi; print(*[f.path for f in HfApi().list_repo_tree('kmy17518/b1k-challenge-2026-pi', path_in_repo='single-task-turning-on-radio-bs512')], sep='\n')"   # or open the repo page
```

W&B: <https://wandb.ai/kmy17518/b1k-challenge-2026-pi> (run `single-task-turning-on-radio-bs512`; `loss`,
`grad_norm`, `param_norm` every 100 steps, `camera_views` of the first batch).

---

## 5. Resume

### On this machine (checkpoints on disk)

The launcher resumes from the newest complete checkpoint in `outputs/checkpoints/pi05_b1k/<EXP_NAME>/` and W&B
continues the same run (`wandb_id.txt` in that directory):

```bash
cd <OPENPI_DIR>
RUN=scripts/b1k/runs/single-task-turning-on-radio-bs512.env
tmux new-session -d -s train-radio-bs512 -c "$PWD" "scripts/b1k/train_b1k_run.sh $RUN resume; exec bash"
# and, if they are not running (tmux ls), the two monitors exactly as in section 3 (their state is on disk; idempotent)
```

To change the run length, edit `NUM_TRAIN_STEPS` in the run definition before resuming (the LR schedule does not
depend on it; the monitors pick the new final step up automatically).

### On another machine (from the Hub)

The full mirror keeps the newest checkpoint in `resume/`; download it into the layout the trainer expects and resume:

```bash
cd <OPENPI_DIR>                                          # same commit, venv built, dataset + norm stats present
EXP=single-task-turning-on-radio-bs512
hf download kmy17518/b1k-challenge-2026-pi --include "$EXP/resume/**" --local-dir /tmp/resume
STEP=$(ls /tmp/resume/$EXP/resume | sed -n 's/^checkpoint-//p')
mkdir -p outputs/checkpoints/pi05_b1k/$EXP
mv /tmp/resume/$EXP/resume/checkpoint-$STEP outputs/checkpoints/pi05_b1k/$EXP/$STEP
cp /tmp/resume/$EXP/resume/wandb_id.txt outputs/checkpoints/pi05_b1k/$EXP/     # to continue the same W&B run
# adapt DATASET_ROOT / OPENPI_DIR / VENV / JAX_CACHE / STAGING_DIR in the run definition, then:
tmux new-session -d -s train-radio-bs512 -c "$PWD" "scripts/b1k/train_b1k_run.sh scripts/b1k/runs/$EXP.env resume; exec bash"
```

`/tmp/resume/$EXP/resume/LATEST.json` records the step, the run definition it was trained with and the openpi commit.

### Stop

```bash
pkill -f "train_b1k_run.sh scripts/b1k/runs/single-task-turning-on-radio-bs512"           # launcher first, or it relaunches
pkill -INT -f "train_b1k.py.*--exp_name=single-task-turning-on-radio-bs512"                 # then the trainer
tmux kill-session -t train-radio-bs512; tmux kill-session -t upload-eval-radio-bs512; tmux kill-session -t upload-full-radio-bs512
```

---

## 6. Serve a checkpoint for evaluation

```bash
cd <OPENPI_DIR>
EXP=single-task-turning-on-radio-bs512; STEP=10000
hf download kmy17518/b1k-challenge-2026-pi --include "$EXP/checkpoint-$STEP/**" --local-dir ckpt
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 uv run scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro --task b1k/turning_on_radio \
    --repo-id behavior-1k/2026-challenge-demos --task-names turning_on_radio \
    policy:checkpoint --policy.config pi05_b1k --policy.dir ckpt/$EXP/checkpoint-$STEP
```

(`--task-names` selects the subset's norm stats inside the checkpoint's `assets/`; the checkpoint was trained with
`max_token_len 144` but serves fine with the default 200 — padding is masked. Local checkpoints work the same way
with `--policy.dir outputs/checkpoints/pi05_b1k/$EXP/$STEP`.) The full `resume/checkpoint-<step>/` serves as well.

---

## 7. Notes on this run

- Started 2026-09-14 23:33 UTC on a 4-GPU aarch64 host; 2.0 s/step → ~55 h for 100k steps. `NUM_TRAIN_STEPS=100000`
  and the unchanged (batch-64-tuned) LR schedule were choices made when the run was set up, not measured optima;
  100k × 512 samples ≈ 119 epochs of the 200 demos.
- The launcher's first trainer process (23:26) was killed by an external `SIGTERM` before step 0 and relaunched
  automatically at 23:33; that left an empty W&B run (`ne0hrpz6`) next to the real one (`j9j68i59`).
- `TrainConfig.max_to_keep` (the "save total limit") and the `video_keys` fix in `scripts/compute_norm_stats.py`
  were added for this run (commit `7f577c2`).
- The machine-level operational record (timeline, decisions, incident notes) is kept outside the repo in
  `/tmp/dev/runs/single-task-turning-on-radio-bs512/README.md`.
