#!/usr/bin/env python
"""Watch a pi0.5 B1K training run, stage eval-only copies of scheduled checkpoints and upload them to the HF Hub.

    python scripts/b1k/hf_checkpoint_uploader.py scripts/b1k/runs/<exp>.env

(any Python >= 3.10 with `huggingface_hub >= 1.0` and `hf_xet`; no JAX needed. Logs go to $B1K_LOG_DIR, default
/tmp/dev/logs, the same directory train_b1k_run.sh writes the training log to.) See docs/b1k_runs.md.

Repo layout (one *project* repo, one folder per experiment -- same convention as kmy17518/b1k-challenge-2026-gr00t):

    <HF_REPO>/                                   e.g. kmy17518/b1k-challenge-2026-pi
        README.md                                project card: table of experiment folders (kept by this script)
        <EXP_NAME>/README.md                     experiment card: recipe + table of uploaded checkpoints
        <EXP_NAME>/checkpoint-<step>/            eval-only copy: params/ (EMA weights), assets/ (norm stats + prompt
                                                 source), _CHECKPOINT_METADATA, training_run.json (provenance)
        <EXP_NAME>/resume/checkpoint-<step>/     the newest FULL checkpoint (hf_latest_full_checkpoint_uploader.py)

Every UPLOADER_POLL_SECONDS the script

1. re-reads run.env (HF_REPO, NUM_TRAIN_STEPS and the UPLOAD_* schedule may be edited while it runs),
2. lists the completed orbax checkpoints under <OPENPI_DIR>/outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<step>/,
3. for every step on the upload schedule -- steps <= UPLOAD_SWITCH_STEP that are multiples of UPLOAD_EVERY_UNTIL,
   later steps that are multiples of UPLOAD_EVERY_AFTER, plus the trainer's final checkpoint (NUM_TRAIN_STEPS - 1)
   when UPLOAD_FINAL=1 -- copies ONLY what serving/eval needs into STAGING_DIR/<generation>/<step>-<identity>/ (`train_state/`, the
   optimizer state needed only to resume, is never copied). The copy happens as soon as the checkpoint is complete,
   so the trainer's `max_to_keep` pruning cannot take a scheduled checkpoint away before it was captured,
4. uploads staged steps as <EXP_NAME>/checkpoint-<step>/ (creating the repo if needed) and refreshes the experiment
   and project READMEs; failures are logged and retried with backoff on later cycles -- staged copies wait on disk,
5. logs a heartbeat (latest checkpoint, trainer alive?, last training-progress line, pending work).

State lives in STAGING_DIR/<generation>/state.json; checkpoint identities prevent stale success on same-step restarts. It exits once the
final checkpoint has been uploaded (or runs forever when UPLOAD_FINAL=0). Log: /tmp/dev/logs/upload-<EXP_NAME>.log.
Needs HF_TOKEN in the environment (source /tmp/dev/env.sh first); the token must have write access to HF_REPO.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub import RepoFolder
from huggingface_hub.utils import HfHubHTTPError

LOG_DIR = pathlib.Path(os.environ.get("B1K_LOG_DIR", "/tmp/dev/logs"))  # must match LOG_DIR of train_b1k_run.sh
MAX_BACKOFF_SECONDS = 900
LOG = logging.getLogger("uploader")

_lifecycle_spec = importlib.util.spec_from_file_location("run_lifecycle", pathlib.Path(__file__).with_name("run_lifecycle.py"))
lifecycle = importlib.util.module_from_spec(_lifecycle_spec)
_lifecycle_spec.loader.exec_module(lifecycle)


# --------------------------------------------------------------------------------------------- run.env / schedule
def read_run_env(path: pathlib.Path) -> dict[str, str]:
    """Parse the shell-syntax KEY=VALUE run file (comments and blank lines ignored, inline ` # ...` stripped)."""
    env: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = re.split(r"\s+#", value, maxsplit=1)[0].strip().strip("'\"")
        env[key.strip()] = value
    return env


class RunSpec:
    def __init__(self, env: dict[str, str]):
        self.exp_name = env["EXP_NAME"]
        self.config_name = env["CONFIG_NAME"]
        self.hf_repo = env["HF_REPO"]
        self.exp_folder = env.get("HF_EXP_FOLDER") or self.exp_name  # folder of this experiment inside the repo
        self.openpi_dir = pathlib.Path(env["OPENPI_DIR"])
        self.ckpt_dir = self.openpi_dir / "outputs" / "checkpoints" / self.config_name / self.exp_name
        self.staging_root = pathlib.Path(env.get("STAGING_DIR", f"/tmp/dev/hf-staging/{self.exp_name}"))
        self.generation = lifecycle.generation(self.ckpt_dir)
        self.staging_dir = self.staging_root / self.generation["id"]
        self.state_path = self.staging_dir / "state.json"
        self.num_train_steps = int(env["NUM_TRAIN_STEPS"])
        self.every_until = int(env.get("UPLOAD_EVERY_UNTIL", 10_000))
        self.switch_step = int(env.get("UPLOAD_SWITCH_STEP", 50_000))
        self.every_after = int(env.get("UPLOAD_EVERY_AFTER", 5_000))
        self.upload_final = env.get("UPLOAD_FINAL", "1") not in ("0", "false", "no", "")
        self.poll_seconds = int(env.get("UPLOADER_POLL_SECONDS", 60))
        self.train_log = LOG_DIR / f"train-{self.exp_name}.log"
        self.raw = env

    @property
    def final_step(self) -> int:
        return self.num_train_steps - 1

    @property
    def target(self) -> str:  # identifies where uploads go; recorded per step so re-targeting is detected
        return f"{self.hf_repo}/{self.exp_folder}"

    def ckpt_path(self, step: int) -> str:
        return f"{self.exp_folder}/checkpoint-{step}"

    def wanted(self, step: int) -> bool:
        if self.upload_final and step == self.final_step:
            return True
        if step <= 0:
            return False
        if step <= self.switch_step:
            return step % self.every_until == 0
        return step % self.every_after == 0

    def planned_steps(self) -> list[int]:
        steps = [s for s in range(self.num_train_steps) if self.wanted(s) and s != self.final_step]
        return steps + ([self.final_step] if self.upload_final else [])

    def describe_schedule(self) -> str:
        final = f", plus the final checkpoint (step {self.final_step})" if self.upload_final else ""
        return (
            f"every {self.every_until} steps up to step {self.switch_step}, then every {self.every_after} steps"
            f"{final}; NUM_TRAIN_STEPS={self.num_train_steps}"
        )


# ------------------------------------------------------------------------------------------------- checkpoints
def completed_steps(ckpt_dir: pathlib.Path) -> list[int]:
    """Steps whose orbax checkpoint is fully committed (final directory name + commit timestamp in the metadata)."""
    steps = []
    if not ckpt_dir.is_dir():
        return steps
    for child in ckpt_dir.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        meta = child / "_CHECKPOINT_METADATA"
        if not meta.is_file() or not (child / "params").is_dir() or not (child / "assets").is_dir():
            continue
        try:
            lifecycle.metadata_identity(meta.read_bytes(), {"id": "legacy", "started_ns": 0})
        except (OSError, ValueError):
            continue
        steps.append(int(child.name))
    return sorted(steps)


def checkpoint_identity(spec, step: int) -> str:
    return lifecycle.checkpoint_identity(spec.ckpt_dir, step, spec.generation)


def current_steps(spec) -> list[int]:
    steps = []
    for step in completed_steps(spec.ckpt_dir):
        try:
            checkpoint_identity(spec, step)
        except (OSError, ValueError, RuntimeError):
            continue
        steps.append(step)
    return steps


def stage_path(spec: RunSpec, step: int, identity: str) -> pathlib.Path:
    return spec.staging_dir / f"{step}-{identity}"


def is_uploaded(spec: RunSpec, rec: dict) -> bool:
    return bool(rec.get("uploaded_at") and rec.get("identity")) and rec.get("target") == spec.target and rec.get("generation") == spec.generation["id"]


def staged_record(spec: RunSpec, step: int, rec: dict) -> bool:
    if rec.get("generation") != spec.generation["id"] or not rec.get("identity") or not rec.get("staged_at"):
        return False
    path = stage_path(spec, step, rec["identity"])
    try:
        provenance = json.loads((path / "training_run.json").read_text())
        return (
            provenance.get("identity") == rec["identity"]
            and provenance.get("generation") == spec.generation["id"]
            and lifecycle.metadata_identity((path / "_CHECKPOINT_METADATA").read_bytes(), spec.generation) == rec["identity"]
            and (path / "params").is_dir() and (path / "assets").is_dir()
        )
    except (OSError, ValueError):
        return False


def tree_stats(root: pathlib.Path) -> tuple[int, int]:
    """(number of files, total bytes) under root."""
    files = bytes_ = 0
    for p in root.rglob("*"):
        if p.is_file():
            files += 1
            bytes_ += p.stat().st_size
    return files, bytes_


def git_commit(repo: pathlib.Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=False)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def train_loss_at(log_path: pathlib.Path, step: int) -> str:
    """The `Step <step>: ... loss=<x>` value from the training log (logged every log_interval steps), or '-'."""
    try:
        text = log_path.read_bytes().decode(errors="replace")
    except OSError:
        return "-"
    m = re.findall(rf"Step {step}: [^\n\r]*?loss=([0-9.]+)", text)
    return m[-1] if m else "-"


def stage(spec: RunSpec, step: int) -> dict:
    """Capture an immutable, generation- and checkpoint-specific eval copy."""
    identity = checkpoint_identity(spec, step)
    src = spec.ckpt_dir / str(step)
    dst = stage_path(spec, step, identity)
    tmp = dst.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    t0 = time.time()
    for name in ("params", "assets"):
        shutil.copytree(src / name, tmp / name, symlinks=False)
    shutil.copy2(src / "_CHECKPOINT_METADATA", tmp / "_CHECKPOINT_METADATA")
    # Verify the copy against the source before declaring the stage complete.
    for name in ("params", "assets"):
        want, got = tree_stats(src / name), tree_stats(tmp / name)
        if want != got:
            shutil.rmtree(tmp)
            raise RuntimeError(f"copy of {src / name} incomplete: source {want} vs copy {got} (files, bytes)")
    if checkpoint_identity(spec, step) != identity or lifecycle.metadata_identity((tmp / "_CHECKPOINT_METADATA").read_bytes(), spec.generation) != identity:
        raise RuntimeError(f"Checkpoint {step} changed while staging")
    provenance = {
        "identity": identity,
        "generation": spec.generation["id"],
        "exp_name": spec.exp_name,
        "config_name": spec.config_name,
        "step": step,
        "train_loss": train_loss_at(spec.train_log, step),
        "checkpoint_source": str(src),
        "contents": "params/ (EMA weights), assets/ (norm stats + prompt source), _CHECKPOINT_METADATA; "
        "train_state/ (optimizer state, resume only) intentionally omitted",
        "run_env": spec.raw,
        "openpi_git_commit": git_commit(spec.openpi_dir),
        "staged_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "serve_example": (
            f"uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/{spec.raw.get('TASK_NAMES', '<task>')} "
            f"--repo-id {spec.raw.get('REPO_ID', '<repo_id>')} --task-names {spec.raw.get('TASK_NAMES', '<task>')} "
            f"policy:checkpoint --policy.config {spec.config_name} --policy.dir ckpt/{spec.ckpt_path(step)}"
        ),
    }
    (tmp / "training_run.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    files, bytes_ = tree_stats(dst)
    LOG.info("staged step %d -> %s (%d files, %.2f GiB) in %.0f s", step, dst, files, bytes_ / 2**30, time.time() - t0)
    return {"staged_at": provenance["staged_at"], "staged_files": files, "staged_bytes": bytes_, "path": str(dst),
            "train_loss": provenance["train_loss"], "identity": identity, "generation": spec.generation["id"]}


# ---------------------------------------------------------------------------------------------------- HF upload
def experiment_readme(spec: RunSpec, steps: dict[str, dict]) -> str:
    env = spec.raw
    task = env.get("TASK_NAMES", "")
    uploaded = sorted(int(s) for s, r in steps.items() if is_uploaded(spec, r))
    rows = "\n".join(
        f"| `checkpoint-{s}/` | {s:,} | {steps[str(s)].get('train_loss', '-')} | {steps[str(s)]['uploaded_at'][:16].replace('T', ' ')} |"
        for s in uploaded
    ) or "| (none yet) | | | |"
    planned = spec.planned_steps()
    return f"""---
license: apache-2.0
base_model: physical-intelligence/pi05_base
tags:
  - robotics
  - vla
  - pi0.5
  - openpi
  - behavior-1k
  - b1k-challenge-2026
datasets:
  - behavior-1k/2026-challenge-demos
---

# {spec.exp_name}

pi0.5 (openpi config `{spec.config_name}`) fine-tuned on the single BEHAVIOR-1K 2026 challenge task **`{task}`**
(task 0 of [behavior-1k/2026-challenge-demos](https://huggingface.co/datasets/behavior-1k/2026-challenge-demos),
200 demos / ~430k frames), starting from the pi0.5 base checkpoint `gs://openpi-assets/checkpoints/pi05_base/params`.

- Global batch size **{env.get("BATCH_SIZE", "?")}** ({env.get("FSDP_DEVICES", "4")} GPUs; gradient accumulation
  {env.get("GRAD_ACCUM_STEPS", "1")}x, FSDP over {env.get("FSDP_DEVICES", "?")} GPUs, remat policy
  `{env.get("REMAT_POLICY", "nothing_saveable")}` -- training-time only), {spec.num_train_steps:,} steps,
  openpi default `CosineDecaySchedule` (warmup 1,000, peak lr 2.5e-5, cosine to 2.5e-6 over 30k steps), AdamW,
  EMA 0.99 (the saved `params/` are the EMA weights), prompt = `task_name` (`{task}`),
  `max_token_len` {env.get("MAX_TOKEN_LEN", "200")} at training time (serve with the default 200), action horizon 32.
- Trained with `scripts/b1k/train_b1k.py {spec.config_name} --data.task-names {task}` of the openpi B1K fork
  (see `docs/b1k.md` there); W&B project `{env.get("WANDB_PROJECT", "")}`, run `{spec.exp_name}`.

## Checkpoints

Each `{spec.exp_folder}/checkpoint-<step>/` is an eval-only openpi checkpoint directory: `params/` (EMA weights as
saved by orbax), `assets/` (normalization statistics + `prompt_source.json` of the task subset),
`_CHECKPOINT_METADATA` and `training_run.json` (provenance: the exact run settings). Optimizer state is **not**
included; the newest full, resumable checkpoint is kept in [`resume/`](./resume) instead. Upload schedule:
{spec.describe_schedule().split(';')[0]} ({len(planned)} checkpoints total when training finishes).

| checkpoint | step | train loss | uploaded (UTC) |
|---|---|---|---|
{rows}

## Serve for evaluation

```bash
hf download {spec.hf_repo} --include "{spec.exp_folder}/checkpoint-<step>/**" --local-dir ckpt
uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/{task} \\
    --repo-id {env.get("REPO_ID", "")} --task-names {task} \\
    policy:checkpoint --policy.config {spec.config_name} --policy.dir ckpt/{spec.exp_folder}/checkpoint-<step>
```

Then run the BEHAVIOR-1K evaluator against the policy server (see `docs/b1k.md`).
"""


def project_readme(api: HfApi, spec: RunSpec) -> str:
    """Root card: one row per experiment folder in the repo (folders holding checkpoint-* subfolders or a README)."""
    rows = []
    try:
        for item in api.list_repo_tree(spec.hf_repo, repo_type="model"):
            if isinstance(item, RepoFolder):
                subs = list(api.list_repo_tree(spec.hf_repo, path_in_repo=item.path, repo_type="model"))
                n = sum(1 for s in subs if isinstance(s, RepoFolder) and s.path.split("/")[-1].startswith("checkpoint-"))
                has_resume = any(isinstance(s, RepoFolder) and s.path.endswith("/resume") for s in subs)
                rows.append(f"| [`{item.path}/`](./{item.path}) | {n} | {'yes' if has_resume else 'no'} |")
    except HfHubHTTPError:
        pass
    if not rows:
        rows = [f"| [`{spec.exp_folder}/`](./{spec.exp_folder}) | 0 | no |"]
    return f"""---
license: apache-2.0
base_model: physical-intelligence/pi05_base
tags:
  - robotics
  - vla
  - pi0.5
  - openpi
  - behavior-1k
  - b1k-challenge-2026
datasets:
  - behavior-1k/2026-challenge-demos
---

# {spec.hf_repo.split('/')[-1]}

pi0.5 (openpi) fine-tunes for the BEHAVIOR-1K 2026 challenge, one folder per experiment. Each folder has its own
`README.md` with the training recipe and a table of its `checkpoint-<step>/` subfolders (eval-only openpi checkpoint
directories: `params/` + `assets/`), and a `resume/checkpoint-<step>/` folder holding the newest full checkpoint
(params + optimizer state) of a run that is still training or was stopped.

| experiment folder | uploaded checkpoints | resume checkpoint |
|---|---|---|
{chr(10).join(rows)}

```bash
hf download {spec.hf_repo} --include "<experiment>/checkpoint-<step>/**" --local-dir ckpt
```
"""


def ensure_cards(api: HfApi, spec: RunSpec, steps: dict[str, dict], *, n_uploaded: int) -> None:
    api.create_repo(spec.hf_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=experiment_readme(spec, steps).encode(), path_in_repo=f"{spec.exp_folder}/README.md",
        repo_id=spec.hf_repo, repo_type="model", commit_message=f"Model card for {spec.exp_name} ({n_uploaded} checkpoints)",
    )
    api.upload_file(
        path_or_fileobj=project_readme(api, spec).encode(), path_in_repo="README.md",
        repo_id=spec.hf_repo, repo_type="model", commit_message="Update experiment index",
    )


def upload(api: HfApi, spec: RunSpec, step: int, steps: dict[str, dict]) -> dict:
    with lifecycle.exclusive_lock(f"publish:{spec.ckpt_dir.resolve()}"):
        return _upload(api, spec, step, steps)


def _upload(api: HfApi, spec: RunSpec, step: int, steps: dict[str, dict]) -> dict:
    """Publish checkpoint content and both cards before recording success."""
    record = steps[str(step)]
    if lifecycle.generation(spec.ckpt_dir) != spec.generation or not staged_record(spec, step, record):
        raise RuntimeError(f"Staged checkpoint {step} does not match the current run")
    src = stage_path(spec, step, record["identity"])
    t0 = time.time()
    api.create_repo(spec.hf_repo, repo_type="model", private=False, exist_ok=True)
    commit = api.upload_folder(
        repo_id=spec.hf_repo,
        repo_type="model",
        folder_path=str(src),
        path_in_repo=spec.ckpt_path(step),
        delete_patterns="*",
        commit_message=f"Add {spec.ckpt_path(step)} ({spec.exp_name})",
    )
    # Sanity check that the commit really holds the checkpoint before recording success.
    if not api.file_exists(spec.hf_repo, f"{spec.ckpt_path(step)}/params/manifest.ocdbt", repo_type="model"):
        raise RuntimeError(f"upload of step {step} returned but {spec.ckpt_path(step)}/params/manifest.ocdbt is not in the repo")
    url = getattr(commit, "commit_url", str(commit))
    rec = {"uploaded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "target": spec.target,
           "repo": spec.hf_repo, "path": spec.ckpt_path(step), "commit": url}
    proposed = {**steps, str(step): {**record, **rec}}
    n_uploaded = sum(1 for r in proposed.values() if is_uploaded(spec, r))
    ensure_cards(api, spec, proposed, n_uploaded=n_uploaded)
    if lifecycle.generation(spec.ckpt_dir) != spec.generation:
        raise RuntimeError("Run generation changed during upload")
    if (spec.ckpt_dir / str(step)).exists() and checkpoint_identity(spec, step) != record["identity"]:
        raise RuntimeError(f"Checkpoint {step} changed during upload")
    steps[str(step)].update(rec)
    LOG.info("uploaded step %d to https://huggingface.co/%s/tree/main/%s in %.0f s (%s)", step, spec.hf_repo, spec.ckpt_path(step), time.time() - t0, url)
    return rec


def explain_hf_error(e: Exception, repo: str) -> str:
    text = str(e)
    if "403" in text and "rights to create" in text:
        ns = repo.split("/")[0]
        return (
            f"the namespace '{ns}' does not exist or HF_TOKEN's user is not a member with write access -- create the "
            f"organization at https://huggingface.co/organizations/new (name: {ns}) and add the token's user to it, "
            f"or change HF_REPO in run.env (picked up automatically). Original error: {text.splitlines()[0]}"
        )
    return text.replace("\n", " ")[:600]


# -------------------------------------------------------------------------------------------------- monitoring
def trainer_alive(spec: RunSpec) -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", f"train_b1k.py.*--exp_name={spec.exp_name}"], capture_output=True, text=True, check=False)
        return out.returncode == 0 and bool(out.stdout.strip())
    except OSError:
        return False


def last_progress_line(log_path: pathlib.Path) -> str:
    if not log_path.is_file():
        return "(no training log yet)"
    try:
        with log_path.open("rb") as f:
            f.seek(max(0, log_path.stat().st_size - 200_000))
            tail = f.read().decode(errors="replace")
    except OSError:
        return "(training log unreadable)"
    # tqdm writes progress lines after carriage returns ("\r\rStep 100: ..."), so split on both \n and \r.
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", tail) if ln.strip()]
    steps = [ln for ln in lines if re.match(r"^Step \d+: ", ln)]
    if steps:
        return steps[-1][:160]
    return lines[-1][:160] if lines else "(training log empty)"


# --------------------------------------------------------------------------------------------------------- main
def load_state(path: pathlib.Path) -> dict:
    if path.is_file():
        try:
            return json.loads(path.read_text())
        except ValueError:
            LOG.warning("state file %s unreadable, starting fresh", path)
    return {"steps": {}}


def save_state(path: pathlib.Path, state: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def setup_logging(exp_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname).1s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / f"upload-{exp_name}.log")):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_env_path = pathlib.Path(sys.argv[1]).resolve()
    spec = RunSpec(read_run_env(run_env_path))
    setup_logging(spec.exp_name)
    if not os.environ.get("HF_TOKEN"):
        LOG.error("HF_TOKEN is not set; run `source /tmp/dev/env.sh` first")
        return 2
    spec.staging_dir.mkdir(parents=True, exist_ok=True)
    state_path = spec.state_path
    state = load_state(state_path)
    state.setdefault("steps", {})
    api = HfApi(token=os.environ["HF_TOKEN"])
    LOG.info("=== uploader for %s: checkpoints %s -> staging %s -> https://huggingface.co/%s/tree/main/%s ===", spec.exp_name, spec.ckpt_dir, spec.staging_dir, spec.hf_repo, spec.exp_folder)
    LOG.info("schedule: %s; poll every %d s", spec.describe_schedule(), spec.poll_seconds)
    LOG.info("HF token user: %s", api.whoami().get("name"))

    last_repo_warning = 0.0
    cards_target = None  # target the cards were last written for (written once per start and after every upload)
    cycle = 0
    while True:
        cycle += 1
        try:
            spec = RunSpec(read_run_env(run_env_path))  # live-reloadable: HF_REPO, schedule, NUM_TRAIN_STEPS
        except (OSError, KeyError, ValueError) as e:
            LOG.error("cannot re-read %s (%s); keeping previous settings", run_env_path, e)
        if state_path != spec.state_path:
            state_path = spec.state_path
            spec.staging_dir.mkdir(parents=True, exist_ok=True)
            state = load_state(state_path)
            state.setdefault("steps", {})
            cards_target = None
        steps = state["steps"]
        on_disk = current_steps(spec)
        latest = on_disk[-1] if on_disk else None
        for step in on_disk:
            if not spec.wanted(step):
                continue
            try:
                identity = checkpoint_identity(spec, step)
            except (OSError, ValueError, RuntimeError):
                continue
            rec = steps.get(str(step), {})
            if rec.get("identity") != identity or rec.get("generation") != spec.generation["id"]:
                steps[str(step)] = {}
        for step, rec in list(steps.items()):
            if rec.get("staged_at") and not staged_record(spec, int(step), rec):
                steps[step] = {}
        save_state(state_path, state)

        # 0. make sure the repo and its cards exist (so the experiment shows up before the first checkpoint)
        if cards_target != spec.target:
            try:
                n_uploaded = sum(1 for r in steps.values() if is_uploaded(spec, r))
                ensure_cards(api, spec, steps, n_uploaded=n_uploaded)
                cards_target = spec.target
                LOG.info("repo https://huggingface.co/%s ready; cards written for %s", spec.hf_repo, spec.exp_folder)
            except (HfHubHTTPError, OSError, ValueError) as e:
                if time.time() - last_repo_warning > 1800:
                    LOG.error("cannot prepare repo %s: %s", spec.hf_repo, explain_hf_error(e, spec.hf_repo))
                    last_repo_warning = time.time()

        # 1. stage every wanted checkpoint that is on disk and not staged yet
        for step in on_disk:
            rec = steps.setdefault(str(step), {}) if spec.wanted(step) else None
            if rec is None or rec.get("staged_at"):
                continue
            try:
                rec.update(stage(spec, step))
                save_state(state_path, state)
            except Exception as e:
                LOG.error("staging step %d failed: %s", step, e)
        # 2. wanted steps that are gone from disk without having been staged: record once
        if latest is not None:
            for step in range(latest + 1):
                if spec.wanted(step) and step not in on_disk and not steps.get(str(step), {}).get("staged_at") and not steps.get(str(step), {}).get("missed"):
                    steps.setdefault(str(step), {})["missed"] = True
                    LOG.warning("step %d was on the upload schedule but is no longer on disk and was never staged", step)
                    save_state(state_path, state)

        # 3. upload staged steps (oldest first), with per-step backoff
        pending = sorted(int(s) for s, r in steps.items() if staged_record(spec, int(s), r) and not is_uploaded(spec, r))
        now = time.time()
        for step in pending:
            rec = steps[str(step)]
            if rec.get("next_attempt_at", 0) > now:
                continue
            try:
                upload(api, spec, step, steps)
                rec.pop("next_attempt_at", None)
                rec.pop("failures", None)
                rec.pop("last_error", None)
                cards_target = spec.target
            except (HfHubHTTPError, RuntimeError, OSError, ValueError) as e:
                n = rec.get("failures", 0) + 1
                backoff = min(spec.poll_seconds * 2 ** min(n, 6), MAX_BACKOFF_SECONDS)
                rec.update({"failures": n, "last_error": str(e)[:300], "next_attempt_at": now + backoff})
                msg = explain_hf_error(e, spec.hf_repo)
                if "namespace" in msg and now - last_repo_warning < 1800 and n > 1:
                    pass  # same actionable warning was printed within the last 30 min
                else:
                    LOG.error("upload of step %d failed (attempt %d, retry in %d s): %s", step, n, backoff, msg)
                    if "namespace" in msg:
                        last_repo_warning = now
                break  # do not hammer the Hub with the remaining steps this cycle
            finally:
                save_state(state_path, state)

        # 4. heartbeat
        uploaded = sorted(int(s) for s, r in steps.items() if is_uploaded(spec, r))
        pending = sorted(int(s) for s, r in steps.items() if r.get("staged_at") and not is_uploaded(spec, r))
        if cycle == 1 or cycle % 10 == 0 or pending:
            LOG.info(
                "heartbeat: latest checkpoint %s (on disk: %s) | trainer alive: %s | %s | uploaded: %s | staged, waiting for upload: %s",
                latest, on_disk, trainer_alive(spec), last_progress_line(spec.train_log), uploaded or "-", pending or "-",
            )
        if spec.upload_final and spec.final_step in uploaded:
            LOG.info("final checkpoint (step %d) uploaded; all done. Exiting.", spec.final_step)
            return 0
        time.sleep(spec.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
