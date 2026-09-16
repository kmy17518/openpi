#!/usr/bin/env python
"""Keep exactly one FULL (resumable) checkpoint -- the newest on disk -- in <EXP_NAME>/resume/ of the project repo.

    python scripts/b1k/hf_latest_full_checkpoint_uploader.py scripts/b1k/runs/<exp>.env

(any Python >= 3.10 with `huggingface_hub >= 1.0` and `hf_xet`; no JAX needed. Logs go to $B1K_LOG_DIR, default
/tmp/dev/logs, the same directory train_b1k_run.sh writes the training log to.) See docs/b1k_runs.md.

Repo layout (shared with hf_checkpoint_uploader.py, same convention as kmy17518/b1k-challenge-2026-gr00t):

    <HF_REPO>/<EXP_NAME>/resume/checkpoint-<step>/   params/ (EMA weights), train_state/ (optimizer state + raw
                                                     params), assets/, _CHECKPOINT_METADATA  (~42 GB for pi05)
    <HF_REPO>/<EXP_NAME>/resume/wandb_id.txt         W&B run id (needed to resume into the same run)
    <HF_REPO>/<EXP_NAME>/resume/LATEST.json          step + provenance;  resume/README.md  how to resume

Every FULL_UPLOADER_POLL_SECONDS the script re-reads run.env, finds the newest completed orbax checkpoint under
<OPENPI_DIR>/outputs/checkpoints/<CONFIG_NAME>/<EXP_NAME>/<step>/ and, if resume/ does not hold that step yet:

1. stages an immutable full copy under STAGING_DIR/<generation>/full-<step>-<identity>/ and uploads it,
2. verifies every path/size and checkpoint provenance, then reconciles LATEST.json, W&B ID and README on every retry,
3. removes lower-numbered resume/checkpoint-*/ folders with normal Hub commits after all metadata is published.
   History, branches, tags and LFS objects remain untouched; storage quota is not reclaimed by this mirror.

Failures (including "namespace does not exist yet") are logged and retried with backoff; the newest checkpoint on
disk is always the target, so a slow upload simply skips intermediate checkpoints. State: STAGING_DIR/<generation>/full-state.json.
Log: /tmp/dev/logs/upload-full-<EXP_NAME>.log. Needs HF_TOKEN (source /tmp/dev/env.sh) with write access to HF_REPO.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import logging
import os
import pathlib
import shutil
import sys
import time

from huggingface_hub import HfApi
from huggingface_hub import RepoFile
from huggingface_hub import RepoFolder
from huggingface_hub.utils import HfHubHTTPError

# Shared helpers (run.env parsing, checkpoint discovery, heartbeat pieces) from the eval-only uploader next to this file.
_spec = importlib.util.spec_from_file_location("hf_checkpoint_uploader", pathlib.Path(__file__).with_name("hf_checkpoint_uploader.py"))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

LOG_DIR = common.LOG_DIR  # $B1K_LOG_DIR, default /tmp/dev/logs
MAX_BACKOFF_SECONDS = 900
LOG = logging.getLogger("full-uploader")


class FullSpec:
    def __init__(self, env: dict[str, str]):
        self.exp_name = env["EXP_NAME"]
        self.config_name = env["CONFIG_NAME"]
        self.repo = env.get("HF_FULL_REPO") or env["HF_REPO"]
        self.private = env.get("HF_FULL_REPO_PRIVATE", "0") in ("1", "true", "yes")
        self.exp_folder = env.get("HF_EXP_FOLDER") or self.exp_name
        self.folder = f"{self.exp_folder}/{env.get('HF_RESUME_FOLDER', 'resume')}"
        self.openpi_dir = pathlib.Path(env["OPENPI_DIR"])
        self.ckpt_dir = self.openpi_dir / "outputs" / "checkpoints" / self.config_name / self.exp_name
        self.staging_root = pathlib.Path(env.get("STAGING_DIR", f"/tmp/dev/hf-staging/{self.exp_name}"))
        self.generation = common.lifecycle.generation(self.ckpt_dir)
        self.staging_dir = self.staging_root / self.generation["id"]
        self.state_path = self.staging_dir / "full-state.json"
        self.poll_seconds = int(env.get("FULL_UPLOADER_POLL_SECONDS", 120))
        self.train_log = LOG_DIR / f"train-{self.exp_name}.log"
        self.raw = env

    @property
    def target(self) -> str:
        return f"{self.repo}/{self.folder}"

    def step_path(self, step: int) -> str:
        return f"{self.folder}/checkpoint-{step}"


# ------------------------------------------------------------------------------------------------ repo inspection
def remote_step_files(api: HfApi, spec: FullSpec) -> dict[int, list[RepoFile]]:
    """{step: files} for every resume/checkpoint-<step>/ folder (recursive listing). {} if the folder does not exist."""
    out: dict[int, list[RepoFile]] = {}
    try:
        items = list(api.list_repo_tree(spec.repo, path_in_repo=spec.folder, repo_type="model"))
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return out
        raise
    for item in items:
        name = item.path.split("/")[-1]
        if isinstance(item, RepoFolder) and name.startswith("checkpoint-") and name[len("checkpoint-"):].isdigit():
            out[int(name[len("checkpoint-"):])] = [
                f for f in api.list_repo_tree(spec.repo, path_in_repo=item.path, recursive=True, repo_type="model")
                if isinstance(f, RepoFile)
            ]
    return out


def local_manifest(step_dir: pathlib.Path) -> dict[str, int]:
    return {str(p.relative_to(step_dir)): p.stat().st_size for p in step_dir.rglob("*") if p.is_file()}


def remote_manifest(files: list[RepoFile], prefix: str) -> dict[str, int]:
    prefix = prefix.rstrip("/") + "/"
    return {f.path[len(prefix):]: f.size for f in files if f.path.startswith(prefix)}


def staged_path(spec: FullSpec, step: int, identity: str) -> pathlib.Path:
    return spec.staging_dir / f"full-{step}-{identity}"


def stage_full(spec: FullSpec, step: int, identity: str) -> pathlib.Path:
    dst = staged_path(spec, step, identity)
    provenance_path = dst / "training_run.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        if provenance.get("identity") == identity and provenance.get("generation") == spec.generation["id"]:
            try:
                if common.lifecycle.metadata_identity((dst / "_CHECKPOINT_METADATA").read_bytes(), spec.generation) == identity:
                    return dst
            except (OSError, ValueError):
                pass
    tmp = dst.with_suffix(".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(spec.ckpt_dir / str(step), tmp)
    if local_manifest(spec.ckpt_dir / str(step)) != local_manifest(tmp):
        raise RuntimeError(f"Incomplete staged copy of checkpoint {step}")
    if common.checkpoint_identity(spec, step) != identity or common.lifecycle.metadata_identity((tmp / "_CHECKPOINT_METADATA").read_bytes(), spec.generation) != identity:
        raise RuntimeError(f"Checkpoint {step} changed while staging")
    (tmp / "training_run.json").write_text(json.dumps({
        "identity": identity, "generation": spec.generation["id"],
        "generation_started_ns": spec.generation["started_ns"], "step": step,
    }) + "\n")
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    return dst


def remote_json(api: HfApi, spec: FullSpec, path: str) -> dict:
    if not api.file_exists(spec.repo, path, repo_type="model"):
        return {}
    local = api.hf_hub_download(spec.repo, path, repo_type="model", force_download=True)
    try:
        value = json.loads(pathlib.Path(local).read_text())
        return value if isinstance(value, dict) else {}
    except ValueError:
        return {}


def remote_identity(api: HfApi, spec: FullSpec, step: int) -> dict:
    return remote_json(api, spec, f"{spec.step_path(step)}/training_run.json")


def check_current_target(api: HfApi, spec: FullSpec, step: int) -> dict[int, list[RepoFile]]:
    if common.lifecycle.generation(spec.ckpt_dir) != spec.generation:
        raise RuntimeError("Run generation changed before publication")
    local = common.current_steps(spec)
    if local and local[-1] > step:
        raise RuntimeError(f"Checkpoint {step} is stale; newest local checkpoint is {local[-1]}")
    remote = remote_step_files(api, spec)
    published = [remote_json(api, spec, f"{spec.folder}/LATEST.json")]
    published.extend(remote_identity(api, spec, remote_step) for remote_step in remote)
    for provenance in published:
        remote_generation = provenance.get("generation")
        remote_step = provenance.get("step")
        if remote_generation == spec.generation["id"]:
            if type(remote_step) is int and remote_step > step:
                raise RuntimeError(f"Checkpoint {step} is stale; checkpoint {remote_step} is already remote")
        elif remote_generation and remote_generation != "legacy":
            started = provenance.get("generation_started_ns")
            if started is None and type(remote_step) is int:
                metadata = remote_json(api, spec, f"{spec.step_path(remote_step)}/_CHECKPOINT_METADATA")
                common.lifecycle.metadata_identity(json.dumps(metadata).encode(), {"id": "legacy", "started_ns": 0})
                started = metadata["commit_timestamp_nsecs"]
            if type(started) is not int or started < 0 or spec.generation["started_ns"] <= started:
                raise RuntimeError("Remote checkpoint belongs to a newer or unordered run generation")
    return remote


def verify(api: HfApi, spec: FullSpec, step: int, identity: str) -> list[RepoFile]:
    """Verify file paths/sizes and immutable checkpoint identity before publishing metadata."""
    files = remote_step_files(api, spec).get(step, [])
    remote = remote_manifest(files, spec.step_path(step))
    local = local_manifest(staged_path(spec, step, identity))
    provenance = remote_identity(api, spec, step)
    if provenance.get("identity") != identity or provenance.get("generation") != spec.generation["id"]:
        raise RuntimeError(f"Remote checkpoint {step} belongs to a different checkpoint generation")
    if remote != local:
        missing = sorted(set(local) - set(remote))[:5]
        extra = sorted(set(remote) - set(local))[:5]
        differ = sorted(k for k in set(local) & set(remote) if local[k] != remote[k])[:5]
        raise RuntimeError(
            f"remote {spec.step_path(step)}/ does not match the local checkpoint: {len(local)} local vs {len(remote)} "
            f"remote files; missing={missing} extra={extra} size-differs={differ}"
        )
    return files


# ------------------------------------------------------------------------------------------------------- upload
def readme_text(spec: FullSpec, step: int, size_gib: float) -> str:
    env = spec.raw
    return f"""# {spec.exp_name} -- latest full checkpoint (resume)

Rolling mirror of the **newest** full training checkpoint of `{spec.exp_name}` (openpi config `{spec.config_name}`,
task `{env.get("TASK_NAMES", "")}`, global batch {env.get("BATCH_SIZE", "?")}`). `LATEST.json` identifies the active
checkpoint and generation. After publication, lower-numbered checkpoint folders are removed by normal Hub commits;
higher-numbered folders from previous generations may remain. History, branches, tags, and LFS objects are retained.

**Current: `checkpoint-{step}/`** ({size_gib:.1f} GiB): `params/` (EMA weights, what serving loads),
`train_state/` (optimizer state + raw params, needed only to resume), `assets/` (norm stats + prompt source),
`_CHECKPOINT_METADATA`; plus `wandb_id.txt` (W&B run id) and `LATEST.json` (provenance) next to it.

## Resume training from it

```bash
cd <OPENPI_DIR>
hf download {spec.repo} --include "{spec.folder}/**" --local-dir /tmp/resume
mkdir -p outputs/checkpoints/{spec.config_name}/{spec.exp_name}
mv /tmp/resume/{spec.folder}/checkpoint-{step} outputs/checkpoints/{spec.config_name}/{spec.exp_name}/{step}
cp /tmp/resume/{spec.folder}/wandb_id.txt outputs/checkpoints/{spec.config_name}/{spec.exp_name}/
# then relaunch with the same flags as the original run (see LATEST.json) plus --resume, e.g.
scripts/b1k/train_b1k_run.sh scripts/b1k/runs/{spec.exp_name}.env resume
```

## Serve / evaluate it

```bash
uv run scripts/b1k/serve_b1k.py --robot b1k/R1Pro --task b1k/{env.get("TASK_NAMES", "")} \\
    --repo-id {env.get("REPO_ID", "")} --task-names {env.get("TASK_NAMES", "")} \\
    policy:checkpoint --policy.config {spec.config_name} --policy.dir /tmp/resume/{spec.step_path(step)}
```
"""


def upload_step(api: HfApi, spec: FullSpec, step: int, replaces: list[int], identity: str) -> tuple[list[RepoFile], float]:
    src = stage_full(spec, step, identity)
    files = local_manifest(src)
    size = sum(files.values())
    t0 = time.time()
    api.create_repo(spec.repo, repo_type="model", private=spec.private, exist_ok=True)
    LOG.info("uploading step %d (%d files, %.1f GiB) to %s/%s ...", step, len(files), size / 2**30, spec.repo, spec.step_path(step))
    suffix = f" (replaces checkpoint-{', checkpoint-'.join(str(s) for s in replaces)})" if replaces else ""
    commit = api.upload_folder(
        repo_id=spec.repo,
        repo_type="model",
        folder_path=str(src),
        path_in_repo=spec.step_path(step),
        delete_patterns="*",
        commit_message=f"{spec.exp_name}: full checkpoint {step} for resume{suffix}",
    )
    remote_files = verify(api, spec, step, identity)
    LOG.info("uploaded and verified step %d in %.0f s (%s)", step, time.time() - t0, getattr(commit, "commit_url", commit))
    return remote_files, time.time() - t0


def publish_metadata(api: HfApi, spec: FullSpec, step: int, identity: str) -> None:
    size = sum(local_manifest(staged_path(spec, step, identity)).values())
    provenance = {
        "identity": identity,
        "generation": spec.generation["id"],
        "generation_started_ns": spec.generation["started_ns"],
        "exp_name": spec.exp_name,
        "config_name": spec.config_name,
        "step": step,
        "path": spec.step_path(step),
        "checkpoint_source": str(spec.ckpt_dir / str(step)),
        "uploaded_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "size_bytes": size,
        "train_loss": common.train_loss_at(spec.train_log, step),
        "run_env": spec.raw,
        "openpi_git_commit": common.git_commit(spec.openpi_dir),
    }
    api.upload_file(
        path_or_fileobj=json.dumps(provenance, indent=2).encode(), path_in_repo=f"{spec.folder}/LATEST.json",
        repo_id=spec.repo, repo_type="model", commit_message=f"{spec.exp_name}: resume/LATEST.json -> checkpoint-{step}",
    )
    wandb_id = spec.ckpt_dir / "wandb_id.txt"
    if wandb_id.is_file():
        api.upload_file(path_or_fileobj=str(wandb_id), path_in_repo=f"{spec.folder}/wandb_id.txt", repo_id=spec.repo,
                        repo_type="model", commit_message=f"{spec.exp_name}: resume/wandb_id.txt")
    elif api.file_exists(spec.repo, f"{spec.folder}/wandb_id.txt", repo_type="model"):
        api.delete_file(path_in_repo=f"{spec.folder}/wandb_id.txt", repo_id=spec.repo, repo_type="model",
                        commit_message=f"{spec.exp_name}: remove stale W&B run id")
    api.upload_file(
        path_or_fileobj=readme_text(spec, step, size / 2**30).encode(), path_in_repo=f"{spec.folder}/README.md",
        repo_id=spec.repo, repo_type="model", commit_message=f"{spec.exp_name}: resume/README.md -> checkpoint-{step}",
    )


# -------------------------------------------------------------------------------------------------------- purge
def purge_others(api: HfApi, spec: FullSpec, keep_step: int) -> dict:
    """Remove only superseded resume paths by normal commits; retain all history and LFS objects."""
    deleted_steps = []
    for step in sorted(remote_step_files(api, spec)):
        if step >= keep_step:
            continue
        api.delete_folder(path_in_repo=spec.step_path(step), repo_id=spec.repo, repo_type="model",
                          commit_message=f"{spec.exp_name}: remove resume/checkpoint-{step} (superseded by checkpoint-{keep_step})")
        deleted_steps.append(step)
        LOG.info("deleted folder %s/ from %s", spec.step_path(step), spec.repo)
    return {"deleted_steps": deleted_steps}


def sync_step(api: HfApi, spec: FullSpec, step: int, state: dict) -> None:
    with (
        common.lifecycle.exclusive_lock(f"publish:{spec.ckpt_dir.resolve()}"),
        common.lifecycle.exclusive_lock(f"publish-full:{spec.target}"),
    ):
        remote = check_current_target(api, spec, step)
        identity = common.checkpoint_identity(spec, step)
        stage_full(spec, step, identity)
        check_current_target(api, spec, step)
        secs = 0.0
        try:
            verify(api, spec, step, identity)
        except (RuntimeError, HfHubHTTPError):
            _, secs = upload_step(api, spec, step, replaces=sorted(s for s in remote if s < step), identity=identity)
        if (spec.ckpt_dir / str(step)).exists() and common.checkpoint_identity(spec, step) != identity:
            raise RuntimeError(f"Checkpoint {step} changed during upload")
        check_current_target(api, spec, step)
        publish_metadata(api, spec, step, identity)
        purge = purge_others(api, spec, step)
        state.update({"remote_step": step, "remote_target": spec.target, "remote_repo": spec.repo,
                      "identity": identity, "generation": spec.generation["id"],
                      "updated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), **purge})
        state["history"] = [*state.get("history", []), {"step": step, "identity": identity, "at": state["updated_at"],
                                                      "upload_seconds": round(secs), **purge}][-50:]
        state.pop("last_error", None)
        common.save_state(spec.state_path, state)
        for old in spec.staging_dir.glob("full-*"):
            if old.is_dir() and old != staged_path(spec, step, identity):
                shutil.rmtree(old)


# --------------------------------------------------------------------------------------------------------- main
def setup_logging(exp_name: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname).1s] %(message)s", "%Y-%m-%d %H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_DIR / f"upload-full-{exp_name}.log")):
        handler.setFormatter(fmt)
        LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    run_env_path = pathlib.Path(sys.argv[1]).resolve()
    spec = FullSpec(common.read_run_env(run_env_path))
    setup_logging(spec.exp_name)
    if not os.environ.get("HF_TOKEN"):
        LOG.error("HF_TOKEN is not set; run `source /tmp/dev/env.sh` first")
        return 2
    spec.state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path = spec.state_path
    state = common.load_state(state_path) if state_path.is_file() else {}
    state.setdefault("history", [])
    api = HfApi(token=os.environ["HF_TOKEN"])
    LOG.info("=== latest-full-checkpoint mirror for %s: %s -> https://huggingface.co/%s/tree/main/%s (%s repo) ===", spec.exp_name, spec.ckpt_dir, spec.repo, spec.folder, "private" if spec.private else "public")
    LOG.info("HF token user: %s; poll every %d s", api.whoami().get("name"), spec.poll_seconds)

    failures, next_attempt, last_ns_warning, cycle = 0, 0.0, 0.0, 0
    while True:
        cycle += 1
        try:
            spec = FullSpec(common.read_run_env(run_env_path))
        except (OSError, KeyError, ValueError) as e:
            LOG.error("cannot re-read %s (%s); keeping previous settings", run_env_path, e)
        if state_path != spec.state_path:
            state_path = spec.state_path
            spec.staging_dir.mkdir(parents=True, exist_ok=True)
            state = common.load_state(state_path)
            failures, next_attempt = 0, 0.0
        on_disk = common.current_steps(spec)
        latest = on_disk[-1] if on_disk else None
        try:
            identity = common.checkpoint_identity(spec, latest) if latest is not None else None
        except (OSError, ValueError, RuntimeError):
            identity = None
        synchronized = identity is not None and state.get("identity") == identity and state.get("remote_target") == spec.target and state.get("generation") == spec.generation["id"]

        if identity is not None and not synchronized and time.time() >= next_attempt:
            try:
                sync_step(api, spec, latest, state)
                failures, next_attempt = 0, 0.0
            except (HfHubHTTPError, RuntimeError, OSError, ValueError) as e:
                failures += 1
                backoff = min(spec.poll_seconds * 2 ** min(failures, 6), MAX_BACKOFF_SECONDS)
                next_attempt = time.time() + backoff
                msg = common.explain_hf_error(e, spec.repo)
                if "namespace" in msg and failures > 1 and time.time() - last_ns_warning < 1800:
                    pass
                else:
                    LOG.error("mirroring step %d failed (attempt %d, retry in %d s): %s", latest, failures, backoff, msg)
                    if "namespace" in msg:
                        last_ns_warning = time.time()
                state["last_error"] = {"step": latest, "at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "error": str(e)[:300]}

        if cycle == 1 or cycle % 5 == 0 or (latest is not None and latest != state.get("remote_step")):
            LOG.info("heartbeat: newest checkpoint on disk %s (all: %s) | in %s: %s | trainer alive: %s | %s",
                     latest, on_disk, spec.folder, state.get("remote_step") if state.get("remote_target") == spec.target else None,
                     common.trainer_alive(spec), common.last_progress_line(spec.train_log))
        time.sleep(spec.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
