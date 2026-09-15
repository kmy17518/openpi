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

1. uploads the whole checkpoint directory as resume/checkpoint-<step>/ in one commit (straight from the checkpoint
   directory; orbax checkpoints are immutable once committed and survive max_to_keep * save_interval more steps),
2. verifies the uploaded tree against the local one (every file, by path and size) before anything is deleted,
3. removes every other resume/checkpoint-*/ folder and then PERMANENTLY deletes orphaned LFS objects with history
   rewrite (`HfApi.permanently_delete_lfs_files(rewrite_history=True)`), so superseded checkpoints do not linger
   as blobs that count against the storage quota. "Orphaned" = not referenced (by sha256) from ANY file in the
   repo's current tree -- the eval-only copies in <EXP_NAME>/checkpoint-<step>/params/ are byte-identical to the
   full checkpoint's params/ and therefore share LFS objects with it; those stay. Objects pushed within the last
   PURGE_MIN_AGE_SECONDS are never touched either, so an upload the other uploader has in flight cannot be hit.
   The bytes the Hub still lists for the repo are logged after every hand-over.

Failures (including "namespace does not exist yet") are logged and retried with backoff; the newest checkpoint on
disk is always the target, so a slow upload simply skips intermediate checkpoints. State: STAGING_DIR/full-state.json.
Log: /tmp/dev/logs/upload-full-<EXP_NAME>.log. Needs HF_TOKEN (source /tmp/dev/env.sh) with write access to HF_REPO.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import logging
import os
import pathlib
import sys
import time

from huggingface_hub import HfApi, RepoFile, RepoFolder
from huggingface_hub.utils import HfHubHTTPError

# Shared helpers (run.env parsing, checkpoint discovery, heartbeat pieces) from the eval-only uploader next to this file.
_spec = importlib.util.spec_from_file_location("hf_checkpoint_uploader", pathlib.Path(__file__).with_name("hf_checkpoint_uploader.py"))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

LOG_DIR = common.LOG_DIR  # $B1K_LOG_DIR, default /tmp/dev/logs
MAX_BACKOFF_SECONDS = 900
PURGE_MIN_AGE_SECONDS = 1800  # never purge LFS objects pushed more recently than this (other uploader may be mid-commit)
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
        self.state_path = pathlib.Path(env.get("STAGING_DIR", f"/tmp/dev/hf-staging/{self.exp_name}")) / "full-state.json"
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


def verify(api: HfApi, spec: FullSpec, step: int) -> list[RepoFile]:
    """Raise unless resume/checkpoint-<step>/ matches the local checkpoint file-for-file (path + size). Returns the files."""
    files = remote_step_files(api, spec).get(step, [])
    remote, local = remote_manifest(files, spec.step_path(step)), local_manifest(spec.ckpt_dir / str(step))
    if remote != local:
        missing = sorted(set(local) - set(remote))[:5]
        extra = sorted(set(remote) - set(local))[:5]
        differ = sorted(k for k in set(local) & set(remote) if local[k] != remote[k])[:5]
        raise RuntimeError(
            f"remote {spec.step_path(step)}/ does not match the local checkpoint: {len(local)} local vs {len(remote)} "
            f"remote files; missing={missing} extra={extra} size-differs={differ}"
        )
    return files


def referenced_lfs_oids(api: HfApi, repo: str) -> tuple[set[str], int]:
    """sha256 of every LFS object referenced from the repo's current tree, and their total bytes."""
    oids, total = set(), 0
    for f in api.list_repo_tree(repo, recursive=True, repo_type="model"):
        if isinstance(f, RepoFile) and f.lfs is not None:
            if f.lfs.sha256 not in oids:
                total += f.lfs.size
            oids.add(f.lfs.sha256)
    return oids, total


# ------------------------------------------------------------------------------------------------------- upload
def readme_text(spec: FullSpec, step: int, size_gib: float) -> str:
    env = spec.raw
    return f"""# {spec.exp_name} -- latest full checkpoint (resume)

Rolling mirror of the **newest** full training checkpoint of `{spec.exp_name}` (openpi config `{spec.config_name}`,
task `{env.get("TASK_NAMES", "")}`, global batch {env.get("BATCH_SIZE", "?")}). This folder holds exactly one step at a
time; when a newer checkpoint is saved it is uploaded, the older one is deleted and its LFS objects are purged from
the repo history (objects shared with the eval-only `../checkpoint-<step>/` copies are kept).

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


def upload_step(api: HfApi, spec: FullSpec, step: int, replaces: list[int]) -> tuple[list[RepoFile], float]:
    src = spec.ckpt_dir / str(step)
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
        commit_message=f"{spec.exp_name}: full checkpoint {step} for resume{suffix}",
    )
    remote_files = verify(api, spec, step)
    provenance = {
        "exp_name": spec.exp_name,
        "config_name": spec.config_name,
        "step": step,
        "path": spec.step_path(step),
        "checkpoint_source": str(src),
        "uploaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
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
    api.upload_file(
        path_or_fileobj=readme_text(spec, step, size / 2**30).encode(), path_in_repo=f"{spec.folder}/README.md",
        repo_id=spec.repo, repo_type="model", commit_message=f"{spec.exp_name}: resume/README.md -> checkpoint-{step}",
    )
    LOG.info("uploaded and verified step %d in %.0f s (%s)", step, time.time() - t0, getattr(commit, "commit_url", commit))
    return remote_files, time.time() - t0


# -------------------------------------------------------------------------------------------------------- purge
def purge_others(api: HfApi, spec: FullSpec, keep_step: int) -> dict:
    """Delete every other resume/checkpoint-* folder, then permanently purge LFS objects nothing in the repo references."""
    deleted_steps = []
    for step in sorted(remote_step_files(api, spec)):
        if step == keep_step:
            continue
        api.delete_folder(path_in_repo=spec.step_path(step), repo_id=spec.repo, repo_type="model",
                          commit_message=f"{spec.exp_name}: remove resume/checkpoint-{step} (superseded by checkpoint-{keep_step})")
        deleted_steps.append(step)
        LOG.info("deleted folder %s/ from %s", spec.step_path(step), spec.repo)
    # Orphans = LFS objects (deduplicated by content hash on the Hub) that no file in the current tree references.
    # `all_lfs` is fetched BEFORE the tree so an object committed in between is seen as referenced, and anything pushed
    # within PURGE_MIN_AGE_SECONDS is left alone: the eval-only uploader may have pre-uploaded it and not committed yet.
    all_lfs = list(api.list_lfs_files(spec.repo, repo_type="model"))
    referenced, referenced_bytes = referenced_lfs_oids(api, spec.repo)
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=PURGE_MIN_AGE_SECONDS)
    orphans, young = [], []
    for f in all_lfs:
        if f.file_oid in referenced:
            continue
        pushed = f.pushed_at
        if pushed is not None and pushed.tzinfo is None:
            pushed = pushed.replace(tzinfo=dt.timezone.utc)
        (young if pushed is not None and pushed > cutoff else orphans).append(f)
    purged_bytes = sum(f.size for f in orphans)
    if orphans:
        api.permanently_delete_lfs_files(spec.repo, orphans, rewrite_history=True, repo_type="model")
        LOG.info("permanently purged %d unreferenced LFS objects (%.1f GiB) with history rewrite: %s%s", len(orphans),
                 purged_bytes / 2**30, ", ".join(sorted({"/".join(f.filename.split("/")[:3]) for f in orphans})[:6]),
                 " ..." if len(orphans) > 6 else "")
    if young:
        LOG.info("left %d unreferenced LFS objects (%.1f GiB) pushed < %d min ago for the next hand-over",
                 len(young), sum(f.size for f in young) / 2**30, PURGE_MIN_AGE_SECONDS // 60)
    remaining = list(api.list_lfs_files(spec.repo, repo_type="model"))
    remaining_bytes = sum(f.size for f in remaining)
    if remaining_bytes > referenced_bytes * 1.01 + sum(f.size for f in young) + 2**20:
        LOG.warning("Hub still lists %.1f GiB of LFS objects for %s vs %.1f GiB referenced by its tree -- storage accounting can lag; will re-check next hand-over",
                    remaining_bytes / 2**30, spec.repo, referenced_bytes / 2**30)
    else:
        LOG.info("Hub LFS storage for %s: %.1f GiB in %d objects; %.1f GiB referenced by the tree (resume/checkpoint-%d + eval-only copies)",
                 spec.repo, remaining_bytes / 2**30, len(remaining), referenced_bytes / 2**30, keep_step)
    return {"deleted_steps": deleted_steps, "purged_objects": len(orphans), "purged_bytes": purged_bytes,
            "remaining_lfs_bytes": remaining_bytes, "remaining_lfs_objects": len(remaining), "referenced_lfs_bytes": referenced_bytes}


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
    state = common.load_state(spec.state_path) if spec.state_path.is_file() else {}
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
        on_disk = common.completed_steps(spec.ckpt_dir)
        latest = on_disk[-1] if on_disk else None
        remote_step = state.get("remote_step") if state.get("remote_target") == spec.target else None

        if latest is not None and latest != remote_step and time.time() >= next_attempt:
            try:
                try:
                    remote = remote_step_files(api, spec)
                except HfHubHTTPError:
                    remote = {}  # repo unreadable / missing: upload_step creates it or reports the error
                if latest in remote:
                    # e.g. an earlier cycle uploaded but failed during the purge, or a restart: verify instead of re-uploading
                    LOG.info("repo already holds %s/, verifying instead of re-uploading", spec.step_path(latest))
                    verify(api, spec, latest)
                    secs = 0.0
                else:
                    _, secs = upload_step(api, spec, latest, replaces=sorted(s for s in remote if s != latest))
                purge = purge_others(api, spec, latest)
                state.update({"remote_step": latest, "remote_target": spec.target, "remote_repo": spec.repo,
                              "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), **purge})
                state["history"] = (state["history"] + [{"step": latest, "at": state["updated_at"], "upload_seconds": round(secs), **purge}])[-50:]
                common.save_state(spec.state_path, state)
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
                state["last_error"] = {"step": latest, "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "error": str(e)[:300]}
                common.save_state(spec.state_path, state)

        if cycle == 1 or cycle % 5 == 0 or (latest is not None and latest != state.get("remote_step")):
            LOG.info("heartbeat: newest checkpoint on disk %s (all: %s) | in %s: %s | trainer alive: %s | %s",
                     latest, on_disk, spec.folder, state.get("remote_step") if state.get("remote_target") == spec.target else None,
                     common.trainer_alive(spec), common.last_progress_line(spec.train_log))
        time.sleep(spec.poll_seconds)


if __name__ == "__main__":
    sys.exit(main())
