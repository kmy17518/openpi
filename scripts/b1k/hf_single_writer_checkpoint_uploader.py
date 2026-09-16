#!/usr/bin/env python
"""Sole-writer, physically bounded Orbax checkpoints in a fresh private Hub repo.

Call stage_local_checkpoint(run_dir / str(step), staging_dir, step) synchronously
AFTER CheckpointManager.wait_until_finished(), before the next save/prune. Steps
are explicit completed-update labels; this module never loads JAX or train_state.
Keep owner.json and journal.json across restarts. Never share the destination with
other uploaders, UI writers, branches, tags or PRs. flock is only a same-host lock;
Hub has no conditional physical-delete API, so exclusive remote ownership is a
lifetime requirement. Historical full revisions intentionally become unreadable.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time
import uuid

from huggingface_hub import CommitOperationAdd
from huggingface_hub import CommitOperationDelete
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
from huggingface_hub.utils import RepositoryNotFoundError

OWNER = "kmy17518"
SCRATCH_REPO = "kmy17518/pi05-radio-uploader-validation-20260916"
LOCK_ROOT = Path("/tmp/dev/hf-staging/.single-writer-locks")
CHECKPOINT = re.compile(r"checkpoint-([1-9]\d*)")
REQUIRED = {"params/manifest.ocdbt", "train_state/manifest.ocdbt", "_CHECKPOINT_METADATA"}
TOP_LEVEL = {"params", "train_state", "assets", "_CHECKPOINT_METADATA"}
# Small OCDBT manifests and metadata must also be physically collectable, not Git blobs.
ATTRIBUTES = b"eval/** filter=lfs diff=lfs merge=lfs -text\nresume/** filter=lfs diff=lfs merge=lfs -text\n"


class BlockedError(RuntimeError):
    """An ownership, completeness, or storage invariant requires intervention."""


Blocked = BlockedError


def canonical(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, indent=2).encode() + b"\n"


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as handle:
        tmp = Path(handle.name)
        try:
            handle.write(canonical(data))
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (OSError, ValueError) as exc:
        raise Blocked(f"Unreadable state: {path}; restore it, do not discard it") from exc


@contextmanager
def exclusive_lock(path: Path, *, blocking: bool = False):
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise Blocked(f"Another local writer holds {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def repo_lock(repo_id: str) -> Path:
    return LOCK_ROOT / (hashlib.sha256(repo_id.encode()).hexdigest() + ".lock")


def snapshot(path: Path) -> dict[str, tuple]:
    if path.is_symlink() or not path.is_dir():
        raise Blocked(f"Not a private checkpoint directory: {path}")
    result = {}
    for src in sorted(path.rglob("*")):
        info = src.lstat()
        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
            raise Blocked(f"Unsupported checkpoint entry: {src}")
        if stat.S_ISREG(info.st_mode):
            result[src.relative_to(path).as_posix()] = (
                info.st_dev,
                info.st_ino,
                info.st_size,
                info.st_mtime_ns,
                info.st_ctime_ns,
            )
    return result


def validate_checkpoint(path: Path, step: int, *, started_ns: int = 0) -> dict[str, tuple]:
    """Check the committed Orbax envelope, without interpreting serialized train state."""
    if type(step) is not int or step <= 0:
        raise Blocked("Checkpoint step must be an explicit positive completed-update integer")
    files = snapshot(path)
    if not files.keys() >= REQUIRED or {p.name for p in path.iterdir()} != TOP_LEVEL:
        raise Blocked(f"Incomplete/unsupported Orbax checkpoint: {path}")
    for name in ("params", "train_state", "assets"):
        if not (path / name).is_dir() or not any(p.startswith(name + "/") for p in files):
            raise Blocked(f"Missing nonempty {name} directory")
    if any(files[name][2] == 0 for name in REQUIRED):
        raise Blocked("Empty Orbax manifest or commit metadata")
    for name in ("params", "train_state"):
        if f"{name}/_METADATA" not in files or not any(p.startswith(f"{name}/d/") for p in files):
            raise Blocked(f"Missing Orbax {name} metadata or OCDBT data")
    metadata = read_json(path / "_CHECKPOINT_METADATA")
    timestamp = metadata.get("commit_timestamp_nsecs")
    if type(timestamp) is not int or timestamp <= 0 or timestamp < started_ns:
        raise Blocked("Checkpoint is not committed or predates this run owner")
    for key in ("step", "global_step"):
        if key in metadata and (type(metadata[key]) is not int or metadata[key] != step):
            raise Blocked("Checkpoint step disagrees with commit metadata")
    # item_handlers/metadata may contain serialized strings in Orbax; do not parse state internals.
    return files


def scheduled(owner: dict, step: int) -> bool:
    return (
        type(step) is int
        and 0 < step <= owner["max_steps"]
        and (step == owner["first_step"] or step % owner["full_every"] == 0)
    )


def load_journal(path: Path, owner: dict) -> dict:
    envelope = read_json(path)
    payload = envelope.get("state")
    if not isinstance(payload, dict) or envelope.get("sha256") != hashlib.sha256(canonical(payload)).hexdigest():
        raise Blocked("Corrupt journal; restore it before any remote mutation")
    if payload.get("owner") != owner:
        raise Blocked("Destination/run configuration changed; refusing journal reuse")
    return payload


def checkpoint_identity(path: Path) -> str:
    return hashlib.sha256((path / "_CHECKPOINT_METADATA").read_bytes()).hexdigest()


def stage_local_checkpoint(src: Path, staging_root: Path, step: int) -> Path:
    """Durably copy a completed save before training resumes; never access the Hub.

    The numeric source directory and explicit public step must match. An already
    published identical save is a no-op; the returned queue path may then be absent.
    Queue exhaustion raises Blocked rather than losing an unpublished eval snapshot.
    """
    src, staging_root = Path(src), Path(staging_root)
    owner = read_json(staging_root / "owner.json")
    if src.is_symlink() or src.resolve() != Path(owner["run_dir"]) / str(step):
        raise Blocked("Checkpoint is outside the immutable run destination or has a different step")
    if not scheduled(owner, step):
        raise Blocked(f"Unscheduled checkpoint {step}")
    queue = staging_root / "queue"
    queue.mkdir(exist_ok=True)
    dst = queue / f"checkpoint-{step}"
    with exclusive_lock(staging_root / ".stage.lock", blocking=True):
        if owner.get("wandb_id"):
            identity_path = Path(owner["run_dir"]) / "wandb_id.txt"
            if identity_path.is_symlink() or identity_path.read_text().strip() != owner["wandb_id"]:
                raise Blocked("Local W&B resume identity differs from the immutable run owner")
        before = validate_checkpoint(src, step, started_ns=owner["created_ns"])
        marker = {"owner_id": owner["owner_id"], "step": step, "identity": checkpoint_identity(src)}
        if (staging_root / "journal.json").exists():
            state = load_journal(staging_root / "journal.json", owner)
            if step <= state["latest_full_step"]:
                if state["published_identities"].get(str(step)) != marker["identity"]:
                    raise Blocked("A published checkpoint step was reused or replaced")
                return dst
        if dst.exists():
            if read_json(dst / "ready.json") != marker:
                raise Blocked("Staged checkpoint ownership/identity mismatch")
            validate_checkpoint(dst / "checkpoint", step, started_ns=owner["created_ns"])
            if checkpoint_identity(dst / "checkpoint") != marker["identity"]:
                raise Blocked("Staged commit metadata changed")
            return dst
        pending_bytes = sum(p.stat().st_size for p in queue.rglob("*") if p.is_file())
        if pending_bytes + sum(info[2] for info in before.values()) > owner["max_staging_bytes"]:
            raise Blocked("Staging queue capacity reached; pause training until publication catches up")
        tmp = Path(tempfile.mkdtemp(dir=queue, prefix=".copy-"))
        try:
            data = tmp / "checkpoint"
            for rel in before:
                target = data / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src / rel, target)
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
            if snapshot(src) != before:
                raise Blocked("Checkpoint changed while copying; old remote full is untouched")
            copied = validate_checkpoint(data, step, started_ns=owner["created_ns"])
            if {k: v[2] for k, v in copied.items()} != {k: v[2] for k, v in before.items()}:
                raise Blocked("Incomplete checkpoint copy")
            for directory in sorted((p for p in data.rglob("*") if p.is_dir()), reverse=True):
                fsync_dir(directory)
            fsync_dir(data)
            atomic_json(tmp / "ready.json", marker)
            os.replace(tmp, dst)
            fsync_dir(queue)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    return dst


def digest_bytes(data: bytes) -> dict:
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "blob_id": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
    }


def file_manifest(path: Path) -> dict[str, dict]:
    before = snapshot(path)
    result = {}
    for rel, info in before.items():
        sha = hashlib.sha256()
        blob = hashlib.sha1(f"blob {info[2]}\0".encode())
        with (path / rel).open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                sha.update(chunk)
                blob.update(chunk)
        result[rel] = {"size": info[2], "sha256": sha.hexdigest(), "blob_id": blob.hexdigest()}
    if snapshot(path) != before:
        raise Blocked("Staged checkpoint changed during hashing")
    return result


def eval_file(rel: str) -> bool:
    return rel == "_CHECKPOINT_METADATA" or rel.startswith(("params/", "assets/"))


def collectable(path: str, old_step: int) -> bool:
    prefix = f"resume/checkpoint-{old_step}/"
    if not path.startswith(prefix):
        return False
    rel = path.removeprefix(prefix)
    return rel == "_CHECKPOINT_METADATA" or rel.startswith(("params/", "train_state/", "assets/"))


def remote_tree(api: HfApi, repo: str, revision: str) -> dict[str, dict]:
    result = {}
    for item in api.list_repo_tree(repo, repo_type="model", revision=revision, recursive=True):
        if not hasattr(item, "blob_id"):
            continue
        entry = {"size": item.size, "blob_id": item.blob_id}
        if item.lfs:
            entry["sha256"] = item.lfs["sha256"] if isinstance(item.lfs, dict) else item.lfs.sha256
            entry["lfs"] = True
        result[item.path] = entry
    return result


def verify_tree(actual: dict, expected: dict) -> None:
    if actual.keys() != expected.keys():
        raise Blocked("Remote tree differs from the journal (unexpected/missing paths)")
    for path, item in actual.items():
        key = "sha256" if item.get("lfs") else "blob_id"
        if item["size"] != expected[path]["size"] or item[key] != expected[path].get(key):
            raise Blocked(f"Remote content verification failed: {path}")
        if path.startswith(("eval/", "resume/")) and not item.get("lfs"):
            raise Blocked(f"Checkpoint content is not physically collectable LFS: {path}")
        if not path.startswith(("eval/", "resume/")) and item.get("lfs"):
            raise Blocked(f"Unexpected LFS data outside checkpoint allowlist: {path}")


def lfs_oids(tree: dict) -> set[str]:
    return {item["sha256"] for item in tree.values() if item.get("lfs")}


class Coordinator:
    def __init__(self, api: HfApi, root: Path, owner: dict):
        self.api, self.root, self.owner = api, Path(root), owner
        self.repo = owner["repo_id"]
        self.path = self.root / "journal.json"
        self.state = load_journal(self.path, owner) if self.path.exists() else None

    def save(self):
        try:
            atomic_json(self.path, {"state": self.state, "sha256": hashlib.sha256(canonical(self.state)).hexdigest()})
        except OSError as exc:
            raise Blocked("Journal durability failed; recover from disk before any remote mutation") from exc

    def inventory(self) -> dict:
        items = {}
        for item in self.api.list_lfs_files(self.repo, repo_type="model"):
            if item.file_oid in items and items[item.file_oid].size != item.size:
                raise Blocked("Inconsistent LFS inventory")
            items[item.file_oid] = item
        return items

    def guard(self, expected_head: str | None = None) -> str:
        info = self.api.repo_info(self.repo, repo_type="model", revision="main")
        if not info.private or info.id != self.repo:
            raise Blocked("Repository is not the expected private owned model repository")
        refs = self.api.list_repo_refs(self.repo, repo_type="model", include_pull_requests=True)
        if (
            len(refs.branches) != 1
            or refs.branches[0].ref != "refs/heads/main"
            or refs.tags
            or refs.converts
            or refs.pull_requests
        ):
            raise Blocked("Unexpected branch, tag, conversion or PR ref; stop all writers")
        head = info.sha
        if refs.branches[0].target_commit != head or (expected_head and head != expected_head):
            raise Blocked("Unexpected HEAD/new writer; irreversible collection refused")
        return head

    def preflight(self, storage_proof: Path | None = None):
        for name in ("list_lfs_files", "permanently_delete_lfs_files", "list_repo_refs"):
            if not callable(getattr(self.api, name, None)):
                raise Blocked(f"Physical storage API unavailable: {name}")
        identity = self.api.whoami()
        if identity.get("name") != OWNER or not self.repo.startswith(OWNER + "/"):
            raise Blocked("Only kmy17518-owned private run repositories are authorized")
        if self.api.endpoint != "https://huggingface.co":
            raise Blocked("Only the verified Hugging Face endpoint is authorized")
        if identity.get("auth", {}).get("accessToken", {}).get("role") != "write":
            raise Blocked("An owner write token is required; fine-grained GC permission is unproven")
        if self.repo != SCRATCH_REPO:
            if storage_proof is None:
                raise Blocked("Successful scratch physical-storage proof is required before launch")
            proof = read_json(storage_proof)
            if not (
                proof.get("repo_id") == SCRATCH_REPO
                and proof.get("owner") == OWNER
                and proof.get("endpoint") == self.api.endpoint
                and proof.get("rotations", 0) >= 2
                and proof.get("deleted_bytes", 0) > 0
                and proof.get("live_download_verified") is True
                and proof.get("inventory_equals_current") is True
                and proof.get("all_checkpoint_files_lfs") is True
                and proof.get("verified_before_retirement") is True
                and proof.get("quota_accounting_confirmed") is True
            ):
                raise Blocked("Invalid or incomplete scratch storage proof")

    def initialize(self, *, adopt_empty: bool = False):
        if self.state is None:
            try:
                self.api.repo_info(self.repo, repo_type="model")
            except RepositoryNotFoundError:
                pass
            else:
                if not adopt_empty:
                    raise Blocked("Repository already exists; explicit empty-repo approval required")
            self.state = {
                "owner": self.owner,
                "phase": "creating",
                "head": None,
                "tree": {},
                "latest_full_step": 0,
                "eval_steps": [],
                "published_identities": {},
                "deleted_bytes": 0,
                "deleted_objects": 0,
                "rotations": 0,
                "pending": None,
            }
            self.save()
        if self.state["phase"] == "creating":
            try:
                self.api.repo_info(self.repo, repo_type="model")
            except RepositoryNotFoundError:
                self.api.create_repo(self.repo, repo_type="model", private=True, exist_ok=False)
            head = self.guard()
            tree = remote_tree(self.api, self.repo, head)
            commits = self.api.list_repo_commits(self.repo, repo_type="model", revision=head)
            if set(tree) - {".gitattributes"} or self.inventory() or len(commits) != 1:
                raise Blocked("Initialization requires an empty new repo, one initial commit and no LFS objects")
            self.state.update(phase="idle", head=head, tree=tree)
            self.save()
        if not self.state["pending"] and ".uploader-owner.json" not in self.state["tree"]:
            self.prepare(None, 0)
        self.finish()

    def metadata(self, step: int, eval_steps: list[int]) -> dict[str, bytes]:
        return {
            **({"wandb_id.txt": (self.owner["wandb_id"] + "\n").encode()} if self.owner.get("wandb_id") else {}),
            ".gitattributes": ATTRIBUTES,
            ".uploader-owner.json": canonical(self.owner),
            "run_config.json": canonical(self.owner["run_config"]),
            "README.md": (
                "---\nlicense: apache-2.0\nbase_model: physical-intelligence/pi05_base\n---\n"
                f"# {self.owner['run_id']}\n\nPrivate OpenPI turning_on_radio run.\n\n"
                f"- W&B: {self.owner['wandb_url'] or 'Not configured'}\n"
                f"- Target: {self.owner['max_steps']} completed updates; full cadence {self.owner['full_every']}; "
                f"eval cadence {self.owner['eval_every']}.\n"
                f"- Latest verified full: `resume/checkpoint-{step}/`.\n"
                f"- Retained eval steps: {', '.join(map(str, eval_steps)) or 'none yet'}.\n\n"
                "Eval contains params/, assets/ and _CHECKPOINT_METADATA. Full also contains train_state/ "
                "(raw parameters and optimizer state); all original asset files are retained. "
                "Restore full checkpoints into the numeric Orbax step directory and put the root wandb_id.txt "
                "beside that directory to resume the same W&B run. See run_config.json for provenance.\n\n"
                "Only this coordinator may write the repository. A replacement full is uploaded and verified "
                "while the previous full is still present, then the previous directory is retired. "
                "Stale checkpoint LFS objects are permanently deleted without rewriting history. "
                "Historical full revisions are NOT resumable. All current eval and full objects are protected. "
                "Storage temporarily includes both old and new full checkpoints; quota accounting may lag.\n"
            ).encode(),
        }

    def prepare(self, staged: Path | None, step: int):
        if self.state["pending"]:
            raise Blocked("Finish the pending transaction before preparing another")
        if staged is None and (step != 0 or ".uploader-owner.json" in self.state["tree"]):
            raise Blocked("Metadata-only initialization cannot replace a checkpoint")
        if staged is not None and (not scheduled(self.owner, step) or step <= self.state["latest_full_step"]):
            raise Blocked("Unscheduled or nonmonotonic checkpoint")
        required = set(range(self.owner["eval_every"], step, self.owner["eval_every"]))
        if required - set(self.state["eval_steps"]):
            raise Blocked("Required eval checkpoint was skipped; restore its ready copy")
        head = self.guard(self.state["head"])
        old = remote_tree(self.api, self.repo, head)
        verify_tree(old, self.state["tree"])
        inventory = self.inventory()
        if set(inventory) != lfs_oids(old):
            raise Blocked("Unjournaled or missing LFS objects exist; refusing unrelated object deletion")
        additions, marker = {}, None
        if staged is not None:
            staged = Path(staged)
            self.validate_stage_path(staged, step)
            marker = read_json(staged / "ready.json")
            expected = {
                "owner_id": self.owner["owner_id"],
                "step": step,
                "identity": checkpoint_identity(staged / "checkpoint"),
            }
            if marker != expected:
                raise Blocked("Ready marker belongs to another run or checkpoint")
            validate_checkpoint(staged / "checkpoint", step, started_ns=self.owner["created_ns"])
            additions = file_manifest(staged / "checkpoint")
        eval_steps = list(self.state["eval_steps"])
        if step and step % self.owner["eval_every"] == 0:
            eval_steps.append(step)
        upload_tree = dict(old)
        for rel, item in additions.items():
            upload_tree[f"resume/checkpoint-{step}/{rel}"] = item
            if step in eval_steps and eval_file(rel):
                upload_tree[f"eval/checkpoint-{step}/{rel}"] = item
        old_step = self.state["latest_full_step"]
        retire_paths = [p for p in old if collectable(p, old_step)]
        if {p for p in old if p.startswith("resume/")} != set(retire_paths):
            raise Blocked("Old full paths do not match the recorded checkpoint allowlist")
        metadata = self.metadata(step, eval_steps)
        final_tree = {p: v for p, v in upload_tree.items() if p not in retire_paths}
        final_tree.update({name: digest_bytes(data) for name, data in metadata.items()})
        if not staged:
            upload_tree = final_tree
        new_objects = {v["sha256"]: v["size"] for v in additions.values()}
        projected = sum(v.size for v in inventory.values()) + sum(
            size for oid, size in new_objects.items() if oid not in inventory
        )
        if projected > self.owner["max_remote_lfs_bytes"]:
            raise Blocked("Remote LFS budget cannot hold retained evals plus both old and replacement full")
        self.state["pending"] = {
            "id": uuid.uuid4().hex,
            "phase": "prepared",
            "parent": head,
            "step": step,
            "staged": str(staged) if staged else None,
            "marker": marker,
            "additions": additions,
            "upload_tree": upload_tree,
            "tree": final_tree,
            "eval_steps": eval_steps,
            "metadata": {k: v.decode() for k, v in metadata.items()},
            "retire_paths": retire_paths,
            "candidates": {old[p]["sha256"]: old[p]["size"] for p in retire_paths if old[p].get("lfs")},
            "upload_commit": None,
            "commit": None,
            "projected_peak_lfs_bytes": projected,
        }
        self.save()

    def validate_stage_path(self, path: Path, step: int):
        expected = self.root.resolve() / "queue" / f"checkpoint-{step}"
        if path.is_symlink() or path.resolve() != expected:
            raise Blocked("Staged path escapes the owned queue")

    def recover_commit(self, parent: str, message: str, expected: dict) -> str | None:
        head = self.guard()
        if head == parent:
            return None
        commits = self.api.list_repo_commits(self.repo, repo_type="model", revision=head)
        if len(commits) < 2 or commits[0].title != message or commits[1].commit_id != parent:
            raise Blocked("Unexpected HEAD during commit recovery; refusing adoption")
        verify_tree(remote_tree(self.api, self.repo, head), expected)
        return head

    def commit(self, parent: str, message: str, operations: list) -> str:
        self.guard(parent)
        result = self.api.create_commit(
            self.repo,
            repo_type="model",
            revision="main",
            parent_commit=parent,
            operations=operations,
            commit_message=message,
            num_threads=2,
        )
        return result.oid

    def finish(self):
        tx = self.state["pending"]
        if tx is None:
            self.guard(self.state["head"])
            return
        message = f"single-writer {self.owner['owner_id']} {tx['id']} step {tx['step']}"
        if tx["phase"] == "prepared":
            head = self.recover_commit(tx["parent"], message + " upload", tx["upload_tree"])
            if head is None:
                operations = []
                if tx["staged"]:
                    data = Path(tx["staged"]) / "checkpoint"
                    if file_manifest(data) != tx["additions"]:
                        raise Blocked("Pending private copy is unavailable or changed; restore it")
                    for rel in tx["additions"]:
                        operations.append(
                            CommitOperationAdd(
                                path_in_repo=f"resume/checkpoint-{tx['step']}/{rel}", path_or_fileobj=str(data / rel)
                            )
                        )
                        if tx["step"] in tx["eval_steps"] and eval_file(rel):
                            operations.append(
                                CommitOperationAdd(
                                    path_in_repo=f"eval/checkpoint-{tx['step']}/{rel}", path_or_fileobj=str(data / rel)
                                )
                            )
                else:
                    operations.extend(
                        CommitOperationAdd(path_in_repo=k, path_or_fileobj=v.encode())
                        for k, v in tx["metadata"].items()
                    )
                head = self.commit(tx["parent"], message + " upload", operations)
            tx.update(phase="uploaded", upload_commit=head)
            self.save()
        if tx["phase"] == "uploaded":
            self.guard(tx["upload_commit"])
            actual = remote_tree(self.api, self.repo, tx["upload_commit"])
            verify_tree(actual, tx["upload_tree"])
            if set(self.inventory()) != lfs_oids(actual):
                raise Blocked("Replacement inventory is incomplete or unknown; previous full retained")
            self.guard(tx["upload_commit"])
            tx.update(phase="verified", upload_tree=actual)
            self.save()
        if tx["phase"] == "verified":
            head = self.recover_commit(tx["upload_commit"], message + " retire", tx["tree"])
            if head is None:
                # Reverify after a restart, before removing any old full paths.
                verify_tree(remote_tree(self.api, self.repo, tx["upload_commit"]), tx["upload_tree"])
                if set(self.inventory()) != lfs_oids(tx["upload_tree"]):
                    raise Blocked("Verified replacement inventory changed; previous full retained")
                operations = [CommitOperationDelete(path_in_repo=p) for p in tx["retire_paths"]]
                operations.extend(
                    CommitOperationAdd(path_in_repo=k, path_or_fileobj=v.encode()) for k, v in tx["metadata"].items()
                )
                head = self.commit(tx["upload_commit"], message + " retire", operations)
            tx.update(phase="committed", commit=head)
            self.save()
        self.guard(tx["commit"])
        actual = remote_tree(self.api, self.repo, tx["commit"])
        verify_tree(actual, tx["tree"])
        live = lfs_oids(actual)
        inventory = self.inventory()
        candidates = set(tx["candidates"]) - live
        if not live <= inventory.keys():
            raise Blocked("Current eval/full LFS data is missing from physical inventory")
        if inventory.keys() - live - candidates:
            raise Blocked("Unknown LFS objects/new writer; irreversible deletion refused")
        targets = sorted(candidates & inventory.keys())
        tx.update(phase="gc", tree=actual, gc_targets=targets)
        self.save()
        # Guard every API batch: physical deletion itself offers no compare-and-swap.
        for start in range(0, len(targets), 100):
            self.guard(tx["commit"])
            self.api.permanently_delete_lfs_files(
                self.repo,
                [inventory[oid] for oid in targets[start : start + 100]],
                repo_type="model",
                rewrite_history=False,
            )
        self.guard(tx["commit"])
        verify_tree(remote_tree(self.api, self.repo, tx["commit"]), actual)
        if set(self.inventory()) != live:
            raise RuntimeError("Physical collection is not confirmed; pending journal retained, rotations paused")
        self.state.update(
            head=tx["commit"],
            tree=actual,
            latest_full_step=tx["step"],
            eval_steps=tx["eval_steps"],
            deleted_bytes=self.state["deleted_bytes"] + sum(tx["candidates"][oid] for oid in candidates),
            deleted_objects=self.state["deleted_objects"] + len(candidates),
            rotations=self.state["rotations"] + bool(candidates),
            pending=None,
        )
        if tx["marker"]:
            self.state["published_identities"][str(tx["step"])] = tx["marker"]["identity"]
        self.save()
        if tx["staged"]:
            with exclusive_lock(self.root / ".stage.lock", blocking=True):
                self.remove_stage(Path(tx["staged"]), tx["step"])

    def remove_stage(self, path: Path, step: int):
        self.validate_stage_path(path, step)
        if not path.exists():
            return
        marker = read_json(path / "ready.json")
        expected = {
            "owner_id": self.owner["owner_id"],
            "step": step,
            "identity": self.state["published_identities"].get(str(step)),
        }
        if marker != expected:
            raise Blocked("Refusing to remove a queue entry not recorded as published by this owner")
        shutil.rmtree(path)
        fsync_dir(path.parent)

    def tick(self):
        self.finish()
        step = self.state["latest_full_step"]
        queued = []
        with exclusive_lock(self.root / ".stage.lock", blocking=True):
            for path in (self.root / "queue").glob("checkpoint-*"):
                match = CHECKPOINT.fullmatch(path.name)
                if not match:
                    raise Blocked("Unexpected checkpoint entry in owned queue")
                queued_step = int(match[1])
                if queued_step <= step:
                    self.remove_stage(path, queued_step)
                else:
                    queued.append((queued_step, path))
        if queued:
            next_step, path = min(queued)
            self.prepare(path, next_step)
            self.finish()
        return self.status()

    def status(self, error: str = "") -> dict:
        inventory = self.inventory()
        tx = self.state["pending"]
        tree = tx["tree"] if tx and tx["phase"] in {"committed", "gc"} else self.state["tree"]
        if tx and tx["phase"] in {"uploaded", "verified"}:
            tree = tx["upload_tree"]
        live = lfs_oids(tree)
        status = {
            "repo_id": self.repo,
            "owner": OWNER,
            "endpoint": self.api.endpoint,
            "latest_full_step": self.state["latest_full_step"],
            "eval_steps": self.state["eval_steps"],
            "pending_step": tx["step"] if tx else None,
            "phase": tx["phase"] if tx else "idle",
            "total_lfs_bytes": sum(i.size for i in inventory.values()),
            "total_lfs_objects": len(inventory),
            "candidates_uncollected": sorted(set(tx["candidates"]) & (inventory.keys() - live)) if tx else [],
            "unreclaimed_lfs_bytes": sum(i.size for oid, i in inventory.items() if oid not in live),
            "inventory_equals_current": not tx and set(inventory) == lfs_oids(self.state["tree"]),
            "quota_accounting": "not independently confirmed; Hub quota accounting may lag LFS inventory",
            "deleted_bytes": self.state["deleted_bytes"],
            "deleted_objects": self.state["deleted_objects"],
            "rotations": self.state["rotations"],
            "queued_steps": sorted(
                int(p.name.split("-")[1])
                for p in (self.root / "queue").glob("checkpoint-*")
                if CHECKPOINT.fullmatch(p.name) and (p / "ready.json").exists()
            ),
            "last_error": error,
            "fatal_error": "",
            "updated_at": time.time(),
        }
        try:
            info = self.api.model_info(self.repo, expand=["usedStorage"])
            used_storage = getattr(info, "used_storage", None)
        except Exception as exc:
            used_storage = None
            status["quota_query_error"] = type(exc).__name__
        status["hub_used_storage_bytes"] = used_storage
        status["quota_accounting_confirmed"] = used_storage == status["total_lfs_bytes"]
        status["quota_unreclaimed_bytes"] = (
            max(0, used_storage - status["total_lfs_bytes"]) if type(used_storage) is int else None
        )
        if status["quota_accounting_confirmed"]:
            status["quota_accounting"] = "Hub usedStorage equals physical LFS inventory"
        status["done"] = (
            not error
            and status["inventory_equals_current"]
            and status["quota_accounting_confirmed"]
            and status["latest_full_step"] == self.owner["max_steps"]
            and status["eval_steps"]
            == list(range(self.owner["eval_every"], self.owner["max_steps"] + 1, self.owner["eval_every"]))
        )
        atomic_json(self.root / "status.json", status)
        return status


def initialize_owner(args) -> dict:
    root, run_dir = args.staging_dir.resolve(), args.run_dir.resolve()
    if root == run_dir or root.is_relative_to(run_dir) or run_dir.is_relative_to(root):
        raise Blocked("Use a separate staging directory outside the trainer output tree")
    if not (
        args.max_steps > 0
        and args.full_every > 0
        and args.eval_every > 0
        and args.eval_every % args.full_every == 0
        and args.max_steps % args.eval_every == 0
        and 0 < args.first_step <= args.full_every
    ):
        raise Blocked("Invalid checkpoint schedule")
    config = read_json(args.run_config)
    if len(canonical(config)) > 1024**2:
        raise Blocked("Run config must be small Git metadata, not checkpoint data")
    desired = {
        "version": 1,
        "repo_id": args.repo_id,
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "staging_dir": str(root),
        "max_steps": args.max_steps,
        "eval_every": args.eval_every,
        "full_every": args.full_every,
        "first_step": args.first_step,
        "max_staging_bytes": getattr(args, "max_staging_bytes", 250 * 1024**3),
        "max_remote_lfs_bytes": getattr(args, "max_remote_lfs_bytes", 600 * 1024**3),
        "wandb_url": args.wandb_url,
        "run_config": config,
        "sole_writer": True,
    }
    wandb_id = getattr(args, "wandb_id", "")
    if wandb_id:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", wandb_id):
            raise Blocked("Invalid W&B run identity")
        desired["wandb_id"] = wandb_id
    if desired["max_staging_bytes"] <= 0 or desired["max_remote_lfs_bytes"] <= 0:
        raise Blocked("Storage budgets must be positive")
    path = root / "owner.json"
    if path.exists():
        owner = read_json(path)
        if {k: v for k, v in owner.items() if k not in {"owner_id", "created_ns"}} != desired:
            raise Blocked("Refusing changed repository destination or run ownership/configuration")
        return owner
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise Blocked("Missing owner.json in nonempty staging directory; restore state")
    desired.update(owner_id=uuid.uuid4().hex, created_ns=time.time_ns())
    atomic_json(path, desired)
    return desired


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--staging-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--wandb-url", default="")
    parser.add_argument("--wandb-id", default="", help="Immutable W&B resume id; checked against run_dir/wandb_id.txt")
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--full-every", type=int, default=2_500)
    parser.add_argument("--first-step", type=int, default=25)
    parser.add_argument("--max-staging-bytes", type=int, default=250 * 1024**3)
    parser.add_argument("--max-remote-lfs-bytes", type=int, default=600 * 1024**3)
    parser.add_argument("--poll-seconds", type=float, default=30)
    parser.add_argument("--storage-proof", type=Path)
    parser.add_argument(
        "--sole-writer",
        action="store_true",
        required=True,
        help="Acknowledge exclusive lifetime ownership; no other Hub writers or refs",
    )
    parser.add_argument("--adopt-empty-repo", action="store_true")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0 or not os.environ.get("HF_TOKEN"):
        parser.error("positive --poll-seconds and HF_TOKEN are required")
    coordinator = None
    try:
        with exclusive_lock(repo_lock(args.repo_id)):
            owner = initialize_owner(args)
            coordinator = Coordinator(HfApi(token=os.environ["HF_TOKEN"]), args.staging_dir, owner)
            coordinator.preflight(args.storage_proof)
            coordinator.initialize(adopt_empty=args.adopt_empty_repo)
            if args.init_only:
                print(json.dumps(coordinator.status()), flush=True)
                return 0
            while True:
                try:
                    status = coordinator.tick()
                    print(json.dumps(status), flush=True)
                    if args.once or status["done"]:
                        return 0
                except Blocked:
                    raise
                except HfHubHTTPError as exc:
                    if exc.response is not None and exc.response.status_code in (401, 403, 404):
                        raise Blocked("Hub storage/write permission or API unavailable; launch blocked") from exc
                    if args.once:
                        raise
                    print(json.dumps(coordinator.status(type(exc).__name__)), flush=True)
                except (OSError, RuntimeError) as exc:
                    if args.once:
                        raise
                    print(json.dumps(coordinator.status(type(exc).__name__)), flush=True)
                time.sleep(args.poll_seconds)
    except Exception as exc:
        # Never include HTTP request headers or credentials in status/logs.
        message = str(exc) if isinstance(exc, Blocked) else type(exc).__name__
        if coordinator and coordinator.state:
            atomic_json(args.staging_dir / "error.json", {"error": message, "at": time.time()})
            status_path = args.staging_dir / "status.json"
            try:
                status = read_json(status_path) if status_path.exists() else {}
            except Blocked:
                status = {}
            status.update(fatal_error=message, last_error=message, done=False, updated_at=time.time())
            atomic_json(status_path, status)
        print(json.dumps({"blocked": True, "fatal_error": message, "error": message}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
