"""Offline ownership, Orbax staging, transactional publication and physical-GC tests."""

import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

from huggingface_hub import CommitOperationDelete
from huggingface_hub.hf_api import RepoFile
import pytest

SPEC = importlib.util.spec_from_file_location(
    "single_writer", Path(__file__).with_name("hf_single_writer_checkpoint_uploader.py")
)
u = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(u)


def repofile(path, data):
    digest = u.digest_bytes(data)
    lfs = (
        {"oid": digest["sha256"], "size": len(data), "pointerSize": 1}
        if path.startswith(("eval/", "resume/"))
        else None
    )
    return RepoFile(path=path, size=len(data), oid=digest["blob_id"], lfs=lfs)


class MemoryHub:
    endpoint = "https://huggingface.co"

    def __init__(self):
        self.head = "initial"
        self.tree = {".gitattributes": repofile(".gitattributes", b"attributes")}
        self.revisions = {self.head: dict(self.tree)}
        self.commits = [SimpleNamespace(commit_id=self.head, title="initial")]
        self.objects = {}
        self.deleted = []
        self.events = []
        self.refs_extra = None
        self.lose_commit_response = False
        self.lose_retire_response = False
        self.fail_delete = False
        self.lose_delete_response = False
        self.leave_objects = False
        self.on_inventory = None
        self.before_commit = None
        self.after_commit = None
        self.role = "write"
        self.name = u.OWNER
        self.private = True

    def whoami(self):
        return {"name": self.name, "auth": {"accessToken": {"role": self.role}}}

    def repo_info(self, repo_id, **kwargs):
        return SimpleNamespace(sha=self.head, private=self.private, id=repo_id)

    def model_info(self, repo_id, **kwargs):
        return SimpleNamespace(used_storage=sum(i.size for i in self.objects.values()))

    def list_repo_refs(self, repo_id, **kwargs):
        refs = SimpleNamespace(
            branches=[SimpleNamespace(ref="refs/heads/main", target_commit=self.head)],
            tags=[],
            converts=[],
            pull_requests=[],
        )
        if self.refs_extra:
            getattr(refs, self.refs_extra).append(SimpleNamespace(ref="unexpected"))
        assert kwargs["include_pull_requests"] is True
        return refs

    def list_repo_commits(self, repo_id, **kwargs):
        return list(self.commits)

    def list_repo_tree(self, repo_id, revision, **kwargs):
        self.events.append(("tree", revision, set(self.revisions[revision])))
        return list(self.revisions[revision].values())

    def list_lfs_files(self, repo_id, **kwargs):
        if self.on_inventory:
            hook, self.on_inventory = self.on_inventory, None
            hook()
        return list(self.objects.values())

    def commit(self, title):
        self.head = f"commit-{len(self.commits)}"
        self.revisions[self.head] = dict(self.tree)
        self.commits.insert(0, SimpleNamespace(commit_id=self.head, title=title))

    def create_commit(self, repo_id, operations, parent_commit, commit_message, **kwargs):
        if self.before_commit:
            hook, self.before_commit = self.before_commit, None
            hook()
        if parent_commit != self.head:
            raise RuntimeError("parent conflict")
        operations = list(operations)
        self.events.append(("commit", commit_message, set(self.tree)))
        for op in operations:
            if isinstance(op, CommitOperationDelete):
                del self.tree[op.path_in_repo]
            else:
                data = op.path_or_fileobj
                if not isinstance(data, bytes):
                    data = Path(data).read_bytes()
                item = repofile(op.path_in_repo, data)
                self.tree[item.path] = item
                if item.lfs:
                    oid = item.lfs.sha256
                    self.objects[oid] = SimpleNamespace(file_oid=oid, size=len(data), filename=item.path)
        self.commit(commit_message)
        if self.after_commit:
            hook, self.after_commit = self.after_commit, None
            hook()
        if self.lose_commit_response:
            self.lose_commit_response = False
            raise RuntimeError("lost commit response")
        if self.lose_retire_response and commit_message.endswith(" retire"):
            self.lose_retire_response = False
            raise RuntimeError("lost retire response")
        return SimpleNamespace(oid=self.head)

    def permanently_delete_lfs_files(self, repo_id, files, rewrite_history, **kwargs):
        assert rewrite_history is False
        if self.fail_delete:
            self.fail_delete = False
            raise RuntimeError("delete failure")
        live = {f.lfs.sha256 for f in self.tree.values() if f.lfs}
        for item in files:
            assert item.file_oid not in live, "attempted to delete live eval/full object"
            self.deleted.append(item.file_oid)
            if not self.leave_objects:
                del self.objects[item.file_oid]
        if self.lose_delete_response:
            self.lose_delete_response = False
            raise RuntimeError("lost delete response")


def owner_args(tmp_path, **overrides):
    config = tmp_path / "run-config.json"
    u.atomic_json(config, {"test": True, "task": "turning_on_radio", "max_to_keep": 3})
    fields = {
        "staging_dir": tmp_path / "staging",
        "run_dir": tmp_path / "run",
        "run_config": config,
        "repo_id": u.SCRATCH_REPO,
        "run_id": "test-run",
        "wandb_url": "",
        "max_steps": 300_000,
        "full_every": 2500,
        "eval_every": 10000,
        "first_step": 25,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def checkpoint(run_dir, step, *, weight=None):
    path = run_dir / str(step)
    for name in ("params", "train_state"):
        (path / name / "d").mkdir(parents=True)
        (path / name / "manifest.ocdbt").write_bytes(f"manifest {name} {step}".encode())
        (path / name / "d" / "data").write_bytes(
            weight if weight is not None and name == "params" else f"{name} {step}".encode()
        )
        # Orbax may serialize strings; validation must never interpret train_state.
        (path / name / "_METADATA").write_text('"serialized-orbax-state"')
    u.atomic_json(path / "assets/task/norm_stats.json", {"mean": [0], "std": [1]})
    u.atomic_json(path / "assets/task/b1k_metadata.json", {"step": step, "test": True})
    u.atomic_json(path / "assets/task/prompt_source.json", {"prompt_source": "task"})
    u.atomic_json(
        path / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": time.time_ns(), "item_handlers": "serialized-string"}
    )
    return path


@pytest.fixture
def setup(tmp_path):
    args = owner_args(tmp_path)
    owner = u.initialize_owner(args)
    hub = MemoryHub()
    coordinator = u.Coordinator(hub, args.staging_dir, owner)
    coordinator.preflight()
    coordinator.initialize(adopt_empty=True)
    return args, hub, coordinator


def publish(setup, step, **kwargs):
    args, _, coordinator = setup
    src = checkpoint(args.run_dir, step, **kwargs)
    staged = u.stage_local_checkpoint(src, args.staging_dir, step)
    return staged, coordinator.tick()


def test_eval_retained_and_entire_stale_full_physically_reclaimed(setup):
    args, hub, coordinator = setup
    publish(setup, 25)
    stale = {p: f.lfs.sha256 for p, f in hub.tree.items() if p.startswith("resume/")}
    publish(setup, 10000)
    eval_oids = {f.lfs.sha256 for p, f in hub.tree.items() if p.startswith("eval/")}
    assert not any(p.startswith("eval/checkpoint-10000/train_state/") for p in hub.tree)
    assert "eval/checkpoint-10000/assets/task/b1k_metadata.json" in hub.tree
    publish(setup, 12500)
    assert eval_oids <= hub.objects.keys()
    assert not eval_oids.intersection(hub.deleted)
    assert {p.split("/")[1] for p in hub.tree if p.startswith("resume/")} == {"checkpoint-12500"}
    assert stale["resume/checkpoint-25/assets/task/b1k_metadata.json"] in hub.deleted
    assert stale["resume/checkpoint-25/params/manifest.ocdbt"] in hub.deleted
    assert stale["resume/checkpoint-25/_CHECKPOINT_METADATA"] in hub.deleted
    assert set(hub.objects) == u.lfs_oids(coordinator.state["tree"])
    assert not list((args.staging_dir / "queue").glob("checkpoint-*"))


def test_previous_full_present_until_replacement_verified(setup):
    args, hub, coordinator = setup
    publish(setup, 25)
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 2500), args.staging_dir, 2500)
    coordinator.prepare(staged, 2500)
    hub.events.clear()
    coordinator.finish()
    upload = next(i for i, event in enumerate(hub.events) if event[0] == "commit" and event[1].endswith(" upload"))
    retire = next(i for i, event in enumerate(hub.events) if event[0] == "commit" and event[1].endswith(" retire"))
    between = [event for event in hub.events[upload + 1 : retire] if event[0] == "tree"]
    assert between
    for event in between:
        assert "resume/checkpoint-25/params/d/data" in event[2]
        assert "resume/checkpoint-2500/params/d/data" in event[2]


def test_private_copy_survives_source_prune_and_mutation(setup):
    args, _, coordinator = setup
    src = checkpoint(args.run_dir, 25)
    staged = u.stage_local_checkpoint(src, args.staging_dir, 25)
    assert (src / "params/d/data").stat().st_ino != (staged / "checkpoint/params/d/data").stat().st_ino
    (src / "params/d/data").write_bytes(b"changed")
    shutil.rmtree(src)
    assert (staged / "checkpoint/params/d/data").read_bytes() == b"params 25"
    coordinator.tick()
    assert coordinator.state["latest_full_step"] == 25


@pytest.mark.parametrize(
    "missing",
    [*sorted(u.REQUIRED), "params/_METADATA", "train_state/_METADATA", "params/d/data", "train_state/d/data", "assets"],
)
def test_incomplete_checkpoint_never_ready(setup, missing):
    args, _, _ = setup
    src = checkpoint(args.run_dir, 25)
    if (src / missing).is_dir():
        shutil.rmtree(src / missing)
    else:
        (src / missing).unlink()
    with pytest.raises(u.Blocked):
        u.stage_local_checkpoint(src, args.staging_dir, 25)
    assert not list((args.staging_dir / "queue").glob("checkpoint-*"))


@pytest.mark.parametrize("timestamp", [None, True, 0, -1, "123", 1])
def test_incomplete_or_other_generation_metadata_blocked(setup, timestamp):
    args, _, _ = setup
    src = checkpoint(args.run_dir, 25)
    u.atomic_json(src / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": timestamp})
    with pytest.raises(u.Blocked, match="not committed or predates"):
        u.stage_local_checkpoint(src, args.staging_dir, 25)


def test_train_state_strings_are_not_read_as_json(setup):
    args, _, _ = setup
    src = checkpoint(args.run_dir, 25)
    (src / "train_state/_METADATA").write_text("not json serialized state")
    u.stage_local_checkpoint(src, args.staging_dir, 25)


def test_explicit_metadata_step_mismatch_is_rejected(setup):
    args, _, _ = setup
    src = checkpoint(args.run_dir, 25)
    u.atomic_json(src / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": time.time_ns(), "step": 24})
    with pytest.raises(u.Blocked, match="step disagrees"):
        u.stage_local_checkpoint(src, args.staging_dir, 25)


@pytest.mark.parametrize("entry", ["unowned.bin", "params/symlink"])
def test_unknown_or_symlink_entries_rejected(setup, entry):
    args, _, _ = setup
    src = checkpoint(args.run_dir, 25)
    if entry.endswith("symlink"):
        (src / entry).symlink_to(src / "params/d/data")
    else:
        (src / entry).write_bytes(b"unrelated")
    with pytest.raises(u.Blocked):
        u.stage_local_checkpoint(src, args.staging_dir, 25)


def test_mutation_during_copy_discards_partial(setup, monkeypatch):
    args, _, _ = setup
    src = checkpoint(args.run_dir, 25)
    copy2 = shutil.copy2

    def changing(source, target):
        result = copy2(source, target)
        if source.name == "manifest.ocdbt":
            source.write_bytes(b"changed manifest")
        return result

    monkeypatch.setattr(shutil, "copy2", changing)
    with pytest.raises(u.Blocked, match="changed while copying"):
        u.stage_local_checkpoint(src, args.staging_dir, 25)
    assert not list((args.staging_dir / "queue").iterdir())


def test_destination_and_corrupt_journal_fail_closed(setup):
    args, hub, coordinator = setup
    args.repo_id = "kmy17518/another-run"
    with pytest.raises(u.Blocked, match="changed repository"):
        u.initialize_owner(args)
    data = u.read_json(coordinator.path)
    data["state"]["head"] = "bad"
    u.atomic_json(coordinator.path, data)
    with pytest.raises(u.Blocked, match="Corrupt journal"):
        u.Coordinator(hub, args.staging_dir, coordinator.owner)


def test_local_exclusive_lock(tmp_path):
    with (
        u.exclusive_lock(tmp_path / "lock"),
        pytest.raises(u.Blocked, match="Another local writer"),
        u.exclusive_lock(tmp_path / "lock"),
    ):
        pass


@pytest.mark.parametrize("ref", ["branches", "tags", "converts", "pull_requests"])
def test_all_unexpected_refs_block_before_retirement_or_deletion(setup, ref):
    args, hub, coordinator = setup
    publish(setup, 25)
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 2500), args.staging_dir, 2500)
    coordinator.prepare(staged, 2500)
    hub.after_commit = lambda: setattr(hub, "refs_extra", ref)
    with pytest.raises(u.Blocked, match="Unexpected branch"):
        coordinator.finish()
    assert "resume/checkpoint-25/params/d/data" in hub.tree
    assert not hub.deleted


def test_new_writer_after_inventory_blocks_gc(setup):
    args, hub, coordinator = setup
    publish(setup, 25)
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 2500), args.staging_dir, 2500)
    coordinator.prepare(staged, 2500)

    def new_writer():
        hub.commit("external writer")

    def after_upload():
        hub.after_commit = lambda: setattr(hub, "on_inventory", new_writer)

    hub.after_commit = after_upload
    with pytest.raises(u.Blocked, match="Unexpected HEAD"):
        coordinator.finish()
    assert not hub.deleted


@pytest.mark.parametrize(
    "failure", ["fail_delete", "lose_delete_response", "lose_commit_response", "lose_retire_response"]
)
def test_journal_recovers_responses_and_delete_failure_after_restart(setup, failure):
    args, hub, coordinator = setup
    publish(setup, 25)
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 2500), args.staging_dir, 2500)
    coordinator.prepare(staged, 2500)
    setattr(hub, failure, True)
    with pytest.raises(RuntimeError):
        coordinator.finish()
    recovered = u.Coordinator(hub, args.staging_dir, coordinator.owner)
    recovered.finish()
    assert recovered.state["latest_full_step"] == 2500
    assert not recovered.state["pending"]
    assert not staged.exists()
    assert recovered.status()["inventory_equals_current"]


def test_failed_physical_accounting_pauses_new_rotations(setup):
    args, hub, coordinator = setup
    publish(setup, 25)
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 2500), args.staging_dir, 2500)
    coordinator.prepare(staged, 2500)
    hub.leave_objects = True
    with pytest.raises(RuntimeError, match="not confirmed"):
        coordinator.finish()
    assert coordinator.state["latest_full_step"] == 25
    assert coordinator.status()["candidates_uncollected"]
    assert coordinator.status()["unreclaimed_lfs_bytes"] > 0
    with pytest.raises(u.Blocked, match="pending transaction"):
        coordinator.prepare(staged, 2500)
    hub.leave_objects = False
    coordinator.finish()
    assert coordinator.status()["inventory_equals_current"]


def test_external_commit_not_adopted_as_lost_response(setup):
    args, hub, coordinator = setup
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    coordinator.prepare(staged, 25)
    hub.commit("unrelated")
    with pytest.raises(u.Blocked, match="Unexpected HEAD during commit recovery"):
        coordinator.finish()
    assert not hub.deleted


def test_parent_compare_and_swap_blocks_race(setup):
    args, hub, coordinator = setup
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    coordinator.prepare(staged, 25)
    hub.before_commit = lambda: hub.commit("external")
    with pytest.raises(RuntimeError, match="parent conflict"):
        coordinator.finish()
    assert not hub.deleted


def test_unknown_unreferenced_objects_are_never_deleted(setup):
    args, hub, coordinator = setup
    oid = hashlib.sha256(b"other").hexdigest()
    hub.objects[oid] = SimpleNamespace(file_oid=oid, size=5)
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    with pytest.raises(u.Blocked, match="Unjournaled"):
        coordinator.prepare(staged, 25)
    assert not hub.deleted


def test_skipped_eval_is_fatal_not_silent(setup):
    args, _, coordinator = setup
    u.stage_local_checkpoint(checkpoint(args.run_dir, 12500), args.staging_dir, 12500)
    with pytest.raises(u.Blocked, match="Required eval checkpoint was skipped"):
        coordinator.tick()


@pytest.mark.parametrize(
    ("field", "value"), [("role", "read"), ("name", "other"), ("endpoint", "https://other.invalid")]
)
def test_api_and_permission_preflight(setup, field, value):
    _, hub, coordinator = setup
    setattr(hub, field, value)
    with pytest.raises(u.Blocked):
        coordinator.preflight()


def test_production_preflight_requires_openpi_storage_proof(setup):
    _, _, coordinator = setup
    coordinator.repo = "kmy17518/pi05-turning-on-radio-1gpu-300k-20260916"
    with pytest.raises(u.Blocked, match="storage proof"):
        coordinator.preflight()


def test_cleanup_holds_staging_lock(setup, monkeypatch):
    args, _, coordinator = setup
    u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    rmtree = shutil.rmtree
    deleted = []

    def locked_cleanup(path, **kwargs):
        with pytest.raises(u.Blocked, match="Another local writer"), u.exclusive_lock(args.staging_dir / ".stage.lock"):
            pass
        deleted.append(path)
        rmtree(path, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", locked_cleanup)
    coordinator.tick()
    assert deleted


def test_capacity_limit_stops_before_copying_unpublished_data(setup):
    args, _, coordinator = setup
    owner = dict(coordinator.owner, max_staging_bytes=1)
    u.atomic_json(args.staging_dir / "owner.json", owner)
    coordinator.state["owner"] = owner
    coordinator.save()
    with pytest.raises(u.Blocked, match="capacity reached"):
        u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    assert not list((args.staging_dir / "queue").iterdir())


def test_peak_remote_budget_blocks_before_upload(setup):
    args, hub, coordinator = setup
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    coordinator.owner["max_remote_lfs_bytes"] = 1
    head = hub.head
    with pytest.raises(u.Blocked, match="budget cannot hold"):
        coordinator.prepare(staged, 25)
    assert hub.head == head


def test_done_requires_all_evals_final_full_and_gc(setup):
    _, _, coordinator = setup
    assert not coordinator.status()["done"]
    coordinator.state["latest_full_step"] = 300000
    assert not coordinator.status()["done"]
    coordinator.state["eval_steps"] = list(range(10000, 300001, 10000))
    assert coordinator.status()["done"]
    assert not coordinator.status("failure")["done"]


def test_invalid_schedule_or_zero_based_path_cannot_publish(setup):
    args, _, _ = setup
    with pytest.raises(u.Blocked, match="Unscheduled"):
        u.stage_local_checkpoint(checkpoint(args.run_dir, 26), args.staging_dir, 26)
    with pytest.raises(u.Blocked, match="different step"):
        u.stage_local_checkpoint(checkpoint(args.run_dir, 9999), args.staging_dir, 10000)


def test_journal_write_failure_fatal_before_commit(setup, monkeypatch):
    args, hub, coordinator = setup
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    head = hub.head

    def full_disk(*args):
        raise OSError("disk full")

    monkeypatch.setattr(u, "atomic_json", full_disk)
    with pytest.raises(u.Blocked, match="Journal durability failed"):
        coordinator.prepare(staged, 25)
    assert hub.head == head
    assert not hub.deleted


def test_retry_same_save_noop_but_step_reuse_blocked(setup):
    args, _, _ = setup
    publish(setup, 25)
    src = args.run_dir / "25"
    u.stage_local_checkpoint(src, args.staging_dir, 25)
    u.atomic_json(src / "_CHECKPOINT_METADATA", {"commit_timestamp_nsecs": time.time_ns()})
    with pytest.raises(u.Blocked, match="reused or replaced"):
        u.stage_local_checkpoint(src, args.staging_dir, 25)


def test_queued_owner_or_path_tampering_blocks_cleanup(setup):
    args, _, coordinator = setup
    publish(setup, 25)
    dst = args.staging_dir / "queue/checkpoint-25"
    dst.mkdir()
    u.atomic_json(dst / "ready.json", {"owner_id": "other", "step": 25})
    with pytest.raises(u.Blocked, match="not recorded as published"):
        coordinator.tick()
    assert dst.exists()


def test_missing_replacement_object_preserves_old_full(setup):
    args, hub, coordinator = setup
    publish(setup, 25)
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 2500), args.staging_dir, 2500)
    coordinator.prepare(staged, 2500)

    def drop_new_object():
        oid = hub.tree["resume/checkpoint-2500/params/d/data"].lfs.sha256
        del hub.objects[oid]

    hub.after_commit = drop_new_object
    with pytest.raises(u.Blocked, match="Replacement inventory"):
        coordinator.finish()
    assert "resume/checkpoint-25/params/d/data" in hub.tree
    assert not hub.deleted


def test_checkpoint_git_blobs_cannot_silently_accumulate(setup):
    args, hub, coordinator = setup
    staged = u.stage_local_checkpoint(checkpoint(args.run_dir, 25), args.staging_dir, 25)
    coordinator.prepare(staged, 25)

    def replace_with_git_blob():
        path = "resume/checkpoint-25/params/manifest.ocdbt"
        item = copy.copy(hub.tree[path])
        item.lfs = None
        hub.tree[path] = item
        hub.revisions[hub.head] = dict(hub.tree)

    hub.after_commit = replace_with_git_blob
    with pytest.raises(u.Blocked, match="not physically collectable"):
        coordinator.finish()
    assert not hub.deleted


def test_quota_endpoint_lag_reports_unreclaimed_without_done(setup):
    _, hub, coordinator = setup
    coordinator.state["latest_full_step"] = 300000
    coordinator.state["eval_steps"] = list(range(10000, 300001, 10000))
    hub.model_info = lambda *args, **kwargs: SimpleNamespace(used_storage=99999)
    status = coordinator.status()
    assert status["inventory_equals_current"]
    assert not status["quota_accounting_confirmed"]
    assert status["quota_unreclaimed_bytes"] == 99999
    assert not status["done"]


def test_wandb_resume_identity_published_and_local_mismatch_blocked(tmp_path):
    args = owner_args(tmp_path, wandb_id="piradio16")
    owner = u.initialize_owner(args)
    hub = MemoryHub()
    coordinator = u.Coordinator(hub, args.staging_dir, owner)
    coordinator.initialize(adopt_empty=True)
    assert hub.tree["wandb_id.txt"].blob_id == u.digest_bytes(b"piradio16\n")["blob_id"]
    src = checkpoint(args.run_dir, 25)
    (args.run_dir / "wandb_id.txt").write_text("piradio16")
    u.stage_local_checkpoint(src, args.staging_dir, 25)
    (args.run_dir / "wandb_id.txt").write_text("another-run")
    with pytest.raises(u.Blocked, match="W&B resume identity"):
        u.stage_local_checkpoint(src, args.staging_dir, 25)


def test_existing_populated_repo_cannot_be_adopted(tmp_path):
    args = owner_args(tmp_path)
    owner = u.initialize_owner(args)
    hub = MemoryHub()
    hub.tree["other.txt"] = repofile("other.txt", b"other run")
    hub.revisions[hub.head] = dict(hub.tree)
    coordinator = u.Coordinator(hub, args.staging_dir, owner)
    with pytest.raises(u.Blocked, match="empty new repo"):
        coordinator.initialize(adopt_empty=True)
    assert "other.txt" in hub.tree
    assert not hub.deleted
