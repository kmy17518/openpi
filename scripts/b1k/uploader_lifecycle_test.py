"""Offline regressions for checkpoint publication, generation reuse and launcher leases."""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts/b1k" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_uploader = load("hf_checkpoint_uploader")
full_uploader = load("hf_latest_full_checkpoint_uploader")
lifecycle = eval_uploader.lifecycle


class MockHub:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.refs = {"release": {"other/model.bin": b"branch"}, "v1": {"other/model.bin": b"tag"}}
        self.events = []
        self.fail_path = None
        self.fail_after = False

    def create_repo(self, *args, **kwargs):
        pass

    def whoami(self):
        return {"name": "offline"}

    def upload_folder(self, *, folder_path, path_in_repo, delete_patterns=None, **kwargs):
        self.events.append(("folder", path_in_repo))
        assert delete_patterns == "*"
        prefix = path_in_repo + "/"
        self.files = {key: value for key, value in self.files.items() if not key.startswith(prefix)}
        for path in pathlib.Path(folder_path).rglob("*"):
            if path.is_file():
                self.files[prefix + str(path.relative_to(folder_path))] = path.read_bytes()
        return SimpleNamespace(commit_url="mock://commit")

    def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
        self.events.append(("metadata", path_in_repo))
        fail = path_in_repo == self.fail_path
        if fail and not self.fail_after:
            self.fail_path = None
            raise RuntimeError("Injected metadata failure")
        data = pathlib.Path(path_or_fileobj).read_bytes() if isinstance(path_or_fileobj, str) else path_or_fileobj
        self.files[path_in_repo] = data
        if fail:
            self.fail_path = None
            raise RuntimeError("Injected post-commit failure")

    def file_exists(self, repo, path, **kwargs):
        return path in self.files

    def hf_hub_download(self, repo, path, **kwargs):
        local = self.root / "download.json"
        local.write_bytes(self.files[path])
        return str(local)

    def list_repo_tree(self, repo, *, path_in_repo="", recursive=False, **kwargs):
        prefix = path_in_repo.rstrip("/") + "/" if path_in_repo else ""
        folders = set()
        out = []
        for path, data in self.files.items():
            if not path.startswith(prefix):
                continue
            relative = path[len(prefix):]
            if not recursive and "/" in relative:
                folders.add(prefix + relative.split("/")[0])
            else:
                out.append(full_uploader.RepoFile(path=path, size=len(data), oid="mock"))
        out.extend(full_uploader.RepoFolder(path=path, oid="mock") for path in folders)
        return out

    def delete_folder(self, *, path_in_repo, **kwargs):
        self.events.append(("delete", path_in_repo))
        assert path_in_repo.startswith("test/resume/checkpoint-")
        self.files = {key: value for key, value in self.files.items() if not key.startswith(path_in_repo + "/")}

    def delete_file(self, *, path_in_repo, **kwargs):
        self.events.append(("delete_file", path_in_repo))
        del self.files[path_in_repo]

    def list_lfs_files(self, *args, **kwargs):
        pytest.fail("Automated mirror must not enumerate repository LFS objects")

    def permanently_delete_lfs_files(self, *args, **kwargs):
        pytest.fail("Automated mirror must not permanently delete objects or rewrite history")


@pytest.fixture
def environment(tmp_path, monkeypatch):
    monkeypatch.setattr(lifecycle, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(full_uploader.common.lifecycle, "LOCK_DIR", tmp_path / "locks")
    monkeypatch.setattr(eval_uploader, "git_commit", lambda _: "offline")
    monkeypatch.setattr(full_uploader.common, "git_commit", lambda _: "offline")
    monkeypatch.setattr(eval_uploader, "setup_logging", lambda _: None)
    monkeypatch.setattr(full_uploader, "setup_logging", lambda _: None)
    monkeypatch.setattr(eval_uploader, "trainer_alive", lambda _: False)
    monkeypatch.setattr(full_uploader.common, "trainer_alive", lambda _: False)
    monkeypatch.setenv("HF_TOKEN", "offline-api-mock")
    return {
        "EXP_NAME": "test", "CONFIG_NAME": "pi05_b1k", "HF_REPO": "test/repo", "OPENPI_DIR": str(tmp_path),
        "STAGING_DIR": str(tmp_path / "staging"), "NUM_TRAIN_STEPS": "11", "UPLOADER_POLL_SECONDS": "0",
        "FULL_UPLOADER_POLL_SECONDS": "0",
    }


def checkpoint(env, *, timestamp=100, content=b"old", step=10):
    spec = eval_uploader.RunSpec(env)
    path = spec.ckpt_dir / str(step)
    for name in ("params", "assets", "train_state"):
        (path / name).mkdir(parents=True, exist_ok=True)
        (path / name / "manifest.ocdbt").write_bytes(content)
    (path / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": timestamp}))
    (spec.ckpt_dir / "wandb_id.txt").write_text("current-wandb")
    return spec


def run_main(module, env, api, monkeypatch, *, cycles=1):
    run_env = pathlib.Path(env["OPENPI_DIR"]) / "run.env"
    run_env.write_text("\n".join(f"{key}={value}" for key, value in env.items()))
    monkeypatch.setattr(module, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(sys, "argv", ["uploader", str(run_env)])
    count = 0

    class EndCyclesError(Exception):
        pass

    def bounded_sleep(_seconds):
        nonlocal count
        count += 1
        if count >= cycles:
            raise EndCyclesError

    monkeypatch.setattr(module.time, "sleep", bounded_sleep)
    try:
        return module.main()
    except EndCyclesError:
        return None


@pytest.mark.parametrize("phase", ["LATEST.json", "wandb_id.txt", "README.md"])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("after_commit", [False, True])
def test_full_metadata_failure_reconciles_before_deletion(environment, tmp_path, monkeypatch, phase, restart, after_commit):
    checkpoint(environment)
    spec = full_uploader.FullSpec(environment)
    api = MockHub(tmp_path)
    api.files["test/resume/checkpoint-5/params/manifest.ocdbt"] = b"previous"
    api.fail_path = f"test/resume/{phase}"
    api.fail_after = after_commit
    run_main(full_uploader, environment, api, monkeypatch, cycles=1 if restart else 2)
    if restart:
        state = json.loads(spec.state_path.read_text()) if spec.state_path.exists() else {}
        assert "remote_step" not in state
        assert not any(event[0] == "delete" for event in api.events)
        assert "test/resume/checkpoint-5/params/manifest.ocdbt" in api.files
        run_main(full_uploader, environment, api, monkeypatch)
    state = json.loads(spec.state_path.read_text())
    assert state["remote_step"] == 10
    assert state["identity"] == full_uploader.common.checkpoint_identity(spec, 10)
    assert sum(event[0] == "folder" for event in api.events) == 1
    assert api.events.count(("metadata", f"test/resume/{phase}")) == 2
    deletion = next(i for i, event in enumerate(api.events) if event[0] == "delete")
    for name in ("LATEST.json", "wandb_id.txt", "README.md"):
        assert ("metadata", f"test/resume/{name}") in api.events[:deletion]
        assert f"test/resume/{name}" in api.files
    assert api.files["test/resume/wandb_id.txt"] == b"current-wandb"
    assert json.loads(api.files["test/resume/LATEST.json"])["identity"] == state["identity"]


@pytest.mark.parametrize("phase", ["test/README.md", "README.md"])
@pytest.mark.parametrize("after_commit", [False, True])
def test_eval_cards_failure_never_marks_success(environment, tmp_path, phase, after_commit):
    spec = checkpoint(environment)
    steps = {"10": eval_uploader.stage(spec, 10)}
    api = MockHub(tmp_path)
    api.fail_path = phase
    api.fail_after = after_commit
    with pytest.raises(RuntimeError, match="failure"):
        eval_uploader.upload(api, spec, 10, steps)
    assert not eval_uploader.is_uploaded(spec, steps["10"])
    eval_uploader.save_state(spec.state_path, {"steps": steps})
    steps = eval_uploader.load_state(spec.state_path)["steps"]
    eval_uploader.upload(api, eval_uploader.RunSpec(environment), 10, steps)
    assert eval_uploader.is_uploaded(spec, steps["10"])
    assert api.events.count(("metadata", phase)) == 2


def test_normal_cleanup_preserves_release_tag_and_unrelated_paths(environment, tmp_path):
    checkpoint(environment)
    spec = full_uploader.FullSpec(environment)
    api = MockHub(tmp_path)
    api.files = {
        "test/resume/checkpoint-5/model": b"old", "test/resume/checkpoint-10/model": b"new",
        "other/resume/checkpoint-5/model": b"other", "test/checkpoint-5/model": b"eval",
        "test/resume/checkpoint-20/model": b"newer",
    }
    retained = copy.deepcopy(api.refs)
    assert full_uploader.purge_others(api, spec, 10) == {"deleted_steps": [5]}
    assert api.refs == retained
    assert set(api.files) == {"test/resume/checkpoint-10/model", "test/resume/checkpoint-20/model",
                              "other/resume/checkpoint-5/model", "test/checkpoint-5/model"}


@pytest.mark.parametrize("module", [eval_uploader, full_uploader])
@pytest.mark.parametrize("explicit_generation", [False, True])
def test_old_state_and_remote_same_step_are_replaced(environment, tmp_path, monkeypatch, module, explicit_generation):
    old_spec = checkpoint(environment)
    api = MockHub(tmp_path)
    run_main(module, environment, api, monkeypatch)
    old_path = old_spec.staging_dir
    old_identity = eval_uploader.checkpoint_identity(old_spec, 10)
    if explicit_generation:
        lifecycle.prepare_generation(old_spec.ckpt_dir, fresh=True)
        current = eval_uploader.RunSpec(environment)
        assert eval_uploader.current_steps(current) == []
        assert current.staging_dir != old_path
        # Old success/staging remain, even if copied into the new namespace by an operator.
        current.staging_dir.mkdir(parents=True)
        for name in ("state.json", "full-state.json"):
            source = old_path / name
            if source.exists():
                (current.staging_dir / name).write_bytes(source.read_bytes())
    checkpoint(environment, timestamp=time.time_ns(), content=b"new")
    new_spec = eval_uploader.RunSpec(environment)
    new_identity = eval_uploader.checkpoint_identity(new_spec, 10)
    assert new_identity != old_identity
    api.events.clear()
    run_main(module, environment, api, monkeypatch)
    prefix = "test/checkpoint-10" if module is eval_uploader else "test/resume/checkpoint-10"
    assert ("folder", prefix) in api.events
    assert api.files[f"{prefix}/params/manifest.ocdbt"] == b"new"
    assert json.loads(api.files[f"{prefix}/training_run.json"])["identity"] == new_identity
    assert old_path.exists()


@pytest.mark.parametrize("module", [eval_uploader, full_uploader])
def test_legacy_unversioned_success_and_same_size_remote_not_trusted(environment, tmp_path, monkeypatch, module):
    spec = checkpoint(environment, content=b"new")
    spec.staging_dir.mkdir(parents=True)
    old_state = {"steps": {"10": {"staged_at": "old", "uploaded_at": "old", "target": spec.target}},
                 "remote_step": 10, "remote_target": "test/repo/test/resume"}
    for root in (spec.staging_root, spec.staging_dir):
        for name in ("state.json", "full-state.json"):
            (root / name).write_text(json.dumps(old_state))
    api = MockHub(tmp_path)
    prefix = "test/checkpoint-10" if module is eval_uploader else "test/resume/checkpoint-10"
    for name in ("params", "assets", "train_state"):
        api.files[f"{prefix}/{name}/manifest.ocdbt"] = b"old"
    api.files[f"{prefix}/_CHECKPOINT_METADATA"] = json.dumps({"commit_timestamp_nsecs": 100}).encode()
    api.files[f"{prefix}/obsolete"] = b"stale-extra"
    run_main(module, environment, api, monkeypatch)
    assert api.files[f"{prefix}/params/manifest.ocdbt"] == b"new"
    assert f"{prefix}/obsolete" not in api.files


def test_staged_eval_survives_checkpoint_pruning(environment, tmp_path, monkeypatch):
    spec = checkpoint(environment)
    record = eval_uploader.stage(spec, 10)
    eval_uploader.save_state(spec.state_path, {"steps": {"10": record}})
    (spec.ckpt_dir / "10").rename(spec.ckpt_dir / "pruned")
    api = MockHub(tmp_path)
    assert run_main(eval_uploader, environment, api, monkeypatch) == 0
    assert api.files["test/checkpoint-10/params/manifest.ocdbt"] == b"old"


def test_generation_changes_reject_inflight_old_staging(environment, tmp_path):
    spec = checkpoint(environment)
    steps = {"10": eval_uploader.stage(spec, 10)}
    lifecycle.prepare_generation(spec.ckpt_dir, fresh=True)
    with pytest.raises(RuntimeError, match="current run"):
        eval_uploader.upload(MockHub(tmp_path), spec, 10, steps)


def test_resume_preserves_generation_and_restores_remote_provenance(environment):
    spec = checkpoint(environment)
    first = lifecycle.prepare_generation(spec.ckpt_dir, fresh=False)
    assert lifecycle.prepare_generation(spec.ckpt_dir, fresh=False) == first
    lifecycle.generation_path(spec.ckpt_dir).unlink()
    (spec.ckpt_dir / "10/training_run.json").write_text(json.dumps({"generation": first["id"]}))
    assert lifecycle.prepare_generation(spec.ckpt_dir, fresh=False)["id"] == first["id"]


def test_full_removes_stale_wandb_without_permanent_deletion(environment, tmp_path):
    checkpoint(environment)
    spec = full_uploader.FullSpec(environment)
    spec.staging_dir.mkdir(parents=True)
    (spec.ckpt_dir / "wandb_id.txt").unlink()
    api = MockHub(tmp_path)
    api.files["test/resume/wandb_id.txt"] = b"old-run"
    full_uploader.sync_step(api, spec, 10, {})
    assert "test/resume/wandb_id.txt" not in api.files


@pytest.mark.parametrize("module", [eval_uploader, full_uploader])
def test_staging_checks_metadata_identity_and_copies_all_assets(environment, tmp_path, module):
    spec = checkpoint(environment)
    sidecar = spec.ckpt_dir / "10/assets/b1k_metadata.json"
    sidecar.write_text('{"representation": "current"}')
    api = MockHub(tmp_path)
    if module is eval_uploader:
        record = eval_uploader.stage(spec, 10)
        path = pathlib.Path(record["path"])
        (path / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 99}')
        assert not eval_uploader.staged_record(spec, 10, record)
        record = eval_uploader.stage(spec, 10)
        eval_uploader.upload(api, spec, 10, {"10": record})
        prefix = "test/checkpoint-10"
    else:
        spec = full_uploader.FullSpec(environment)
        identity = full_uploader.common.checkpoint_identity(spec, 10)
        path = full_uploader.stage_full(spec, 10, identity)
        (path / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 99}')
        full_uploader.sync_step(api, spec, 10, {})
        prefix = "test/resume/checkpoint-10"
    assert api.files[f"{prefix}/assets/b1k_metadata.json"] == sidecar.read_bytes()
    assert json.loads(api.files[f"{prefix}/_CHECKPOINT_METADATA"])["commit_timestamp_nsecs"] == 100


@pytest.mark.parametrize("module", [eval_uploader, full_uploader])
def test_changed_checkpoint_during_upload_is_not_marked_success(environment, tmp_path, module):
    spec = checkpoint(environment)
    api = MockHub(tmp_path)
    original = api.upload_folder

    def upload_and_replace(**kwargs):
        result = original(**kwargs)
        checkpoint(environment, timestamp=200, content=b"new")
        return result

    api.upload_folder = upload_and_replace
    state = {}
    if module is eval_uploader:
        record = eval_uploader.stage(spec, 10)
        with pytest.raises(RuntimeError, match="changed during upload"):
            eval_uploader.upload(api, spec, 10, {"10": record})
    else:
        with pytest.raises(RuntimeError, match="changed during upload"):
            full_uploader.sync_step(api, full_uploader.FullSpec(environment), 10, state)
    assert not state.get("remote_step")
    if module is eval_uploader:
        assert not record.get("uploaded_at")


def test_fresh_generation_cannot_race_old_publication(environment, tmp_path, monkeypatch):
    spec = checkpoint(environment)
    record = eval_uploader.stage(spec, 10)
    api = MockHub(tmp_path)
    original = api.upload_folder

    def interrupted_upload(**kwargs):
        original(**kwargs)
        lifecycle.prepare_generation(spec.ckpt_dir, fresh=True)
        checkpoint(environment, timestamp=time.time_ns(), content=b"new")
        raise RuntimeError("Injected fresh generation")

    # A fresh launch is blocked while an old-generation publication owns the lease.
    api.upload_folder = interrupted_upload
    with pytest.raises(RuntimeError, match="Already locked"):
        eval_uploader.upload(api, spec, 10, {"10": record})
    api.upload_folder = original
    assert not record.get("uploaded_at")
    lifecycle.prepare_generation(spec.ckpt_dir, fresh=True)
    checkpoint(environment, timestamp=time.time_ns(), content=b"new")
    assert run_main(eval_uploader, environment, api, monkeypatch) == 0
    assert api.files["test/checkpoint-10/params/manifest.ocdbt"] == b"new"


def test_delayed_full_target_cannot_roll_back_newer_checkpoint(environment, tmp_path, monkeypatch):
    checkpoint(environment, step=10)
    spec = full_uploader.FullSpec(environment)
    api = MockHub(tmp_path)
    original_lock = full_uploader.common.lifecycle.exclusive_lock
    fired = False
    newer, older = {}, {}

    @contextlib.contextmanager
    def interleaved_lock(key):
        nonlocal fired
        if not fired:
            fired = True
            checkpoint(environment, step=20, timestamp=200, content=b"new")
            full_uploader.sync_step(api, spec, 20, newer)
        with original_lock(key) as descriptor:
            yield descriptor

    monkeypatch.setattr(full_uploader.common.lifecycle, "exclusive_lock", interleaved_lock)
    with pytest.raises(RuntimeError, match="stale"):
        full_uploader.sync_step(api, spec, 10, older)
    assert newer["remote_step"] == 20
    assert not older
    assert json.loads(api.files["test/resume/LATEST.json"])["step"] == 20
    assert api.files["test/resume/checkpoint-20/params/manifest.ocdbt"] == b"new"
    assert ("folder", "test/resume/checkpoint-10") not in api.events
    assert json.loads(spec.state_path.read_text())["remote_step"] == 20


@pytest.mark.parametrize("pointer", [False, True])
def test_full_rechecks_remote_provenance_for_other_local_root(environment, tmp_path, pointer):
    checkpoint(environment, step=20, timestamp=200)
    spec = full_uploader.FullSpec(environment)
    api = MockHub(tmp_path)
    full_uploader.sync_step(api, spec, 20, {})
    if not pointer:
        del api.files["test/resume/LATEST.json"]
    other = {**environment, "OPENPI_DIR": str(tmp_path / "other"), "STAGING_DIR": str(tmp_path / "other-stage")}
    checkpoint(other, step=10)
    api.events.clear()
    with pytest.raises(RuntimeError, match="already remote"):
        full_uploader.sync_step(api, full_uploader.FullSpec(other), 10, {})
    assert not api.events
    assert "test/resume/checkpoint-20/params/manifest.ocdbt" in api.files


def test_full_destination_lease_precedes_shared_staging(environment, tmp_path, monkeypatch):
    checkpoint(environment)
    other = {**environment, "OPENPI_DIR": str(tmp_path / "other")}
    checkpoint(other)
    spec = full_uploader.FullSpec(environment)
    api = MockHub(tmp_path)
    original = full_uploader.stage_full

    def stage_with_contender(current, step, identity):
        with pytest.raises(RuntimeError, match="publish-full"):
            full_uploader.sync_step(api, full_uploader.FullSpec(other), 10, {})
        return original(current, step, identity)

    monkeypatch.setattr(full_uploader, "stage_full", stage_with_contender)
    full_uploader.sync_step(api, spec, 10, {})
    assert json.loads(api.files["test/resume/LATEST.json"])["step"] == 10


@pytest.mark.parametrize("phase", ["stage", "upload"])
def test_full_rechecks_latest_before_upload_and_metadata(environment, tmp_path, monkeypatch, phase):
    checkpoint(environment)
    spec = full_uploader.FullSpec(environment)
    api = MockHub(tmp_path)
    original = full_uploader.stage_full if phase == "stage" else api.upload_folder

    def advance(*args, **kwargs):
        result = original(*args, **kwargs)
        checkpoint(environment, step=20, timestamp=200)
        return result

    monkeypatch.setattr(full_uploader if phase == "stage" else api,
                        "stage_full" if phase == "stage" else "upload_folder", advance)
    with pytest.raises(RuntimeError, match="stale"):
        full_uploader.sync_step(api, spec, 10, {})
    assert not any(event[0] in ("metadata", "delete") for event in api.events)
    if phase == "stage":
        assert not api.events


@pytest.mark.parametrize("legacy_provenance", [False, True])
def test_fresh_lower_full_step_allowed_but_previous_generation_cannot_return(environment, tmp_path, legacy_provenance):
    initial = checkpoint(environment, step=20, timestamp=200)
    lifecycle.prepare_generation(initial.ckpt_dir, fresh=False)
    old = full_uploader.FullSpec(environment)
    api = MockHub(tmp_path)
    full_uploader.sync_step(api, old, 20, {})
    if legacy_provenance:
        for path in ("test/resume/LATEST.json", "test/resume/checkpoint-20/training_run.json"):
            provenance = json.loads(api.files[path])
            provenance.pop("generation_started_ns")
            api.files[path] = json.dumps(provenance).encode()
    other = {**environment, "OPENPI_DIR": str(tmp_path / "other"), "STAGING_DIR": str(tmp_path / "other-stage")}
    root = eval_uploader.RunSpec(other).ckpt_dir
    lifecycle.prepare_generation(root, fresh=True)
    checkpoint(other, timestamp=time.time_ns(), content=b"fresh")
    fresh = full_uploader.FullSpec(other)
    full_uploader.sync_step(api, fresh, 10, {})
    latest = json.loads(api.files["test/resume/LATEST.json"])
    assert latest["step"] == 10
    assert latest["generation"] == fresh.generation["id"]
    assert "test/resume/checkpoint-20/params/manifest.ocdbt" in api.files
    downloaded_env = {**environment, "OPENPI_DIR": str(tmp_path / "downloaded"), "STAGING_DIR": str(tmp_path / "downloaded-stage")}
    downloaded_root = eval_uploader.RunSpec(downloaded_env).ckpt_dir
    prefix = "test/resume/checkpoint-10/"
    for path, contents in api.files.items():
        if path.startswith(prefix):
            destination = downloaded_root / "10" / path[len(prefix):]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
    assert lifecycle.prepare_generation(downloaded_root, fresh=False) == fresh.generation
    downloaded = full_uploader.FullSpec(downloaded_env)
    full_uploader.sync_step(api, downloaded, 10, {})
    assert json.loads(api.files["test/resume/LATEST.json"])["generation"] == fresh.generation["id"]
    api.events.clear()
    with pytest.raises(RuntimeError, match="generation"):
        full_uploader.sync_step(api, old, 20, {})
    assert not api.events


@pytest.mark.parametrize("value", [None, {}, [], True, False, 0, -1, 1.5, "100", "bad"])
def test_bad_checkpoint_timestamp_is_skipped_then_repaired(environment, value):
    spec = checkpoint(environment, timestamp=value)
    marker = spec.ckpt_dir / "10/_CHECKPOINT_METADATA"
    with pytest.raises(ValueError, match="Checkpoint|generation"):
        lifecycle.metadata_identity(marker.read_bytes(), spec.generation)
    assert eval_uploader.completed_steps(spec.ckpt_dir) == []
    assert eval_uploader.current_steps(spec) == []
    assert full_uploader.common.current_steps(full_uploader.FullSpec(environment)) == []
    checkpoint(environment, timestamp=100)
    assert eval_uploader.current_steps(spec) == [10]
    assert full_uploader.common.current_steps(full_uploader.FullSpec(environment)) == [10]


@pytest.mark.parametrize("value", [None, [], True, 0, "100", {}])
def test_nonobject_checkpoint_metadata_is_skipped(environment, value):
    spec = checkpoint(environment)
    (spec.ckpt_dir / "10/_CHECKPOINT_METADATA").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Checkpoint|generation"):
        lifecycle.checkpoint_identity(spec.ckpt_dir, 10, spec.generation)
    assert eval_uploader.current_steps(spec) == []
    assert full_uploader.common.current_steps(full_uploader.FullSpec(environment)) == []


@pytest.mark.parametrize("value", [None, [], True, {}, {"id": []}, {"id": "ok", "started_ns": True},
                                   {"id": "ok", "started_ns": []}, {"id": "ok", "started_ns": {}},
                                   {"id": "ok", "started_ns": -1}, {"id": "ok", "started_ns": "1"}])
def test_invalid_expected_and_persisted_generation_raise_value_error(environment, value):
    spec = checkpoint(environment)
    with pytest.raises(ValueError, match="Checkpoint|generation"):
        lifecycle.metadata_identity(b'{"commit_timestamp_nsecs": 100}', value)
    with pytest.raises(ValueError, match="Checkpoint|generation"):
        lifecycle.checkpoint_identity(spec.ckpt_dir, 10, value)
    lifecycle.generation_path(spec.ckpt_dir).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Checkpoint|generation"):
        lifecycle.generation(spec.ckpt_dir)
    assert eval_uploader.current_steps(spec) == []


@pytest.mark.parametrize("module", [eval_uploader, full_uploader])
def test_monitor_retries_repaired_completion_marker(environment, tmp_path, monkeypatch, module):
    checkpoint(environment, timestamp={})
    api = MockHub(tmp_path)
    run_main(module, environment, api, monkeypatch)
    assert not any(event[0] == "folder" for event in api.events)
    checkpoint(environment, timestamp=100)
    run_main(module, environment, api, monkeypatch)
    assert any(event[0] == "folder" for event in api.events)


def test_launcher_latest_skips_malformed_markers_without_traceback(environment):
    spec = checkpoint(environment)
    marker = spec.ckpt_dir / "10/_CHECKPOINT_METADATA"
    for value in ({"commit_timestamp_nsecs": []}, {"commit_timestamp_nsecs": True}, [], 0):
        marker.write_text(json.dumps(value))
        result = subprocess.run(
            ["bash", "-c", 'source /tmp/dev/env.sh && exec "$@"', "latest", PYTHON,
             str(ROOT / "scripts/b1k/run_lifecycle.py"), "latest", str(spec.ckpt_dir)],
            check=True, capture_output=True, text=True, timeout=3,
        )
        assert result.stdout.strip() == "-1"
        assert not result.stderr


@pytest.fixture
def launcher(tmp_path):
    scripts = tmp_path / "scripts/b1k"
    scripts.mkdir(parents=True)
    for name in ("train_b1k_run.sh", "run_lifecycle.py"):
        (scripts / name).write_bytes((ROOT / "scripts/b1k" / name).read_bytes())
    bin_dir = tmp_path / "mockbin"
    bin_dir.mkdir()
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(PYTHON)
    nvidia = bin_dir / "nvidia-smi"
    nvidia.write_text(f"""#!{PYTHON}
import os, pathlib, sys
args = sys.argv[1:]
if "--query-gpu=uuid" in args:
    device = args[args.index("-i") + 1]
    print(device if device.startswith("GPU-") else "GPU-" + device)
elif "--query-compute-apps=pid" in args:
    pathlib.Path(os.environ["MOCK_ROOT"], "idle-" + os.environ["EXP_NAME"]).touch()
    if os.environ.get("MOCK_BUSY") == "1": print("123")
elif "--query-gpu=memory.used" in args:
    print("0")
""")
    nvidia.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text(f"#!{PYTHON}\nimport time\ntime.sleep(0.02)\n")
    sleep.chmod(0o755)
    (scripts / "train_b1k.py").write_text("""import json, os, pathlib, time
root = pathlib.Path(os.environ["MOCK_ROOT"])
(root / ("trainer-" + os.environ["EXP_NAME"])).write_text(str(os.getpid()))
while not (root / "finish").exists():
    time.sleep(.02)
""")
    processes = []

    def start(name="test", devices="0,1", mode="fresh", *, busy=False):
        env_path = tmp_path / f"{name}.env"
        env = {
            "EXP_NAME": name, "CONFIG_NAME": "pi05_b1k", "OPENPI_DIR": str(tmp_path), "LOG_DIR": str(tmp_path / "logs"),
            "CUDA_VISIBLE_DEVICES": devices, "VENV": str(tmp_path / "venv"), "B1K_LOCK_DIR": str(tmp_path / "locks"),
            "MOCK_ROOT": str(tmp_path), "MOCK_BUSY": str(int(busy)), "JAX_CACHE": str(tmp_path / "cache"),
            "WANDB_PROJECT": "mock", "BATCH_SIZE": "4", "GRAD_ACCUM_STEPS": "1", "FSDP_DEVICES": "2",
            "REMAT_POLICY": "nothing_saveable", "MAX_TOKEN_LEN": "200", "NUM_WORKERS": "0", "NUM_TRAIN_STEPS": "11",
            "SAVE_INTERVAL": "10", "MAX_TO_KEEP": "2", "REPO_ID": "mock/demos", "DATASET_ROOT": str(tmp_path),
            "TASK_NAMES": "task", "PATH": f"{bin_dir}:{os.environ['PATH']}",
        }
        env_path.write_text("\n".join(f"{key}={value}" for key, value in env.items()))
        process = subprocess.Popen(
            ["bash", "-c", 'source /tmp/dev/env.sh && exec bash "$@"', "launcher", str(scripts / "train_b1k_run.sh"), str(env_path), mode],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, "B1K_LAUNCHER_GUARDED": "0"},
        )
        processes.append(process)
        return process

    yield tmp_path, start
    (tmp_path / "finish").touch()
    for process in processes:
        if process.poll() is None:
            process.terminate()
        try:
            process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=3)


def wait_path(path, process):
    deadline = time.monotonic() + 5
    while not path.exists():
        if process.poll() is not None:
            pytest.fail(f"Launcher exited early: {process.communicate(timeout=1)}")
        if time.monotonic() > deadline:
            pytest.fail(f"Timed out waiting for {path}")
        time.sleep(.02)


@pytest.mark.parametrize("waiting", [False, True])
@pytest.mark.parametrize("same_run", [False, True])
def test_launcher_conflicts_fail_immediately_and_release(launcher, waiting, same_run):
    root, start = launcher
    first = start(busy=waiting)
    wait_path(root / ("idle-test" if waiting else "trainer-test"), first)
    marker = root / "outputs/checkpoints/pi05_b1k/.test.generation.json"
    generation = marker.read_bytes()
    second = start(name="test" if same_run else "other", devices="2,3" if same_run else "GPU-1,2")
    stdout, stderr = second.communicate(timeout=3)
    assert second.returncode != 0, (stdout, stderr)
    assert "Already locked" in stderr
    assert marker.read_bytes() == generation
    assert not (root / "trainer-other").exists()
    first.terminate()
    first.communicate(timeout=8)
    # A fresh command can acquire the released run/GPU leases, but the rejected command never queues.
    (root / "finish").touch()
    third = start()
    stdout, stderr = third.communicate(timeout=5)
    assert third.returncode == 0, (stdout, stderr)
    assert marker.read_bytes() != generation


def test_launcher_disjoint_gpu_runs_can_launch(launcher):
    root, start = launcher
    first = start()
    second = start(name="other", devices="2,3")
    wait_path(root / "trainer-test", first)
    wait_path(root / "trainer-other", second)
    (root / "finish").touch()
    for process in (first, second):
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)


def test_launcher_parent_crash_retains_lease_until_trainer_exits(launcher):
    root, start = launcher
    first = start()
    wait_path(root / "trainer-test", first)
    first.kill()
    first.wait(timeout=3)
    second = start()
    stdout, stderr = second.communicate(timeout=3)
    assert second.returncode != 0, (stdout, stderr)
    assert "Already locked" in stderr
    (root / "finish").touch()
    first.communicate(timeout=5)
    third = start()
    stdout, stderr = third.communicate(timeout=5)
    assert third.returncode == 0, (stdout, stderr)


def test_fresh_early_trainer_failures_never_resume_previous_generation(launcher):
    root, start = launcher
    checkpoint({"OPENPI_DIR": str(root), "EXP_NAME": "test", "CONFIG_NAME": "pi05_b1k", "HF_REPO": "mock/repo", "NUM_TRAIN_STEPS": "11", "STAGING_DIR": str(root / "staging")})
    trainer = root / "scripts/b1k/train_b1k.py"
    trainer.write_text("raise SystemExit(9)\n")
    process = start()
    stdout, stderr = process.communicate(timeout=5)
    assert process.returncode == 9, (stdout, stderr)
    commands = [line for line in stdout.splitlines() if "launching:" in line]
    assert len(commands) == 3
    assert all("--overwrite" in command and "--resume" not in command for command in commands)


def test_launcher_resume_keeps_generation_and_mode_conventions(launcher):
    root, start = launcher
    (root / "finish").touch()
    first = start()
    first.communicate(timeout=5)
    assert first.returncode == 0
    marker = root / "outputs/checkpoints/pi05_b1k/.test.generation.json"
    generation = marker.read_bytes()
    (root / "outputs/checkpoints/pi05_b1k/test").mkdir()
    for mode in ("resume", "auto"):
        process = start(mode=mode)
        stdout, stderr = process.communicate(timeout=5)
        assert process.returncode == 0, (stdout, stderr)
        assert "--resume" in stdout
        assert marker.read_bytes() == generation
