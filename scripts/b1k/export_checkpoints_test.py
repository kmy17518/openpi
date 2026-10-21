import dataclasses
import json
import os
import pathlib
import shutil
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.b1k import export_checkpoints as export


def fake_step(checkpoint_dir: pathlib.Path, name: str) -> pathlib.Path:
    """A step directory laid out like an orbax pi05_b1k checkpoint."""
    files = {
        "_CHECKPOINT_METADATA": b'{"commit_timestamp_nsecs": 1}',
        "commit_success.txt": b"Checkpoint commit was successful",
        "assets/turning_on_radio/norm_stats.json": b'{"norm_stats": {}}',
        "params/_METADATA": b"params tree",
        "params/manifest.ocdbt": b"params manifest",
        "params/ocdbt.process_0/d/0123abcd": os.urandom(3000),
        "train_state/_METADATA": b"train state tree",
        "train_state/ocdbt.process_0/d/4567cdef": os.urandom(5000),
    }
    for relative, data in files.items():
        (checkpoint_dir / name / relative).parent.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / name / relative).write_bytes(data)
    return checkpoint_dir / name


def initial_state(checkpoint_dir: pathlib.Path, run_dir: pathlib.Path) -> dict:
    for kind in ("eval", "full"):
        (run_dir / "export_queue" / kind).mkdir(parents=True, exist_ok=True)
    return {"checkpoint_dir": str(checkpoint_dir), "exported": []}


def queued(run_dir: pathlib.Path) -> list[str]:
    """Every file in the queue, temporary files included."""
    queue = run_dir / "export_queue"
    return sorted(str(path.relative_to(queue)) for path in queue.rglob("*") if path.is_file())


def tree(root: pathlib.Path) -> dict:
    return {
        str(path.relative_to(root)): (
            path.stat().st_mode,
            path.stat().st_mtime_ns,
            path.is_file() and path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize("temporary", ["10000.orbax-checkpoint-tmp", "10000.orbax-checkpoint-tmp-1729000000"])
def test_only_finalized_steps_are_exported_with_eval_and_full_members(tmp_path, temporary):
    checkpoints, run = tmp_path / "checkpoints", tmp_path / "run"
    step = fake_step(checkpoints, "5000")
    fake_step(checkpoints, temporary)
    (checkpoints / "wandb_id.txt").write_text("run-id")
    before = tree(checkpoints)
    state = initial_state(checkpoints, run)
    export.export_pending(checkpoints, run, state)

    assert tree(checkpoints) == before
    assert queued(run) == ["eval/step_00005000.tar", "full/step_00005000.tar"]
    assert state["exported"] == [5000]
    assert json.loads((run / "export_state.json").read_text()) == state
    files = {str(path.relative_to(checkpoints)): path.read_bytes() for path in step.rglob("*") if path.is_file()}
    directories = {"5000", *(str(path.relative_to(checkpoints)) for path in step.rglob("*") if path.is_dir())}

    def serving(name):
        return name == "5000" or name.split("/")[1] in ("params", "assets")

    for kind, keep in (("full", lambda name: True), ("eval", serving)):
        with tarfile.open(run / f"export_queue/{kind}/step_00005000.tar") as tar:
            assert tar.getnames()[0] == "5000"
            assert {member.name for member in tar if member.isdir()} == {name for name in directories if keep(name)}
            assert {member.name: tar.extractfile(member).read() for member in tar if member.isfile()} == {
                name: data for name, data in files.items() if keep(name)
            }


def test_newest_full_tar_supersedes_queued_one_and_exported_steps_are_not_written_again(tmp_path):
    checkpoints, run = tmp_path / "checkpoints", tmp_path / "run"
    fake_step(checkpoints, "5000")
    state = initial_state(checkpoints, run)
    export.export_pending(checkpoints, run, state)
    fake_step(checkpoints, "10000.orbax-checkpoint-tmp").rename(checkpoints / "10000")
    export.export_pending(checkpoints, run, state)
    assert queued(run) == ["eval/step_00005000.tar", "eval/step_00010000.tar", "full/step_00010000.tar"]

    # Tars published just before a crash that lost the state update are kept as they are.
    inodes = {name: (run / "export_queue" / name).stat().st_ino for name in queued(run)}
    state["exported"] = [5000]
    export.export_pending(checkpoints, run, state)
    assert {name: (run / "export_queue" / name).stat().st_ino for name in queued(run)} == inodes
    assert state["exported"] == [5000, 10000]

    # Once the uploader has consumed the queue, a restarted exporter writes nothing again.
    for name in queued(run):
        (run / "export_queue" / name).unlink()
    state = json.loads((run / "export_state.json").read_text())
    export.export_pending(checkpoints, run, state)
    assert queued(run) == []
    assert state["exported"] == [5000, 10000]


@pytest.mark.parametrize("change", ["deleted", "modified"])
def test_step_changing_during_export_publishes_nothing(tmp_path, monkeypatch, change):
    checkpoints, run = tmp_path / "checkpoints", tmp_path / "run"
    step = fake_step(checkpoints, "5000")
    state = initial_state(checkpoints, run)
    addfile = tarfile.TarFile.addfile

    def add_and_change(self, member, fileobj=None):
        assert queued(run) == ["eval/.step_00005000.tar.tmp"]
        if member.name == "5000/params/_METADATA":
            if change == "deleted":
                shutil.rmtree(step)
            else:
                (step / "assets/turning_on_radio/norm_stats.json").write_bytes(b"{}")
        return addfile(self, member, fileobj)

    monkeypatch.setattr(tarfile.TarFile, "addfile", add_and_change)
    export.export_pending(checkpoints, run, state)
    assert queued(run) == []
    assert state["exported"] == []
    assert not (run / "export_state.json").exists()

    monkeypatch.undo()
    export.export_pending(checkpoints, run, state)
    if change == "deleted":
        assert queued(run) == []
        assert state["exported"] == []
    else:
        assert queued(run) == ["eval/step_00005000.tar", "full/step_00005000.tar"]
        assert state["exported"] == [5000]


def test_waits_until_the_next_checkpoint_still_fits(tmp_path, monkeypatch):
    checkpoints, run = tmp_path / "checkpoints", tmp_path / "run"
    step = fake_step(checkpoints, "5000")
    state = initial_state(checkpoints, run)
    size = sum(path.stat().st_size for path in step.rglob("*") if path.is_file())
    usage = shutil.disk_usage(run)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=2 * size - 1))
    export.export_pending(checkpoints, run, state)
    assert queued(run) == ["eval/step_00005000.tar"]
    assert state["exported"] == []
    monkeypatch.setattr(shutil, "disk_usage", lambda path: usage._replace(free=2 * size))
    export.export_pending(checkpoints, run, state)
    assert queued(run) == ["eval/step_00005000.tar", "full/step_00005000.tar"]
    assert state["exported"] == [5000]


def test_main_exits_once_the_final_step_is_exported(tmp_path):
    checkpoints, run = tmp_path / "checkpoints", tmp_path / "run"
    for name in ("95000", "99999", "100000"):
        fake_step(checkpoints, name)
    argv = ["--checkpoint-dir", str(checkpoints), "--run-dir", str(run), "--max-steps", "99999", "--poll-seconds", "1"]
    assert export.main(argv) == 0
    assert queued(run) == ["eval/step_00095000.tar", "eval/step_00099999.tar", "full/step_00099999.tar"]
    assert json.loads((run / "export_state.json").read_text())["exported"] == [95000, 99999]
    assert export.main(argv) == 0
    with pytest.raises(SystemExit, match="exports"):
        export.main(["--checkpoint-dir", str(tmp_path / "other"), "--run-dir", str(run), "--max-steps", "99999"])
    with pytest.raises(SystemExit):
        export.main(["--checkpoint-dir", str(checkpoints), "--run-dir", str(checkpoints / "export")])


def test_tars_of_a_real_orbax_step_restore_for_resume_and_serving(tmp_path, monkeypatch):
    from openpi.models import model as _model
    from openpi.shared import normalize
    from openpi.training import b1k_artifacts
    from openpi.training import checkpoints
    from openpi.training import config

    model = config.get_config("pi05_b1k").model
    dc = config.LeRobotB1KDataConfig(repo_id="org/demos", robot_config_name="b1k/R1Pro").create(
        tmp_path / "config", model
    )
    metadata = b1k_artifacts.inference_metadata(dc, model, {"turning_on_radio": "Switch on this radio."})
    stats = {"state": normalize.NormStats(mean=np.zeros(23), std=np.ones(23))}
    loader = SimpleNamespace(data_config=lambda: dataclasses.replace(dc, norm_stats=stats, inference_metadata=metadata))
    kernel = np.arange(6, dtype=np.float32).reshape(2, 3)
    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ({"step": np.asarray(5001)}, {"kernel": kernel}))
    monkeypatch.setattr(checkpoints, "_merge_params", lambda state, params: (state, params))
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "orbax", keep_period=None, max_to_keep=3, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, object(), loader, 5000)
        manager.wait_until_finished()
    finally:
        manager.close()
    run = tmp_path / "run"
    export.export_pending(tmp_path / "orbax", run, initial_state(tmp_path / "orbax", run))
    for kind in ("eval", "full"):
        with tarfile.open(run / f"export_queue/{kind}/step_00005000.tar") as tar:
            tar.extractall(tmp_path / kind, filter="data")

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "full", keep_period=None, max_to_keep=3, overwrite=False, resume=True
    )
    try:
        assert resuming
        restored, params = checkpoints.restore_state(manager, object(), loader)
        assert restored["step"] == 5001
        np.testing.assert_array_equal(params["params"]["kernel"], kernel)
    finally:
        manager.close()
    served = tmp_path / "eval/5000"
    assert sorted(path.name for path in served.iterdir()) == ["assets", "params"]
    np.testing.assert_array_equal(_model.restore_params(served / "params", restore_type=np.ndarray)["kernel"], kernel)
    np.testing.assert_array_equal(checkpoints.load_norm_stats(served / "assets", dc.asset_id)["state"].std, np.ones(23))
