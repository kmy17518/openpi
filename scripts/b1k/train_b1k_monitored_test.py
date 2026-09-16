import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from scripts.b1k.train_b1k_monitored import RunObserver
from scripts.b1k.train_b1k_monitored import parse_cpus


def test_cpu_affinity_limit():
    assert parse_cpus("30-59") == set(range(30, 60))
    with pytest.raises(ValueError, match="CPUs"):
        parse_cpus("0-30")


def test_checkpoint_schedule_and_stage(tmp_path):
    stage = Mock()
    settings = {
        "status_file": str(tmp_path / "status.json"),
        "staging_dir": str(tmp_path / "staging"),
        "first_checkpoint_step": 25,
        "max_steps": 300000,
        "save_interval": 2500,
    }
    observer = RunObserver(settings, stage)
    for step in [25, 2500, 10000, 300000]:
        assert observer.should_save(step)
    for step in [0 + 1, 24, 2501, 299999]:
        assert not observer.should_save(step)
    observer.on_step(25, {"loss": 1.0, "grad_norm": 2.0}, 3.0)
    observer.on_checkpoint(Path("/tmp/checkpoint/25"), 25)
    stage.assert_called_once_with(Path("/tmp/checkpoint/25"), tmp_path / "staging", 25)
    assert json.loads((tmp_path / "status.json").read_text())["staged_step"] == 25
    with pytest.raises(RuntimeError, match="Nonfinite"):
        observer.on_step(26, {"loss": float("nan")}, 1.0)


def test_resume_reconciles_unstaged_completed_saves(tmp_path):
    settings = {
        "status_file": str(tmp_path / "status.json"),
        "staging_dir": str(tmp_path / "staging"),
        "first_checkpoint_step": 25,
        "max_steps": 300000,
        "save_interval": 2500,
    }
    stage = Mock()
    observer = RunObserver(settings, stage)
    observer.reconcile_checkpoints(tmp_path, [300000, 10000, 25])
    assert [call.args[2] for call in stage.call_args_list] == [25, 10000, 300000]


def test_stage_failure_keeps_visible_status(tmp_path):
    settings = {
        "status_file": str(tmp_path / "status.json"),
        "staging_dir": str(tmp_path / "staging"),
        "first_checkpoint_step": 25,
        "max_steps": 300000,
        "save_interval": 2500,
    }
    observer = RunObserver(settings, Mock(side_effect=RuntimeError("stage failure")))
    with pytest.raises(RuntimeError, match="stage failure"):
        observer.on_checkpoint(tmp_path / "25", 25)
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "staging_checkpoint"
