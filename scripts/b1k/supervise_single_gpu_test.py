from pathlib import Path
import signal
from unittest.mock import Mock

from scripts.b1k import supervise_single_gpu
from scripts.b1k.supervise_single_gpu import command_lines


def test_supervisor_uses_one_writer_and_exact_schedule():
    settings = {
        "python": "/tmp/python",
        "hf_repo": "kmy17518/test",
        "checkpoint_dir": "/tmp/checkpoints",
        "staging_dir": "/tmp/staging",
        "wandb_run_id": "run",
        "wandb_url": "https://wandb.ai/test",
        "max_steps": 300000,
        "save_interval": 2500,
        "first_checkpoint_step": 25,
        "storage_proof": "/tmp/proof.json",
    }
    trainer, publisher = command_lines(settings, Path("/tmp/run.json"))
    assert trainer == ["/tmp/python", "scripts/b1k/train_b1k_monitored.py", "--run-config", "/tmp/run.json"]
    assert publisher[publisher.index("--eval-every") + 1] == "10000"
    assert publisher[publisher.index("--full-every") + 1] == "2500"
    assert "--sole-writer" in publisher
    assert publisher[publisher.index("--max-staging-bytes") + 1] == str(250 * 1024**3)


def test_shutdown_signals_group_even_if_parent_exited(monkeypatch):
    process = Mock(pid=123)
    process.poll.return_value = 0
    killpg = Mock()
    monkeypatch.setattr(supervise_single_gpu.os, "killpg", killpg)
    supervise_single_gpu.stop(process, timeout=0)
    assert killpg.call_args_list[0].args == (123, signal.SIGTERM)
    assert killpg.call_args_list[1].args == (123, signal.SIGKILL)
    process.wait.assert_called_once()
