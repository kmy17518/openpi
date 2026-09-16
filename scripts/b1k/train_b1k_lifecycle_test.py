import contextlib
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from scripts.b1k import train_b1k


@pytest.mark.parametrize("prefetch_batches", [0, 1])
@pytest.mark.parametrize(
    "failure", ["first_batch", "images", "init", "restore", "jit", "train", "checkpoint_wait", "none", "empty_loop"]
)
def test_main_closes_iterator_on_startup_failure_training_failure_and_completion(
    monkeypatch, tmp_path, prefetch_batches, failure
):
    closed = threading.Event()
    close_threads = []
    batch = (SimpleNamespace(images={"camera": np.zeros((1, 2, 2, 3))}), np.zeros((1, 1, 1)))

    def source():
        try:
            if failure == "first_batch":
                raise RuntimeError(failure)
            while True:
                yield batch
        finally:
            close_threads.append(threading.current_thread().name)
            closed.set()

    source_iter = source()

    class Loader:
        def __iter__(self):
            return source_iter

    config = SimpleNamespace(
        batch_size=1,
        grad_accum_steps=1,
        seed=0,
        fsdp_devices=1,
        checkpoint_dir=tmp_path,
        keep_period=None,
        overwrite=False,
        resume=failure == "restore",
        max_to_keep=1,
        wandb_enabled=False,
        prefetch_batches=prefetch_batches,
        num_train_steps=0 if failure == "empty_loop" else 1,
        log_interval=1,
        val_log_interval=0,
        save_interval=1,
    )
    state = SimpleNamespace(step=0, params={})
    manager = mock.Mock()
    if failure == "checkpoint_wait":
        manager.wait_until_finished.side_effect = RuntimeError(failure)
    monkeypatch.setattr(train_b1k, "init_logging", lambda: None)
    monkeypatch.setattr(train_b1k._xla_gpu_compat, "configure_xla_flags", lambda: None)  # noqa: SLF001
    monkeypatch.setattr(train_b1k, "init_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_b1k._checkpoints,  # noqa: SLF001
        "initialize_checkpoint_dir",
        lambda *args, **kwargs: (manager, failure == "restore"),
    )
    monkeypatch.setattr(train_b1k._checkpoints, "save_state", lambda *args: None)  # noqa: SLF001
    monkeypatch.setattr(train_b1k._data_loader, "create_b1k_data_loader", lambda *args, **kwargs: Loader())  # noqa: SLF001
    monkeypatch.setattr(train_b1k.training_utils, "array_tree_to_info", lambda _: "")
    monkeypatch.setattr(train_b1k.wandb, "Image", lambda image: image)
    monkeypatch.setattr(train_b1k.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(train_b1k.common_utils, "stack_forest", lambda _: {})

    def log(*args, **kwargs):
        if failure == "images":
            raise RuntimeError(failure)

    def init(*args, **kwargs):
        if failure == "init":
            raise RuntimeError(failure)
        return state, None

    def restore(*args, **kwargs):
        raise RuntimeError("restore")

    def step(*args):
        if failure == "train":
            raise RuntimeError(failure)
        return state, {}

    def jit(*args, **kwargs):
        if failure == "jit":
            raise RuntimeError(failure)
        return step

    monkeypatch.setattr(train_b1k.wandb, "log", log)
    monkeypatch.setattr(train_b1k, "init_train_state", init)
    monkeypatch.setattr(train_b1k._checkpoints, "restore_state", restore)  # noqa: SLF001
    monkeypatch.setattr(train_b1k.jax, "jit", jit)

    if failure in {"none", "empty_loop"}:
        train_b1k.main(config)
    else:
        with pytest.raises(RuntimeError, match=failure):
            train_b1k.main(config)
    assert closed.wait(5)
    before_prefetch = failure in {"first_batch", "images", "init", "restore"}
    expected_thread = threading.current_thread().name if before_prefetch or not prefetch_batches else "batch-prefetch"
    assert close_threads == [expected_thread]


@pytest.mark.parametrize("failure", ["next", "step", "empty", "none"])
def test_validation_closes_iterator_on_early_return_or_failure(monkeypatch, failure):
    closed = threading.Event()

    def source():
        try:
            if failure == "next":
                raise RuntimeError("validation read failed")
            if failure == "empty":
                return
            while True:
                yield (None, None)
        finally:
            closed.set()

    def step(*args):
        if failure == "step":
            raise RuntimeError("validation step failed")
        return np.asarray(1.0)

    monkeypatch.setattr(train_b1k.jax, "jit", lambda *args, **kwargs: step)
    monkeypatch.setattr(train_b1k.sharding, "set_mesh", lambda _: contextlib.nullcontext())
    config = SimpleNamespace(seed=0, val_num_batches=2)
    if failure == "next":
        with pytest.raises(RuntimeError, match="validation read failed"):
            train_b1k._compute_validation_losses(source(), None, None, None, None, config)  # noqa: SLF001
    else:
        result = train_b1k._compute_validation_losses(source(), None, None, None, None, config)  # noqa: SLF001
        if failure in {"step", "empty"}:
            assert result is None
        else:
            assert result == 1.0
    assert closed.is_set()
