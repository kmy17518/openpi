import contextlib
import gc
import itertools
import queue
import threading
import time
import weakref

import jax
import numpy as np
import pytest

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _next_with_timeout(iterator, timeout=5.0):
    results = queue.Queue()

    def consume():
        try:
            results.put((True, next(iterator)))
        except BaseException as exc:
            results.put((False, exc))

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    try:
        success, result = results.get(timeout=timeout)
    except queue.Empty:
        pytest.fail("next() did not finish")
    thread.join(timeout)
    if not success:
        raise result
    return result


def _assert_exhausted(iterator):
    for _ in range(3):
        with pytest.raises(StopIteration):
            _next_with_timeout(iterator)


def test_finite_exhaustion_is_remembered():
    with _data_loader.PrefetchIterator(iter(range(3)), depth=1) as iterator:
        assert [_next_with_timeout(iterator) for _ in range(3)] == [0, 1, 2]
        _assert_exhausted(iterator)
    assert iterator.close()


def test_producer_exception_is_forwarded_once_after_queued_items():
    closed = threading.Event()
    error = ValueError("producer failed")

    def source():
        try:
            yield 42
            raise error
        finally:
            closed.set()

    with _data_loader.PrefetchIterator(source()) as iterator:
        assert _next_with_timeout(iterator) == 42
        with pytest.raises(ValueError, match="producer failed") as raised:
            _next_with_timeout(iterator)
        assert raised.value is error
        assert closed.wait(5)
        _assert_exhausted(iterator)


def test_exception_objects_are_valid_items():
    value = ValueError("data, not an error")
    with _data_loader.PrefetchIterator(iter([value])) as iterator:
        assert _next_with_timeout(iterator) is value
        _assert_exhausted(iterator)


@pytest.mark.parametrize("consumer_error", [False, True])
def test_early_break_or_consumer_exception_closes_generator(*, consumer_error):
    closed = threading.Event()

    def source():
        try:
            yield from itertools.count()
        finally:
            closed.set()

    def consume():
        with _data_loader.PrefetchIterator(source(), depth=1) as iterator:
            for _ in iterator:
                if consumer_error:
                    raise RuntimeError("consumer failed")
                break
        return iterator

    if consumer_error:
        with pytest.raises(RuntimeError, match="consumer failed"):
            consume()
    else:
        _assert_exhausted(consume())
    assert closed.wait(5)


class _TrackedIterator:
    def __init__(self):
        self.read = threading.Event()
        self.closed = threading.Event()
        self.close_calls = 0
        self.producer_thread = None
        self.close_thread = None

    def __iter__(self):
        return self

    def __next__(self):
        self.producer_thread = threading.current_thread()
        self.read.set()
        return 42

    def close(self):
        self.close_calls += 1
        self.close_thread = threading.current_thread()
        self.closed.set()


def test_full_queue_cancellation_closes_upstream_once_on_producer_thread():
    source = _TrackedIterator()
    iterator = _data_loader.PrefetchIterator(source, depth=1)
    try:
        assert source.read.wait(5)
        # Wait for a queued item without consuming it, leaving the producer without capacity.
        state = iterator._state  # noqa: SLF001
        with state.condition:
            assert state.condition.wait_for(lambda: len(state.items) == 1, timeout=5)
        assert iterator.close(timeout=5)
        assert source.closed.is_set()
        assert source.close_calls == 1
        assert source.close_thread is source.producer_thread
        assert iterator.close(timeout=5)
        assert source.close_calls == 1
        assert not state.items
        _assert_exhausted(iterator)
    finally:
        iterator.close()


def test_blocked_upstream_close_is_bounded_and_does_not_close_executing_generator():
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def source():
        try:
            entered.set()
            release.wait(10)
            yield 42
        finally:
            closed.set()

    iterator = _data_loader.PrefetchIterator(source())
    try:
        assert entered.wait(5)
        start = time.monotonic()
        assert not iterator.close(timeout=0.01)
        assert time.monotonic() - start < 1.0
        assert not closed.is_set()
        _assert_exhausted(iterator)
    finally:
        release.set()
        assert iterator.close(timeout=5)
    assert closed.is_set()


def test_close_wakes_waiting_consumer():
    entered = threading.Event()
    release = threading.Event()
    consumer_done = threading.Event()
    errors = []

    def source():
        entered.set()
        release.wait(10)
        yield 42

    iterator = _data_loader.PrefetchIterator(source())

    def consume():
        try:
            next(iterator)
        except StopIteration:
            pass
        except BaseException as exc:
            errors.append(exc)
        else:
            errors.append(AssertionError("received an item after cancellation"))
        finally:
            consumer_done.set()

    consumer = threading.Thread(target=consume, daemon=True)
    try:
        assert entered.wait(5)
        consumer.start()
        iterator.close(timeout=0)
        assert consumer_done.wait(5)
        assert not errors
    finally:
        release.set()
        iterator.close(timeout=5)
        consumer.join(timeout=5)


def test_garbage_collection_cancels_producer_without_retaining_consumer():
    source = _TrackedIterator()
    iterator = _data_loader.PrefetchIterator(source, depth=1)
    reference = weakref.ref(iterator)
    thread = iterator._thread  # noqa: SLF001
    assert source.read.wait(5)
    del iterator
    gc.collect()
    try:
        assert reference() is None
        assert source.closed.wait(5)
        thread.join(5)
        assert not thread.is_alive()
    finally:
        if (remaining := reference()) is not None:
            remaining.close()


def test_thread_start_failure_closes_upstream_without_masking_error(monkeypatch):
    source = _TrackedIterator()

    def start(_thread):
        raise RuntimeError("thread start failed")

    def close():
        source.closed.set()
        raise ValueError("close failed")

    monkeypatch.setattr(threading.Thread, "start", start)
    monkeypatch.setattr(source, "close", close)
    with pytest.raises(RuntimeError, match="thread start failed"):
        _data_loader.PrefetchIterator(source)
    assert source.closed.is_set()


def test_upstream_close_error_is_forwarded():
    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        def close(self):
            raise RuntimeError("close failed")

    with _data_loader.PrefetchIterator(Source()) as iterator:
        with pytest.raises(RuntimeError, match="close failed"):
            _next_with_timeout(iterator)
        _assert_exhausted(iterator)


def test_primary_producer_error_is_not_masked_by_close_error(caplog):
    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise ValueError("read failed")

        def close(self):
            raise RuntimeError("close failed")

    with _data_loader.PrefetchIterator(Source()) as iterator:
        with pytest.raises(ValueError, match="read failed"):
            _next_with_timeout(iterator)
        _assert_exhausted(iterator)
    assert "close failed" in caplog.text


def test_data_loader_wrapper_forwards_close(monkeypatch):
    closed = threading.Event()

    class Loader:
        def __iter__(self):
            try:
                while True:
                    yield {"actions": 42}
            finally:
                closed.set()

    monkeypatch.setattr(_data_loader._model.Observation, "from_dict", lambda batch: batch)  # noqa: SLF001
    loader = _data_loader.DataLoaderImpl(_config.DataConfig(), Loader())
    with _data_loader.PrefetchIterator(iter(loader)) as iterator:
        assert _next_with_timeout(iterator) == ({"actions": 42}, 42)
    assert closed.wait(5)


class _WorkerDataset:
    def __init__(self, *, fail=False):
        self.fail = fail

    def __len__(self):
        return 8

    def __getitem__(self, index):
        if self.fail and index >= 2:
            raise ValueError("worker sample failed")
        return {"sample": np.asarray(index, dtype=np.float32)}


@pytest.mark.parametrize("completion", ["exhaustion", "close", "worker_error"])
def test_persistent_workers_shutdown(completion):
    loader = _data_loader.TorchDataLoader(
        _WorkerDataset(fail=completion == "worker_error"),
        local_batch_size=2,
        num_workers=1,
        num_batches=1 if completion == "exhaustion" else None,
        framework="pytorch",
    )
    upstream = iter(loader)
    # Match trainer startup: spawn workers on the main thread before ownership transfers to prefetch.
    next(upstream)
    workers = list(loader.torch_loader._iterator._workers)  # noqa: SLF001
    try:
        with _data_loader.PrefetchIterator(upstream, depth=1) as iterator:
            if completion == "exhaustion":
                _assert_exhausted(iterator)
            elif completion == "worker_error":
                with pytest.raises(ValueError, match="worker sample failed"):
                    _next_with_timeout(iterator, timeout=30)
                _assert_exhausted(iterator)
            else:
                _next_with_timeout(iterator, timeout=30)
        assert iterator.close(timeout=30)
        assert loader.torch_loader._iterator is None  # noqa: SLF001
        assert all(not worker.is_alive() for worker in workers)
        if completion != "worker_error":
            # A closed loader remains reusable, with a fresh set of persistent workers.
            with contextlib.closing(iter(loader)) as restarted:
                np.testing.assert_array_equal(next(restarted)["sample"], [0, 1])
            assert loader.torch_loader._iterator is None  # noqa: SLF001
    finally:
        # Avoid leaving worker processes behind even if a lifecycle assertion fails.
        if (torch_iterator := loader.torch_loader._iterator) is not None:  # noqa: SLF001
            torch_iterator._shutdown_workers()  # noqa: SLF001


@pytest.mark.parametrize("micro_batches", [1, 2])
def test_prefetched_microbatches_preserve_data_and_sharding(micro_batches):
    loader = _data_loader.TorchDataLoader(
        _WorkerDataset(), local_batch_size=4, num_batches=3, micro_batches=micro_batches
    )
    with _data_loader.PrefetchIterator(iter(loader), depth=2) as iterator:
        batches = [_next_with_timeout(iterator) for _ in range(3)]
        _assert_exhausted(iterator)
    for index, batch in enumerate(batches):
        values = batch["sample"]
        assert values.shape == ((4,) if micro_batches == 1 else (2, 2))
        expected_spec = jax.sharding.PartitionSpec("B") if micro_batches == 1 else jax.sharding.PartitionSpec(None, "B")
        assert values.sharding.spec == expected_spec
        np.testing.assert_array_equal(np.asarray(values).reshape(-1), np.arange((index % 2) * 4, (index % 2 + 1) * 4))
