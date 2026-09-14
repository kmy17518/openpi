import numpy as np
import pytest

import openpi.shared.normalize as normalize


def test_normalize_update():
    arr = np.arange(12).reshape(4, 3)  # 4 vectors of length 3

    stats = normalize.RunningStats()
    for i in range(len(arr)):
        stats.update(arr[i : i + 1])  # Update with one vector at a time
    results = stats.get_statistics()

    assert np.allclose(results.mean, np.mean(arr, axis=0))
    assert np.allclose(results.std, np.std(arr, axis=0))


def test_serialize_deserialize():
    stats = normalize.RunningStats()
    stats.update(np.arange(12).reshape(4, 3))  # 4 vectors of length 3

    norm_stats = {"test": stats.get_statistics()}
    norm_stats2 = normalize.deserialize_json(normalize.serialize_json(norm_stats))
    assert np.allclose(norm_stats["test"].mean, norm_stats2["test"].mean)
    assert np.allclose(norm_stats["test"].std, norm_stats2["test"].std)


class _ReferenceRunningStats(normalize.RunningStats):
    """RunningStats with the original (per-dimension np.histogram) histogram update, used as the oracle."""

    def _update_histograms(self, batch: np.ndarray) -> None:
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist


def _assert_same_stats(a: normalize.RunningStats, b: normalize.RunningStats) -> None:
    for name in ["_mean", "_mean_of_squares", "_min", "_max", "_histograms", "_bin_edges"]:
        x, y = getattr(a, name), getattr(b, name)
        assert x.dtype == y.dtype, name
        assert np.array_equal(x, y), name
    stats_a, stats_b = a.get_statistics(), b.get_statistics()
    for name in ["mean", "std", "q01", "q99"]:
        x, y = getattr(stats_a, name), getattr(stats_b, name)
        assert x.dtype == y.dtype, name
        assert np.array_equal(x, y), name


@pytest.mark.parametrize(
    "make_batch",
    [
        # Values that fall exactly on bin edges (running min / max reappear in later batches).
        lambda rng, k: rng.integers(-3, 4, size=(32, 5)).astype(np.float32),
        # Growing range, i.e. frequent histogram re-binning.
        lambda rng, k: (rng.standard_normal((64, 7)) * (1 + k)).astype(np.float32),
        # Constant and mixed constant/random dimensions (all bin edges coincide).
        lambda rng, k: np.ones((16, 3), np.float32),
        lambda rng, k: np.concatenate([np.ones((16, 1), np.float32), rng.random((16, 2)).astype(np.float32)], axis=1),
        # float64 inputs, including ranges of a few ulps where consecutive bin edges are duplicated.
        lambda rng, k: rng.standard_normal((32, 4)),
        lambda rng, k: 1.0 + rng.integers(0, 3, size=(32, 3)) * 2.0**-52,
        lambda rng, k: 1.0 + rng.integers(0, 20000, size=(32, 3)) * 2.0**-52,
        # Large magnitudes and repeated values.
        lambda rng, k: (rng.integers(-1000, 1000, size=(64, 6)) * 1e3).astype(np.float32),
        lambda rng, k: rng.integers(0, 50, size=(128, 8)).astype(np.float32) / 7,
    ],
)
def test_histogram_update_matches_np_histogram(make_batch):
    rng = np.random.default_rng(0)
    batches = [make_batch(rng, k) for k in range(60)]

    stats, reference = normalize.RunningStats(), _ReferenceRunningStats()
    for batch in batches:
        stats.update(batch)
        reference.update(batch)
    _assert_same_stats(stats, reference)


def test_histogram_counts_semantics():
    # Half-open bins, closed last bin, out-of-range values and NaNs dropped, duplicated edges.
    edges = np.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 2.0], [5.0, 5.0, 5.0, 5.0]])
    values = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 1.0, 5.0],
            [3.0, 2.0, 4.0],
            [-1.0, 0.5, 6.0],
            [2.5, 1.5, np.nan],
        ]
    )
    counts = normalize._histogram_counts(values, edges)  # noqa: SLF001
    expected = np.stack([np.histogram(values[:, i], bins=edges[i])[0] for i in range(edges.shape[0])])
    assert np.array_equal(counts, expected)
    assert np.array_equal(counts, [[1, 1, 2], [2, 0, 3], [0, 0, 2]])


def test_multiple_batch_dimensions():
    # Test with multiple batch dimensions: (2, 3, 4) where 4 is vector dimension
    batch_shape = (2, 3, 4)
    arr = np.random.rand(*batch_shape)

    stats = normalize.RunningStats()
    stats.update(arr)  # Should handle (2, 3, 4) -> reshape to (6, 4)
    results = stats.get_statistics()

    # Flatten batch dimensions and compute expected stats
    flattened = arr.reshape(-1, arr.shape[-1])  # (6, 4)
    expected_mean = np.mean(flattened, axis=0)
    expected_std = np.std(flattened, axis=0)

    assert np.allclose(results.mean, expected_mean)
    assert np.allclose(results.std, expected_std)
