import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile


class RunningStats:
    """Compute running statistics of a batch of vectors.

    The statistics depend on the order and partitioning of the batches (the running mean is updated batch by
    batch and the quantile histograms are re-binned whenever the observed range grows), so feeding the same
    batches in the same order always yields bit-identical results.
    """

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        # Per-dimension histograms used to approximate quantiles: (vector_length, num_bins) counts and
        # (vector_length, num_bins + 1) bin edges.
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = np.zeros((vector_length, self._num_quantile_bins))
            self._bin_edges = np.stack(
                [
                    np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                    for i in range(vector_length)
                ]
            )
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        new_histograms, new_bin_edges = [], []
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            new_histograms.append(new_hist)
            new_bin_edges.append(new_edges)
        self._histograms = np.stack(new_histograms)
        self._bin_edges = np.stack(new_bin_edges)

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors.

        Equivalent to `np.histogram(batch[:, i], bins=self._bin_edges[i])` for every dimension i, but bins all
        dimensions at once instead of calling np.histogram (which sorts the data) once per dimension.
        """
        self._histograms += _histogram_counts(batch, self._bin_edges)

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


def _histogram_counts(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Per-dimension histogram counts, bit-identical to `np.histogram` with explicit bin edges.

    Args:
        values: Array of shape (num_samples, vector_length).
        bin_edges: Monotonically non-decreasing bin edges of shape (vector_length, num_bins + 1).

    Returns:
        Integer counts of shape (vector_length, num_bins) equal to
        `np.stack([np.histogram(values[:, i], bins=bin_edges[i])[0] for i in range(vector_length)])`: bin b
        contains the values with `edges[b] <= x < edges[b + 1]`, except for the last bin which also includes its
        right edge; values outside `[edges[0], edges[-1]]` (and NaNs) are not counted.

    np.histogram sorts the data and searches the edges in it, once per dimension. Since the edges are (nearly)
    uniformly spaced, this instead computes each value's bin arithmetically for all dimensions at once and then
    corrects the guess against the true edges, so the result is exact.
    """
    vector_length, num_bins = bin_edges.shape[0], bin_edges.shape[1] - 1
    # np.histogram compares data and edges in their common dtype; float64 comparisons are exact for float32/float16
    # and integer inputs, so this matches whichever dtype the data has.
    values = values.astype(np.float64, copy=False)
    lower, upper = bin_edges[:, 0], bin_edges[:, -1]
    valid = (values >= lower) & (values <= upper)

    # Initial guess assuming perfectly uniform bins.
    width = (upper - lower) / num_bins
    with np.errstate(divide="ignore", invalid="ignore"):
        guess = np.floor((values - lower) / width)
    guess = np.where(np.isfinite(guess), guess, 0.0)
    bins = np.clip(guess, 0, num_bins - 1).astype(np.intp)
    bins[:, width == 0] = num_bins - 1  # All edges coincide: every (valid) value belongs to the closed last bin.
    bins[~valid] = 0

    # Correct the guess (normally off by at most one bin, due to rounding) until every valid value satisfies
    # `edges[b] <= x < edges[b + 1]`, or `b` is the last bin and `x <= edges[-1]`. This is exactly the bin that
    # np.histogram assigns, also when consecutive edges are equal. The edges are looked up through the flattened
    # table, which is considerably faster than 2-D fancy indexing.
    flat_edges = bin_edges.ravel()
    edge_row_offsets = np.arange(vector_length) * (num_bins + 1)
    unresolved = valid
    for _ in range(4):
        flat_bins = bins + edge_row_offsets
        left = flat_edges[flat_bins]
        right = flat_edges[np.minimum(flat_bins + 1, edge_row_offsets + num_bins)]
        too_high = unresolved & (values < left)
        too_low = unresolved & (values >= right) & (bins < num_bins - 1)
        unresolved = too_high | too_low
        if not unresolved.any():
            break
        bins[too_high] -= 1
        bins[too_low] += 1
    else:
        # Long runs of duplicated edges may need more steps; finish those values with a binary search.
        rows, cols = np.nonzero(unresolved)
        for i in np.unique(cols):
            sel = rows[cols == i]
            bins[sel, i] = np.minimum(np.searchsorted(bin_edges[i], values[sel, i], side="right") - 1, num_bins - 1)

    flat_bins = bins + np.arange(vector_length) * num_bins
    return np.bincount(flat_bins[valid], minlength=vector_length * num_bins).reshape(vector_length, num_bins)


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())
