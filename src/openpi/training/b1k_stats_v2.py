"""Exact or reproducibly sampled B1K statistics, excluding padded chunk steps."""
import contextlib
import json
from pathlib import Path
import numpy as np
from openpi.shared.b1k_data_core import StreamingStats, fingerprint, revision
from openpi.shared.normalize import NormStats
from openpi.training import lerobot_lowdim


def compute(dataset, transform, horizon, *, estimator='exact', sample_size=10_000_000,
            max_frames=None, seed=0, representation=None):
    if len(dataset.delta_indices) != 1:
        raise ValueError('v2 statistics require one action sequence key')
    action_key = next(iter(dataset.delta_indices))
    offsets = np.asarray(dataset.delta_indices[action_key], dtype=np.int64)
    if not np.array_equal(offsets, np.arange(horizon)):
        raise ValueError('unexpected action chunk offsets')
    if max_frames is not None and max_frames < 1:
        raise ValueError('max_frames must be positive')
    total = len(dataset)
    selected = None if max_frames is None or max_frames >= total else np.sort(
        np.random.default_rng(seed).choice(total, max_frames, replace=False))
    def positions(span):
        if selected is None:
            return np.arange(len(span))
        lo, hi = np.searchsorted(selected, [span.offset, span.offset + len(span)])
        return selected[lo:hi] - span.offset
    frames = total if selected is None else len(selected)
    steps = sum(int(np.minimum(horizon, len(span) - positions(span)).sum()) for span in dataset.spans)
    per_frame = not lerobot_lowdim.batched_transform_matches_per_frame(dataset, transform, ['state', 'actions'])
    with contextlib.ExitStack() as stack:
        trackers = None
        for span in dataset.spans:
            indices = positions(span)
            for start in range(0, len(indices), 2048):
                raw, shared = dataset.episode_items(span, indices[start:start+2048])
                batch = lerobot_lowdim.transform_episode(raw, shared, transform, ['state','actions'], per_frame_mode=per_frame)
                mask = ~np.asarray(raw[action_key + '_is_pad'], dtype=bool)
                state, actions = np.asarray(batch['state']), np.asarray(batch['actions'])
                if actions.shape[:-1] != mask.shape:
                    raise ValueError('transformed actions and padding mask disagree')
                if trackers is None:
                    trackers = {name: stack.enter_context(StreamingStats(dim, count, estimator=estimator,
                                                                         sample_size=sample_size))
                                for name,dim,count in [('state',state.shape[-1],frames), ('actions',actions.shape[-1],steps)]}
                trackers['state'].update(state)
                trackers['actions'].update(actions[mask])
        if trackers is None:
            raise ValueError('empty normalization selection')
        result, metadata = {}, {}
        for name, tracker in trackers.items():
            values = tracker.finish()
            result[name] = NormStats(**{k:values[k] for k in ('mean','std','q01','q99')})
            metadata[name] = tracker.metadata()
    key = fingerprint(dataset.root,
                      [(s.episode_index,s.first_index,s.end_index) for s in dataset.spans],
                      [s.data_file for s in dataset.spans], horizon=horizon, representation=representation,
                      max_frames=max_frames, seed=seed, estimator=estimator, sample_size=sample_size)
    source_sizes = {str(Path(s.data_file).absolute().relative_to(Path(dataset.root).absolute())): Path(s.data_file).stat().st_size for s in dataset.spans}
    return result, dict(version=2, fingerprint=key, dataset_revision=revision(dataset.root), source_sizes=source_sizes,
                        estimator=metadata, frames=frames, action_count=steps,
                        action_representation=representation, max_frames=max_frames, seed=seed)


def validate_manifest(root, directory):
    """Validate v2 assets on a local training dataset; old checkpoint assets still load."""
    path = Path(directory)/'stats_manifest.json'
    if not path.is_file() or root is None or not Path(root).is_dir():
        return
    manifest=json.loads(path.read_text())
    if 'dataset_revision' not in manifest:
        return
    if manifest['dataset_revision'] != revision(root):
        raise ValueError('normalization dataset revision mismatch; recompute statistics for this dataset')
    for name,size in manifest['source_sizes'].items():
        source=Path(root)/name
        if not source.is_file() or source.stat().st_size != size:
            raise ValueError(f'normalization source size mismatch: {name}')
