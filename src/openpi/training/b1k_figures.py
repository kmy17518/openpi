"""Validation action figures, in the format of ACT's `val/actions/sample_<i>` (act b1k_training.py)."""

import numpy as np


def action_chunk_figure(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, title: str):
    """Matplotlib figure of one (chunk, action_dim) chunk: one small plot per action dimension over the valid steps."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.flatnonzero(valid)
    dims = target.shape[-1]
    figure, axes = plt.subplots(-(-dims // 6), 6, figsize=(18, 10), squeeze=False, layout="constrained")
    for dim, axis in enumerate(axes.flat):
        if dim >= dims:
            axis.remove()
            continue
        axis.plot(steps, target[steps, dim], color="tab:blue", linewidth=1, label="ground truth")
        axis.plot(steps, prediction[steps, dim], color="tab:orange", linewidth=1, label="prediction")
        axis.set_title(f"action {dim}", fontsize=8)
        axis.tick_params(labelsize=6)
    axes.flat[0].legend(fontsize=6)
    figure.suptitle(title, fontsize=10)
    return figure


def figure_image(figure) -> np.ndarray:
    """RGB uint8 array of a matplotlib figure, which is closed."""
    import matplotlib.pyplot as plt

    try:
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
    finally:
        plt.close(figure)
