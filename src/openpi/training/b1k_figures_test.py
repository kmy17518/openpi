import numpy as np

from openpi.training import b1k_figures


def test_action_chunk_figure_renders_every_action_dimension():
    rng = np.random.default_rng(0)
    target, prediction = rng.normal(size=(2, 50, 23))
    valid = np.arange(50) < 30
    image = b1k_figures.figure_image(
        b1k_figures.action_chunk_figure(prediction, target, valid, "task  episode 3  frame 7")
    )
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[-1] == 3
    assert image.shape[0] > 100 and image.shape[1] > 100
    assert image.min() < 255  # something was drawn
