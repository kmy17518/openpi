from types import SimpleNamespace

import numpy as np
import pytest

from openpi.training import checkpoints


@pytest.mark.parametrize(("keep_period", "kept"), [(None, [3, 4, 5]), (2, [2, 3, 4, 5])])
def test_max_to_keep_keeps_latest_checkpoints_plus_keep_period(tmp_path, monkeypatch, keep_period, kept):
    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ({"step": np.asarray(1)}, {"kernel": np.ones(2)}))
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, action_representation=None))
    directory = tmp_path / "orbax"
    manager, _ = checkpoints.initialize_checkpoint_dir(
        directory, keep_period=keep_period, max_to_keep=3, overwrite=False, resume=False
    )
    try:
        for step in range(1, 6):
            checkpoints.save_state(manager, object(), loader, step)
        manager.wait_until_finished()
        assert sorted(manager.all_steps()) == kept
        assert sorted(int(path.name) for path in directory.iterdir() if path.name.isdigit()) == kept
    finally:
        manager.close()
