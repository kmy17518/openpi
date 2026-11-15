#!/usr/bin/env python3
"""Build the goal tables a B1K training run will read, before the run (isg-train224 `prepare`).

    prepare_goal_table.py pi05_b1k <the training run's flags>

Constructs the run's training and validation datasets exactly as train_b1k.py does, which decodes every selected
episode's goal image once into a GoalTable .npy under $B1K_GOAL_TABLE_DIR (default ~/.cache/openpi/b1k_goal_tables);
the training run (and each of its spawned loader workers) then memory-maps it instead of decoding again. A run without
goal views (--text-only) has no goal table and this is a no-op.
"""

import logging

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def main(config: _config.TrainConfig) -> None:
    data_config = config.data.create(config.assets_dirs, config.model)
    if not data_config.goal_views:
        logging.info("No goal views: nothing to prepare")
        return
    for subset in ("train", "val") if data_config.val_fraction else ("train",):
        dataset = _data_loader.create_b1k_dataset(data_config, config.model.action_horizon, config.model, subset=subset)
        logging.info("%s: %d frames; goal table ready", subset, len(dataset))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(_config.cli())
