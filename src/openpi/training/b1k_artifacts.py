"""Checkpoint conditioning and normalization provenance for BEHAVIOR policies."""

import dataclasses
import json
from typing import Any

from etils import epath
import numpy as np

METADATA_FILENAME = "b1k_metadata.json"
SCHEMA_VERSION = 1
MODEL_FIELDS = (
    "action_dim",
    "action_horizon",
    "max_token_len",
    "pi05",
    "discrete_state_input",
    "paligemma_variant",
    "action_expert_variant",
)


def action_representation(robot_config: Any, *, extra_delta_transform: bool) -> dict[str, Any]:
    def group(config):
        return {
            "name": config.name,
            "indices": list(config.indices),
            "is_eef": config.is_eef,
            "needs_delta_comp": config.needs_delta_comp,
            "delta_state_indices": config.delta_state_indices,
        }

    return {
        "robot_type": robot_config.robot_type,
        "action_dim": robot_config.action_dim,
        "extra_delta_transform": extra_delta_transform,
        "actions": [group(config) for config in robot_config.action],
        "proprio": [group(config) for config in robot_config.proprio],
    }


def load_metadata(assets_dir: Any) -> dict[str, Any] | None:
    path = epath.Path(assets_dir) / METADATA_FILENAME
    if not path.exists():
        return None
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported BEHAVIOR metadata in {path}")
    if not isinstance(metadata.get("action_representation"), dict):
        raise ValueError(f"Missing action representation in {path}")
    return metadata


def save_metadata(assets_dir: Any, metadata: dict[str, Any]) -> None:
    path = epath.Path(assets_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def norm_metadata(representation: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": SCHEMA_VERSION, "action_representation": representation}


def inference_metadata(data_config: Any, model_config: Any, prompts: dict[str, str]) -> dict[str, Any]:
    metadata = norm_metadata(data_config.action_representation)
    metadata.update(
        prompt_source=data_config.prompt_source,
        task_prompts=prompts,
        model_type=model_config.model_type.value,
        model={field: getattr(model_config, field) for field in MODEL_FIELDS if hasattr(model_config, field)},
    )
    return metadata


def validate_representation(
    metadata: dict[str, Any] | None,
    expected: dict[str, Any] | None,
    *,
    allow_legacy_assets: bool = False,
    context: str = "BEHAVIOR assets",
) -> None:
    if expected is None:
        return
    if metadata is None:
        if allow_legacy_assets:
            return
        raise ValueError(
            f"{context} lack action-representation metadata. Recompute normalization statistics, or explicitly "
            "allow unversioned assets with --data.allow-legacy-assets (training) / --allow-legacy-assets (serving) "
            "only after verifying the robot configuration matches the artifact. Old four-joint torso-delta "
            "checkpoints require b1k/R1Pro-legacy-torso-delta."
        )
    if metadata.get("action_representation") != expected:
        raise ValueError(
            f"{context} use an incompatible action representation. Select the matching robot configuration "
            "or recompute statistics/retrain; legacy-asset permission does not override a known mismatch."
        )


def restore_model_config(model_config: Any, metadata: dict[str, Any] | None, *, max_token_len: int | None = None):
    updates = {}
    if metadata is not None and "model" in metadata:
        if metadata.get("model_type") != model_config.model_type.value:
            raise ValueError("Checkpoint model type does not match the selected training configuration")
        stored = metadata["model"]
        if not isinstance(stored, dict) or set(stored) - set(MODEL_FIELDS):
            raise ValueError("Invalid inference model settings in BEHAVIOR checkpoint")
        allowed = {field.name for field in dataclasses.fields(model_config)}
        if set(stored) - allowed:
            raise ValueError("Checkpoint model settings are incompatible with the selected model")
        updates.update(stored)
    if max_token_len is not None:
        if isinstance(max_token_len, bool) or not isinstance(max_token_len, int) or max_token_len < 1:
            raise ValueError("max_token_len must be a positive integer")
        updates["max_token_len"] = max_token_len
    for field in ("action_dim", "action_horizon", "max_token_len"):
        value = updates.get(field, getattr(model_config, field))
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Invalid checkpoint model setting {field}")
    return dataclasses.replace(model_config, **updates)


def validate_norm_stats(saved: dict, current: dict | None) -> None:
    if current is None or saved.keys() != current.keys():
        raise ValueError("Resume normalization statistics differ from the checkpoint")
    for key, stats in saved.items():
        for field in ("mean", "std", "q01", "q99"):
            left, right = getattr(stats, field), getattr(current[key], field)
            if left is None and right is None:
                continue
            if left is None or right is None or not np.array_equal(left, right):
                raise ValueError(f"Resume normalization statistics differ for {key}.{field}; use checkpoint assets")


def state_dimension(robot_config: Any) -> int:
    return sum(1 if group.is_eef else len(group.indices) for group in robot_config.proprio)
