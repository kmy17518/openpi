import dataclasses
import json
from types import SimpleNamespace

from etils import epath
import numpy as np
import pytest

from openpi.configs.robots import ROBOT_REGISTRY
from openpi.training import b1k_artifacts
from openpi.training import b1k_dataset
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import data_loader
from openpi.training.b1k_dataset_test import _write_synthetic_root
from scripts.b1k import serve_b1k


def factory(robot="b1k/R1Pro", **kwargs):
    return config.LeRobotB1KDataConfig(repo_id="org/demos", robot_config_name=robot, **kwargs)


def model_config():
    return config.get_config("pi05_b1k").model


def other_convention_factory(**kwargs):
    """A data config with a different action representation (no extra delta transform)."""
    return factory(extra_delta_transform=False, **kwargs)


def test_statistics_representation_guard_and_explicit_legacy(tmp_path):
    current = factory().create(tmp_path, model_config())
    legacy = other_convention_factory().create(tmp_path, model_config())
    stats = {"state": SimpleNamespace()}
    with pytest.raises(ValueError, match="lack action-representation"):
        data_loader.transform_dataset([], dataclasses.replace(current, norm_stats=stats))
    b1k_artifacts.validate_representation(None, current.action_representation, allow_legacy_assets=True)
    metadata = b1k_artifacts.norm_metadata(current.action_representation)
    b1k_artifacts.save_metadata(tmp_path / "org/demos", metadata)
    loaded = factory().create(tmp_path, model_config())
    assert loaded.norm_stats_metadata == metadata
    b1k_artifacts.validate_representation(loaded.norm_stats_metadata, current.action_representation)
    with pytest.raises(ValueError, match="incompatible action representation"):
        b1k_artifacts.validate_representation(metadata, legacy.action_representation, allow_legacy_assets=True)
    assert current.action_representation != legacy.action_representation


def test_representation_records_this_checkouts_convention(tmp_path):
    # All delta groups are matched to proprio slices in order (no explicit `delta_state_indices` here), which is what
    # the `my` branch registers as `b1k/R1Pro-legacy-torso-delta`: its tooling can identify these artifacts.
    representation = factory().create(tmp_path, model_config()).action_representation
    assert representation["robot_type"] == ROBOT_REGISTRY["b1k/R1Pro"].robot_type
    assert representation["extra_delta_transform"] is True
    torso = [group for group in representation["actions"] if group["name"] == "torso"]
    assert torso == [{"name": "torso", "indices": [3, 4, 5, 6], "is_eef": False, "needs_delta_comp": True,
                      "delta_state_indices": None}]
    assert b1k_artifacts.state_dimension(ROBOT_REGISTRY["b1k/R1Pro"]) == 23


def test_dataset_prompt_and_token_settings_survive_checkpoint_assets(tmp_path, monkeypatch):
    root = tmp_path / "demos"
    _write_synthetic_root(root, chunks=[2])
    (root / "meta/tasks.jsonl").write_text(
        json.dumps({"task_index": 2, "task_name": "chop_an_onion", "task": "Dice the onion."}) + "\n"
    )
    model = dataclasses.replace(model_config(), max_token_len=256)
    dc = factory(
        dataset_root=str(root),
        task_names=["chop_an_onion"],
        prompt_source="task_description",
        base_config=config.DataConfig(prompt_from_task=True),
    ).create(tmp_path / "assets", model)
    dataset = data_loader.create_b1k_dataset(dc, model.action_horizon, model)
    dc = dataclasses.replace(dc, norm_stats={}, inference_metadata=dataset.inference_metadata)

    class Loader:
        def data_config(self):
            return dc

    class Manager:
        def save(self, step, items):
            items["assets"](epath.Path(tmp_path / "checkpoint/assets"))

    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ({}, {}))
    checkpoints.save_state(Manager(), object(), Loader(), 1)
    path = tmp_path / "checkpoint/assets" / dc.asset_id
    metadata = b1k_artifacts.load_metadata(path)
    assert metadata["task_prompts"] == {"chop_an_onion": "Dice the onion."}
    assert metadata["model"]["max_token_len"] == 256
    args = serve_b1k.Args(
        robot="b1k/R1Pro", task="b1k/chop_an_onion", policy=serve_b1k.Checkpoint("pi05_b1k", "unused")
    )
    assert serve_b1k.resolve_prompt(args, path)[0] == "Dice the onion."
    restored = b1k_artifacts.restore_model_config(model_config(), metadata)
    assert restored.max_token_len == 256
    metadata["task_prompts"] = {"custom_task": "Do the custom action."}
    b1k_artifacts.save_metadata(path, metadata)
    assert (
        serve_b1k.resolve_prompt(dataclasses.replace(args, task="b1k/custom_task"), path)[0] == "Do the custom action."
    )
    assert serve_b1k.resolve_prompt(dataclasses.replace(args, text_prompt="Override"), path)[0] == "Override"
    assert serve_b1k.resolve_prompt(dataclasses.replace(args, prompt_source="task_name"), path)[0] == "chop_an_onion"


def test_real_orbax_asset_save_and_resume_validation(tmp_path, monkeypatch):
    from openpi.shared import normalize

    dc = factory().create(tmp_path / "config", model_config())
    stats = {"state": normalize.NormStats(mean=np.zeros(23), std=np.ones(23))}
    metadata = b1k_artifacts.inference_metadata(dc, model_config(), {"turning_on_radio": "Switch on this radio."})
    dc = dataclasses.replace(dc, norm_stats=stats, inference_metadata=metadata)
    loader = SimpleNamespace(data_config=lambda: dc)
    monkeypatch.setattr(checkpoints, "_split_params", lambda state: ({"step": np.asarray(1)}, {"kernel": np.ones(2)}))
    monkeypatch.setattr(checkpoints, "_merge_params", lambda state, params: (state, params))
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "orbax", keep_period=None, overwrite=False, resume=False
    )
    try:
        checkpoints.save_state(manager, object(), loader, 1)
        manager.wait_until_finished()
        assets = tmp_path / "orbax/1/assets" / dc.asset_id
        assert b1k_artifacts.load_metadata(assets) == metadata
        restored, params = checkpoints.restore_state(manager, object(), loader)
        assert restored["step"] == 1
        np.testing.assert_array_equal(params["params"]["kernel"], np.ones(2))
    finally:
        manager.close()


def test_serve_restores_prompt_and_budget_before_loading_policy(tmp_path, monkeypatch):
    from openpi import transforms
    from openpi.policies import policy_config

    model = dataclasses.replace(model_config(), max_token_len=256)
    dc = factory().create(tmp_path / "source", model)
    metadata = b1k_artifacts.inference_metadata(dc, model, {"custom_task": "Custom instruction."})
    checkpoint = tmp_path / "checkpoint"
    assets = checkpoint / "assets/org/demos"
    assets.mkdir(parents=True)
    from openpi.shared import normalize

    normalize.save(assets, {"state": transforms.NormStats(mean=np.zeros(23), std=np.ones(23))})
    b1k_artifacts.save_metadata(assets, metadata)
    seen = {}

    def create_policy(train_config, directory, **kwargs):
        seen["model"] = train_config.model
        seen.update(kwargs)
        return SimpleNamespace(metadata={})

    monkeypatch.setattr(policy_config, "create_trained_policy", create_policy)
    monkeypatch.setattr(serve_b1k.socket, "gethostbyname", lambda name: "127.0.0.1")
    monkeypatch.setattr(serve_b1k.websocket_b1k_server.WebsocketPolicyServer, "serve_forever", lambda self: None)
    serve_b1k.main(
        serve_b1k.Args(
            robot="b1k/R1Pro",
            task="b1k/custom_task",
            repo_id="org/demos",
            policy=serve_b1k.Checkpoint("pi05_b1k", str(checkpoint)),
        )
    )
    assert seen["model"].max_token_len == 256
    assert seen["default_prompt"] == "Custom instruction."
    assert seen["b1k_metadata"] == metadata


def test_long_prompt_budget_restored_and_checked(tmp_path):
    dc = factory().create(tmp_path, model_config())
    long_prompt = "Prepare the pizza with all the requested toppings. " * 35
    model = dataclasses.replace(model_config(), max_token_len=1024)
    metadata = b1k_artifacts.inference_metadata(dc, model, {"make_pizza": long_prompt})
    restored = b1k_artifacts.restore_model_config(model_config(), metadata)
    state_dim = b1k_artifacts.state_dimension(ROBOT_REGISTRY["b1k/R1Pro"])
    assert state_dim == 23
    b1k_dataset.check_prompt_token_lengths({0: long_prompt}, restored, state_dim=state_dim)
    with pytest.raises(ValueError, match="would be truncated"):
        b1k_dataset.check_prompt_token_lengths({0: long_prompt}, model_config(), state_dim=state_dim)
    assert b1k_artifacts.restore_model_config(model_config(), None, max_token_len=1024).max_token_len == 1024


def test_resume_rejects_representation_or_conditioning_change_before_restore(tmp_path):
    dc = factory().create(tmp_path, model_config())
    metadata = b1k_artifacts.inference_metadata(dc, model_config(), {"x": "train prompt"})
    dc = dataclasses.replace(dc, inference_metadata=metadata)
    path = tmp_path / "10/assets" / dc.asset_id
    b1k_artifacts.save_metadata(path, {**metadata, "task_prompts": {"x": "different prompt"}})
    manager = SimpleNamespace(directory=epath.Path(tmp_path), latest_step=lambda: 10)
    loader = SimpleNamespace(data_config=lambda: dc)
    with pytest.raises(ValueError, match="task_prompts differs"):
        checkpoints.restore_state(manager, object(), loader)
    legacy = other_convention_factory().create(tmp_path / "legacy", model_config())
    b1k_artifacts.save_metadata(path, b1k_artifacts.norm_metadata(legacy.action_representation))
    with pytest.raises(ValueError, match="incompatible action representation"):
        checkpoints.restore_state(manager, object(), loader)


def test_resume_rejects_changed_stats_despite_identical_metadata(tmp_path):
    from openpi.shared import normalize

    dc = factory().create(tmp_path, model_config())
    metadata = b1k_artifacts.inference_metadata(dc, model_config(), {"x": "same prompt"})
    saved = {"state": normalize.NormStats(mean=np.zeros(23), std=np.ones(23))}
    changed = {"state": normalize.NormStats(mean=np.ones(23) * 100, std=np.ones(23) * 2)}
    path = tmp_path / "10/assets" / dc.asset_id
    normalize.save(path, saved)
    b1k_artifacts.save_metadata(path, metadata)
    dc = dataclasses.replace(dc, inference_metadata=metadata, norm_stats=changed)
    manager = SimpleNamespace(directory=epath.Path(tmp_path), latest_step=lambda: 10)
    with pytest.raises(ValueError, match="normalization statistics differ"):
        checkpoints.restore_state(manager, object(), SimpleNamespace(data_config=lambda: dc))
    b1k_artifacts.validate_norm_stats(saved, saved)


@pytest.mark.parametrize("metadata", [{}, {"schema_version": 999}, {"schema_version": 1}])
def test_invalid_metadata_is_not_treated_as_legacy(tmp_path, metadata):
    (tmp_path / b1k_artifacts.METADATA_FILENAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata|representation"):
        b1k_artifacts.load_metadata(tmp_path)
