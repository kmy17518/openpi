"""Goal-image conditioning (PI-SLOT / PI-ROLE): input adapter, config validation, prompt scaffolds, model prefix."""

import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms
from openpi.configs.robots.base_config import ROBOT_REGISTRY
import openpi.configs.robots.b1k  # noqa: F401  (registers b1k/R1Pro)
from openpi.models import model as _model
import openpi.models.pi0_config as _pi0_config
from openpi.policies import b1k_policy
from openpi.training import config as _config


def _example(goal_views=()):
    data = b1k_policy.make_b1k_example()
    data["observation/state"] = np.random.rand(61)
    for view in goal_views:
        data[f"observation/{view}"] = np.random.randint(256, size=(240, 240, 3), dtype=np.uint8)
    return data


def test_b1k_inputs_emit_goal_slots_after_the_cameras_in_fixed_order():
    robot = ROBOT_REGISTRY["b1k/R1Pro"]
    assert list(robot.goals) == ["goal_image_0", "goal_image_1", "goal_image_2"]
    assert robot.goals["goal_image_0"].obs_key == "goal::robot_r1::robot_r1:zed_link:Camera:0::rgb"
    assert robot.goals["goal_image_0"].dataset_key == "observation.goal_rgb.zed_link_camera_0"
    plain = b1k_policy.B1KInputs(robot_config=robot, model_type=_model.ModelType.PI05)(_example())
    assert list(plain["image"]) == list(_model.IMAGE_KEYS)
    conditioned = b1k_policy.B1KInputs(robot_config=robot, model_type=_model.ModelType.PI05, goal_views=("goal_image_0",))(
        _example(("goal_image_0",))
    )
    assert list(conditioned["image"]) == [*_model.IMAGE_KEYS, "goal_0_rgb"]
    assert conditioned["image_mask"]["goal_0_rgb"] == np.True_ and conditioned["image"]["goal_0_rgb"].shape == (240, 240, 3)
    two = b1k_policy.B1KInputs(robot_config=robot, model_type=_model.ModelType.PI05, goal_views=("goal_image_0", "goal_image_2"))(
        _example(("goal_image_0", "goal_image_2"))
    )
    assert list(two["image"])[-2:] == ["goal_0_rgb", "goal_1_rgb"]  # slot order follows the configured view order
    with pytest.raises(ValueError, match="pi0-FAST"):
        b1k_policy.B1KInputs(robot_config=robot, model_type=_model.ModelType.PI0_FAST, goal_views=("goal_image_0",))(
            _example(("goal_image_0",))
        )


def test_pi0_config_goal_slots_and_role_validation():
    config = _pi0_config.Pi0Config(pi05=True, goal_image_keys=("goal_0_rgb",))
    spec, _ = config.inputs_spec(batch_size=2)
    assert list(spec.images) == [*_model.IMAGE_KEYS, "goal_0_rgb"] and list(spec.image_masks) == list(spec.images)
    plain, _ = _pi0_config.Pi0Config(pi05=True).inputs_spec()
    assert list(plain.images) == list(_model.IMAGE_KEYS)
    with pytest.raises(ValueError, match="prefix"):
        _pi0_config.Pi0Config(goal_image_keys=("goal_1_rgb",))
    with pytest.raises(ValueError, match="prefix"):
        _pi0_config.Pi0Config(goal_image_keys=("base_0_rgb",))
    with pytest.raises(ValueError, match="requires at least one goal image key"):
        _pi0_config.Pi0Config(goal_role_embedding=True)
    assert _pi0_config.Pi0Config(goal_image_keys=["goal_0_rgb", "goal_1_rgb"]).goal_image_keys == ("goal_0_rgb", "goal_1_rgb")


def test_prompt_scaffolds():
    goal_only = transforms.ComposePrompt(_config.GOAL_PROMPT_SCAFFOLD, include_task=False)
    assert goal_only({"prompt": "turning_on_radio"})["prompt"] == "Reach the configuration shown in the goal image."
    assert goal_only({})["prompt"] == "Reach the configuration shown in the goal image."
    both = transforms.ComposePrompt(_config.GOAL_PROMPT_SCAFFOLD, include_task=True)
    assert both({"prompt": np.asarray("turning_on_radio")})["prompt"] == (
        "Reach the configuration shown in the goal image. turning_on_radio"
    )
    with pytest.raises(ValueError, match="needs the task prompt"):
        both({})
    assert transforms.ComposePrompt(_config.PLAIN_PROMPT_SCAFFOLD, include_task=False)({"prompt": "x"})["prompt"] == "Perform the task."


def _data_config(tmp_path, model_config, **kwargs):
    factory = _config.LeRobotB1KDataConfig(
        repo_id="fake",
        base_config=_config.DataConfig(prompt_from_task=True),
        robot_config_name="b1k/R1Pro",
        **kwargs,
    )
    return factory.create(tmp_path, model_config)


def test_b1k_data_config_validates_goal_views_against_the_model_and_regime(tmp_path):
    dummy = dict(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy")
    with pytest.raises(ValueError, match="need model goal_image_keys"):
        _data_config(tmp_path, _pi0_config.Pi0Config(**dummy), goal_views=("goal_image_0",))
    with pytest.raises(ValueError, match="need model goal_image_keys"):
        _data_config(tmp_path, _pi0_config.Pi0Config(**dummy, goal_image_keys=("goal_0_rgb",)))
    with pytest.raises(ValueError, match="distinct entries"):
        _data_config(tmp_path, _pi0_config.Pi0Config(**dummy), goal_views=("goal_image_9",))
    with pytest.raises(ValueError, match="requires goal_views"):
        _data_config(tmp_path, _pi0_config.Pi0Config(**dummy), conditioning_regime="image")
    with pytest.raises(ValueError, match="excludes goal_views"):
        _data_config(
            tmp_path, _pi0_config.Pi0Config(**dummy, goal_image_keys=("goal_0_rgb",)), goal_views=("goal_image_0",),
            conditioning_regime="language",
        )
    config = _data_config(
        tmp_path, _pi0_config.Pi0Config(**dummy, goal_image_keys=("goal_0_rgb",)), goal_views=("goal_image_0",),
        conditioning_regime="image",
    )
    assert config.goal_views == ("goal_image_0",) and config.conditioning_regime == "image"
    assert config.dataset_kwargs["video_keys"][-1] == "observation.goal_rgb.zed_link_camera_0"
    repack = config.repack_transforms.inputs[0]
    assert repack.structure["observation/goal_image_0"] == "observation.goal_rgb.zed_link_camera_0"
    scaffold = [t for t in config.data_transforms.inputs if isinstance(t, transforms.ComposePrompt)]
    assert len(scaffold) == 1 and scaffold[0].include_task is False and scaffold[0].scaffold == _config.GOAL_PROMPT_SCAFFOLD
    # image regime: the task prompt never reaches the tokenizer
    sample = {"observation.rgb.zed_link_camera_0": np.zeros((240, 240, 3), np.uint8),
              "observation.rgb.left_realsense_link_camera_0": np.zeros((240, 240, 3), np.uint8),
              "observation.rgb.right_realsense_link_camera_0": np.zeros((240, 240, 3), np.uint8),
              "observation.goal_rgb.zed_link_camera_0": np.full((240, 240, 3), 7, np.uint8),
              "observation.state": np.zeros(61, np.float32), "action": np.zeros((32, 23), np.float32),
              "prompt": "turning_on_radio-pick_up_radio"}
    out = sample
    for transform in [*config.repack_transforms.inputs, *config.data_transforms.inputs]:
        out = transform(out)
    assert out["prompt"] == _config.GOAL_PROMPT_SCAFFOLD and out["image"]["goal_0_rgb"][0, 0, 0] == 7
    language = _data_config(
        tmp_path, _pi0_config.Pi0Config(**dummy, goal_image_keys=("goal_0_rgb",)), goal_views=("goal_image_0",),
        conditioning_regime="image_language",
    )
    out = sample
    for transform in [*language.repack_transforms.inputs, *language.data_transforms.inputs]:
        out = transform(out)
    assert out["prompt"] == f"{_config.GOAL_PROMPT_SCAFFOLD} turning_on_radio-pick_up_radio"
    none = _data_config(tmp_path, _pi0_config.Pi0Config(**dummy), conditioning_regime="none")
    out = sample
    for transform in [*none.repack_transforms.inputs, *none.data_transforms.inputs]:
        out = transform(out)
    assert out["prompt"] == _config.PLAIN_PROMPT_SCAFFOLD and "goal_0_rgb" not in out["image"]


def test_pi0_slot_adds_no_parameters_and_role_adds_exactly_one():
    dummy = dict(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4, action_dim=8,
                 max_token_len=16)
    plain = nnx.eval_shape(_pi0_config.Pi0Config(**dummy).create, jax.random.key(0))
    slot = nnx.eval_shape(_pi0_config.Pi0Config(**dummy, goal_image_keys=("goal_0_rgb",)).create, jax.random.key(0))
    role = nnx.eval_shape(
        _pi0_config.Pi0Config(**dummy, goal_image_keys=("goal_0_rgb",), goal_role_embedding=True).create, jax.random.key(0)
    )
    plain_params = nnx.state(plain, nnx.Param).flat_state()
    slot_params = nnx.state(slot, nnx.Param).flat_state()
    role_params = nnx.state(role, nnx.Param).flat_state()
    # PI-SLOT adds no parameters (a pretrained checkpoint loads as-is); PI-ROLE adds exactly the role vector
    assert set(slot_params) == set(plain_params)
    assert set(role_params) - set(plain_params) == {("goal_role_embed",)}
    assert slot.image_keys == (*_model.IMAGE_KEYS, "goal_0_rgb") and plain.goal_role_embed is None
    assert plain.image_keys == _model.IMAGE_KEYS


def test_goal_role_embedding_is_zero_initialised_and_changes_goal_tokens_only():
    dummy = dict(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy", action_horizon=4, action_dim=8,
                 max_token_len=16)
    config = _pi0_config.Pi0Config(**dummy, goal_image_keys=("goal_0_rgb",), goal_role_embedding=True)
    model = config.create(jax.random.key(0))
    assert not jnp.any(model.goal_role_embed.value)
    obs = config.fake_obs(batch_size=1)
    before, mask, ar_mask = model.embed_prefix(obs)
    # one 224 px image = (224 / 14)^2 = 256 SigLIP tokens: three cameras, one goal slot, then the prompt tokens
    assert before.shape[1] == 4 * 256 + config.max_token_len == mask.shape[1] == ar_mask.shape[0]
    plain = _pi0_config.Pi0Config(**dummy).fake_obs(batch_size=1)
    assert 4 * 256 - 3 * 256 == 256 and len(plain.images) == 3 and len(obs.images) == 4
    model.goal_role_embed.value = jnp.ones_like(model.goal_role_embed.value)
    after, _, _ = model.embed_prefix(obs)
    delta = jnp.abs(after - before).max(axis=-1)[0]
    assert not jnp.any(delta[: 3 * 256]), "camera tokens must not change"
    assert jnp.all(delta[3 * 256 : 4 * 256] > 0), "every goal token carries the role vector"
    assert not jnp.any(delta[4 * 256 :]), "language tokens must not change"
