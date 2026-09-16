import dataclasses

import pytest

from scripts.b1k import benchmark_single_gpu as benchmark


def test_benchmark_defaults_keep_physical_batch_and_production_model(tmp_path):
    args = benchmark.parse_args(["--output", str(tmp_path / "result.json")])
    config = benchmark.make_config(args)
    assert config.batch_size == 64
    assert config.grad_accum_steps == config.fsdp_devices == 1
    assert config.model.max_token_len == 112
    assert config.model.action_horizon == 32
    assert config.ema_decay == 0.99
    assert config.data.task_names == ("turning_on_radio",)
    assert config.data.prompt_source == "task_name"
    assert config.model.paligemma_variant == "gemma_2b"


@pytest.mark.parametrize("arguments", [
    ["--batch", "0"],
    ["--steps", "0"],
    ["--workers", "25"],
    ["--workers", "-1"],
    ["--prefetch", "-1"],
    ["--weights", "/tmp/absent-benchmark-weights"],
])
def test_invalid_or_unavailable_benchmark_inputs_fail_before_launch(tmp_path, arguments):
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--output", str(tmp_path / "result.json"), *arguments])


def test_benchmark_output_cannot_escape_tmp():
    with pytest.raises(SystemExit):
        benchmark.parse_args(["--output", "/var/result.json"])


def test_radio_budget_uses_actual_tokenizer_when_cached():
    from openpi.configs.robots.b1k import R1Pro
    from openpi.shared.download import get_cache_dir
    from openpi.training import b1k_artifacts
    from openpi.training import b1k_dataset
    from openpi.training import config

    if not (get_cache_dir() / "big_vision/paligemma_tokenizer.model").is_file():
        pytest.skip("Tokenizer not cached; unit test does not download")
    state_dim = b1k_artifacts.state_dimension(R1Pro)
    model = dataclasses.replace(config.get_config("pi05_b1k").model, max_token_len=112)
    b1k_dataset.check_prompt_token_lengths({0: "turning_on_radio"}, model, state_dim=state_dim)
    with pytest.raises(ValueError, match="106"):
        b1k_dataset.check_prompt_token_lengths(
            {0: "turning_on_radio"}, dataclasses.replace(model, max_token_len=105), state_dim=state_dim
        )
