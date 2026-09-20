import dataclasses
import logging
import pathlib
import socket

import tyro

from openpi.configs.robots import ROBOT_REGISTRY
from openpi.configs.tasks import TASK_REGISTRY
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_b1k_server
import openpi.shared.download as _download
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
import openpi.shared.normalize as _normalize
from openpi.training import config as _config
import openpi.training.b1k_artifacts as _b1k_artifacts
import openpi.training.b1k_dataset as _b1k_dataset


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    robot: str
    task: str
    # Specifies how to load the policy.
    policy: Checkpoint

    # LeRobot repo id whose checkpoint assets contain the norm stats.
    repo_id: str | None = None
    # Task subset the checkpoint was trained on (`--data.task-names` of train_b1k.py); selects that subset's norm
    # stats (`<repo_id>/task_subsets/<key>`) inside the checkpoint assets. When the requested norm stats are not in
    # the checkpoint but it holds exactly one norm-stats file (the one training used), that file is served instead.
    task_names: list[str] | None = None
    # Which text of the task to prompt the policy with: `task_name` (the snake_case id of `--task`, e.g.
    # `turning_on_radio`) or `task_description` (its natural-language instruction from the task registry,
    # `configs/tasks/b1k.py`). Default: what the checkpoint was trained with (`--data.prompt-source` of train_b1k.py,
    # recorded in its assets); checkpoints that predate that record are served with `task_description`, as before.
    prompt_source: _b1k_dataset.PromptSource | None = None
    # Prompt the policy with exactly this text instead (overrides --prompt-source and the task registry).
    text_prompt: str | None = None
    # Explicit budget for older checkpoints; new checkpoints restore their training value.
    max_token_len: int | None = None
    # Goal-conditioned checkpoints: fixed goal image per robot-config goal view, `goal_image_0=PATH` (PNG/JPEG, RGB),
    # used when a request carries no goal::<camera key> image; repeat per view.
    goal_image: list[str] | None = None
    # Permit unversioned artifacts only after verifying their action representation matches --robot.
    allow_legacy_assets: bool = False
    control_mode: str = "receding_horizon"
    # Required client chunk size n; the model predicts m=config.model.action_horizon >= n.
    action_horizon: int = 16
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False


# Served text for checkpoints whose assets do not record a prompt source (trained before it existed): this script
# has always prompted with the task description, so keep doing that; pass --prompt-source task_name to serve such a
# checkpoint with the text it was actually trained on.
LEGACY_PROMPT_SOURCE: _b1k_dataset.PromptSource = "task_description"


def resolve_assets_dir(config: _config.TrainConfig, checkpoint_dir: str) -> pathlib.Path:
    """The checkpoint's assets directory (norm stats + recorded prompt source) to serve with.

    `assets/<asset_id>` when the checkpoint holds it. Otherwise falls back to the checkpoint's only norm-stats
    directory -- a checkpoint written by train_b1k.py saves exactly one, the one training used -- so a checkpoint
    trained with `--data.task-names` also serves without `--task-names`. Ambiguous or empty assets raise with what
    is there.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    asset_id = data_config.asset_id if isinstance(data_config.asset_id, str) else data_config.asset_id[0]
    # maybe_download yields a local directory (remote checkpoints are cached locally first).
    assets_dir = pathlib.Path(_download.maybe_download(str(checkpoint_dir))) / "assets"
    requested = assets_dir / asset_id
    if (requested / "norm_stats.json").exists():
        return requested
    candidates = sorted(path.parent for path in assets_dir.rglob("norm_stats.json")) if assets_dir.is_dir() else []
    found = [path.relative_to(assets_dir).as_posix() for path in candidates]
    if len(candidates) == 1:
        logging.warning(
            "Norm stats not found at %s; serving the checkpoint's only norm stats (%s). Pass --repo-id / --task-names "
            "matching the training run to silence this.",
            requested,
            found[0],
        )
        return candidates[0]
    raise FileNotFoundError(
        f"Norm stats not found at {requested}; the checkpoint holds {found or 'no norm stats'}. Pass --repo-id and, "
        "for a checkpoint trained on a task subset, --task-names matching the training run."
    )


def resolve_prompt(args: Args, assets_dir: pathlib.Path) -> tuple[str, str]:
    """(prompt text, how it was chosen) for `--task <bucket>/<task_name>`.

    Precedence: --text-prompt, --prompt-source, the prompt source recorded in the checkpoint assets, then
    LEGACY_PROMPT_SOURCE.
    """
    task_bucket, task_name = args.task.split("/")
    if args.text_prompt is not None:
        return args.text_prompt, "--text-prompt"
    metadata = _b1k_artifacts.load_metadata(assets_dir)
    if metadata is not None and args.prompt_source is None:
        prompts = metadata.get("task_prompts")
        if not isinstance(prompts, dict) or task_name not in prompts or not isinstance(prompts[task_name], str):
            raise ValueError(f"Checkpoint has no recorded prompt for {task_name!r}; pass an explicit prompt override")
        return prompts[task_name], f"exact training prompt from {assets_dir / _b1k_artifacts.METADATA_FILENAME}"
    if args.prompt_source is not None:
        prompt_source, origin = args.prompt_source, "--prompt-source"
    elif (recorded := _b1k_dataset.load_prompt_source(assets_dir)) is not None:
        prompt_source, origin = (
            recorded,
            f"recorded in the checkpoint ({assets_dir / _b1k_dataset.PROMPT_SOURCE_FILENAME})",
        )
    else:
        prompt_source, origin = LEGACY_PROMPT_SOURCE, "default for checkpoints without a recorded prompt source"
    if prompt_source == "task_name":
        return task_name, f"task_name ({origin})"
    tasks = TASK_REGISTRY[task_bucket]
    if task_name not in tasks:
        raise KeyError(
            f"No description for task {task_name!r} in TASK_REGISTRY[{task_bucket!r}] "
            f"(src/openpi/configs/tasks/{task_bucket}.py); add it, or pass --prompt-source task_name / --text-prompt."
        )
    return tasks[task_name], f"task_description ({origin})"


def main(args: Args) -> None:
    # Load training config and override request-specific fields.
    config = _config.get_config(args.policy.config)
    if args.control_mode != "receding_horizon":
        raise ValueError("BEHAVIOR chunk serving requires --control-mode receding_horizon")
    if isinstance(args.action_horizon, bool) or not isinstance(args.action_horizon, int):
        raise ValueError("--action-horizon must be a positive integer")
    norm_stats_repo_id = args.repo_id or args.task
    logging.info("Using norm stats for repo: %s (task subset: %s)", norm_stats_repo_id, args.task_names)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            repo_id=norm_stats_repo_id,
            robot_config_name=args.robot,
            task_names=args.task_names,
            allow_legacy_assets=args.allow_legacy_assets,
        ),
    )

    assets_dir = resolve_assets_dir(config, args.policy.dir)
    metadata = _b1k_artifacts.load_metadata(assets_dir)
    config = dataclasses.replace(
        config, model=_b1k_artifacts.restore_model_config(config.model, metadata, max_token_len=args.max_token_len)
    )
    conditioning = (metadata or {}).get("conditioning")
    if conditioning is not None:
        # Restore the goal views / prompt regime the checkpoint trained with (the model's goal slots come from
        # restore_model_config above); the data config validates that both agree.
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                goal_views=tuple(conditioning.get("goal_views") or ()),
                conditioning_regime=conditioning.get("regime"),
            ),
        )
    if not 1 <= args.action_horizon <= config.model.action_horizon:
        raise ValueError(f"Require 1 <= --action-horizon <= prediction horizon {config.model.action_horizon}")
    data_config = config.data.create(config.assets_dirs, config.model)
    _b1k_artifacts.validate_representation(
        metadata,
        data_config.action_representation,
        allow_legacy_assets=args.allow_legacy_assets,
        context="Serving checkpoint",
    )
    task_prompt, prompt_origin = resolve_prompt(args, assets_dir)
    _b1k_dataset.check_prompt_token_lengths(
        {0: task_prompt}, config.model, state_dim=_b1k_artifacts.state_dimension(ROBOT_REGISTRY[args.robot])
    )
    logging.info("Using robot: %s, prompt: %r [%s]", args.robot, task_prompt, prompt_origin)

    policy = _policy_config.create_trained_policy(
        config,
        args.policy.dir,
        default_prompt=task_prompt,
        norm_stats=_normalize.load(assets_dir),
        b1k_metadata=metadata,
        max_token_len=args.max_token_len,
    )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    fixed_goals = {}
    for spec in args.goal_image or []:
        view, _, path = spec.partition("=")
        if not path or view not in ROBOT_REGISTRY[args.robot].goals:
            raise ValueError(f"--goal-image expects <goal view>=PATH with a robot-config goal view, got {spec!r}")
        import av

        with av.open(path) as container:
            fixed_goals[view] = next(container.decode(video=0)).to_ndarray(format="rgb24")
    if data_config.goal_views:
        logging.info("Goal-conditioned policy: views %s (regime %s), fixed goal images for %s",
                     list(data_config.goal_views), data_config.conditioning_regime, sorted(fixed_goals))

    policy = B1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=task_prompt,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
        goal_views=tuple(data_config.goal_views),
        fixed_goals=fixed_goals,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_b1k_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
