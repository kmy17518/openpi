import dataclasses
import logging
import pathlib
import socket

import tyro

from openpi.configs.tasks import TASK_REGISTRY
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_b1k_server
import openpi.shared.download as _download
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
import openpi.shared.normalize as _normalize
from openpi.training import config as _config


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
    control_mode: str = "receding_horizon"
    # Number of actions to execute before replanning.
    action_horizon: int = 16
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False


def resolve_norm_stats(config: _config.TrainConfig, checkpoint_dir: str) -> dict[str, _normalize.NormStats] | None:
    """Norm stats to serve with.

    Returns None when the checkpoint holds the norm stats the config asks for (`assets/<asset_id>`, which
    `create_trained_policy` then loads itself). Otherwise falls back to the checkpoint's only norm-stats file --
    a checkpoint written by train_b1k.py saves exactly one, the one training used -- so a checkpoint trained with
    `--data.task-names` also serves without `--task-names`. Ambiguous or empty assets raise with what is there.
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    asset_id = data_config.asset_id if isinstance(data_config.asset_id, str) else data_config.asset_id[0]
    # maybe_download yields a local directory (remote checkpoints are cached locally first).
    assets_dir = pathlib.Path(_download.maybe_download(str(checkpoint_dir))) / "assets"
    requested = assets_dir / asset_id
    if (requested / "norm_stats.json").exists():
        return None
    candidates = sorted(path.parent for path in assets_dir.rglob("norm_stats.json")) if assets_dir.is_dir() else []
    found = [path.relative_to(assets_dir).as_posix() for path in candidates]
    if len(candidates) == 1:
        logging.warning(
            "Norm stats not found at %s; serving the checkpoint's only norm stats (%s). Pass --repo-id / --task-names "
            "matching the training run to silence this.",
            requested,
            found[0],
        )
        return _normalize.load(candidates[0])
    raise FileNotFoundError(
        f"Norm stats not found at {requested}; the checkpoint holds {found or 'no norm stats'}. Pass --repo-id and, "
        "for a checkpoint trained on a task subset, --task-names matching the training run."
    )


def main(args: Args) -> None:
    # Load task from registry
    task_bucket, task_name = args.task.split("/")
    task_prompt = TASK_REGISTRY[task_bucket][task_name]
    # log the prompt used
    logging.info(f"Using robot: {args.robot}, prompt: {task_prompt}")

    # Load training config and override request-specific fields.
    config = _config.get_config(args.policy.config)
    norm_stats_repo_id = args.repo_id or args.task
    logging.info("Using norm stats for repo: %s (task subset: %s)", norm_stats_repo_id, args.task_names)
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data, repo_id=norm_stats_repo_id, robot_config_name=args.robot, task_names=args.task_names
        ),
    )

    policy = _policy_config.create_trained_policy(
        config, args.policy.dir, default_prompt=task_prompt, norm_stats=resolve_norm_stats(config, args.policy.dir)
    )
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    policy = B1KPolicyWrapper(
        policy=policy,
        robot=args.robot,
        text_prompt=task_prompt,
        control_mode=args.control_mode,
        action_horizon=args.action_horizon,
        max_len=config.model.action_horizon,
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
