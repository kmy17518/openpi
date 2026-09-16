"""Run pi0.5 with explicit resource limits, health status and synchronous checkpoint staging."""

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, allow_nan=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
        tmp = Path(stream.name)
    os.replace(tmp, path)


def parse_cpus(text: str) -> set[int]:
    result = set()
    for item in text.split(","):
        bounds = [int(part) for part in item.split("-")]
        if len(bounds) == 1:
            result.add(bounds[0])
        elif len(bounds) == 2 and 0 <= bounds[0] <= bounds[1]:
            result.update(range(bounds[0], bounds[1] + 1))
        else:
            raise ValueError("Invalid CPU affinity")
    if not result or min(result) < 0 or len(result) > 30:
        raise ValueError("Reserve between1 and30 CPUs")
    return result


class RunObserver:
    def __init__(self, settings: dict, stage):
        self.settings = settings
        self.stage = stage
        self.started = time.time()
        self.status = {"state": "starting", "pid": os.getpid(), "step": 0, "started_at": self.started}
        self.last_write = 0.0
        self.update("starting")

    def update(self, state: str, **values) -> None:
        self.status.update(values, state=state, updated_at=time.time())
        atomic_json(Path(self.settings["status_file"]), self.status)
        self.last_write = time.monotonic()

    def on_step(self, step: int, info: dict, seconds: float) -> None:
        values = {key: float(value) for key, value in info.items()}
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"Nonfinite training metrics at step{step}")
        self.status.update(step=step, metrics=values, step_seconds=seconds)
        if step <= 5 or step % self.settings.get("status_interval", 10) == 0 or time.monotonic() - self.last_write > 30:
            self.update("training")

    def should_save(self, step: int) -> bool:
        return (
            step in (self.settings["first_checkpoint_step"], self.settings["max_steps"])
            or step % self.settings["save_interval"] == 0
        )

    def on_checkpoint(self, path: Path, step: int) -> None:
        self.update("staging_checkpoint", checkpoint_step=step)
        self.stage(path, Path(self.settings["staging_dir"]), step)
        self.update("training", staged_step=step)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.run_config.read_text())
    allowed = parse_cpus(settings["cpu_affinity"])
    if not allowed <= os.sched_getaffinity(0):
        raise ValueError("Requested CPUs are not available")
    os.sched_setaffinity(0, allowed)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(settings["gpu"]):
        raise ValueError("CUDA_VISIBLE_DEVICES must select exactly the reserved GPU")
    if os.environ.get("WANDB_MODE") != "online" or os.environ.get("WANDB_RUN_ID") != settings["wandb_run_id"]:
        raise ValueError("Online W&B and the recorded run ID are required")
    if settings["max_steps"] != 300_000 or settings["max_to_keep"] != 3:
        raise ValueError("Run requires300k steps and exactly3 retained local checkpoints")
    if settings["grad_accum_steps"] != 1:
        raise ValueError("This run uses the measured physical batch without gradient accumulation")

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    import wandb

    from openpi.training import config as training_config
    from scripts.b1k import run_lifecycle
    from scripts.b1k import train_b1k
    from scripts.b1k.hf_single_writer_checkpoint_uploader import stage_local_checkpoint

    observer = RunObserver(settings, stage_local_checkpoint)
    try:
        base = training_config.get_config("pi05_b1k")
        model = dataclasses.replace(
            base.model, max_token_len=settings["max_token_len"], remat_policy=settings["remat_policy"]
        )
        config = dataclasses.replace(
            base,
            model=model,
            data=dataclasses.replace(
                base.data,
                repo_id="behavior-1k/2026-challenge-demos",
                dataset_root=settings["dataset_root"],
                task_names=("turning_on_radio",),
                prompt_source="task_name",
                base_config=dataclasses.replace(
                    base.data.base_config,
                    dataset_kwargs={**base.data.base_config.dataset_kwargs, "video_backend": "pyav"},
                ),
            ),
            exp_name=settings["exp_name"],
            project_name=settings["wandb_project"],
            batch_size=settings["batch_size"],
            grad_accum_steps=1,
            fsdp_devices=1,
            num_workers=settings["num_workers"],
            prefetch_batches=settings["prefetch_batches"],
            num_train_steps=settings["max_steps"],
            save_interval=settings["save_interval"],
            keep_period=None,
            max_to_keep=3,
            log_interval=settings.get("log_interval", 10),
            val_log_interval=None,
            wandb_enabled=True,
            resume=args.resume,
            overwrite=False,
        )
        if config.checkpoint_dir.resolve() != Path(settings["checkpoint_dir"]).resolve():
            raise ValueError("Checkpoint directory differs from the monitored run")
        if not args.resume and config.checkpoint_dir.exists() and any(config.checkpoint_dir.iterdir()):
            raise ValueError("Fresh run directory is not empty; refusing overwrite")
        with (
            run_lifecycle.exclusive_lock(f"run:{config.checkpoint_dir.resolve()}"),
            run_lifecycle.exclusive_lock(f"gpu:{settings['gpu_uuid']}"),
        ):
            run_lifecycle.prepare_generation(config.checkpoint_dir, fresh=not args.resume)
            train_b1k.main(config, observer=observer)
        wandb.finish(exit_code=0)
        observer.update("completed", step=settings["max_steps"])
    except BaseException as exc:
        observer.update("failed", error=f"{type(exc).__name__}: {exc}")
        wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()
