"""Keep a single-GPU trainer and its sole checkpoint publisher in one process lifetime."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.b1k.train_b1k_monitored import atomic_json
from scripts.b1k.train_b1k_monitored import parse_cpus


def stop(process, timeout: float = 30) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            process.wait()
            return
        time.sleep(0.1)
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def command_lines(settings: dict, config_path: Path):
    """Trainer and publisher command lines; the publisher is None when the run config has no `hf_repo`."""
    python = settings["python"]
    trainer = [python, "scripts/b1k/train_b1k_monitored.py", "--run-config", str(config_path)]
    if not settings.get("hf_repo"):
        return trainer, None
    publisher = [
        python,
        "scripts/b1k/hf_single_writer_checkpoint_uploader.py",
        "--repo-id",
        settings["hf_repo"],
        "--run-dir",
        settings["checkpoint_dir"],
        "--staging-dir",
        settings["staging_dir"],
        "--run-id",
        settings["wandb_run_id"],
        "--run-config",
        str(config_path),
        "--wandb-url",
        settings["wandb_url"],
        "--wandb-id",
        settings["wandb_run_id"],
        "--max-steps",
        str(settings["max_steps"]),
        "--eval-every",
        "10000",
        "--full-every",
        str(settings["save_interval"]),
        "--first-step",
        str(settings["first_checkpoint_step"]),
        "--storage-proof",
        settings["storage_proof"],
        "--max-staging-bytes",
        str(settings.get("max_staging_bytes", 250 * 1024**3)),
        "--max-remote-lfs-bytes",
        str(settings.get("max_remote_lfs_bytes", 600 * 1024**3)),
        "--sole-writer",
        "--poll-seconds",
        "30",
    ]
    return trainer, publisher


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    path = args.run_config.resolve()
    settings = json.loads(path.read_text())
    allowed = parse_cpus(settings["cpu_affinity"])
    if not allowed <= os.sched_getaffinity(0):
        raise ValueError("Unavailable CPU affinity")
    os.sched_setaffinity(0, allowed)
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    env = dict(os.environ)
    # Import openpi from this checkout's sources even when the shared venv's editable install points at another
    # checkout (this branch is meant to run from a git worktree next to the main one).
    env["PYTHONPATH"] = os.pathsep.join([str(repo / "src"), *filter(None, [env.get("PYTHONPATH")])])
    env.update(
        CUDA_VISIBLE_DEVICES=str(settings["gpu"]),
        WANDB_MODE="online",
        WANDB_ENTITY="kmy17518",
        WANDB_RUN_ID=settings["wandb_run_id"],
        WANDB_BASE_URL="https://api.wandb.ai",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_PYTHON_CLIENT_MEM_FRACTION=str(settings.get("memory_fraction", 0.95)),
        JAX_PLATFORMS="cuda",
        PYTHONUNBUFFERED="1",
    )
    trainer_cmd, publisher_cmd = command_lines(settings, path)
    if args.resume:
        trainer_cmd.append("--resume")
    status_path = Path(settings["supervisor_status"])
    started = time.time()
    atomic_json(status_path, {"state": "initializing", "pid": os.getpid(), "updated_at": started})
    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, handle_signal)
    processes = []
    try:
        with (
            Path(settings["uploader_log"]).open("a") as upload_log,
            Path(settings["training_log"]).open("a") as train_log,
        ):
            uploader = None
            if publisher_cmd is not None:
                initialized = subprocess.Popen(
                    [*publisher_cmd, "--init-only"],
                    env=env,
                    stdout=upload_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                processes.append(initialized)
                initialization_deadline = time.monotonic() + settings.get("initialization_timeout_seconds", 7200)
                while initialized.poll() is None:
                    if stopped:
                        raise RuntimeError("Cancelled during checkpoint recovery")
                    if time.monotonic() > initialization_deadline:
                        raise RuntimeError("Checkpoint initialization/recovery exceeded its time budget")
                    atomic_json(status_path, {"state": "initializing", "pid": os.getpid(), "updated_at": time.time()})
                    time.sleep(5)
                if initialized.returncode:
                    raise RuntimeError("Publisher ownership/storage initialization failed; trainer not started")
                processes.remove(initialized)
                uploader = subprocess.Popen(
                    publisher_cmd, env=env, stdout=upload_log, stderr=subprocess.STDOUT, start_new_session=True
                )
                processes.append(uploader)
            else:
                # hf_repo is null: no checkpoint publisher; the trainer alone decides completion.
                upload_log.write("No hf_repo in the run config: checkpoint publication disabled for this run.\n")
                upload_log.flush()
            trainer = subprocess.Popen(
                trainer_cmd, env=env, stdout=train_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(trainer)
            trainer_finished = None
            while not stopped:
                tcode = trainer.poll()
                ucode = uploader.poll() if uploader is not None else None
                atomic_json(
                    status_path,
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "trainer_pid": trainer.pid,
                        "uploader_pid": uploader.pid if uploader is not None else None,
                        "trainer_exit": tcode,
                        "uploader_exit": ucode,
                        "started_at": started,
                        "updated_at": time.time(),
                    },
                )
                if tcode not in (None, 0) or ucode not in (None, 0):
                    raise RuntimeError(f"Required process failed: trainer={tcode}, publisher={ucode}")
                if tcode == 0 and (uploader is None or ucode == 0):
                    atomic_json(status_path, {"state": "completed", "updated_at": time.time()})
                    return 0
                if uploader is not None and ucode == 0 and tcode is None:
                    status = json.loads((Path(settings["staging_dir"]) / "status.json").read_text())
                    if not status.get("done") or status.get("latest_full_step") != settings["max_steps"]:
                        raise RuntimeError("Publisher exited before verifying the final checkpoint")
                if tcode == 0:
                    trainer_finished = trainer_finished or time.monotonic()
                    if time.monotonic() - trainer_finished > 7200:
                        raise RuntimeError("Final checkpoint publication exceeded two hours")
                time.sleep(5)
            atomic_json(status_path, {"state": "cancelled", "updated_at": time.time()})
            return 130
    except BaseException as exc:
        atomic_json(status_path, {"state": "failed", "error": str(exc), "updated_at": time.time()})
        raise
    finally:
        for process in reversed(processes):
            stop(process)


if __name__ == "__main__":
    sys.exit(main())
