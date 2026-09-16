"""Keep a single-GPU trainer and its sole checkpoint publisher in one process lifetime."""

import argparse
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


def stop(process) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def command_lines(settings: dict, config_path: Path):
    python = settings["python"]
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
        "--sole-writer",
        "--poll-seconds",
        "30",
    ]
    trainer = [python, "scripts/b1k/train_b1k_monitored.py", "--run-config", str(config_path)]
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
    os.chdir(Path(__file__).resolve().parents[2])
    env = dict(os.environ)
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
            initialized = subprocess.run(
                [*publisher_cmd, "--init-only"],
                env=env,
                stdout=upload_log,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
            if initialized.returncode:
                raise RuntimeError("Publisher ownership/storage initialization failed; trainer not started")
            uploader = subprocess.Popen(
                publisher_cmd, env=env, stdout=upload_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(uploader)
            trainer = subprocess.Popen(
                trainer_cmd, env=env, stdout=train_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            processes.append(trainer)
            trainer_finished = None
            while not stopped:
                tcode, ucode = trainer.poll(), uploader.poll()
                atomic_json(
                    status_path,
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "trainer_pid": trainer.pid,
                        "uploader_pid": uploader.pid,
                        "trainer_exit": tcode,
                        "uploader_exit": ucode,
                        "started_at": started,
                        "updated_at": time.time(),
                    },
                )
                if tcode not in (None, 0) or ucode not in (None, 0):
                    raise RuntimeError(f"Required process failed: trainer={tcode}, publisher={ucode}")
                if tcode == 0 and ucode == 0:
                    atomic_json(status_path, {"state": "completed", "updated_at": time.time()})
                    return 0
                if ucode == 0 and tcode is None:
                    status = json.loads(Path(settings["status_file"]).read_text())
                    if status.get("step", 0) < settings["max_steps"]:
                        raise RuntimeError("Publisher exited before the final training step")
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
