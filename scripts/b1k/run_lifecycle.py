"""Local launch leases and checkpoint generation identities (no training dependencies)."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import uuid

LOCK_DIR = pathlib.Path(os.environ.get("B1K_LOCK_DIR", "/tmp/openpi-b1k-locks"))


def generation_path(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    return checkpoint_dir.parent / f".{checkpoint_dir.name}.generation.json"


def validate_generation(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("Run generation must be an object")
    if not isinstance(value.get("id"), str) or not value["id"].isalnum():
        raise ValueError("Run generation must have an alphanumeric id")
    # Zero is the lower bound for legacy or adopted resume checkpoints.
    if type(value.get("started_ns")) is not int or value["started_ns"] < 0:
        raise ValueError("Run generation started_ns must be a non-negative integer")


def generation(checkpoint_dir: pathlib.Path) -> dict:
    path = generation_path(checkpoint_dir)
    if not path.exists():
        return {"id": "legacy", "started_ns": 0}
    value = json.loads(path.read_text())
    validate_generation(value)
    return value


def prepare_generation(checkpoint_dir: pathlib.Path, *, fresh: bool) -> dict:
    path = generation_path(checkpoint_dir)
    if not fresh and path.exists():
        return generation(checkpoint_dir)
    value = {"id": uuid.uuid4().hex, "started_ns": time.time_ns() if fresh else 0}
    if not fresh and checkpoint_dir.is_dir():
        steps = sorted((p for p in checkpoint_dir.iterdir() if p.name.isdigit()), key=lambda p: int(p.name))
        for step in reversed(steps):
            provenance = step / "training_run.json"
            if provenance.is_file():
                recorded = json.loads(provenance.read_text())
                if not isinstance(recorded, dict):
                    raise ValueError(f"Invalid run provenance in {provenance}")
                previous = recorded.get("generation")
                if previous:
                    value = {"id": previous, "started_ns": recorded.get("generation_started_ns", 0)}
                    validate_generation(value)
                    break
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value) + "\n")
    with exclusive_lock(f"publish:{checkpoint_dir.resolve()}"):
        tmp.replace(path)
    return generation(checkpoint_dir)


def checkpoint_identity(checkpoint_dir: pathlib.Path, step: int, expected_generation: dict) -> str:
    validate_generation(expected_generation)
    if generation(checkpoint_dir) != expected_generation:
        raise RuntimeError("Run generation changed; retry with the current generation")
    metadata = (checkpoint_dir / str(step) / "_CHECKPOINT_METADATA").read_bytes()
    return metadata_identity(metadata, expected_generation)


def metadata_identity(metadata: bytes, expected_generation: dict) -> str:
    validate_generation(expected_generation)
    value = json.loads(metadata)
    if not isinstance(value, dict):
        raise ValueError("Checkpoint metadata must be an object")
    timestamp = value.get("commit_timestamp_nsecs")
    if type(timestamp) is not int or timestamp <= 0:
        raise ValueError("Checkpoint commit_timestamp_nsecs must be a positive integer")
    if timestamp < expected_generation["started_ns"]:
        raise ValueError("Checkpoint predates the current run generation or is not committed")
    return hashlib.sha256(expected_generation["id"].encode() + b"\0" + metadata).hexdigest()


@contextlib.contextmanager
def exclusive_lock(key: str):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCK_DIR / (hashlib.sha256(key.encode()).hexdigest() + ".lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Already locked: {key}; refusing a competing launch") from exc
        yield handle.fileno()


def launch(checkpoint_dir: pathlib.Path, command: list[str]) -> int:
    with contextlib.ExitStack() as stack:
        descriptors = [stack.enter_context(exclusive_lock(f"run:{checkpoint_dir.resolve()}"))]
        devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if not all(device.strip() for device in devices):
            raise ValueError("CUDA_VISIBLE_DEVICES must explicitly list the GPUs to lock")
        physical_devices = set()
        for device in devices:
            result = subprocess.run(
                ["nvidia-smi", "-i", device.strip(), "--query-gpu=uuid", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=10,
            )
            identities = result.stdout.strip().splitlines()
            if len(identities) != 1 or not identities[0].startswith("GPU-"):
                raise ValueError(f"Cannot resolve a physical GPU UUID for {device!r}")
            physical_devices.add(identities[0].strip())
        descriptors.extend(stack.enter_context(exclusive_lock(f"gpu:{device}")) for device in sorted(physical_devices))
        env = dict(os.environ, B1K_LAUNCHER_GUARDED="1")
        child = subprocess.Popen(command, env=env, start_new_session=True, pass_fds=descriptors)

        def stop(signum, _frame):
            raise SystemExit(128 + signum)

        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, stop)
        try:
            return child.wait()
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            except ProcessLookupError:
                pass


if __name__ == "__main__":
    try:
        action, directory, *args = sys.argv[1:]
        if action == "launch":
            sys.exit(launch(pathlib.Path(directory), args))
        elif action == "generation":
            prepare_generation(pathlib.Path(directory), fresh=args == ["fresh"])
        elif action == "latest":
            root = pathlib.Path(directory)
            current = generation(root)
            completed = []
            for path in root.iterdir() if root.is_dir() else []:
                if not path.name.isdigit():
                    continue
                try:
                    checkpoint_identity(root, int(path.name), current)
                except (OSError, ValueError, RuntimeError):
                    continue
                completed.append(int(path.name))
            print(max(completed, default=-1))
        else:
            raise ValueError(f"Unknown lifecycle action {action}")
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"[launcher] {exc}", file=sys.stderr)
        sys.exit(2)
