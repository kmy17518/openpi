#!/usr/bin/env python3
"""Queue the finalized orbax steps of an openpi run for the ACT repo's Hugging Face checkpoint uploader.

    export_checkpoints.py --checkpoint-dir outputs/checkpoints/<config>/<exp_name> --run-dir <run dir>
                          [--poll-seconds 60] [--max-steps N]

Orbax writes step <S> as <S>.orbax-checkpoint-tmp[-<timestamp>] and renames it to <S> once it is complete. Every
finalized step becomes two uncompressed tars whose members are <S>/...:

    <run dir>/export_queue/eval/step_<S:08d>.tar    <S>/params and <S>/assets, what serving needs
    <run dir>/export_queue/full/step_<S:08d>.tar    all of <S>/: extract it into the checkpoint directory to resume

which `scripts/b1k/upload_checkpoints.py --run-dir <run dir>` (ACT repo) publishes as eval/step-<S:08d>.tar and
resume/step-<S:08d>.tar and deletes once uploaded. As with ACT's trainer, only the newest full tar stays queued.
<run dir>/export_state.json lists the exported steps, so consumed tars are not written again. The checkpoint
directory is only read.
"""

import argparse
import errno
import fcntl
import json
import logging
import os
import pathlib
import re
import shutil
import stat
import tarfile
import time

STEP_ITEMS = ("params", "train_state", "assets")
EVAL_ITEMS = ("params", "assets")


def _sync_dir(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def save_state(path: pathlib.Path, state: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(state, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_dir(path.parent)


def finalized_steps(checkpoint_dir: pathlib.Path) -> list[int]:
    if not checkpoint_dir.is_dir():
        return []
    with os.scandir(checkpoint_dir) as entries:
        return sorted(
            int(entry.name)
            for entry in entries
            if re.fullmatch(r"\d+", entry.name) and entry.is_dir(follow_symlinks=False)
        )


def members(
    path: pathlib.Path, name: str, items: tuple[str, ...] | None = None
) -> list[tuple[str, pathlib.Path, os.stat_result]]:
    """(member name, path, lstat) of `path` and everything below it, parents first, sorted by name; `items` restricts
    the top level."""
    info = path.lstat()
    result = [(name, path, info)]
    if stat.S_ISDIR(info.st_mode):
        for child in sorted(os.listdir(path)):
            if items is None or child in items:
                result += members(path / child, f"{name}/{child}")
    elif not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{path} is neither a regular file nor a directory")
    return result


def _data_bytes(entries: list[tuple[str, pathlib.Path, os.stat_result]]) -> int:
    return sum(info.st_size for _, _, info in entries if stat.S_ISREG(info.st_mode))


def write_tar(target: pathlib.Path, step_dir: pathlib.Path, items: tuple[str, ...] | None, reserve: int) -> None:
    """Stream the step into `target` through an fsynced temporary file. OSError if fewer than `reserve` bytes would
    stay free or if the step changes (e.g. retention deletes it) meanwhile; nothing is published then."""
    entries = members(step_dir, step_dir.name, items)
    free = shutil.disk_usage(target.parent).free
    if free < _data_bytes(entries) + reserve:
        raise OSError(errno.ENOSPC, f"{free / 1e9:.1f} GB free: no room for {target.name} and one more checkpoint")
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with temporary.open("wb") as stream:
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT, copybufsize=8 << 20) as tar:
                for name, path, info in entries:
                    member = tarfile.TarInfo(name)
                    member.mode = stat.S_IMODE(info.st_mode)
                    member.mtime = int(info.st_mtime)
                    if stat.S_ISDIR(info.st_mode):
                        member.type = tarfile.DIRTYPE
                        tar.addfile(member)
                    else:
                        member.size = info.st_size
                        with path.open("rb") as source:
                            tar.addfile(member, source)
            stream.flush()
            os.fsync(stream.fileno())
        signature = [(name, info.st_mode, info.st_size, info.st_mtime_ns) for name, _, info in entries]
        if signature != [
            (name, info.st_mode, info.st_size, info.st_mtime_ns)
            for name, _, info in members(step_dir, step_dir.name, items)
        ]:
            raise OSError(f"{step_dir} changed while it was exported")
        os.replace(temporary, target)
        _sync_dir(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def export_pending(
    checkpoint_dir: pathlib.Path, run_dir: pathlib.Path, state: dict, max_steps: int | None = None
) -> None:
    queue = run_dir / "export_queue"
    steps = [step for step in finalized_steps(checkpoint_dir) if max_steps is None or step <= max_steps]
    for step in steps:
        if step in state["exported"]:
            continue
        step_dir = checkpoint_dir / str(step)
        # A full tar of an older step would be superseded before it is uploaded.
        kinds = [("eval", EVAL_ITEMS), ("full", None)] if step == steps[-1] else [("eval", EVAL_ITEMS)]
        try:
            if missing := [item for item in STEP_ITEMS if not (step_dir / item).is_dir()]:
                raise FileNotFoundError(f"{step_dir} has no {', '.join(missing)}")
            # Room for the trainer's next checkpoint stays free.
            reserve = 0
            if run_dir.stat().st_dev == step_dir.stat().st_dev:
                reserve = _data_bytes(members(step_dir, step_dir.name))
            for kind, items in kinds:
                target = queue / kind / f"step_{step:08d}.tar"
                if not target.exists():
                    start = time.monotonic()
                    write_tar(target, step_dir, items, reserve)
                    logging.info(
                        "Wrote %s (%.2f GB, %.0f s)", target, target.stat().st_size / 1e9, time.monotonic() - start
                    )
        except OSError as exc:
            logging.warning("Step %d not exported, retrying at the next poll: %s", step, exc)
            continue
        if step == steps[-1]:
            for old in (queue / "full").glob("step_*.tar"):
                if int(old.stem.removeprefix("step_")) < step:
                    old.unlink()
            _sync_dir(queue / "full")
        state["exported"].append(step)
        save_state(run_dir / "export_state.json", state)
        logging.info("Exported step %d", step)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True, help="the run's orbax directory")
    parser.add_argument("--run-dir", type=pathlib.Path, required=True, help="the uploader's --run-dir")
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument(
        "--max-steps", type=int, help="exit once this step is exported (openpi's last one is num_train_steps - 1)"
    )
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0 or (args.max_steps is not None and args.max_steps <= 0):
        parser.error("--poll-seconds and --max-steps must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    checkpoint_dir, run_dir = args.checkpoint_dir.resolve(), args.run_dir.resolve()
    if run_dir == checkpoint_dir or checkpoint_dir in run_dir.parents:
        parser.error("--run-dir must be outside --checkpoint-dir")
    for kind in ("eval", "full"):
        (run_dir / "export_queue" / kind).mkdir(parents=True, exist_ok=True)
    with (run_dir / ".checkpoint-export.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another exporter holds {lock.name}") from None
        for stale in run_dir.glob("export_queue/*/.step_*.tar.tmp"):
            stale.unlink()
        state_path = run_dir / "export_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {"checkpoint_dir": str(checkpoint_dir)}
        if state["checkpoint_dir"] != str(checkpoint_dir):
            raise SystemExit(f"{run_dir} exports {state['checkpoint_dir']}, not {checkpoint_dir}")
        state.setdefault("exported", [])
        logging.info("Exporting the finalized steps of %s to %s", checkpoint_dir, run_dir / "export_queue")
        while True:
            export_pending(checkpoint_dir, run_dir, state, args.max_steps)
            if args.max_steps is not None and args.max_steps in state["exported"]:
                logging.info("Step %d is exported", args.max_steps)
                return 0
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
