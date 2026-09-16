"""Quiet status watcher for a monitored single-GPU training/publisher pair."""

import argparse
import json
from pathlib import Path
import time


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", type=Path, required=True)
    args = parser.parse_args()
    settings = read(args.run_config)
    previous_failure = None
    while True:
        supervisor = read(settings["supervisor_status"])
        training = read(settings["status_file"])
        uploader = read(Path(settings["staging_dir"]) / "status.json")
        error = read(Path(settings["staging_dir"]) / "error.json")
        reason = None
        if supervisor.get("state") == "failed":
            reason = supervisor.get("error", "supervisor failed")
        elif training.get("state") == "failed":
            reason = training.get("error", "training failed")
        elif uploader.get("fatal_error") or error.get("error"):
            reason = uploader.get("fatal_error") or error["error"]
        elif (
            supervisor
            and supervisor.get("state") not in ("completed", "cancelled")
            and time.time() - supervisor.get("updated_at", 0) > 120
        ):
            reason = "supervisor heartbeat missing"
        elif training.get("state") == "training" and time.time() - training.get("updated_at", 0) > 900:
            reason = "training progress stalled for fifteen minutes"
        elif uploader.get("last_error"):
            reason = "checkpoint publisher reports " + str(uploader["last_error"])
        if reason and reason != previous_failure:
            print("FAILED " + reason, flush=True)
        previous_failure = reason
        if supervisor.get("state") == "completed":
            print("DONE training and final checkpoint publication complete", flush=True)
            return
        if supervisor.get("state") == "cancelled":
            print("CANCELLED run supervisor stopped", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
