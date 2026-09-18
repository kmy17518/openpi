"""Measure the throughput of the B1K training data loader (spawned torch workers), without a GPU.

Builds the dataset exactly as `scripts/b1k/train_b1k.py` does for the given config and `--data.*` flags, then times
the torch DataLoader at a batch size for several worker counts. Steady state is measured beyond the loader's prefetch
depth (2 batches per worker), which is what a trainer sees once the first batches are consumed.

    JAX_PLATFORMS=cpu taskset -c 30-59 uv run scripts/b1k/benchmark_loader.py pi05_b1k \\
        --data.repo_id=behavior-1k/2026-challenge-demos --data.dataset-root=<DATASET_ROOT> \\
        --data.task-names turning_on_radio -- --batch-size 512 --workers 8 16 24

Arguments before `--` are the training config overrides (tyro, as for train_b1k.py); after it, the benchmark's own.
Add `--lerobot-decoder` to time lerobot's video path for comparison.
"""

import argparse
import dataclasses
import sys
import time

import torch

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main() -> None:
    argv = sys.argv[1:]
    split = argv.index("--") if "--" in argv else len(argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--batches", type=int, default=30, help="timed batches beyond the prefetch depth")
    parser.add_argument("--lerobot-decoder", action="store_true", help="decode every stream with lerobot's decoder")
    args = parser.parse_args(argv[split + 1 :])
    sys.argv = [sys.argv[0], *argv[:split]]
    if "--exp_name" not in " ".join(sys.argv):
        sys.argv.append("--exp_name=benchmark_loader")

    config = _config.cli()
    data_config = config.data.create(config.assets_dirs, config.model)
    if args.lerobot_decoder:
        kwargs = {k: v for k, v in data_config.dataset_kwargs.items() if k != "video_keys"}
        data_config = dataclasses.replace(data_config, dataset_kwargs={**kwargs, "fast_video_reader": False})
    raw = _data_loader.create_b1k_dataset(data_config, config.model.action_horizon, config.model)
    data_config = dataclasses.replace(data_config, inference_metadata=getattr(raw, "inference_metadata", None))
    dataset = _data_loader.transform_dataset(raw, data_config)
    print(f"{len(dataset)} samples, batch {args.batch_size}, decoding {getattr(raw._dataset, 'video_keys', '?')}", flush=True)  # noqa: SLF001

    for workers in args.workers:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=workers,
            multiprocessing_context=torch.multiprocessing.get_context("spawn"),
            persistent_workers=True,
            collate_fn=_data_loader._collate_fn,  # noqa: SLF001
            worker_init_fn=_data_loader._worker_init_fn,  # noqa: SLF001
            drop_last=True,
            generator=torch.Generator().manual_seed(0),
        )
        iterator = iter(loader)
        started = time.perf_counter()
        arrivals = []
        total = 2 * workers + args.batches
        for _ in range(total):
            batch = next(iterator)
            arrivals.append(time.perf_counter() - started)
        warm = 2 * workers
        steady = (arrivals[-1] - arrivals[warm]) / (total - 1 - warm)
        image = next(iter(batch["image"].values()))
        print(
            f"workers={workers:2d}: first batch after {arrivals[0]:5.1f} s (spawn + fill) | steady state "
            f"{args.batch_size / steady:7.1f} samples/s = {steady:5.2f} s per batch | overall incl. spawn "
            f"{total * args.batch_size / arrivals[-1]:6.1f} samples/s | image {tuple(image.shape)} {image.dtype}",
            flush=True,
        )
        del iterator, loader


if __name__ == "__main__":
    main()
