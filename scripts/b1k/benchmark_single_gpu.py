"""Bounded pi05_b1k throughput probes; no checkpoints or remote writes by default."""

import argparse
import contextlib
import dataclasses
import functools
import json
import logging
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time


class ResourceSampler:
    """Sample only the selected GPU and this process tree."""

    def __init__(self, gpu: str, cpus: set[int]):
        import psutil

        self.process = psutil.Process()
        self.gpu = gpu
        self.cpus = cpus
        self.stop = threading.Event()
        self.peak_gpu_mib = 0
        self.peak_rss_bytes = 0
        self.peak_threads = 0
        self.affinity_violations = set()
        self.cpu_seconds = {}
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        import psutil

        while not self.stop.is_set():
            rss = threads = 0
            for process in [self.process, *self.process.children(recursive=True)]:
                try:
                    rss += process.memory_info().rss
                    threads += process.num_threads()
                    for thread in process.threads():
                        self.affinity_violations.update(set(os.sched_getaffinity(thread.id)) - self.cpus)
                    cpu = process.cpu_times()
                    self.cpu_seconds[process.pid] = cpu.user + cpu.system
                except (psutil.Error, ProcessLookupError):
                    continue
            self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
            self.peak_threads = max(self.peak_threads, threads)
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                )
                self.peak_gpu_mib = max(self.peak_gpu_mib, int(value.strip()))
            except (subprocess.SubprocessError, ValueError):
                pass
            self.stop.wait(1)

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)

    def summary(self):
        return {
            "gpu_peak_mib_sampled": self.peak_gpu_mib,
            "process_tree_peak_rss_gib": self.peak_rss_bytes / 2**30,
            "process_tree_peak_threads": self.peak_threads,
            "cpu_seconds": sum(self.cpu_seconds.values()),
            "mean_cpu_cores": sum(self.cpu_seconds.values()) / (time.perf_counter() - self.started),
            "affinity_violations": sorted(self.affinity_violations),
        }


def emit(event, **fields):
    print(json.dumps({"event": event, "time": time.time(), **fields}, sort_keys=True), flush=True)


def make_config(args):
    from openpi.training import config
    from openpi.training import weight_loaders

    cfg = config.get_config("pi05_b1k")
    loader = (
        weight_loaders.NoOpWeightLoader()
        if args.weights is None
        else weight_loaders.CheckpointWeightLoader(str(args.weights))
    )
    return dataclasses.replace(
        cfg,
        exp_name="benchmark-single-gpu",
        model=dataclasses.replace(cfg.model, max_token_len=args.tokens, remat_policy=args.remat),
        batch_size=args.batch,
        grad_accum_steps=1,
        fsdp_devices=1,
        num_workers=args.workers,
        prefetch_batches=args.prefetch,
        wandb_enabled=False,
        weight_loader=loader,
        data=dataclasses.replace(
            cfg.data,
            repo_id="behavior-1k/2026-challenge-demos",
            dataset_root=str(args.dataset_root),
            task_names=("turning_on_radio",),
            prompt_source="task_name",
            base_config=dataclasses.replace(
                cfg.data.base_config,
                dataset_kwargs={**cfg.data.base_config.dataset_kwargs, "video_backend": args.video_backend},
            ),
        ),
    )


def synthetic_batch(cfg, data_sharding):
    import jax
    import numpy as np

    from openpi.models.tokenizer import PaligemmaTokenizer

    rng = np.random.default_rng(cfg.seed)
    batch = jax.tree.map(lambda s: np.zeros(s.shape, s.dtype), cfg.model.inputs_spec(batch_size=cfg.batch_size))
    observation, actions = batch
    for key in observation.images:
        observation.images[key] = rng.uniform(-1, 1, observation.images[key].shape).astype(np.float32)
        observation.image_masks[key][:] = True
    tokens, mask = PaligemmaTokenizer(cfg.model.max_token_len).tokenize("turning_on_radio", np.zeros(23))
    observation.tokenized_prompt[:] = tokens
    observation.tokenized_prompt_mask[:] = mask
    actions[:] = rng.normal(0, 0.1, actions.shape)
    return jax.device_put(batch, data_sharding)


def run_train(args, cfg):
    import importlib.util

    import jax
    import numpy as np

    from openpi.training import sharding

    spec = importlib.util.spec_from_file_location("benchmark_train_b1k", Path(__file__).with_name("train_b1k.py"))
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
    start = time.perf_counter()
    state, state_sharding = train.init_train_state(cfg, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    emit("initialized", seconds=time.perf_counter() - start)
    batch = synthetic_batch(cfg, data_sharding)
    jax.block_until_ready(batch)
    step = jax.jit(
        functools.partial(train.train_step, cfg),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    start = time.perf_counter()
    emit("compile_start")
    from openpi.shared import array_typing

    # Explicit lowering reconstructs TrainState with JAX ArgInfo rather than array leaves.
    with sharding.set_mesh(mesh), array_typing.disable_typechecking():
        executable = step.lower(rng, state, batch).compile()
    compile_seconds = time.perf_counter() - start
    memory = executable.memory_analysis()
    memory_fields = {
        key: int(getattr(memory, key))
        for key in ("argument_size_in_bytes", "output_size_in_bytes", "alias_size_in_bytes", "temp_size_in_bytes")
    }
    estimate = memory.argument_size_in_bytes + memory.output_size_in_bytes - memory.alias_size_in_bytes
    estimate += memory.temp_size_in_bytes
    emit("compiled", seconds=compile_seconds, estimated_peak_gib=estimate / 2**30, **memory_fields)
    if estimate > args.memory_limit_gib * 2**30:
        return {"status": "memory_rejected", "estimated_peak_gib": estimate / 2**30, "compile_seconds": compile_seconds}
    # Separate buffers model the trainer's device prefetch queue without changing the measured graph.
    reserve = [jax.tree.map(lambda x: x.copy(), batch) for _ in range(args.prefetch)]
    jax.block_until_ready(reserve)
    times = []
    for index in range(args.warmup + args.steps):
        start = time.perf_counter()
        with sharding.set_mesh(mesh):
            state, info = executable(rng, state, batch)
        jax.block_until_ready((state, info))
        seconds = time.perf_counter() - start
        loss = float(info["loss"])
        if not np.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at iteration {index}")
        phase = "warmup" if index < args.warmup else "timed"
        emit(phase, index=index, seconds=seconds, loss=loss)
        if phase == "timed":
            times.append(seconds)
    stats = jax.devices()[0].memory_stats()
    result = {
        "status": "ok",
        "compile_seconds": compile_seconds,
        "seconds": times,
        "median_seconds": statistics.median(times),
        "samples_per_second": cfg.batch_size / statistics.median(times),
        "estimated_peak_gib": estimate / 2**30,
        "jax_memory_stats": stats,
        "last_loss": loss,
    }
    if args.checkpoint_snapshot:
        start = time.perf_counter()
        host_state = jax.device_get(state)
        result["checkpoint_host_snapshot_seconds"] = time.perf_counter() - start
        result["checkpoint_host_snapshot_bytes"] = sum(x.nbytes for x in jax.tree.leaves(host_state))
        emit("checkpoint_host_snapshot", **{k: v for k, v in result.items() if k.startswith("checkpoint_")})
    return result


def run_data(args, cfg):
    import jax

    from openpi.training import data_loader

    start = time.perf_counter()
    loader = data_loader.create_b1k_data_loader(
        cfg, shuffle=True, num_batches=args.warmup + args.steps, skip_norm_stats=args.skip_norm_stats
    )
    emit("data_initialized", seconds=time.perf_counter() - start)
    times = []
    with contextlib.closing(iter(loader)) as batches:
        for index in range(args.warmup + args.steps):
            start = time.perf_counter()
            batch = next(batches)
            jax.block_until_ready(batch)
            seconds = time.perf_counter() - start
            phase = "warmup" if index < args.warmup else "timed"
            emit(phase, index=index, seconds=seconds)
            if phase == "timed":
                times.append(seconds)
    return {
        "status": "ok",
        "seconds": times,
        "median_seconds": statistics.median(times),
        "samples_per_second": cfg.batch_size / statistics.median(times),
        "aggregate_samples_per_second": cfg.batch_size * len(times) / sum(times),
        "normalization_skipped": args.skip_norm_stats,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "data"), default="train")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--tokens", type=int, default=112)
    parser.add_argument("--remat", choices=("nothing_saveable", "dots_with_no_batch_dims_saveable", "none"), default="nothing_saveable")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--memory-limit-gib", type=float, default=250)
    parser.add_argument("--weights", type=Path, help="Existing local params directory; omission means random initialization.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/tmp/dev/datasets/2026-challenge-demos"))
    parser.add_argument("--video-backend", choices=("pyav", "torchcodec"), default="pyav")
    parser.add_argument("--skip-norm-stats", action="store_true")
    parser.add_argument("--checkpoint-snapshot", action="store_true", help="Measure a host state snapshot, not checkpoint disk IO.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch < 1 or args.steps < 1 or args.warmup < 0 or not 0 <= args.workers <= 24 or args.prefetch < 0:
        parser.error("Invalid batch, timing, worker (0..24), or prefetch count")
    if not args.output.resolve().is_relative_to("/tmp"):
        parser.error("Output must be under /tmp")
    if args.weights is not None and not args.weights.is_dir():
        parser.error("Weights must be an existing local directory; benchmark never downloads model weights")
    return args


def main(argv=None):
    args = parse_args(argv)
    cpus = set(range(30, 60))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1" or not set(os.sched_getaffinity(0)) <= cpus:
        raise RuntimeError("Requires CUDA_VISIBLE_DEVICES=1 and taskset -c 30-59")
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("Requires XLA_PYTHON_CLIENT_PREALLOCATE=false")
    from openpi.shared.xla_gpu_compat import configure_xla_flags

    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")
    configure_xla_flags()
    import jax
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Exactly one visible GPU is required")
    cfg = make_config(args)
    from openpi.training.b1k_dataset import check_prompt_token_lengths

    check_prompt_token_lengths({0: "turning_on_radio"}, cfg.model, state_dim=23)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "pid": os.getpid(),
        "jax_version": jax.__version__,
        "device": str(jax.devices()[0]),
        "mode": args.mode,
        "batch": args.batch,
        "tokens": args.tokens,
        "remat": args.remat,
        "workers": args.workers,
        "video_backend": args.video_backend,
        "dataset_root": str(args.dataset_root),
        "prefetch_batches_reserved": args.prefetch,
        "initialization": "random" if args.weights is None else str(args.weights),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
    }
    emit("start", **result)
    sampler = ResourceSampler("1", cpus)
    try:
        with sampler:
            result.update(run_train(args, cfg) if args.mode == "train" else run_data(args, cfg))
    except BaseException as error:
        result.update(status="failed", error=repr(error))
        raise
    finally:
        result.update(sampler.summary())
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        emit("result", **result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
