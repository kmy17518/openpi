# pi0.5 turning_on_radio — single GPU, 300k updates

Run configuration: `scripts/b1k/runs/turning-on-radio-1gpu-bs576-300k-20260916.json`.

## Resource and training settings

- GPU: physical index1, UUID `GPU-2246c972-301d-6778-7a38-bdf1d0e05687`.
- CPU affinity: **30–59**, shared by trainer, data workers, W&B service, publisher and local watchers. The other three training jobs' GPU/CPU allocations are unchanged.
- Model: pretrained pi0.5 base, full fine-tuning, prediction horizon32, token budget112, `nothing_saveable` rematerialization, EMA0.99.
- Batch: **576 physical samples**, no gradient accumulation, one GPU, eight PyAV workers, two prefetched batches.
- Data: all200 turning_on_radio episodes (429,928 frames); task-name prompting; newly computed representation-versioned norm statistics.
- Schedule: **300,000 completed optimizer updates**. Local full checkpoints at update25, every2,500 updates, and300,000; **latest3 retained**, no periodic keep exemptions. The default optimizer/learning-rate schedule is unchanged.
- Runtime: existing `.venv`, JAX/jaxlib0.5.3, Flax0.10.2, Orbax0.11.13. The Blackwell compatibility helper disables unsupported Triton GEMM fusion. TorchCodec is incompatible with this installed Torch, so this run explicitly uses PyAV.

Batch576 is the largest **validated conservative** batch, not a mathematical maximum. Synthetic measurements ranged from62.9samples/s at batch64 to61.2samples/s at576; batch256 reached65.4samples/s. The lighter dots rematerialization policy reached66.8samples/s at64 but exceeded the reserved memory budget at128. Largest-batch priority therefore selected576/nothing_saveable. Batch640 required262.64GiB estimated live memory and768 required292.65GiB; both were rejected by a250GiB safety guard. Batch608 was not tested.

The final **pretrained, real-data** check at576 measured median9.832s/update (58.58samples/s), peak236.58GiB live JAX allocations,263.65GiB allocator footprint,122.1GiB process-tree host RAM, and no CPU-affinity violations. An actual44.89GB full checkpoint saved in10.90s. Validation with batch32 was finite; routine production validation is disabled. At this measured speed the target is approximately34days before outages and publication overhead.

That standalone benchmark completed computations/save but aborted during interpreter shutdown. Production starts workers before prefetch and now waits up to60seconds for producer cleanup before closing the checkpoint manager; regression tests cover these cleanup paths. Long-duration stability and final production shutdown are not claimed from the benchmark.

## Logging and launch

- W&B: https://wandb.ai/kmy17518/b1k-challenge-2026-pi/runs/piradio16
- Training status: `/tmp/dev/runs/pi05-turning-on-radio-1gpu-300k-20260916/training-status.json`
- Supervisor status: same directory, `supervisor-status.json`.
- Logs: `/tmp/dev/logs/pi05-turning-on-radio-1gpu-300k-20260916.{train,uploader,monitor}.log`.
- tmux sessions: `pi-radio-1gpu-300k` and `pi-radio-monitor`.

From this checkout, after `source /tmp/dev/env.sh`:

```bash
taskset -c 30-59 .venv/bin/python -u scripts/b1k/supervise_single_gpu.py \
    --run-config scripts/b1k/runs/turning-on-radio-1gpu-bs576-300k-20260916.json
```

Use the same command with `--resume` only after the previous supervisor/process group is stopped. Preserve the run config, local generation marker, checkpoint assets, `wandb_id.txt`, staging `owner.json` and journal. Recovery reconciles committed local checkpoints that were interrupted before staging, including the final save.

## Hugging Face retention and quota

Dedicated private repository: https://huggingface.co/kmy17518/pi05-turning-on-radio-1gpu-300k-20260916

One `hf_single_writer_checkpoint_uploader.py` process handles both:

- `eval/checkpoint-10000`, `20000`, …, `300000`: parameters/EMA weights, assets and Orbax metadata, excluding optimizer/train state.
- `resume/checkpoint-<latest>`: the latest full resumable checkpoint, with parameters, raw train state/optimizer and all assets.

The trainer synchronously stages each completed checkpoint before proceeding, so pruning cannot race the copy. The publisher uploads and verifies the replacement while the old full is still available, then removes the old paths and permanently deletes only journaled stale LFS hashes absent from **all retained live files**. All checkpoint files, including small manifests, use LFS. Hub physical inventory and usedStorage accounting are checked; stale quota is reported rather than hidden. During replacement there is temporarily an old and new full copy; steady state contains one full checkpoint plus all scheduled eval checkpoints.

Exclusive repository ownership is required for the entire run: **no other uploader, UI commit, branch, tag or PR** may write this repository. Unexpected refs, paths, identity or revision changes stop publication. The local lease is not a cross-host lock and Hub physical deletion has no conditional-delete API. Historical full revisions intentionally become unreadable after their stale blobs are reclaimed; current eval/full snapshots remain protected.

Scratch validation on `kmy17518/pi05-radio-uploader-validation-20260916` performed two rotations, permanently deleted2,621,676bytes/8objects, downloaded/hash-verified all17 retained files, and confirmed Hub usedStorage equals the2,097,642-byte live inventory with zero unreclaimed bytes. Proof and events are under `/tmp/dev/audits/openpi-single-radio-20260916/`.

The supervisor stops its owned process groups if either required component fails. A persistent local watcher reports failures/stalls, and the interactive session also registered periodic health checks. These do not modify the other GPU jobs.
