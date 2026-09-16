# BEHAVIOR-compatible action chunks

Both openpi and GR00T's `serve_b1k.py` predict `m` actions and return the first `n`, configured by **`--action-horizon n`**. The model configuration/checkpoint determines `m`; require `1 <= n <= m`. The default execution length is 16. A GR00T checkpoint with 16-step predictions can execute a shorter prefix, such as `n=8`; larger predictions require a matching trained checkpoint.

## Existing client and wire format

Use the existing BEHAVIOR `vector` evaluator with **`--replay-action-chunk-size n`**, matching the server. No evaluator patch or replacement client is required.

For example, with `m=32` and `n=16`:

- Model server: `--action-horizon 16`.
- BEHAVIOR evaluator: append `--replay-action-chunk-size 16` to the usual evaluation command.
- Request: current observations plus `"__action_chunk_size__": 16`.
- Response preserves Andi's format:

```python
{
    "action": action_chunk[..., 0, :],
    "action_chunk": action_chunk,
    "server_timing": {...},
}
```

`action_chunk` is `(n,D)` for a single environment or `(B,n,D)` for a batch; `action` is `(D,)` or `(B,D)`. `B` is the environment batch size, `n` is execution length, and `m` is prediction length. Each response comes from one fresh model prediction and discards its unused tail. The existing metadata handshake is unchanged; there is no protocol-version negotiation.

The BEHAVIOR client consumes one buffered action per control step and sends another observation after `n` steps. It does not report a separate count of actions actually executed. Its `reset()` clears the buffer and sends `{"reset": true}` without waiting for a response; servers preserve that fire-and-forget behavior.

The server requires the requested integer chunk size to equal its configured `n`. BEHAVIOR's default chunk flag `0` disables chunking; it does not negotiate `n`. For `n=1`, the client omits the field and the server accepts it as one action. For `n>1`, configure the evaluator explicitly.

## Vectorized evaluation

The inspected public `StanfordVL/BEHAVIOR-1K` branch `vector`, commit `f0e7109f8cf8292070d5a5ef767b2b79d7f86298`, uses one batched client with a shared chunk cursor. All batch slots advance together; finished slots stay inactive until the complete batch ends. This aligned use is supported without changes to the evaluator. Keep row ordering and batch membership stable until reset.

The earlier `vec-eval` approach uses separate connections per environment, each with its own buffer. These connections can reset independently. Independently restarting one row inside an existing shared-cursor batch is not supported by that existing client and is not introduced here.

## Compatibility tests

Set `BEHAVIOR_REPO` to an unmodified checkout of the above `vector` commit. The tests load its `network_utils.py` unchanged, replacing only the unrelated OmniGibson macro import so Isaac Sim is not required. HTTP health checks, metadata, WebSocket requests/responses, MessagePack, action buffering, and reset behavior run against actual localhost servers.

From openpi's environment:

```bash
BEHAVIOR_REPO=/path/to/BEHAVIOR-1K python -m pytest -q \
    src/openpi/serving/websocket_b1k_server_test.py \
    examples/b1k/test_action_chunk_servers.py
```

From GR00T's environment:

```bash
BEHAVIOR_REPO=/path/to/BEHAVIOR-1K B1K_TEST_BACKEND=gr00t \
    python -m pytest -q /path/to/openpi/examples/b1k/test_action_chunk_servers.py
```

Coverage includes single-env and three-env batches, `m32/n16`, `m16/n8`, non-divisible `m7/n3`, `m1/n1`, fresh replanning at each `n` boundary, fire-and-forget resets, independent connections, exact response-format validation, and no chunk fallback. Full simulator dynamics and model task success require a separate rollout.
