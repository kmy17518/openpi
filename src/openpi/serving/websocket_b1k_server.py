import asyncio
import copy
import functools
import http
import logging
import time
import traceback
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as _server

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


__all__ = ["WebsocketPolicyServer"]

ACTION_CHUNK_REQUEST_KEY = "__action_chunk_size__"


class WebsocketPolicyServer:
    """Serve fixed execution chunks, with one fresh prediction per observation request."""

    def __init__(
        self,
        policy: Any,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._validate_action_chunk_request(policy, policy.action_horizon)
        self._metadata = dict(metadata or {})

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        logger.info(f"Starting websocket server on {self._host}:{self._port}...")
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    @staticmethod
    def _clear_policy_state(policy) -> None:
        """Reset only the per-connection receding-horizon bookkeeping (NOT the shared jax model)."""
        policy.reset_connection_state()

    @staticmethod
    def _validate_action_chunk_request(policy, action_chunk_size: int) -> None:
        if policy.control_mode != "receding_horizon":
            raise ValueError("Action chunks require receding_horizon control")
        for name, value in (
            ("execution_horizon", policy.action_horizon),
            ("prediction_horizon", policy.prediction_horizon),
            (ACTION_CHUNK_REQUEST_KEY, action_chunk_size),
        ):
            if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if policy.action_horizon > policy.prediction_horizon:
            raise ValueError("execution_horizon must not exceed prediction_horizon")
        if action_chunk_size != policy.action_horizon:
            raise ValueError(f"Requested action chunk must equal execution_horizon={policy.action_horizon}")

    async def _handler(self, websocket):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = Packer()

        # Model weights are shared; connection resets never reset another client's model or state.
        conn_policy = copy.copy(self._policy)
        self._clear_policy_state(conn_policy)

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                result = unpackb(await websocket.recv(), strict_map_key=False)
                if not isinstance(result, dict):
                    raise ValueError("Expected an observation dictionary")
                if "reset" in result:
                    if result["reset"] is not True or len(result) != 1:
                        raise ValueError("Reset requests must be exactly {'reset': True}")
                    self._clear_policy_state(conn_policy)
                    continue

                action_chunk_size = result.pop(ACTION_CHUNK_REQUEST_KEY, 1)
                self._validate_action_chunk_request(conn_policy, action_chunk_size)

                infer_time = time.monotonic()
                chunk = conn_policy.act_chunk(result)
                infer_time = time.monotonic() - infer_time

                action_chunk = chunk.cpu().numpy()
                action = {"action": action_chunk[..., 0, :], "action_chunk": action_chunk}
                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                logger.error(f"Error in connection from {websocket.remote_address}:\n{traceback.format_exc()}")
                try:
                    # Try new websockets API first
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error. Traceback included in previous frame.",
                    )
                except AttributeError:
                    # Fallback for older websockets versions
                    await websocket.close(code=1011, reason="Internal server error")
                raise


def _health_check(connection, request) -> Any | None:
    if hasattr(request, "path") and request.path == "/healthz":
        if hasattr(connection, "respond"):
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        # For older websockets versions, return a simple response
        return http.HTTPStatus.OK, {"Content-Type": "text/plain"}, b"OK\n"
    # Continue with the normal request handling.
    return None


"""
Adds NumPy array and PyTorch tensor support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

This module supports serializing both NumPy arrays and PyTorch tensors. PyTorch tensors are converted to
NumPy arrays (zero-copy when possible) before serialization. On deserialization, arrays are returned as NumPy arrays.

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""


def pack_data(obj):
    if isinstance(obj, torch.Tensor):
        data = obj.detach().cpu().numpy()
        return {
            b"__ndarray__": True,
            b"data": data.tobytes(),
            b"dtype": data.dtype.str,
            b"shape": data.shape,
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_data(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_data)
packb = functools.partial(msgpack.packb, default=pack_data)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_data)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_data)
