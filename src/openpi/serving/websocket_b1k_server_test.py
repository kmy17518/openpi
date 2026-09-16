import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import websockets
import websockets.asyncio.server

from openpi.configs.robots import ROBOT_REGISTRY
from openpi.serving import websocket_b1k_server as wire
from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper


def _policy(**overrides):
    values = {"control_mode": "receding_horizon", "action_horizon": 16, "prediction_horizon": 32}
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("size", [1, 10, 15, 17, 0, -1, None, True, 16.0, "16"])
def test_action_chunk_rejects_unaligned_requests(size):
    with pytest.raises(ValueError, match="positive integer|execution_horizon|receding_horizon"):
        wire.WebsocketPolicyServer._validate_action_chunk_request(_policy(), size)  # noqa: SLF001


def test_action_chunk_accepts_exact_receding_horizon():
    wire.WebsocketPolicyServer._validate_action_chunk_request(_policy(), 16)  # noqa: SLF001


@pytest.mark.parametrize(
    "policy", [_policy(control_mode="temporal_ensemble"), _policy(prediction_horizon=8), _policy(action_horizon=True)]
)
def test_action_chunk_rejects_invalid_server_configuration(policy):
    with pytest.raises(ValueError, match="positive integer|execution_horizon|receding_horizon"):
        wire.WebsocketPolicyServer(policy)


class MarkerPolicy:
    def __init__(self, horizon):
        self.horizon = horizon
        self.calls = []

    def infer_batch(self, obs_list):
        markers = [float(obs["observation/state"][0]) for obs in obs_list]
        self.calls.append(markers)
        return [
            {
                "actions": np.broadcast_to(marker * 1000 + np.arange(self.horizon)[:, None], (self.horizon, 23)).astype(
                    np.float32
                )
            }
            for marker in markers
        ]


def _wrapper(m=32, n=16):
    model = MarkerPolicy(m)
    return B1KPolicyWrapper(model, "b1k/R1Pro", "test", "receding_horizon", n, m, obs_size=(4, 4)), model


def _observation(markers):
    state = np.zeros((len(markers), 61), np.float32)
    state[:, 0] = markers
    return {
        "robot_r1::proprio": state,
        "task_id": np.zeros((len(markers), 1), np.int64),
        **{
            camera.obs_key: np.zeros((len(markers), 4, 4, 3), np.uint8)
            for camera in ROBOT_REGISTRY["b1k/R1Pro"].observations.values()
        },
    }


@pytest.mark.parametrize(("m", "n"), [(32, 16), (16, 8), (7, 3), (1, 1)])
def test_chunk_predicts_once_and_discards_tail(m, n):
    wrapper, model = _wrapper(m, n)
    chunk = wrapper.act_chunk(_observation([1, 2])).numpy()
    assert chunk.shape == (2, n, 23)
    assert model.calls == [[1, 2]]
    assert chunk[:, :, 0].tolist() == [list(range(1000, 1000 + n)), list(range(2000, 2000 + n))]
    assert wrapper.action_buffer is None
    single = {k: v[0] for k, v in _observation([3]).items()}
    assert wrapper.act_chunk(single).shape == (n, 23)
    assert model.calls == [[1, 2], [3]]


def test_bad_prediction_and_observation_fail():
    wrapper, model = _wrapper()
    model.horizon = 8
    with pytest.raises(ValueError, match="Expected actions"):
        wrapper.act_chunk(_observation([1]))
    model.horizon = 32
    bad = _observation([1, 2])
    key = next(iter(ROBOT_REGISTRY["b1k/R1Pro"].observations.values())).obs_key
    bad[key] = bad[key][:1]
    with pytest.raises(ValueError, match="batch size"):
        wrapper.act_chunk(bad)


@pytest.mark.parametrize("batched", [False, True])
def test_andi_response_and_fire_and_forget_reset(batched):
    async def run():
        wrapper, model = _wrapper(m=32, n=16)
        service = wire.WebsocketPolicyServer(wrapper, metadata={"robot": "R1Pro"})
        async with websockets.asyncio.server.serve(service._handler, "127.0.0.1", 0) as listener:  # noqa: SLF001
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
                assert wire.unpackb(await client.recv()) == {"robot": "R1Pro"}
                for step in [0, 16, 32]:
                    obs = _observation([step, step + 100] if batched else [step])
                    if not batched:
                        obs = {key: value[0] for key, value in obs.items()}
                    await client.send(wire.packb({"reset": True}))
                    await client.send(wire.packb({**obs, "__action_chunk_size__": 16}))
                    reply = wire.unpackb(await asyncio.wait_for(client.recv(), timeout=5))
                    assert reply["action_chunk"].shape == ((2, 16, 23) if batched else (16, 23))
                    np.testing.assert_array_equal(reply["action"], reply["action_chunk"][..., 0, :])
                    assert "reset" not in reply
                assert len(model.calls) == 3
                await client.send(wire.packb({**obs, "__action_chunk_size__": 10}))
                with pytest.raises(websockets.ConnectionClosed):
                    await client.recv()
                assert len(model.calls) == 3

    asyncio.run(run())


def test_single_action_client_omits_chunk_size_when_n_is_one():
    async def run():
        wrapper, model = _wrapper(m=8, n=1)
        service = wire.WebsocketPolicyServer(wrapper)
        async with websockets.asyncio.server.serve(service._handler, "127.0.0.1", 0) as listener:  # noqa: SLF001
            port = listener.sockets[0].getsockname()[1]
            async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
                await client.recv()
                await client.send(wire.packb(_observation([9])))
                reply = wire.unpackb(await client.recv())
                assert reply["action"].shape == (1, 23)
                assert reply["action_chunk"].shape == (1, 1, 23)
                assert model.calls == [[9]]

    asyncio.run(run())
