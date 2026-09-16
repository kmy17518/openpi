"""Cross-repository CPU integration: run from GR00T's environment with both source roots on PYTHONPATH."""

import asyncio
import http
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
from websockets.asyncio.server import serve

CAMERAS = (
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)
DIMS = {"base": 3, "torso": 4, "left_arm": 7, "left_gripper": 1, "right_arm": 7, "right_gripper": 1}


def observation(markers):
    state = np.zeros((len(markers), 61), np.float32)
    state[:, 0] = markers
    return {
        "robot_r1::proprio": state,
        "task_id": np.zeros((len(markers), 1), np.int64),
        **{camera: np.zeros((len(markers), 4, 4, 4), np.uint8) for camera in CAMERAS},
    }


class Model:
    language_key = "annotation.human.task_name"

    def __init__(self, m):
        self.m = m
        self.calls = []
        self.modality_configs = {"action": SimpleNamespace(delta_indices=list(range(m)))}

    def actions(self, markers, dim):
        return (
            np.broadcast_to(
                np.asarray(markers)[:, None, None] * 1000 + np.arange(self.m)[None, :, None],
                (len(markers), self.m, dim),
            )
            .astype(np.float32)
            .copy()
        )

    def infer_batch(self, observations):
        markers = [obs["observation/state"][0] for obs in observations]
        self.calls.append(list(markers))
        return [{"actions": array} for array in self.actions(markers, 23)]

    def get_action(self, batch):
        markers = batch["state"]["base_qvel"][:, 0, 0]
        self.calls.append(list(markers))
        return {key: self.actions(markers, dim) for key, dim in DIMS.items()}, {}


def make_server(backend, m, n):
    model = Model(m)
    if backend == "openpi":
        from openpi.serving.websocket_b1k_server import WebsocketPolicyServer
        from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper

        wrapper = B1KPolicyWrapper(model, "b1k/R1Pro", "task", "receding_horizon", n, m, obs_size=(4, 4))
    else:
        import gr00t
        from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.eval.eval_b1k_wrapper import B1KPolicyWrapper
        from gr00t.eval.eval_b1k_wrapper import load_modality_config
        from gr00t.policy.websocket_b1k_server import WebsocketPolicyServer

        root = Path(gr00t.__file__).resolve().parents[1]
        if EmbodimentTag.NEW_EMBODIMENT.value not in MODALITY_CONFIGS:
            load_modality_config(str(root / "examples/b1k/r1pro.py"))
        wrapper = B1KPolicyWrapper(
            model,
            EmbodimentTag.NEW_EMBODIMENT,
            json.loads((root / "examples/b1k/r1pro.json").read_text()),
            text_prompt="task",
            control_mode="receding_horizon",
            action_horizon=n,
            obs_size=(4, 4),
        )
    return WebsocketPolicyServer(wrapper), model


def load_behavior_client():
    root = os.environ.get("BEHAVIOR_REPO")
    if not root:
        pytest.skip("Set BEHAVIOR_REPO to an unmodified BEHAVIOR vector checkout")
    path = Path(root) / "OmniGibson/omnigibson/eval/utils/network_utils.py"
    spec = importlib.util.spec_from_file_location("behavior_client_compat_test", path)
    module = importlib.util.module_from_spec(spec)
    macros = ModuleType("omnigibson.macros")
    macros.gm = SimpleNamespace(DEBUG=False)
    previous = sys.modules.get("omnigibson.macros")
    sys.modules["omnigibson.macros"] = macros
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is None:
            sys.modules.pop("omnigibson.macros", None)
        else:
            sys.modules["omnigibson.macros"] = previous
    return module.WebsocketClientPolicy


@pytest.mark.parametrize("backend", [os.environ.get("B1K_TEST_BACKEND", "openpi")])
@pytest.mark.parametrize(("m", "n"), [(32, 16), (16, 8), (7, 3), (1, 1)])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_unmodified_behavior_client(backend, m, n, batch_size):
    client_type = load_behavior_client()

    async def run():
        service, model = make_server(backend, m, n)

        def health_check(connection, request):
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        async with serve(service._handler, "127.0.0.1", 0, process_request=health_check) as listener:  # noqa: SLF001
            port = listener.sockets[0].getsockname()[1]

            def execute():
                first = client_type(host="127.0.0.1", port=port, action_chunk_size=n)
                second = client_type(host="127.0.0.1", port=port, action_chunk_size=n)
                expected_calls = []
                try:
                    first.reset()
                    for step in range(3 * n + 3):
                        markers = [100 * i + step for i in range(batch_size)]
                        obs = observation(markers)
                        if batch_size == 1:
                            obs = {key: value[0] for key, value in obs.items()}
                        result = first.act(obs).numpy()
                        plan_step = step // n * n
                        expected = (np.arange(batch_size) * 100 + plan_step) * 1000 + step % n
                        expected_actions = np.broadcast_to(expected[:, None], (batch_size, 23))
                        np.testing.assert_array_equal(
                            result, expected_actions[0] if batch_size == 1 else expected_actions
                        )
                        if step % n == 0:
                            expected_calls.append([100 * i + step for i in range(batch_size)])
                        if step == 2:
                            second.reset()
                            assert second.act(observation([900])).numpy()[0, 0] == 900000
                            expected_calls.append([900])
                            second.reset()
                            assert second.act(observation([901])).numpy()[0, 0] == 901000
                            expected_calls.append([901])
                    first.reset()
                    assert first.act(observation([2000])).numpy()[0, 0] == 2000000
                    expected_calls.append([2000])
                    assert model.calls == expected_calls
                finally:
                    for client in (first, second):
                        if client._ws is not None:  # noqa: SLF001
                            client._ws.close()  # noqa: SLF001

            await asyncio.to_thread(execute)

    asyncio.run(run())
