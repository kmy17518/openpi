from types import SimpleNamespace

import pytest

from openpi.serving.websocket_b1k_server import WebsocketPolicyServer


def _policy(**overrides):
    values = {"control_mode": "receding_horizon", "action_horizon": 16, "max_len": 32}
    values.update(overrides)
    return SimpleNamespace(**values)


def test_action_chunk_accepts_exact_receding_horizon():
    WebsocketPolicyServer._validate_action_chunk_request(_policy(), 16)  # noqa: SLF001


@pytest.mark.parametrize(
    ("policy", "chunk_size"),
    [
        (_policy(control_mode="temporal_ensemble"), 16),
        (_policy(), 17),
        (_policy(max_len=8), 16),
    ],
)
def test_action_chunk_rejects_semantic_boundary_crossing(policy, chunk_size):
    with pytest.raises(ValueError, match="Action-chunk|Requested"):
        WebsocketPolicyServer._validate_action_chunk_request(policy, chunk_size)  # noqa: SLF001
