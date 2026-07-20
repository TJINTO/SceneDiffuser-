import numpy as np
import pytest
import sumolib

from topoworld.scenediffuserpp.roadgraph import load_roadgraph
from topoworld.scenediffuserpp.roadgraph import map_sumo_signal
from topoworld.scenediffuserpp.roadgraph import validate_tls_state
from topoworld.scenediffuserpp.schema import LightState


def test_roadgraph_excludes_internal_edges_and_keeps_successors(tiny_signal_net_file):
    graph = load_roadgraph(tiny_signal_net_file, point_spacing_m=2.0)

    assert graph.lane_tokens
    assert all(not token.edge_id.startswith(":") for token in graph.lane_tokens)
    assert [token.lane_id for token in graph.lane_tokens] == sorted(
        token.lane_id for token in graph.lane_tokens
    )
    assert graph.successors["e0_0"] == ("e1_0", "e2_0")
    assert all(token.xy.ndim == 2 and token.xy.shape[1] == 2 for token in graph.lane_tokens)
    assert all(token.tangent.shape == token.xy.shape for token in graph.lane_tokens)


def test_resampled_lane_keeps_both_endpoints(tiny_signal_net_file):
    graph = load_roadgraph(tiny_signal_net_file, point_spacing_m=30.0)
    lane = next(token for token in graph.lane_tokens if token.lane_id == "e0_0")
    source_lane = sumolib.net.readNet(str(tiny_signal_net_file)).getLane("e0_0")
    source_shape = np.asarray(source_lane.getShape())

    np.testing.assert_allclose(lane.xy[0], source_shape[0], atol=1e-6)
    np.testing.assert_allclose(lane.xy[-1], source_shape[-1], atol=1e-6)
    np.testing.assert_allclose(lane.tangent, np.tile([1.0, 0.0], (len(lane.xy), 1)))


def test_tls_tokens_are_one_per_controlled_link(tiny_signal_net_file):
    graph = load_roadgraph(tiny_signal_net_file, point_spacing_m=2.0)

    identities = {(x.tls_id, x.link_index) for x in graph.light_tokens}
    assert len(graph.light_tokens) == len(identities) == 2
    assert all(x.stop_line_xy.shape == (2,) for x in graph.light_tokens)
    assert all(x.incoming_lane_id == "e0_0" for x in graph.light_tokens)


@pytest.mark.parametrize(
    ("char", "turn", "expected"),
    [
        ("G", "s", LightState.GREEN),
        ("g", "l", LightState.GREEN_ARROW),
        ("y", "r", LightState.YELLOW_ARROW),
        ("r", "s", LightState.RED),
        ("R", "l", LightState.RED_ARROW),
        ("o", "s", LightState.UNKNOWN),
    ],
)
def test_sumo_signal_mapping_uses_turn_semantics(char, turn, expected):
    assert map_sumo_signal(char, turn) is expected


def test_tls_state_validation_rejects_short_state_string(tiny_signal_net_file):
    graph = load_roadgraph(tiny_signal_net_file, point_spacing_m=2.0)
    tls_id = graph.light_tokens[0].tls_id

    with pytest.raises(ValueError, match="state length"):
        validate_tls_state(tls_id=tls_id, state="r", light_tokens=graph.light_tokens)
