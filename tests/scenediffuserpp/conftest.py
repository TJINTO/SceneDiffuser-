"""Shared deterministic fixtures for SceneDiffuser++ tests."""

from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def tiny_signal_net_file(tmp_path: Path) -> Path:
    nodes = tmp_path / "tiny.nod.xml"
    edges = tmp_path / "tiny.edg.xml"
    net = tmp_path / "tiny.net.xml"
    nodes.write_text(
        """<nodes>
  <node id="n0" x="0" y="0" type="priority"/>
  <node id="n1" x="100" y="0" type="traffic_light"/>
  <node id="n2" x="200" y="0" type="priority"/>
  <node id="n3" x="100" y="100" type="priority"/>
</nodes>
""",
        encoding="utf-8",
    )
    edges.write_text(
        """<edges>
  <edge id="e0" from="n0" to="n1" numLanes="1" speed="13.9" type="arterial"/>
  <edge id="e1" from="n1" to="n2" numLanes="1" speed="13.9" type="arterial"/>
  <edge id="e2" from="n1" to="n3" numLanes="1" speed="8.0" type="collector"/>
</edges>
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "netconvert",
            "--node-files",
            str(nodes),
            "--edge-files",
            str(edges),
            "--output-file",
            str(net),
            "--tls.set",
            "n1",
            "--no-turnarounds",
            "true",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.fail(f"netconvert failed: {result.stderr}")
    return net
