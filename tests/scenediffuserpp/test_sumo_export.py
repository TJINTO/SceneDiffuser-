from pathlib import Path

import pytest

from topoworld.scenediffuserpp.sumo_export import SumoTeacherConfig
from topoworld.scenediffuserpp.sumo_export import build_sumo_command
from topoworld.scenediffuserpp.sumo_export import tls_snapshot


class _FakeTrafficLightDomain:
    def getIDList(self):
        return ("tls1", "tls0")

    def getRedYellowGreenState(self, tls_id):
        return {"tls0": "Gr", "tls1": "rG"}[tls_id]

    def getProgram(self, tls_id):
        return f"program_{tls_id}"

    def getPhase(self, tls_id):
        return {"tls0": 2, "tls1": 1}[tls_id]

    def getNextSwitch(self, tls_id):
        return {"tls0": 9.0, "tls1": 7.0}[tls_id]


class _FakeTraci:
    trafficlight = _FakeTrafficLightDomain()


def _config(tmp_path: Path) -> SumoTeacherConfig:
    net_file = tmp_path / "n.net.xml"
    route_file = tmp_path / "r.rou.xml"
    net_file.write_text("<net/>", encoding="utf-8")
    route_file.write_text("<routes/>", encoding="utf-8")
    return SumoTeacherConfig(
        net_file=net_file,
        route_file=route_file,
        output_dir=tmp_path / "run",
        seed=7,
        begin_s=0.0,
        end_s=20.0,
    )


def test_teacher_command_fixes_micro_step_and_fcd_period(tmp_path):
    command = build_sumo_command(_config(tmp_path), sumo_binary="sumo")

    assert command[command.index("--step-length") + 1] == "0.1"
    assert command[command.index("--device.fcd.period") + 1] == "0.1"
    assert "--fcd-output.period" not in command
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--begin") + 1] == "0"
    assert command[command.index("--end") + 1] == "20"
    assert command[command.index("--fcd-output") + 1].endswith("fcd.xml")
    assert "--collision.action" in command


def test_teacher_command_runs_warmup_but_delays_fcd_recording(tmp_path):
    base = _config(tmp_path)
    config = SumoTeacherConfig(
        net_file=base.net_file,
        route_file=base.route_file,
        output_dir=base.output_dir,
        seed=base.seed,
        begin_s=0.0,
        end_s=80.0,
        recording_begin_s=20.0,
    )

    command = build_sumo_command(config, sumo_binary="sumo")

    assert command[command.index("--begin") + 1] == "0"
    assert command[command.index("--end") + 1] == "80"
    assert command[command.index("--device.fcd.begin") + 1] == "20"


def test_teacher_config_rejects_recording_begin_outside_simulation(tmp_path):
    base = _config(tmp_path)

    with pytest.raises(ValueError, match="recording_begin_s"):
        SumoTeacherConfig(
            net_file=base.net_file,
            route_file=base.route_file,
            output_dir=base.output_dir,
            seed=base.seed,
            begin_s=0.0,
            end_s=20.0,
            recording_begin_s=20.0,
        )


def test_teacher_command_uses_resolved_input_paths(tmp_path):
    config = _config(tmp_path)
    command = build_sumo_command(config, sumo_binary="sumo")

    assert command[command.index("--net-file") + 1] == str(config.net_file.resolve())
    assert command[command.index("--route-files") + 1] == str(config.route_file.resolve())


def test_teacher_config_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError, match="SUMO net file"):
        SumoTeacherConfig(
            net_file=tmp_path / "missing.net.xml",
            route_file=tmp_path / "missing.rou.xml",
            output_dir=tmp_path / "run",
            seed=0,
            begin_s=0.0,
            end_s=1.0,
        )


def test_tls_snapshot_preserves_sorted_ids_state_and_phase_metadata():
    rows = tls_snapshot(_FakeTraci(), simulation_time=1.2)

    assert [row["tls_id"] for row in rows] == ["tls0", "tls1"]
    assert rows[0] == {
        "time_s": 1.2,
        "tls_id": "tls0",
        "program_id": "program_tls0",
        "phase_index": 2,
        "state": "Gr",
        "next_switch_s": 9.0,
    }
