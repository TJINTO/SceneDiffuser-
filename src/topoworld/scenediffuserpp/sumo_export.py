from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any


STEP_LENGTH_S = 0.1


@dataclass(frozen=True)
class SumoTeacherConfig:
    net_file: Path
    route_file: Path
    output_dir: Path
    seed: int
    begin_s: float
    end_s: float
    recording_begin_s: float | None = None

    def __post_init__(self) -> None:
        net_file = Path(self.net_file)
        route_file = Path(self.route_file)
        if not net_file.is_file():
            raise FileNotFoundError(f"SUMO net file does not exist: {net_file}")
        if not route_file.is_file():
            raise FileNotFoundError(f"SUMO route file does not exist: {route_file}")
        if self.begin_s < 0.0:
            raise ValueError("begin_s must be non-negative")
        if self.end_s <= self.begin_s:
            raise ValueError("end_s must be greater than begin_s")
        if not self.begin_s <= self.recording_start_s < self.end_s:
            raise ValueError(
                "recording_begin_s must be within [begin_s, end_s)"
            )

    @property
    def recording_start_s(self) -> float:
        return (
            self.begin_s
            if self.recording_begin_s is None
            else float(self.recording_begin_s)
        )

    @property
    def fcd_file(self) -> Path:
        return Path(self.output_dir) / "fcd.xml"

    @property
    def tls_file(self) -> Path:
        return Path(self.output_dir) / "tls_states.jsonl"

    @property
    def manifest_file(self) -> Path:
        return Path(self.output_dir) / "manifest.json"


def build_sumo_command(
    config: SumoTeacherConfig, sumo_binary: str = "sumo"
) -> list[str]:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    return [
        str(sumo_binary),
        "--net-file",
        str(Path(config.net_file).resolve()),
        "--route-files",
        str(Path(config.route_file).resolve()),
        "--begin",
        _format_number(config.begin_s),
        "--end",
        _format_number(config.end_s),
        "--step-length",
        _format_number(STEP_LENGTH_S),
        "--seed",
        str(config.seed),
        "--fcd-output",
        str(config.fcd_file.resolve()),
        "--device.fcd.period",
        _format_number(STEP_LENGTH_S),
        "--device.fcd.begin",
        _format_number(config.recording_start_s),
        "--fcd-output.geo",
        "false",
        "--collision.action",
        "warn",
        "--collision.check-junctions",
        "true",
        "--no-step-log",
        "true",
    ]


def tls_snapshot(traci_module: Any, simulation_time: float) -> list[dict[str, Any]]:
    domain = traci_module.trafficlight
    return [
        {
            "time_s": float(simulation_time),
            "tls_id": str(tls_id),
            "program_id": str(domain.getProgram(tls_id)),
            "phase_index": int(domain.getPhase(tls_id)),
            "state": str(domain.getRedYellowGreenState(tls_id)),
            "next_switch_s": float(domain.getNextSwitch(tls_id)),
        }
        for tls_id in sorted(domain.getIDList())
    ]


def run_teacher(
    config: SumoTeacherConfig,
    sumo_binary: str = "sumo",
    traci_module: Any | None = None,
    resume_complete: bool = False,
) -> dict[str, Any]:
    command = build_sumo_command(config, sumo_binary=sumo_binary)
    signature = _run_signature(config, command)
    output_dir = Path(config.output_dir)
    if resume_complete and _matching_complete_manifest(config.manifest_file, signature):
        return json.loads(config.manifest_file.read_text(encoding="utf-8"))
    _require_clean_output(config, allow_manifest=config.manifest_file.exists())

    if traci_module is None:
        import traci as traci_module

    temporary_tls = config.tls_file.with_suffix(".jsonl.tmp")
    started = False
    step_count = 0
    recorded_step_count = 0
    tls_record_count = 0
    started_at = time.perf_counter()
    try:
        traci_module.start(command)
        started = True
        with temporary_tls.open("w", encoding="utf-8", newline="\n") as handle:
            while float(traci_module.simulation.getTime()) < config.end_s - 1e-9:
                traci_module.simulationStep()
                simulation_time = float(traci_module.simulation.getTime())
                step_count += 1
                if simulation_time < config.recording_start_s - 1e-9:
                    continue
                recorded_step_count += 1
                for row in tls_snapshot(traci_module, simulation_time):
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    tls_record_count += 1
    finally:
        if started:
            traci_module.close()
    runtime_s = max(time.perf_counter() - started_at, 1e-12)

    if not config.fcd_file.is_file():
        raise RuntimeError(f"SUMO did not create FCD output: {config.fcd_file}")
    temporary_tls.replace(config.tls_file)
    manifest = {
        "status": "complete",
        "signature": signature,
        "config": _serialized_config(config),
        "command": command,
        "sumo_version": _sumo_version(sumo_binary),
        "step_length_s": STEP_LENGTH_S,
        "actual_step_count": step_count,
        "recorded_step_count": recorded_step_count,
        "tls_record_count": tls_record_count,
        "runtime_s": runtime_s,
        "input_sha256": {
            "net_file": _sha256(Path(config.net_file)),
            "route_file": _sha256(Path(config.route_file)),
        },
        "output_sha256": {
            "fcd_file": _sha256(config.fcd_file),
            "tls_file": _sha256(config.tls_file),
        },
    }
    _write_json_atomic(config.manifest_file, manifest)
    return manifest


def _require_clean_output(config: SumoTeacherConfig, allow_manifest: bool) -> None:
    conflicts = [path for path in (config.fcd_file, config.tls_file) if path.exists()]
    if conflicts:
        raise FileExistsError(
            "SUMO teacher output already exists: " + ", ".join(map(str, conflicts))
        )
    if config.manifest_file.exists() and not allow_manifest:
        raise FileExistsError(f"SUMO teacher manifest already exists: {config.manifest_file}")


def _matching_complete_manifest(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return manifest.get("status") == "complete" and manifest.get("signature") == signature


def _serialized_config(config: SumoTeacherConfig) -> dict[str, Any]:
    result = asdict(config)
    for key in ("net_file", "route_file", "output_dir"):
        result[key] = str(Path(result[key]).resolve())
    return result


def _run_signature(config: SumoTeacherConfig, command: list[str]) -> str:
    payload = {
        "config": _serialized_config(config),
        "command": command,
        "net_sha256": _sha256(Path(config.net_file)),
        "route_sha256": _sha256(Path(config.route_file)),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sumo_version(binary: str) -> str | None:
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.splitlines()[0].strip() if result.stdout else None


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _format_number(value: float) -> str:
    return f"{float(value):g}"
