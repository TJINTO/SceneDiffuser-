from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from topoworld.scenediffuserpp.sumo_export import SumoTeacherConfig
from topoworld.scenediffuserpp.sumo_export import run_teacher


def main() -> int:
    parser = argparse.ArgumentParser(description="Record 10 Hz SUMO SceneDiffuser++ teachers.")
    parser.add_argument("--net", required=True, type=Path)
    route_group = parser.add_mutually_exclusive_group()
    route_group.add_argument("--routes", type=Path)
    route_group.add_argument("--random-trips-period", type=float)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--begin", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=60.0)
    parser.add_argument("--recording-begin", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sumo-binary", default="sumo")
    parser.add_argument("--resume-complete", action="store_true")
    args = parser.parse_args()

    if args.config:
        manifests = _run_config_grid(args)
    else:
        if args.routes is None and args.random_trips_period is None:
            parser.error("provide --routes or --random-trips-period")
        manifests = [
            _run_one(
                net_file=args.net,
                route_file=args.routes,
                random_trips_period=args.random_trips_period,
                output_dir=args.out,
                begin=args.begin,
                end=args.end,
                recording_begin=args.recording_begin,
                seed=args.seed,
                sumo_binary=args.sumo_binary,
                resume_complete=args.resume_complete,
            )
        ]
    print(json.dumps({"runs": manifests}, indent=2))
    return 0


def _run_config_grid(args) -> list[dict]:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = config["dataset"]
    run_grid = config["runs"]
    duration = float(dataset["duration_s"])
    warmup = float(dataset.get("warmup_s", 0.0))
    if warmup < 0.0 or duration <= 0.0:
        raise ValueError("warmup must be nonnegative and duration must be positive")
    manifests = []
    for period in run_grid["departure_period_s"]:
        for seed in run_grid["seeds"]:
            run_dir = args.out / f"period_{float(period):g}_seed_{int(seed)}"
            manifests.append(
                _run_one(
                    net_file=args.net,
                    route_file=None,
                    random_trips_period=float(period),
                    output_dir=run_dir,
                    begin=0.0,
                    end=warmup + duration,
                    recording_begin=warmup,
                    seed=int(seed),
                    sumo_binary=args.sumo_binary,
                    resume_complete=args.resume_complete,
                )
            )
    return manifests


def _run_one(
    net_file: Path,
    route_file: Path | None,
    random_trips_period: float | None,
    output_dir: Path,
    begin: float,
    end: float,
    recording_begin: float | None,
    seed: int,
    sumo_binary: str,
    resume_complete: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if route_file is None:
        if random_trips_period is None or random_trips_period <= 0.0:
            raise ValueError("random-trips period must be positive")
        route_file = output_dir / "routes.rou.xml"
        if not route_file.exists():
            _generate_random_routes(
                net_file=net_file,
                route_file=route_file,
                period=random_trips_period,
                begin=begin,
                end=end,
                seed=seed,
            )
    return run_teacher(
        SumoTeacherConfig(
            net_file=net_file,
            route_file=route_file,
            output_dir=output_dir,
            seed=seed,
            begin_s=begin,
            end_s=end,
            recording_begin_s=recording_begin,
        ),
        sumo_binary=sumo_binary,
        resume_complete=resume_complete,
    )


def _generate_random_routes(
    net_file: Path,
    route_file: Path,
    period: float,
    begin: float,
    end: float,
    seed: int,
) -> None:
    script = _random_trips_script()
    command = [
        sys.executable,
        str(script),
        "-n",
        str(net_file.resolve()),
        "-r",
        str(route_file.resolve()),
        "--begin",
        f"{begin:g}",
        "--end",
        f"{end:g}",
        "--period",
        f"{period:g}",
        "--seed",
        str(seed),
        "--min-distance",
        "300",
        "--validate",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    (route_file.parent / "randomTrips.stdout.log").write_text(
        result.stdout, encoding="utf-8"
    )
    (route_file.parent / "randomTrips.stderr.log").write_text(
        result.stderr, encoding="utf-8"
    )
    if result.returncode != 0:
        raise RuntimeError(f"randomTrips failed with exit code {result.returncode}")


def _random_trips_script() -> Path:
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        candidate = Path(sumo_home) / "tools" / "randomTrips.py"
        if candidate.is_file():
            return candidate
    try:
        import sumolib

        candidate = Path(sumolib.__file__).resolve().parents[1] / "tools" / "randomTrips.py"
        if candidate.is_file():
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError("cannot locate SUMO tools/randomTrips.py; set SUMO_HOME")


if __name__ == "__main__":
    raise SystemExit(main())
