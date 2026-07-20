from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scenediffuserpp.roadgraph import load_roadgraph
from scenediffuserpp.normalization import POSITION_SCALE
from scenediffuserpp.scene_builder import build_window
from scenediffuserpp.scene_builder import candidate_windows
from scenediffuserpp.scene_builder import count_light_state_transitions
from scenediffuserpp.scene_builder import parse_fcd
from scenediffuserpp.scene_builder import parse_tls_jsonl
from scenediffuserpp.schema import dataset_build_config_from_mapping
from scenediffuserpp.storage import SCHEMA_VERSION
from scenediffuserpp.storage import assign_split
from scenediffuserpp.storage import canonical_json_sha256
from scenediffuserpp.storage import file_sha256
from scenediffuserpp.storage import write_shard


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AV-centric SceneDiffuser++ windows.")
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/scenediffuserpp/data_nanjing_10hz.yaml",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--max-windows-per-run", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--min-reference-agents", type=int)
    parser.add_argument("--min-light-state-transitions", type=int)
    parser.add_argument(
        "--require-reference-light",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    args = parser.parse_args()
    if args.max_windows is not None and args.max_windows <= 0:
        parser.error("--max-windows must be positive")
    if args.max_windows_per_run is not None and args.max_windows_per_run <= 0:
        parser.error("--max-windows-per-run must be positive")

    manifests = _find_manifests(args.runs)
    if not manifests:
        raise FileNotFoundError(f"no SUMO teacher manifests found under {args.runs}")
    raw_config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    build_config = dataset_build_config_from_mapping(raw_config)
    overrides = {
        "window_stride_steps": args.stride,
        "minimum_reference_agents": args.min_reference_agents,
        "minimum_light_state_transitions": args.min_light_state_transitions,
        "require_reference_light": args.require_reference_light,
    }
    build_config = replace(
        build_config,
        **{key: value for key, value in overrides.items() if value is not None},
    )
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    buffer = []
    buffer_rows = []
    shard_index = 0
    total = 0
    spec = build_config.scene_spec
    roadgraph_cache = {}
    for manifest_path in manifests:
        run_total = 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"SUMO teacher run is incomplete: {manifest_path}")
        config = manifest["config"]
        run_dir = manifest_path.parent
        tracks = parse_fcd(
            run_dir / "fcd.xml",
            route_file=config["route_file"],
            frequency_hz=spec.frequency_hz,
        )
        tls_states = parse_tls_jsonl(
            run_dir / "tls_states.jsonl", frequency_hz=spec.frequency_hz
        )
        roadgraph = _load_roadgraph_cached(
            roadgraph_cache,
            config["net_file"],
            point_spacing_m=build_config.map_point_spacing_m,
        )
        for av_id, start_step in candidate_windows(
            tracks,
            spec,
            stride=build_config.window_stride_steps,
            light_tokens=roadgraph.light_tokens,
            min_reference_agents=build_config.minimum_reference_agents,
            require_reference_light=build_config.require_reference_light,
            observation_radius_m=build_config.observation_radius_m,
            minimum_travel_m=build_config.minimum_av_travel_m,
        ):
            sample_id = f"{run_dir.name}_{av_id}_{start_step}"
            window = build_window(
                tracks,
                av_id=av_id,
                start_step=start_step,
                spec=spec,
                light_tokens=roadgraph.light_tokens,
                tls_states=tls_states,
                roadgraph=roadgraph,
                observation_radius_m=build_config.observation_radius_m,
                meta={
                    "sample_id": sample_id,
                    "run_id": run_dir.name,
                    "scenario_id": run_dir.name,
                    "seed": int(config["seed"]),
                    "net_file": str(Path(config["net_file"]).resolve()),
                    "route_file": str(Path(config["route_file"]).resolve()),
                    "tls_file": str((run_dir / "tls_states.jsonl").resolve()),
                },
            )
            light_state_transitions = count_light_state_transitions(window.lights)
            if (
                light_state_transitions
                < build_config.minimum_light_state_transitions
            ):
                continue
            buffer.append(window)
            buffer_rows.append(
                {
                    "sample_id": sample_id,
                    **window.meta,
                    "truncated_agents": window.truncated_agents,
                    "truncated_lights": window.truncated_lights,
                    "light_state_transitions": light_state_transitions,
                    "split": assign_split(
                        run_dir.name, seed=build_config.split_seed
                    ),
                }
            )
            total += 1
            run_total += 1
            if len(buffer) == build_config.shard_size:
                _flush_shard(
                    args.out,
                    shard_index,
                    buffer,
                    buffer_rows,
                    rows,
                    maximum_map_points=build_config.maximum_map_points,
                    maximum_map_lanes=build_config.maximum_map_lanes,
                    maximum_map_connections=build_config.maximum_map_connections,
                    map_radius_m=build_config.map_radius_m,
                )
                buffer.clear()
                buffer_rows.clear()
                shard_index += 1
            if _window_limit_reached(
                total=total,
                run_total=run_total,
                max_windows=args.max_windows,
                max_windows_per_run=args.max_windows_per_run,
            ):
                break
        if args.max_windows is not None and total >= args.max_windows:
            break
    if buffer:
        _flush_shard(
            args.out,
            shard_index,
            buffer,
            buffer_rows,
            rows,
            maximum_map_points=build_config.maximum_map_points,
            maximum_map_lanes=build_config.maximum_map_lanes,
            maximum_map_connections=build_config.maximum_map_connections,
            map_radius_m=build_config.map_radius_m,
        )
    if not rows:
        raise RuntimeError("no scene windows satisfied the configured filters")
    shards = sorted(args.out.glob("shard_*.h5"))
    split_counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "validation", "test")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_config": raw_config,
        "dataset_config_sha256": canonical_json_sha256(raw_config),
        "dataset_config_file": str(args.config.resolve()),
        "dataset_config_file_sha256": file_sha256(args.config),
        "normalization": {"position_scale_m": float(POSITION_SCALE)},
        "effective_build_config": asdict(build_config),
        "source_manifests": {
            str(path.resolve()): file_sha256(path) for path in manifests
        },
        "split_counts": split_counts,
        "samples": rows,
        "shards": {path.name: file_sha256(path) for path in shards},
    }
    _write_json_atomic(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {"windows": total, "runs": len(manifests), "splits": split_counts},
            indent=2,
        )
    )
    return 0


def _flush_shard(
    output_dir,
    shard_index,
    windows,
    pending_rows,
    rows,
    *,
    maximum_map_points,
    maximum_map_lanes,
    maximum_map_connections,
    map_radius_m,
) -> None:
    shard_path = output_dir / f"shard_{shard_index:05d}.h5"
    write_shard(
        shard_path,
        windows,
        max_map_points=maximum_map_points,
        max_map_lanes=maximum_map_lanes,
        max_map_connections=maximum_map_connections,
        map_radius_m=map_radius_m,
    )
    for sample_index, row in enumerate(pending_rows):
        rows.append(
            {**row, "shard": shard_path.name, "sample_index": sample_index}
        )


def _find_manifests(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    direct = root / "manifest.json"
    if direct.is_file():
        return [direct]
    return sorted(root.rglob("manifest.json"))


def _window_limit_reached(
    *,
    total: int,
    run_total: int,
    max_windows: int | None,
    max_windows_per_run: int | None,
) -> bool:
    if max_windows is not None and total >= max_windows:
        return True
    if max_windows_per_run is not None and run_total >= max_windows_per_run:
        return True
    return False


def _load_roadgraph_cached(
    cache: dict[tuple[str, float], object],
    net_file: str,
    *,
    point_spacing_m: float,
):
    key = (str(Path(net_file).resolve()), float(point_spacing_m))
    if key not in cache:
        cache[key] = load_roadgraph(key[0], point_spacing_m=point_spacing_m)
    return cache[key]


def _write_json_atomic(path: Path, values: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())
