from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scenediffuserpp.audit import audit_dataset
from scenediffuserpp.diagnostics import plot_scene
from scenediffuserpp.storage import SceneDataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit SceneDiffuser++ SUMO tensors.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report = audit_dataset(args.dataset, verify_hashes=not args.skip_hashes)
    (args.out / "audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    shard_paths = [args.dataset / name for name in sorted(manifest.get("shards", {}))]
    if shard_paths:
        dataset = SceneDataset(shard_paths)
        if len(dataset):
            plot_scene(dataset[0], args.out / "scene_000000.png")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
