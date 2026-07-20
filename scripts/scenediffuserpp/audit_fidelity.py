from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scenediffuserpp.fidelity_audit import write_audit_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Write the SceneDiffuser++ clean-room fidelity audit."
    )
    parser.add_argument("--json-out", required=True, type=Path)
    parser.add_argument("--markdown-out", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    json_path, markdown_path = write_audit_artifacts(
        args.json_out,
        args.markdown_out,
        git_revision=revision,
        force=args.force,
    )
    print(json_path.resolve())
    print(markdown_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
