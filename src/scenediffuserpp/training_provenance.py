from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Mapping

from scenediffuserpp.evaluation import verify_training_log


def build_checkpoint_run_config(
    *,
    training: Mapping[str, Any],
    model: Mapping[str, Any],
    dataset: str,
    tensor_contract: Mapping[str, Any],
    incomplete_corpus_override: bool,
    optimizer_provenance: Mapping[str, Any] | None,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "training": dict(training),
        "model": dict(model),
        "dataset": str(dataset),
        "tensor_contract": dict(tensor_contract),
        "incomplete_corpus_override": bool(incomplete_corpus_override),
        "execution": dict(execution),
    }
    if optimizer_provenance is not None:
        result["optimizer_provenance"] = dict(optimizer_provenance)
    return result


def verify_training_start(
    log_path: str | Path, *, checkpoint_step: int | None
) -> None:
    path = Path(log_path)
    if not path.is_file():
        if checkpoint_step is None:
            return
        raise ValueError("checkpoint resume requires an existing training log")
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except json.JSONDecodeError as exc:
        raise ValueError(f"training log is not valid JSONL: {exc}") from exc
    if checkpoint_step is None:
        if rows:
            raise ValueError(
                "fresh training output already contains a training log; "
                "use --resume or a new output directory"
            )
        return
    if not rows:
        raise ValueError("checkpoint resume requires a nonempty training log")
    verify_training_log(rows, checkpoint_step=checkpoint_step)


def verify_resume_contract(
    checkpoint_run_config: Mapping[str, Any],
    *,
    training: Mapping[str, Any],
    model: Mapping[str, Any],
    tensor_contract: Mapping[str, Any],
) -> None:
    expected = (
        ("training", dict(training), "training configuration"),
        ("model", dict(model), "model configuration"),
        ("tensor_contract", dict(tensor_contract), "tensor contract"),
    )
    for key, current, label in expected:
        checkpoint_value = checkpoint_run_config.get(key)
        if checkpoint_value != current:
            raise ValueError(f"resume checkpoint {label} differs")
