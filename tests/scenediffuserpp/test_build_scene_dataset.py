from pathlib import Path
import importlib.util


def _load_build_scene_dataset():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "scenediffuserpp"
        / "build_scene_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("build_scene_dataset", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_window_limit_respects_per_run_cap_before_global_cap():
    module = _load_build_scene_dataset()

    assert module._window_limit_reached(
        total=4,
        run_total=1,
        max_windows=5,
        max_windows_per_run=2,
    ) is False
    assert module._window_limit_reached(
        total=4,
        run_total=2,
        max_windows=5,
        max_windows_per_run=2,
    ) is True


def test_window_limit_respects_global_cap_without_per_run_cap():
    module = _load_build_scene_dataset()

    assert module._window_limit_reached(
        total=4,
        run_total=99,
        max_windows=5,
        max_windows_per_run=None,
    ) is False
    assert module._window_limit_reached(
        total=5,
        run_total=0,
        max_windows=5,
        max_windows_per_run=None,
    ) is True
