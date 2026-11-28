"""Pipeline, voxelize and registry-introspection tests for the rvegen bindings (#4)."""

from __future__ import annotations

import json
from pathlib import Path
import signal
import threading
import time
import warnings

import numpy as np
import pytest

import rvegen

CIRCLES = {
    "domain_box": [1.0, 1.0, 0.0],
    "seed": 2026,
    "distributions": {
        "x": {"type": "uniform_real", "a": 0.1, "b": 0.9},
        "y": {"type": "uniform_real", "a": 0.1, "b": 0.9},
        "r": {"type": "uniform_real", "a": 0.03, "b": 0.05},
    },
    "shapes": [
        {"type": "circle_input", "pos_x_dist": "x", "pos_y_dist": "y", "radius_dist": "r"}
    ],
    "generator": {"type": "only_inside", "max_attempts": 100000},
    "termination": {"type": "number_of_inclusions", "target": 10},
}


# ---- registry introspection ----------------------------------------------

@pytest.mark.parametrize(
    "category, expected",
    [
        (rvegen.distributions, "uniform_real"),
        (rvegen.direction_distributions, "bingham"),
        (rvegen.inputs, "circle_input"),
        (rvegen.inputs, "polyline_tube_oriented_input"),
        (rvegen.generators, "only_inside"),
        (rvegen.generators, "weave"),
        (rvegen.generators, "voronoi_3d"),
        (rvegen.terminations, "number_of_inclusions"),
        (rvegen.post_processes, "voxel"),
    ],
)
def test_keys(category, expected):
    keys = category.keys()
    assert expected in keys
    assert keys == sorted(keys)


def test_shapes_alias():
    assert rvegen.shapes.keys() == rvegen.inputs.keys()


def test_schema_fields():
    s = rvegen.distributions.schema("uniform_real")
    assert set(s) == {"a", "b"}
    assert s["a"]["type"] == "double"
    assert s["a"]["required"] is True
    assert s["a"]["description"]

    w = rvegen.generators.schema("weave")
    assert w["n_warp_yarns"]["min"] == 1
    assert w["n_segments_per_yarn"]["has_default"] is True
    assert w["n_segments_per_yarn"]["required"] is False


def test_schema_unknown_type():
    with pytest.raises(KeyError):
        rvegen.generators.schema("no_such_generator")


# ---- run ------------------------------------------------------------------

def test_run_dict():
    res = rvegen.run(CIRCLES)
    assert len(res) == 10
    assert res.summary["n_shapes"] == 10
    assert res.summary["seed"] == 2026
    assert res.summary["domain_box"] == [1.0, 1.0, 0.0]
    assert 0.0 < res.summary["volume_fraction"] < 1.0
    for s in res:
        assert isinstance(s, rvegen.Circle)
        assert isinstance(s, rvegen.Shape)
        assert s.kind == "circle"
        assert s.area() > 0
        assert 0.0 <= s.center[0] <= 1.0
    assert res[0].kind == "circle"


def test_run_seed_override_and_determinism():
    a = [s.center for s in rvegen.run(CIRCLES, seed=7)]
    b = [s.center for s in rvegen.run(CIRCLES, seed=7)]
    c = [s.center for s in rvegen.run(CIRCLES, seed=8)]
    assert a == b
    assert a != c
    assert rvegen.run(CIRCLES, seed=7).summary["seed"] == 7


def test_run_path_and_postprocess(tmp_path):
    cfg = dict(CIRCLES)
    out = tmp_path / "grid.vox"
    cfg["postprocess"] = [
        {"type": "voxel", "nx": 8, "ny": 8, "nz": 1, "output_path": str(out)}
    ]
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    res = rvegen.run(cfg_path)               # os.PathLike
    assert out.exists()
    assert res.summary["postprocess"] == ["voxel"]

    out.unlink()
    rvegen.run(str(cfg_path), postprocess=False)   # str path, no writers
    assert not out.exists()


def test_run_deterministic_generators_need_no_termination():
    res = rvegen.run(
        {"domain_box": [1.0, 1.0, 0.0],
         "generator": {"type": "voronoi_2d", "n_seeds": 9, "seed": 3}}
    )
    assert len(res) == 9
    assert all(s.kind == "convex_polygon" for s in res)
    assert sum(s.area() for s in res) == pytest.approx(1.0)

    weave = rvegen.run(
        {"domain_box": [1.0, 1.0, 0.4],
         "generator": {"type": "weave", "n_warp_yarns": 4, "n_weft_yarns": 2,
                       "yarn_radius": 0.05, "amplitude": 0.06}}
    )
    assert len(weave) == 6
    assert all(s.kind == "polyline_tube" for s in weave)


def test_run_phases():
    cfg = dict(CIRCLES)
    cfg["phases"] = [{"name": "fibre"}]
    cfg["phase_assignment"] = {"all_inclusions": "fibre"}
    res = rvegen.run(cfg)
    assert res.summary["phases"] == ["fibre"]
    assert all(s.phase_name == "fibre" for s in res)


def test_run_errors():
    with pytest.raises(RuntimeError):
        rvegen.run({"generator": {"type": "nope"}})
    with pytest.raises(ValueError):
        rvegen.run(42)
    with pytest.raises(RuntimeError):
        rvegen.run("/no/such/config.json")


# ---- voxelize -------------------------------------------------------------

def test_voxelize_2d():
    shapes = [rvegen.Circle(0.5, 0.5, 0.25)]
    grid = rvegen.voxelize(shapes, [1.0, 1.0, 0.0], 64, 64, 1)
    assert grid.dtype == np.uint32
    assert grid.shape == (64, 64, 1)
    assert grid[32, 32, 0] == 1 and grid[0, 0, 0] == 0
    # area fraction ~ pi r^2
    assert (grid > 0).mean() == pytest.approx(np.pi * 0.25**2, abs=0.01)


def test_voxelize_3d_orientation_and_ids():
    shapes = [rvegen.Sphere(0.25, 0.5, 0.5, 0.2), rvegen.Sphere(0.75, 0.5, 0.5, 0.2)]
    grid = rvegen.voxelize(shapes, [1.0, 1.0, 1.0], 16, 8, 4)
    assert grid.shape == (16, 8, 4)          # indexed [i, j, k]
    assert grid[4, 4, 2] == 1                # x = 0.28 -> first sphere
    assert grid[12, 4, 2] == 2               # x = 0.78 -> second sphere
    assert grid[8, 4, 2] == 0                # between them


def test_voxelize_layout_is_fortran_and_documented():
    grid = rvegen.voxelize([rvegen.Circle(0.5, 0.5, 0.25)], [1.0, 1.0, 0.0], 8, 6, 1)
    assert grid.flags.f_contiguous
    assert "F-contiguous" in rvegen.voxelize.__doc__
    c = np.ascontiguousarray(grid)
    assert c.flags.c_contiguous and np.array_equal(c, grid)


def test_voxelize_rejects_zero_dims():
    for dims in [(0, 4, 1), (4, 0, 1), (4, 4, 0)]:
        with pytest.raises(ValueError):
            rvegen.voxelize([rvegen.Circle(0.5, 0.5, 0.25)], [1.0, 1.0, 0.0], *dims)


def test_config_accepts_numpy_scalars_and_rejects_nan():
    cfg = {"domain_box": [np.float64(1.0), 1.0, 0.0],
           "generator": {"type": "voronoi_2d", "n_seeds": np.int64(4), "seed": 2}}
    assert len(rvegen.run(cfg, postprocess=False)) == 4
    cfg["domain_box"] = [float("nan"), 1.0, 0.0]
    with pytest.raises(ValueError, match="NaN"):
        rvegen.run(cfg, postprocess=False)


def test_voxelize_run_output_and_shape_reuse():
    shapes = list(rvegen.run(CIRCLES))
    grid = rvegen.voxelize(shapes, [1.0, 1.0, 0.0], 32, 32, 1)
    assert grid.max() <= len(shapes)
    assert (grid > 0).any()
    again = rvegen.voxelize(shapes, [1.0, 1.0, 0.0], 32, 32, 1)   # shapes untouched
    assert np.array_equal(grid, again)


# ---- interruption, warnings, quiet stdout, domain validation, seeds -------

def _long_config():
    cfg = json.loads(json.dumps(CIRCLES))
    cfg["distributions"]["r"] = {"type": "constant", "value": 0.0005}
    cfg["generator"]["max_attempts"] = 100_000_000
    cfg["termination"]["target"] = 3_000_000
    return cfg


@pytest.mark.skipif(not hasattr(signal, "raise_signal"), reason="needs signal.raise_signal")
def test_run_is_interruptible():
    # A long run must raise KeyboardInterrupt soon after SIGINT arrives.
    cfg = _long_config()
    timer = threading.Timer(0.5, lambda: signal.raise_signal(signal.SIGINT))
    timer.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            rvegen.run(cfg, postprocess=False)
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < 5.0


@pytest.mark.skipif(not hasattr(signal, "raise_signal"), reason="needs signal.raise_signal")
def test_voxelize_is_interruptible():
    # The job is sized from timed runs, so the signal lands inside the call
    # however fast the machine is: add shapes until an uninterrupted call
    # takes at least a second, then interrupt a quarter of the way in.
    domain, n = [1.0, 1.0, 1.0], 160
    shapes = [rvegen.Sphere(0.5, 0.5, 0.5, 0.45)] * 8
    while True:
        t0 = time.monotonic()
        rvegen.voxelize(shapes, domain, n, n, n)
        full = time.monotonic() - t0
        if full >= 1.0:
            break
        assert len(shapes) < 100_000, "voxelize is too fast to interrupt"
        shapes = shapes * 2
    timer = threading.Timer(full / 4, lambda: signal.raise_signal(signal.SIGINT))
    timer.start()
    t0 = time.monotonic()
    try:
        with pytest.raises(KeyboardInterrupt):
            rvegen.voxelize(shapes, domain, n, n, n)
    finally:
        timer.cancel()
    assert time.monotonic() - t0 < full


def test_run_not_interrupted_without_signal():
    assert len(rvegen.run(CIRCLES, postprocess=False)) == 10


def test_warnings_as_errors_propagate():
    cfg = json.loads(json.dumps(CIRCLES))
    cfg["bogus_key"] = 1   # unknown top-level key: UserWarning with strict=False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="bogus_key"):
            rvegen.run(cfg, postprocess=False, strict=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rvegen.run(cfg, postprocess=False, strict=False)
    assert any("bogus_key" in str(w.message) for w in caught)


def test_unknown_top_level_key_is_an_error():
    cfg = json.loads(json.dumps(CIRCLES))
    cfg["post_process"] = []
    with pytest.raises(ValueError, match=r"'post_process' \(did you mean 'postprocess'\?\)"):
        rvegen.run(cfg, postprocess=False)
    with pytest.raises(ValueError, match="post_process"):
        rvegen.check(cfg)
    cfg = json.loads(json.dumps(CIRCLES))
    cfg["_comment"] = "keys starting with an underscore are annotations"
    assert rvegen.check(cfg) is None


def test_check_validates_without_generating(tmp_path):
    cfg = json.loads(json.dumps(CIRCLES))
    out = tmp_path / "a.svg"
    cfg["postprocess"] = [
        {"type": "svg", "canvas_width": 200, "output_path": str(out)},
        {"type": "svg", "output_path": str(tmp_path / "b.svg")},   # no canvas_width
    ]
    # A missing key is a ValueError, as everywhere else in a config.
    with pytest.raises(ValueError, match=r"postprocess\[1\] \('svg'\).*canvas_width"):
        rvegen.check(cfg)
    with pytest.raises(ValueError, match=r"postprocess\[1\]"):
        rvegen.run(cfg)
    assert not out.exists()            # the first writer did not run either
    cfg["postprocess"][1]["canvas_width"] = 100
    assert rvegen.check(cfg) is None
    assert list(tmp_path.iterdir()) == []   # a check writes nothing
    cfg["postprocess"][1]["output_path"] = str(tmp_path / "nowhere" / "b.svg")
    with pytest.raises(RuntimeError, match="does not exist"):
        rvegen.check(cfg)
    # A path to a config file works as for run().
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(CIRCLES))
    assert rvegen.check(str(path)) is None
    assert rvegen.check(CIRCLES, seed=3) is None


def test_run_prints_nothing_to_stdout(capfd):
    cfg = json.loads(json.dumps(CIRCLES))   # relies on defaulted keys
    rvegen.run(cfg, postprocess=False)
    out, err = capfd.readouterr()
    assert out == ""
    assert "using default" not in err


@pytest.mark.parametrize("domain", [
    [0.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, -0.5],
    [float("nan"), 1.0, 0.0], [1.0, float("inf"), 0.0],
])
def test_voxelize_rejects_invalid_domain(domain):
    with pytest.raises(ValueError):
        rvegen.voxelize([rvegen.Circle(0.5, 0.5, 0.25)], domain, 4, 4, 1)


def test_voxelize_accepts_2d_and_3d_domain():
    assert rvegen.voxelize([], [1.0, 1.0, 0.0], 4, 4, 1).shape == (4, 4, 1)
    assert rvegen.voxelize([], [1.0, 2.0, 0.5], 4, 4, 2).shape == (4, 4, 2)


def test_voronoi_follows_run_seed():
    cfg = {"domain_box": [1.0, 1.0, 0.0],
           "generator": {"type": "voronoi_2d", "n_seeds": 8}}
    a = rvegen.run(cfg, seed=1, postprocess=False)
    b = rvegen.run(cfg, seed=2, postprocess=False)
    a2 = rvegen.run(cfg, seed=1, postprocess=False)
    assert a.summary["seed"] == 1 and b.summary["seed"] == 2
    areas = lambda r: [s.area() for s in r]
    assert areas(a) != areas(b)
    assert areas(a) == areas(a2)
    # An explicit generator seed overrides the run seed.
    cfg["generator"]["seed"] = 7
    assert areas(rvegen.run(cfg, seed=1, postprocess=False)) == \
        areas(rvegen.run(cfg, seed=2, postprocess=False))
    # Top-level config seed behaves the same way.
    cfg["generator"].pop("seed")
    cfg["seed"] = 5
    c1 = areas(rvegen.run(cfg, postprocess=False))
    cfg["seed"] = 6
    assert c1 != areas(rvegen.run(cfg, postprocess=False))


def test_periodic_copies_are_linked_to_their_primary():
    cfg = {
        "domain_box": [1.0, 1.0, 0.0], "seed": 3,
        "distributions": {"x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
                          "r": {"type": "uniform_real", "a": 0.05, "b": 0.09}},
        "shapes": [{"type": "circle_input", "pos_x_dist": "x",
                    "pos_y_dist": "x", "radius_dist": "r"}],
        "generator": {"type": "periodic", "max_attempts": 100000},
        "termination": {"type": "number_of_inclusions", "target": 20},
    }
    res = rvegen.run(cfg, postprocess=False)
    primaries = [s for s in res if s.is_primary]
    copies = [s for s in res if not s.is_primary]
    assert len(primaries) == 20 and copies          # some cross a face
    by_id = {p.inclusion_id: p for p in primaries if p.inclusion_id is not None}
    for c in copies:
        p = by_id[c.inclusion_id]
        assert c.image_shift != (0, 0, 0) and p.image_shift == (0, 0, 0)
        for k in range(2):                              # copy = primary + shift * L
            assert abs(c.center[k] - (p.center[k] + c.image_shift[k] * 1.0)) < 1e-12
        assert c.area() == p.area()
    # A shape without copies has no id; the summary counts each inclusion once.
    assert any(p.inclusion_id is None for p in primaries)
    import math
    assert math.isclose(res.summary["volume_fraction"],
                        sum(p.area() for p in primaries), rel_tol=1e-12)


# ---- #129: paths relative to the config file, output_dir, log callback ----

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.mark.skipif(not (EXAMPLES / "mesh_inclusions_3d.json").is_file(),
                    reason="examples/ not available (installed wheel)")
def test_config_file_runs_from_any_directory(tmp_path, monkeypatch):
    elsewhere = tmp_path / "cwd"
    out = tmp_path / "out"
    elsewhere.mkdir()
    out.mkdir()
    monkeypatch.chdir(elsewhere)
    # "mesh_path": "data/octahedron.stl" is found next to the config file.
    res = rvegen.run(EXAMPLES / "mesh_inclusions_3d.json", output_dir=out)
    assert len(res) == 40
    assert (out / "mesh_inclusions_3d.vtk").is_file()
    assert list(elsewhere.iterdir()) == []
    assert rvegen.check(str(EXAMPLES / "mesh_inclusions_3d.json")) is None

    # The same config as a dict has no file to be relative to: input_dir.
    cfg = json.loads((EXAMPLES / "mesh_inclusions_3d.json").read_text())
    with pytest.raises(RuntimeError, match="octahedron.stl"):
        rvegen.run(cfg, postprocess=False)
    res2 = rvegen.run(cfg, postprocess=False, input_dir=EXAMPLES)
    assert [s.center for s in res2] == [s.center for s in res]
    with pytest.raises(RuntimeError, match="does not exist"):
        rvegen.check(cfg, input_dir=EXAMPLES, output_dir=tmp_path / "missing")


def test_log_callback(capfd):
    cfg = json.loads(json.dumps(CIRCLES))
    lines = []
    res = rvegen.run(cfg, postprocess=False, log=lines.append)
    assert lines and all(isinstance(l, str) for l in lines)
    assert any(f"generated {len(res)} shapes" in l for l in lines)
    assert not any("seed" in l for l in lines)          # detail lines: verbose only

    detailed = []
    rvegen.run(cfg, postprocess=False, log=detailed.append, verbose=True)
    assert any(l.startswith("rvegen: seed ") for l in detailed)
    assert set(lines) <= set(detailed)

    checked = []
    rvegen.check(cfg, log=checked.append, verbose=True)
    assert any(l.startswith("rvegen: seed ") for l in checked)

    # Nothing is printed by rvegen itself, with or without a callback.
    out, err = capfd.readouterr()
    assert out == "" and err == ""

    # verbose without a callback stays silent.
    rvegen.run(cfg, postprocess=False, verbose=True)
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_log_callback_exception_propagates():
    cfg = json.loads(json.dumps(CIRCLES))

    class Boom(Exception):
        pass

    def bad(line):
        raise Boom(line)

    with pytest.raises(Boom):
        rvegen.run(cfg, postprocess=False, log=bad, verbose=True)
    with pytest.raises(Boom):
        rvegen.check(cfg, log=bad, verbose=True)
    with pytest.raises(TypeError):
        rvegen.run(cfg, None, False, True, ".")     # input_dir is keyword-only
    # The module is still usable afterwards.
    assert len(rvegen.run(cfg, postprocess=False)) > 0


def test_log_callback_failure_stops_before_writers(tmp_path):
    cfg = json.loads(json.dumps(CIRCLES))
    cfg["postprocess"] = [{"type": "svg", "canvas_width": 200, "output_path": "late.svg"}]

    def bad(line):
        if "generated" in line:
            raise KeyError(line)

    with pytest.raises(KeyError):
        rvegen.run(cfg, output_dir=tmp_path, log=bad)
    assert not (tmp_path / "late.svg").exists()        # no writer was started
    rvegen.run(cfg, output_dir=tmp_path, log=lambda line: None)
    assert (tmp_path / "late.svg").is_file()

    # The callable's exception is not hidden by a later pipeline error
    # (the output directory is missing, reported after the first debug line).
    def always(line):
        raise KeyError(line)

    with pytest.raises(KeyError):
        rvegen.check(cfg, output_dir=tmp_path / "missing", log=always, verbose=True)

    # Not a callable: rejected before anything runs.
    for not_callable in (True, 5, "print"):
        with pytest.raises(TypeError, match="callable"):
            rvegen.run(cfg, postprocess=False, log=not_callable)
        with pytest.raises(TypeError, match="callable"):
            rvegen.check(cfg, log=not_callable)


@pytest.mark.skipif(not (EXAMPLES / "mesh_inclusions_3d.json").is_file(),
                    reason="examples/ not available (installed wheel)")
def test_explicit_input_dir_never_falls_back_to_cwd(tmp_path, monkeypatch):
    cfg = json.loads((EXAMPLES / "mesh_inclusions_3d.json").read_text())
    monkeypatch.chdir(EXAMPLES)            # data/octahedron.stl exists here
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="octahedron.stl"):
        rvegen.check(cfg, input_dir=empty)
    assert rvegen.check(cfg) is None       # a dict: the working directory


def test_metadata_object_and_string_forms_agree():
    def run(metadata):
        cfg = json.loads(json.dumps(CIRCLES))
        cfg["shapes"][0]["metadata"] = metadata
        return rvegen.run(cfg, postprocess=False)

    obj = {"batch": "A7", "angle_deg": 42.5}
    a, b = run(obj), run(json.dumps(obj))
    assert [s.info for s in a] == [s.info for s in b]
    assert all(s.info["batch"] == "A7" for s in a)
    with pytest.raises(ValueError, match="'metadata'"):
        run([1, 2])
