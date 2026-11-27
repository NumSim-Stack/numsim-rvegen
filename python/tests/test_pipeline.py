"""Pipeline, voxelize and registry-introspection tests for the rvegen bindings (#4)."""

from __future__ import annotations

import json

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
        (rvegen.post_processes, "damask_material_config"),
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
         "generator": {"type": "weave", "n_warp_yarns": 3, "n_weft_yarns": 3,
                       "yarn_radius": 0.05, "amplitude": 0.05}}
    )
    assert len(weave) == 6
    assert all(s.kind == "polyline_tube" for s in weave)


def test_run_phases():
    cfg = dict(CIRCLES)
    cfg["phases"] = [{"name": "fibre", "material": {"E": 1.0, "nu": 0.3}}]
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
