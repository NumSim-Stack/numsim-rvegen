"""VoronoiGenerator3d: bounded / periodic / weighted, both backends."""

import random

import pytest

import rvegen


def _seeds(n, L, rng_seed=1):
    rng = random.Random(rng_seed)
    return [(rng.random() * L[0], rng.random() * L[1], rng.random() * L[2])
            for _ in range(n)]


def test_public_api():
    assert hasattr(rvegen, "VoronoiGenerator3d")
    assert hasattr(rvegen, "lognormal_grain_seeds_3d")
    assert isinstance(rvegen.voro_pp_available(), bool)


def test_two_seeds_split_box():
    gen = rvegen.VoronoiGenerator3d(
        1.0, 1.0, 1.0, [(0.25, 0.5, 0.5), (0.75, 0.5, 0.5)])
    assert gen.seed_count == 2
    assert not gen.periodic
    cells = gen.build()
    assert len(cells) == 2
    for c in cells:
        assert isinstance(c, rvegen.VoronoiCell)
        assert c.volume() == pytest.approx(0.5)
        assert len(c.vertices) == 8
        assert len(c.faces) == 6
    assert cells[0].is_inside(0.25, 0.5, 0.5)
    assert not cells[0].is_inside(0.75, 0.5, 0.5)


@pytest.mark.parametrize("periodic", [False, True])
def test_partition_of_unity(periodic):
    L = (1.0, 1.5, 2.0)
    gen = rvegen.VoronoiGenerator3d(*L, _seeds(8, L), periodic=periodic)
    total = sum(c.volume() for c in gen.build())
    assert total == pytest.approx(L[0] * L[1] * L[2], rel=1e-9)


def test_periodic_cells_may_extend_outside_box():
    gen = rvegen.VoronoiGenerator3d(
        1.0, 1.0, 1.0, [(0.05, 0.5, 0.5), (0.55, 0.5, 0.5)], periodic=True)
    cells = gen.build()
    xs = [v[0] for v in cells[0].vertices]
    assert min(xs) < 0.0   # whole convex cell around the seed
    assert sum(c.volume() for c in cells) == pytest.approx(1.0)


def test_coincident_seeds_rejected():
    with pytest.raises(RuntimeError):
        rvegen.VoronoiGenerator3d(1, 1, 1, [(0.3, 0.3, 0.3)] * 2)
    # opposite faces coincide modulo the periodic box
    with pytest.raises(RuntimeError):
        rvegen.VoronoiGenerator3d(
            1, 1, 1, [(0.0, 0.5, 0.5), (1.0, 0.5, 0.5)], periodic=True)


def test_bad_backend_and_weights():
    with pytest.raises(ValueError):
        rvegen.VoronoiGenerator3d(1, 1, 1, [(0.5, 0.5, 0.5)], backend="nope")
    with pytest.raises(RuntimeError):
        rvegen.VoronoiGenerator3d(1, 1, 1, [(0.5, 0.5, 0.5)], weights=[0.0, 1.0])


def test_voro_pp_backend_selection():
    s = _seeds(10, (1, 1, 1))
    if not rvegen.voro_pp_available():
        with pytest.raises(RuntimeError):
            rvegen.VoronoiGenerator3d(1, 1, 1, s, backend="voro_pp")
        return
    a = rvegen.VoronoiGenerator3d(1, 1, 1, s, periodic=True).build()
    b = rvegen.VoronoiGenerator3d(1, 1, 1, s, periodic=True,
                                  backend="voro_pp").build()
    for ca, cb in zip(a, b):
        assert ca.volume() == pytest.approx(cb.volume(), abs=1e-9)


def test_weighted_partition():
    s = _seeds(6, (1, 1, 1), 4)
    gen = rvegen.VoronoiGenerator3d(
        1, 1, 1, s, periodic=True, weights=[0, .02, -.01, .015, 0, .03])
    assert sum(c.volume() for c in gen.build()) == pytest.approx(1.0, abs=1e-8)


def test_lognormal_grain_seeds():
    r = rvegen.lognormal_grain_seeds_3d(2.2, 1.0, 0.25, rng_seed=5)
    n = len(r["seeds"])
    assert n > 5
    assert r["max_rel_volume_error"] <= 0.01
    mean = sum(r["diameters"]) / n
    assert mean == pytest.approx(1.0, rel=0.1)
    cells = rvegen.VoronoiGenerator3d(
        2.2, 2.2, 2.2, r["seeds"], periodic=True,
        weights=r["weights"]).build()
    assert sum(c.volume() for c in cells) == pytest.approx(2.2 ** 3, rel=1e-8)
