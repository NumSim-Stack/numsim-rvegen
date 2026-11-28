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


def test_in_house_limit_and_override():
    s = _seeds(301, (1, 1, 1))
    with pytest.raises(ValueError, match="allow_slow_backend"):
        rvegen.VoronoiGenerator3d(1, 1, 1, s, backend="in_house")
    gen = rvegen.VoronoiGenerator3d(1, 1, 1, s, backend="in_house",
                                    allow_slow_backend=True)
    assert gen.seed_count == 301


def test_ten_thousand_grains_with_voro_pp():
    """A size the in-house backend cannot do (#123)."""
    if not rvegen.voro_pp_available():
        pytest.skip("built without Voro++")
    L = (1.0, 1.0, 1.0)
    cells = rvegen.VoronoiGenerator3d(*L, _seeds(10000, L), periodic=True).build()
    assert len(cells) == 10000
    assert abs(sum(c.volume() for c in cells) - 1.0) < 1e-6


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


def test_lognormal_grain_report_fields_and_robustness():
    for sd in range(1, 4):
        r = rvegen.lognormal_grain_seeds_3d(3.5, 1.0, 0.8, rng_seed=sd,
                                            backend="in_house")
        assert r["converged"]
        assert r["empty_cells"] == 0
        assert min(r["diameters"]) > 0
        assert r["max_rel_volume_error"] <= 0.01


def test_lognormal_grain_nonconvergence_warns():
    with pytest.warns(RuntimeWarning, match="did not reach"):
        r = rvegen.lognormal_grain_seeds_3d(
            3.5, 1.0, 0.8, rng_seed=1, backend="in_house",
            tolerance=1e-12, max_iterations=3, max_restarts=0)
    assert not r["converged"]


def test_default_backend_is_auto():
    # The default must work without naming a backend (auto picks Voro++
    # when built in, in-house otherwise).
    gen = rvegen.VoronoiGenerator3d(1.0, 1.0, 1.0,
                                    _seeds(20, (1.0, 1.0, 1.0), 3),
                                    periodic=True)
    assert sum(c.volume() for c in gen.build()) == pytest.approx(1.0, rel=1e-8)


def test_build_releases_gil():
    import threading
    import time
    gen = rvegen.VoronoiGenerator3d(
        1.0, 1.0, 1.0, _seeds(150, (1.0, 1.0, 1.0), 3), periodic=True,
        backend="in_house")
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        t0 = time.perf_counter()
        gen.build()
        elapsed = time.perf_counter() - t0
    finally:
        stop.set()
        t.join()
    assert elapsed > 0.1, "build too fast to observe GIL behaviour"
    # With the GIL held for the whole build the ticker would get ~0 ticks.
    assert len(ticks) > 10


def test_weighted_empty_cell_warns():
    seeds = [(0.25, 0.5, 0.5), (0.75, 0.5, 0.5), (0.5, 0.2, 0.8)]
    gen = rvegen.VoronoiGenerator3d(1.0, 1.0, 1.0, seeds, periodic=True,
                                    weights=[-10.0, 0.0, 0.0])
    with pytest.warns(RuntimeWarning, match="empty cell"):
        cells = gen.build()
    assert cells[0].volume() == 0.0
