# Performance

How long rvegen takes, how that grows with the size of the problem, and how
to measure it again.

## Running the benchmarks

    scripts/run_benchmarks.py --rvegen build/rvegen

runs the cases of [`bench/cases.py`](../bench/cases.py), whole pipeline runs
of the command-line binary, three times each, and writes `results.md` (the
tables below) and `results.json` to `bench-results/`. The whole set takes
about a minute. Build the binary in `Release`; the times of an unoptimised
build say nothing.

| option | |
|---|---|
| `--quick` | leave out the largest sizes |
| `--filter REGEX` | only the cases whose name matches |
| `--repeat N` | runs per case (default 3); the fastest counts |
| `--check-scaling` | exit 1 if a family grows faster than `--max-exponent` (default 1.7) |
| `--compare OLD.json` | print new / old per case, exit 1 beyond `--max-ratio` (default 1.5) |
| `--emit DIR` | write the configs and input files to `DIR` without running |
| `--list` | print the case names |

To see what a change does, keep the `results.json` of the commit before and
pass it to `--compare` on the same machine.

## Reference

rvegen 0.1.0, 2026-11-28, AMD Ryzen 7 PRO 6850U with Radeon Graphics (16 logical cores), Linux 7.0.0-43-generic; fastest of 3 run(s) per case.
| case | size | shapes | attempts | time [s] | per unit [µs] |
|---|---:|---:|---:|---:|---:|
| **startup** | | | | | |
| `startup` | 1 | 1 | 1 | 0.003 |  |
| **circles, periodic, 40 %** | | | | | |
| `circles_periodic_1000` | 1 000 | 1 069 | 5 991 | 0.006 | 1.08 |
| `circles_periodic_10000` | 10 000 | 10 420 | 59 186 | 0.046 | 0.77 |
| `circles_periodic_100000` | 100 000 | 103 456 | 600 285 | 0.643 | 1.07 |
| **circles, only_inside, 30 %** | | | | | |
| `circles_inside_10000` | 10 000 | 10 078 | 26 786 | 0.019 | 0.69 |
| `circles_inside_100000` | 100 000 | 101 212 | 270 427 | 0.365 | 1.35 |
| **circles, periodic, 40 %, min_gap** | | | | | |
| `circles_min_gap_10000` | 10 000 | 10 641 | 254 771 | 0.099 | 0.39 |
| `circles_min_gap_100000` | 100 000 | 105 671 | 2 605 964 | 2.404 | 0.92 |
| **circles + rectangles + ellipses, 25 %** | | | | | |
| `mixed_2d_10000` | 10 000 | 11 255 | 24 245 | 0.035 | 1.46 |
| `mixed_2d_100000` | 100 000 | 111 717 | 247 426 | 0.561 | 2.27 |
| **spheres, periodic, 30 %** | | | | | |
| `spheres_periodic_1000` | 1 000 | 1 384 | 34 395 | 0.018 | 0.53 |
| `spheres_periodic_10000` | 10 000 | 12 464 | 313 979 | 0.165 | 0.52 |
| `spheres_periodic_100000` | 100 000 | 117 993 | 3 260 845 | 4.443 | 1.36 |
| **spheres + boxes, 15 %** | | | | | |
| `mixed_3d_10000` | 10 000 | 11 563 | 27 668 | 0.033 | 1.21 |
| `mixed_3d_100000` | 100 000 | 110 601 | 277 192 | 0.732 | 2.64 |
| **straight fibres, 5 %** | | | | | |
| `fibres_100` | 100 | 100 | 1 664 | 0.003 | 1.85 |
| `fibres_400` | 400 | 400 | 41 306 | 0.043 | 1.04 |
| **mesh inclusions, 20 480 triangles each** | | | | | |
| `mesh_inclusions_200` | 200 | 200 | 414 | 0.074 | 178.61 |
| `mesh_inclusions_2000` | 2 000 | 2 000 | 3 551 | 0.360 | 101.47 |
| **voronoi_2d, periodic** | | | | | |
| `voronoi_2d_1000` | 1 000 | 1 088 |  | 0.018 | 16.73 |
| `voronoi_2d_10000` | 10 000 | 10 256 |  | 0.206 | 20.04 |
| `voronoi_2d_100000` | 100 000 | 100 794 |  | 2.755 | 27.33 |
| **voronoi_3d, periodic** | | | | | |
| `voronoi_3d_500` | 500 | 810 |  | 0.012 | 14.27 |
| `voronoi_3d_2000` | 2 000 | 2 763 |  | 0.041 | 14.69 |
| `voronoi_3d_8000` | 8 000 | 9 887 |  | 0.178 | 18.03 |
| **twill weave, 16 x 16 yarns** | | | | | |
| `weave_twill_16` | 16 | 32 |  | 0.002 |  |
| **writers** | | | | | |
| `write_voxel_2d_2048` | 4 194 304 | 10 420 | 59 186 | 0.198 |  |
| `write_voxel_3d_128` | 2 097 152 | 1 384 | 34 395 | 0.098 |  |
| `write_voxel_3d_256` | 16 777 216 | 1 384 | 34 395 | 0.649 |  |
| `write_voxel_3d_256_dense` | 16 777 216 | 12 464 | 313 979 | 0.851 |  |
| `write_vtk_3d_128` | 2 097 152 | 1 384 | 34 395 | 0.100 |  |
| `write_gmsh_geo_2d` | 1 000 | 1 069 | 5 991 | 0.016 |  |
| `write_svg_2d` | 10 000 | 10 420 | 59 186 | 0.041 |  |
| `write_statistics_2d` | 10 000 | 10 420 | 59 186 | 0.545 |  |
| `write_three_js_3d` | 1 000 | 1 384 | 34 395 | 0.020 |  |
| `write_voxel_weave` | 4 194 304 | 32 |  | 0.732 |  |

Scaling (exponent of the time over the work, 1 = linear, 2 = quadratic):

| family | from | to | exponent | |
|---|---|---|---:|---|
| circles, periodic, 40 % | `circles_periodic_1000` | `circles_periodic_10000` |  | not checked: circles_periodic_1000 takes under 10 ms |
| circles, periodic, 40 % | `circles_periodic_10000` | `circles_periodic_100000` | 1.17 | ok |
| circles, only_inside, 30 % | `circles_inside_10000` | `circles_inside_100000` | 1.36 | ok |
| circles, periodic, 40 %, min_gap | `circles_min_gap_10000` | `circles_min_gap_100000` | 1.39 | ok |
| circles + rectangles + ellipses, 25 % | `mixed_2d_10000` | `mixed_2d_100000` | 1.23 | ok |
| spheres, periodic, 30 % | `spheres_periodic_1000` | `spheres_periodic_10000` | 1.07 | ok |
| spheres, periodic, 30 % | `spheres_periodic_10000` | `spheres_periodic_100000` | 1.42 | ok |
| spheres + boxes, 15 % | `mixed_3d_10000` | `mixed_3d_100000` | 1.38 | ok |
| straight fibres, 5 % | `fibres_100` | `fibres_400` |  | not checked: fibres_100 takes under 10 ms |
| mesh inclusions, 20 480 triangles each | `mesh_inclusions_200` | `mesh_inclusions_2000` | 0.75 | ok |
| voronoi_2d, periodic | `voronoi_2d_1000` | `voronoi_2d_10000` | 1.15 | ok |
| voronoi_2d, periodic | `voronoi_2d_10000` | `voronoi_2d_100000` | 1.14 | ok |
| voronoi_3d, periodic | `voronoi_3d_500` | `voronoi_3d_2000` |  | not checked: voronoi_3d_500 takes under 10 ms |
| voronoi_3d, periodic | `voronoi_3d_2000` | `voronoi_3d_8000` | 1.21 | ok |

The time is the wall-clock time of the whole process: reading the config,
generating, writing. "per unit" is that time per placement attempt, or per
shape for the tessellations. The writer cases are whole runs too; the
placement case of the same size next to them is the part that is not
writing.

What the numbers say:

- A placement attempt of a circle or sphere costs between half a
  microsecond and a microsecond and a half, and that stays within a factor
  of three from 10^3 to 10^5 inclusions. The rise towards 10^5 (exponent
  1.2 to 1.4) has not been traced to a cause; a quadratic step would show
  as a factor of ten per decade.
- How many attempts a run needs is a property of the packing, not of the
  code: 40 % of circles take about 6 attempts per shape, 30 % of spheres
  about 25, 400 thin fibres about 100, and a minimum gap multiplies it.
  Near the densest packing a generator reaches, the attempts are the run
  time.
- `voronoi_2d` costs 17 to 27 µs per cell and `voronoi_3d` 14 to 18 µs, both
  close to linear in the number of seeds.
- A voxel grid costs 30 to 40 ns per voxel; the statistics writer takes half
  a second for its 512 x 512 grid.

## Scaling check

Absolute times depend on the machine, and on a shared CI runner they are
too noisy to gate anything. How the time grows with the size is not. For
two sizes of one family the script computes

    exponent = log(t2 / t1) / log(w2 / w1)

with `t` the time less the start-up time of the binary and `w` the work of
the run (placement attempts, or shapes for the tessellations). Linear work
gives 1, a pass over everything per inclusion gives 2. A pair whose smaller
run is under 10 ms is not judged. `--check-scaling` fails above 1.7, and CI
runs it on every pull request (the table is attached to the run as the
`benchmarks` artifact and shown in the job summary).

This is the check that would have caught the two quadratic paths found by
hand before it existed: a `volume_fraction` target re-summed all inclusions
after every acceptance (#153; 10^5 circles 147 s, now 0.6 s), and
`voronoi_2d` clipped every cell against every seed (#154; 10^4 seeds 38 s,
now 0.2 s).

## Mesh inclusions, per call

`rvegen_mesh_bench` (built when Google Benchmark is installed, not run by
ctest) times the single operations on a mesh inclusion. The test mesh is a
sphere of radius 1; same machine as above.

| operation | 1 280 triangles | 20 480 | 81 920 |
|---|---:|---:|---:|
| new mesh from its triangles (checks, components, tree) | 1.29 ms | 14.4 ms | 70.4 ms |
| new orientation (bounds of the rotated mesh) | 7.2 µs | 40 µs | 91 µs |
| point inside? | 0.097 µs | 0.13 µs | 0.16 µs |
| against a sphere near the surface | 0.48 µs | 2.3 µs | 5.7 µs |
| against a box near the surface | 0.49 µs | 1.1 µs | 2.5 µs |
| against a mesh, deep overlap | 6.5 µs | 11 µs | 30 µs |
| against a mesh, surfaces just crossing | 9 µs | 64 µs | 95 µs |
| against a mesh, 0.02 apart | 0.14 ms | 0.51 ms | 0.75 ms |
| against a mesh, bounding boxes apart | 5 ns | 5 ns | 5 ns |
| minimum gap 0.05, meshes 0.02 apart | 8.6 µs | 82 µs | 0.53 ms |
| minimum gap 0.05, meshes 0.2 apart | 71 µs | 86 µs | 73 µs |

The pair tests go through a tree over the triangles (#126), so they grow
with the patch of surface that is near the other shape, not with the mesh.
The expensive case is two meshes that nearly touch without touching: no
pair of triangles ends the search early.

`rvegen_dispatch_bench` compares ways of dispatching a pair test on the
dynamic types of two shapes; it documents why the dispatcher is a table.

## Known slow spots

- `voronoi_2d` with a great many seeds at the same distance from one point
  (all seeds on one circle): every cell reaches that point, every seed may
  cut every cell, and the cost is quadratic (20 000 seeds: 26 s).
- Two mesh inclusions that nearly touch: 0.1 to 0.8 ms per test.
- Dense packings: the attempts grow steeply towards the densest fill a
  random sequential placement reaches; `max_attempts` is the bound.
- A voxel grid of 256^3 takes 0.6 to 0.7 s whatever is in it.
