// Pair tests and queries on mesh inclusions (#126), per call.
//
// What one placement attempt of a mesh inclusion costs: the tests against a
// sphere, a box and another mesh through the collision dispatcher, the
// point query, and what a new orientation and a new mesh cost. The argument
// is the subdivision level of the test mesh, a sphere of 20 * 4^level
// triangles (3: 1280, 5: 20 480, 6: 81 920): with the triangle tree the
// pair tests should grow with the logarithm of that number, not with the
// number.
//
// Not run by ctest. Build with Google Benchmark installed and run
//   ./rvegen_mesh_bench

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include <benchmark/benchmark.h>

#include "rvegen/rvegen.h"

namespace {

using D = double;
using V = gte::Vector3<D>;
using triangle_list = std::vector<gte::Triangle3<D>>;
using mesh = rvegen::mesh_inclusion<D>;

// Unit sphere of 20 * 4^level triangles.
triangle_list icosphere(int level) {
  const D t = (1 + std::sqrt(5.0)) / 2;
  std::vector<V> v{{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
                   {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
                   {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
  for (auto& p : v) gte::Normalize(p);
  std::vector<std::array<int, 3>> f{
      {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
      {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
      {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
      {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
  for (int l = 0; l < level; ++l) {
    std::map<std::pair<int, int>, int> mid;
    std::vector<std::array<int, 3>> next;
    auto middle = [&](int a, int b) {
      const auto key = std::minmax(a, b);
      if (auto it = mid.find(key); it != mid.end()) return it->second;
      V p = v[a] + v[b];
      gte::Normalize(p);
      v.push_back(p);
      return mid[key] = static_cast<int>(v.size()) - 1;
    };
    for (auto const& t3 : f) {
      const int a = middle(t3[0], t3[1]);
      const int b = middle(t3[1], t3[2]);
      const int c = middle(t3[2], t3[0]);
      next.push_back({t3[0], a, c});
      next.push_back({t3[1], b, a});
      next.push_back({t3[2], c, b});
      next.push_back({a, b, c});
    }
    f = std::move(next);
  }
  triangle_list out;
  out.reserve(f.size());
  for (auto const& t3 : f) {
    gte::Triangle3<D> x;
    for (int k = 0; k < 3; ++k) x.v[k] = v[t3[k]];
    out.push_back(x);
  }
  return out;
}

// One mesh per level, built once: copies share triangles and tree.
mesh const& prototype(int level) {
  static std::map<int, mesh> cache;
  auto it = cache.find(level);
  if (it == cache.end()) it = cache.emplace(level, mesh{icosphere(level)}).first;
  return it->second;
}

mesh placed(int level, std::array<D, 3> const& at, D angle) {
  mesh m{prototype(level)};
  m.set_rotation_axis_angle({0.6, 0.0, 0.8}, angle);
  m.set_middle_point(at);
  m.make_bounding_box();
  return m;
}

struct points {
  std::mt19937_64 engine{155};
  D uniform(D a, D b) {
    return a + (b - a) * (static_cast<D>(engine() >> 11) * 0x1.0p-53);
  }
  // On the sphere of the given radius around the origin.
  std::array<D, 3> on_sphere(D radius) {
    for (;;) {
      const std::array<D, 3> d{uniform(-1, 1), uniform(-1, 1), uniform(-1, 1)};
      const D n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      if (n > 0.1 && n <= 1)
        return {radius * d[0] / n, radius * d[1] / n, radius * d[2] / n};
    }
  }
};

constexpr std::size_t probes = 256;

void set_triangles(benchmark::State& state) {
  state.counters["triangles"] =
      static_cast<double>(20ull << (2 * state.range(0)));
}

// A new mesh from its triangles: orientation check, components, tree.
void mesh_build(benchmark::State& state) {
  const auto triangles = icosphere(static_cast<int>(state.range(0)));
  for (auto _ : state) {
    mesh m{triangles};
    benchmark::DoNotOptimize(m);
  }
  set_triangles(state);
}

// A new orientation: the bounds of the rotated mesh.
void mesh_set_rotation(benchmark::State& state) {
  auto m = placed(static_cast<int>(state.range(0)), {0, 0, 0}, 0.3);
  D angle = 0;
  for (auto _ : state) {
    angle += 0.37;
    m.set_rotation_axis_angle({0.6, 0.0, 0.8}, angle);
    m.make_bounding_box();
    benchmark::DoNotOptimize(m);
  }
  set_triangles(state);
}

// Points near the surface, half of them inside.
void mesh_point_inside(benchmark::State& state) {
  const auto m = placed(static_cast<int>(state.range(0)), {0, 0, 0}, 0.3);
  points p;
  std::vector<std::array<D, 3>> at;
  for (std::size_t k = 0; k < probes; ++k)
    at.push_back(p.on_sphere(p.uniform(0.9, 1.1)));
  std::size_t k = 0, inside = 0;
  for (auto _ : state) {
    inside += m.is_inside(at[k++ % probes]) ? 1 : 0;
  }
  benchmark::DoNotOptimize(inside);
  set_triangles(state);
}

// Small spheres around the surface: touching or not by a small margin.
void mesh_sphere(benchmark::State& state) {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  const auto m = placed(static_cast<int>(state.range(0)), {0, 0, 0}, 0.3);
  points p;
  std::vector<rvegen::sphere<D>> others;
  for (std::size_t k = 0; k < probes; ++k) {
    const auto c = p.on_sphere(p.uniform(1.0, 1.2));
    others.emplace_back(c[0], c[1], c[2], 0.1);
  }
  std::size_t k = 0, hits = 0;
  for (auto _ : state) {
    hits += dispatch(m, others[k++ % probes]) ? 1 : 0;
  }
  benchmark::DoNotOptimize(hits);
  set_triangles(state);
}

void mesh_box(benchmark::State& state) {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  const auto m = placed(static_cast<int>(state.range(0)), {0, 0, 0}, 0.3);
  points p;
  std::vector<rvegen::box<D>> others;
  for (std::size_t k = 0; k < probes; ++k) {
    const auto c = p.on_sphere(p.uniform(1.0, 1.25));
    others.emplace_back(c[0], c[1], c[2], 0.2, 0.2, 0.2);
  }
  std::size_t k = 0, hits = 0;
  for (auto _ : state) {
    hits += dispatch(m, others[k++ % probes]) ? 1 : 0;
  }
  benchmark::DoNotOptimize(hits);
  set_triangles(state);
}

// Two meshes at a given distance of their centres (radius 1 each):
//   1.0   deep overlap: a crossing pair of triangles is found at once
//   1.98  the surfaces cross in a small lens
//   2.02  no contact, 0.02 apart: the most work, every near pair is looked at
//   3.0   the bounding boxes do not overlap
void mesh_mesh(benchmark::State& state, D distance) {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  const int level = static_cast<int>(state.range(0));
  const auto a = placed(level, {0, 0, 0}, 0.3);
  points p;
  std::vector<mesh> others;
  for (std::size_t k = 0; k < 16; ++k)
    others.push_back(placed(level, p.on_sphere(distance), p.uniform(0, 6)));
  std::size_t k = 0, hits = 0;
  for (auto _ : state) {
    hits += dispatch(a, others[k++ % others.size()]) ? 1 : 0;
  }
  benchmark::DoNotOptimize(hits);
  set_triangles(state);
}

// With a minimum gap of 0.05 between the surfaces (distance query).
void mesh_mesh_gap(benchmark::State& state, D distance) {
  auto& dispatch = rvegen::collision_dispatcher<D>::instance();
  const int level = static_cast<int>(state.range(0));
  const auto a = placed(level, {0, 0, 0}, 0.3);
  points p;
  std::vector<mesh> others;
  for (std::size_t k = 0; k < 16; ++k)
    others.push_back(placed(level, p.on_sphere(distance), p.uniform(0, 6)));
  std::size_t k = 0, hits = 0;
  for (auto _ : state) {
    hits += dispatch(a, others[k++ % others.size()], D{0.05}) ? 1 : 0;
  }
  benchmark::DoNotOptimize(hits);
  set_triangles(state);
}

BENCHMARK(mesh_build)->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMillisecond);
BENCHMARK(mesh_set_rotation)->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK(mesh_point_inside)->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK(mesh_sphere)->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK(mesh_box)->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(mesh_mesh, deep_overlap, 1.0)
    ->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(mesh_mesh, crossing, 1.98)
    ->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(mesh_mesh, near_miss, 2.02)
    ->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(mesh_mesh, apart, 3.0)
    ->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(mesh_mesh_gap, inside_gap, 2.02)
    ->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);
BENCHMARK_CAPTURE(mesh_mesh_gap, outside_gap, 2.2)
    ->Arg(3)->Arg(5)->Arg(6)->Unit(benchmark::kMicrosecond);

} // namespace

int main(int argc, char** argv) {
  rvegen::register_all_collision_pairs<D>();
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
