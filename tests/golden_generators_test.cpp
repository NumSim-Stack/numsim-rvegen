// Golden test per generator (#130): a fixed config must give the same
// shape list on every toolchain.
//
// Each config below runs through the pipeline; the result is reduced to an
// FNV-1a hash over the bit patterns of every shape's primary flag, centre,
// area and volume, in order. CI runs this test
// against libstdc++, libc++ and the MSVC standard library; the hashes are
// the same on all of them.
//
// Two classes of config:
//   * drawn: the hashed values are the drawn values themselves or products
//     of them (circles, spheres, ellipses). These hashes hold everywhere,
//     including builds where the compiler fuses multiply-add (FMA).
//   * derived: the hashed values come out of further geometry (tube
//     centre lines, Voronoi cells, the grain-size fit). They are the same
//     across standard libraries, but FMA contraction changes their last
//     bits, so they are checked only when this translation unit is not
//     contracted (a run-time probe decides).
//
// If this test fails after an intentional change, rerun with `--dump` and
// replace the table, and say so in the change log: seeded RVEs change.

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/pipeline.h"

namespace {

int failures = 0;
bool dump = false;

struct fnv {
  std::uint64_t h = 1469598103934665603ull;
  void add(std::uint64_t bits) {
    for (int i = 0; i < 8; ++i) {
      h ^= (bits >> (8 * i)) & 0xffu;
      h *= 1099511628211ull;
    }
  }
  void add(double v) { add(std::bit_cast<std::uint64_t>(v)); }
};

// True if the compiler fuses a * b + c in this translation unit.
bool contracts() {
  volatile double va = 1.0 + 0x1p-30;
  volatile double vc = -1.0;
  const double a = va;
  const double c = vc;
  // a * a = 1 + 2^-29 + 2^-60; the last term is lost unless fused.
  const double r = a * a + c;
  return r != 0x1p-29;
}

struct golden {
  char const* name;
  bool derived;          // see the header comment
  std::size_t shapes;    // expected number of shapes
  std::uint64_t hash;
  char const* config;
};

constexpr golden cases[] = {
  {"only_inside_circles", false, 40, 0x1cfc57d387d446e8ull, R"({
     "domain_box": [1.0, 1.0], "seed": 11,
     "distributions": {
       "x": {"type": "uniform_real", "a": 0.05, "b": 0.95},
       "r": {"type": "normal", "mean": 0.04, "stddev": 0.008}},
     "shapes": [{"type": "circle_input", "pos_x_dist": "x", "pos_y_dist": "x",
                 "radius_dist": "r"}],
     "generator": {"type": "only_inside", "max_attempts": 100000},
     "termination": {"type": "number_of_inclusions", "target": 40}})"},
  {"random_spheres", false, 129, 0xd5578f16e22aeac0ull, R"({
     "domain_box": [1.0, 1.0, 1.0], "seed": 12,
     "distributions": {
       "x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
       "r": {"type": "uniform_real", "a": 0.04, "b": 0.09}},
     "shapes": [{"type": "sphere_input", "pos_x_dist": "x", "pos_y_dist": "x",
                 "pos_z_dist": "x", "radius_dist": "r"}],
     "generator": {"type": "random", "max_attempts": 100000},
     "termination": {"type": "volume_fraction", "target_fraction": 0.15}})"},
  {"periodic_circles", false, 49, 0xf7d195d6fb3892beull, R"({
     "domain_box": [1.0, 1.0], "seed": 13,
     "distributions": {
       "x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
       "r": {"type": "uniform_real", "a": 0.03, "b": 0.07}},
     "shapes": [{"type": "circle_input", "pos_x_dist": "x", "pos_y_dist": "x",
                 "radius_dist": "r"}],
     "generator": {"type": "periodic", "max_attempts": 100000},
     "termination": {"type": "volume_fraction", "target_fraction": 0.3}})"},
  {"periodic_spheres", false, 42, 0xe687c3ccd140c84dull, R"({
     "domain_box": [1.0, 1.0, 1.0], "seed": 14,
     "distributions": {
       "x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
       "r": {"type": "normal", "mean": 0.08, "stddev": 0.01}},
     "shapes": [{"type": "sphere_input", "pos_x_dist": "x", "pos_y_dist": "x",
                 "pos_z_dist": "x", "radius_dist": "r"}],
     "generator": {"type": "periodic", "max_attempts": 100000},
     "termination": {"type": "number_of_inclusions", "target": 30}})"},
  {"periodic_ellipses_von_mises", false, 29, 0xa9e3768e02c5c017ull, R"({
     "domain_box": [1.0, 1.0], "seed": 15,
     "distributions": {
       "x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
       "a": {"type": "uniform_real", "a": 0.06, "b": 0.1},
       "b": {"type": "uniform_real", "a": 0.025, "b": 0.05},
       "angle": {"type": "von_mises", "mean_angle": 0.4, "kappa": 3.0}},
     "shapes": [{"type": "ellipse_input", "pos_x_dist": "x", "pos_y_dist": "x",
                 "radius_a_dist": "a", "radius_b_dist": "b",
                 "rotation_dist": "angle"}],
     "generator": {"type": "periodic", "max_attempts": 100000},
     "termination": {"type": "volume_fraction", "target_fraction": 0.2}})"},
  {"random_tubes_bingham", true, 15, 0x4f180cdbe8e42019ull, R"({
     "domain_box": [1.0, 1.0, 1.0], "seed": 16,
     "distributions": {
       "pos": {"type": "uniform_real", "a": 0.2, "b": 0.8},
       "length": {"type": "uniform_real", "a": 0.15, "b": 0.25},
       "radius": {"type": "constant", "value": 0.02}},
     "direction_distributions": {
       "dir": {"type": "bingham", "mean_axis_x": 1.0, "mean_axis_y": 0.5,
               "mean_axis_z": 0.0, "kappa": 8.0}},
     "shapes": [{"type": "polyline_tube_oriented_input",
                 "position_x_dist": "pos", "position_y_dist": "pos",
                 "position_z_dist": "pos", "direction_dist": "dir",
                 "length_dist": "length", "radius_dist": "radius"}],
     "generator": {"type": "random", "max_attempts": 10000},
     "termination": {"type": "number_of_inclusions", "target": 15}})"},
  {"voronoi_2d", true, 24, 0xf089560ab02f73c0ull, R"({
     "domain_box": [1.0, 1.0],
     "generator": {"type": "voronoi_2d", "n_seeds": 24, "seed": 7}})"},
  {"voronoi_3d", true, 12, 0xcc547fa18b6441f2ull, R"({
     "domain_box": [1.0, 1.0, 1.0],
     "generator": {"type": "voronoi_3d", "n_seeds": 12, "seed": 3,
                   "backend": "in_house"}})"},
  {"voronoi_3d_grain_size", true, 14, 0xcb57e9a879a9cabeull, R"({
     "domain_box": [1.0, 1.0, 1.0], "seed": 5,
     "generator": {"type": "voronoi_3d", "backend": "in_house",
                   "grain_diameter_mean": 0.5, "grain_diameter_sigma": 0.2,
                   "grain_size_tolerance": 0.1,
                   "grain_size_max_iterations": 50}})"},
};

void run(golden const& g, bool contracted) {
  rvegen::pipeline_options opts;
  opts.info = [](std::string const&) {};
  opts.warn = [](std::string const&) {};
  const auto result = rvegen::generate(nlohmann::json::parse(g.config), opts);
  fnv h;
  for (auto const& s : result.shapes) {
    h.add(static_cast<std::uint64_t>(s->is_primary() ? 1 : 0));
    for (double c : s->get_middle_point()) h.add(c);
    h.add(s->area());
    h.add(s->volume());
  }
  if (dump) {
    std::printf("  %-28s %s shapes %zu hash 0x%016llxull\n", g.name,
                g.derived ? "derived" : "drawn  ", result.shapes.size(),
                static_cast<unsigned long long>(h.h));
    return;
  }
  if (g.derived && contracted) {
    std::printf("  %s: skipped (derived geometry, FMA contraction is on)\n",
                g.name);
    return;
  }
  if (result.shapes.size() != g.shapes || h.h != g.hash) {
    std::fprintf(stderr,
                 "%s: expected %zu shapes, hash 0x%016llx; got %zu shapes, "
                 "hash 0x%016llx\n",
                 g.name, g.shapes, static_cast<unsigned long long>(g.hash),
                 result.shapes.size(), static_cast<unsigned long long>(h.h));
    ++failures;
  }
}

} // namespace

int main(int argc, char** argv) {
  dump = argc > 1 && std::strcmp(argv[1], "--dump") == 0;
  rvegen::register_all_types();
  const bool contracted = contracts();
  if (dump) std::printf("contraction: %s\n", contracted ? "on" : "off");
  for (auto const& g : cases) {
    try {
      run(g, contracted);
    } catch (std::exception const& e) {
      std::fprintf(stderr, "%s: %s\n", g.name, e.what());
      ++failures;
    }
  }
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  if (!dump) std::printf("golden_generators_test: all passed\n");
  return 0;
}
