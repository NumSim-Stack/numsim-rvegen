// Exact two-point statistics by FFT (#127).
//
// 1. voxel_correlator against a reference that walks over every pair of
//    voxels: periodic and bounded boxes, 2D and 3D, two different sets,
//    odd and even and prime sizes, boxes that are not cubes. The numbers
//    are ratios of integers and must be equal, not close.
// 2. S2 at lag 0 is the volume fraction of the grid, exactly.
// 3. One disc and one sphere reproduce the analytic overlap curve.
// 4. A periodic random packing tends to phi^2; its profiles along the axes
//    agree. Aligned fibres: the profile along the fibres stays at phi, the
//    one across drops.
// 5. Phases: every phase and pair, and the sum rule of a periodic box.
// 6. The statistics post-process: columns, the r = 0 row, the boundary it
//    picks from the generator, phase columns.
// 7. A program's own pocketfft next to rvegen's (fft_isolation_*.cpp).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"
#include "rvegen/pipeline.h"
#include "rvegen/postprocess/statistics_post_process.h"
#include "rvegen/statistics/microstructure_stats.h"
#include "rvegen/statistics/two_point_fft.h"

double fft_isolation_first();
double fft_isolation_second();

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

using json = nlohmann::json;
using rvegen::phase_id;
using shape_ptr = std::unique_ptr<rvegen::shape_base<double>>;
using shape_vector = std::vector<shape_ptr>;

template <typename S, typename... A>
shape_ptr make(A&&... a) {
  auto s = std::make_unique<S>(std::forward<A>(a)...);
  s->make_bounding_box();
  return s;
}

bool same(double a, double b) {   // equal, NaN included
  return (std::isnan(a) && std::isnan(b)) || a == b;
}

// The reference: every ordered pair of voxels (p, q), its lag q - p.
rvegen::two_point_profile brute_force(
    std::vector<phase_id> const& grid, std::array<std::size_t, 3> const& n,
    std::array<double, 3> const& extent, bool periodic, double max_r,
    std::size_t nbins, phase_id a, phase_id b) {
  using rvegen::portable_detail::rounded;
  std::vector<std::uint64_t> count(nbins, 0), pairs(nbins, 0);
  std::array<std::vector<std::uint64_t>, 3> axis_count, axis_pairs;
  for (auto& v : axis_count) v.assign(nbins, 0);
  for (auto& v : axis_pairs) v.assign(nbins, 0);
  std::uint64_t zero_count = 0, zero_pairs = 0;
  const std::array<double, 3> d{extent[0] / double(n[0]),
                                extent[1] / double(n[1]),
                                extent[2] / double(n[2])};
  auto lag = [&](std::size_t p, std::size_t q, std::size_t len) {
    auto l = static_cast<std::ptrdiff_t>(q) - static_cast<std::ptrdiff_t>(p);
    if (periodic) {   // the shortest of the equivalent lags, +len/2 on a tie
      const auto m = static_cast<std::ptrdiff_t>(len);
      if (2 * l > m) l -= m;
      if (2 * l <= -m) l += m;
    }
    return l;
  };
  auto at = [&](std::size_t i, std::size_t j, std::size_t k) {
    return grid[(k * n[1] + j) * n[0] + i];
  };
  const double dr = max_r / double(nbins);
  for (std::size_t k0 = 0; k0 < n[2]; ++k0)
    for (std::size_t j0 = 0; j0 < n[1]; ++j0)
      for (std::size_t i0 = 0; i0 < n[0]; ++i0)
        for (std::size_t k1 = 0; k1 < n[2]; ++k1)
          for (std::size_t j1 = 0; j1 < n[1]; ++j1)
            for (std::size_t i1 = 0; i1 < n[0]; ++i1) {
              const auto lx = lag(i0, i1, n[0]);
              const auto ly = lag(j0, j1, n[1]);
              const auto lz = lag(k0, k1, n[2]);
              const double x = double(lx) * d[0], y = double(ly) * d[1],
                           z = double(lz) * d[2];
              const double yz2 = rounded(y * y) + rounded(z * z);
              const double r = std::sqrt(rounded(x * x) + yz2);
              if (!(r < max_r)) continue;
              const std::uint64_t hit =
                  at(i0, j0, k0) == a && at(i1, j1, k1) == b ? 1 : 0;
              if (lx == 0 && ly == 0 && lz == 0) {
                zero_count += hit;
                ++zero_pairs;
                continue;
              }
              const std::size_t bin =
                  std::min(static_cast<std::size_t>(r / dr), nbins - 1);
              count[bin] += hit;
              ++pairs[bin];
              const int zeros = (lx == 0) + (ly == 0) + (lz == 0);
              if (zeros == 2) {
                const std::size_t ax = lx != 0 ? 0 : (ly != 0 ? 1 : 2);
                axis_count[ax][bin] += hit;
                ++axis_pairs[ax][bin];
              }
            }
  auto ratio = [&](std::vector<std::uint64_t> const& c,
                   std::vector<std::uint64_t> const& p) {
    std::vector<double> v(nbins, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t bin = 0; bin < nbins; ++bin)
      if (p[bin] > 0) v[bin] = double(c[bin]) / double(p[bin]);
    return v;
  };
  rvegen::two_point_profile out;
  out.zero = double(zero_count) / double(zero_pairs);
  out.radial = ratio(count, pairs);
  for (std::size_t ax = 0; ax < 3; ++ax)
    if (n[ax] > 1) out.axis[ax] = ratio(axis_count[ax], axis_pairs[ax]);
  return out;
}

bool same(rvegen::two_point_profile const& a,
          rvegen::two_point_profile const& b) {
  if (!same(a.zero, b.zero) || a.radial.size() != b.radial.size()) return false;
  for (std::size_t i = 0; i < a.radial.size(); ++i)
    if (!same(a.radial[i], b.radial[i])) return false;
  for (std::size_t ax = 0; ax < 3; ++ax) {
    if (a.axis[ax].size() != b.axis[ax].size()) return false;
    for (std::size_t i = 0; i < a.axis[ax].size(); ++i)
      if (!same(a.axis[ax][i], b.axis[ax][i])) return false;
  }
  return true;
}

void test_against_every_pair() {
  std::mt19937 engine{7};
  struct setup {
    std::array<std::size_t, 3> n;
    std::array<double, 3> extent;
  };
  const std::vector<setup> setups{
      {{12, 9, 1}, {1.3, 0.7, 0.0}},     // 2D, even and odd
      {{7, 7, 1}, {1.0, 1.0, 0.0}},      // prime sizes
      {{16, 1, 1}, {2.0, 1.0, 0.0}},     // a single row
      {{1, 11, 1}, {1.0, 3.0, 0.0}},     // a single column
      {{6, 5, 4}, {1.0, 0.8, 1.6}},      // 3D
      {{5, 8, 1}, {1.0, 1.0, 2.0}},      // one slice of a 3D box
      {{4, 4, 7}, {0.5, 0.5, 2.0}},
  };
  std::size_t compared = 0;
  for (auto const& s : setups) {
    std::vector<phase_id> grid(s.n[0] * s.n[1] * s.n[2]);
    for (auto& v : grid) v = static_cast<phase_id>(engine() % 3);
    double smallest = std::min(s.extent[0], s.extent[1]);
    if (s.extent[2] > 0.0) smallest = std::min(smallest, s.extent[2]);
    // Half the smallest edge (the default), a short range, and one beyond
    // the box (every lag there is).
    for (double max_r : {0.5 * smallest, 0.21 * smallest, 10.0}) {
      for (std::size_t nbins :
           {std::size_t{1}, std::size_t{5}, std::size_t{40}}) {
        for (bool periodic : {true, false}) {
          const rvegen::voxel_correlator c{
              s.n, s.extent, {periodic, periodic, periodic}, max_r, nbins};
          const auto A = c.transform(grid, [](phase_id v) { return v == 1; });
          const auto B = c.transform(grid, [](phase_id v) { return v == 2; });
          const auto aa = c.correlate(A, A);
          const auto ab = c.correlate(A, B);
          REQUIRE(same(aa, brute_force(grid, s.n, s.extent, periodic, max_r,
                                       nbins, 1, 1)));
          // A against B: a bin holds a lag and its opposite, so the order
          // of the two sets does not matter.
          const auto ref = brute_force(grid, s.n, s.extent, periodic, max_r,
                                       nbins, 1, 2);
          REQUIRE(same(ab, ref));
          REQUIRE(same(c.correlate(B, A), ref));
          REQUIRE(ab.zero == 0.0);   // no voxel is in both sets
          compared += 3;
        }
      }
    }
  }
  REQUIRE(compared == 7 * 3 * 3 * 2 * 3);
}

void test_zero_lag_is_the_voxel_fraction() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  shape_vector shapes;
  std::mt19937 engine{3};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  for (int s = 0; s < 30; ++s)
    shapes.push_back(make<rvegen::circle<double>>(u(engine), u(engine),
                                                  0.02 + 0.05 * u(engine)));
  for (auto boundary : {rvegen::two_point_boundary::periodic,
                        rvegen::two_point_boundary::bounded}) {
    const auto s2 =
        rvegen::two_point_correlations(shapes, box, 97, 64, 1, 20, boundary);
    const auto grid = rvegen::sample_voxel_grid(shapes, box, 97, 64, 1);
    const auto filled = static_cast<double>(std::count_if(
        grid.begin(), grid.end(), [](phase_id v) { return v != 0; }));
    REQUIRE(s2.inclusion.zero == filled / double(97 * 64));   // exactly
    REQUIRE(s2.nbins == 20 && s2.max_r == 0.5);
    REQUIRE((s2.grid == std::array<std::size_t, 3>{97, 64, 1}));
    REQUIRE(s2.phases.empty());
    REQUIRE(s2.inclusion.axis[2].empty());   // no lags along z in 2D
    // The wrappers return the radial average.
    const auto radial =
        rvegen::two_point_correlation_2d(shapes, box, 97, 64, 20, boundary);
    REQUIRE(radial.size() == 20);
    for (std::size_t b = 0; b < 20; ++b) {
      REQUIRE(radial[b].s2 == s2.inclusion.radial[b]);
      REQUIRE(radial[b].r == 0.5 * (double(b) + 0.5) / 20.0);
      // In a periodic box every lag has the same number of pairs, and S2
      // cannot exceed the volume fraction.
      if (boundary == rvegen::two_point_boundary::periodic)
        REQUIRE(radial[b].s2 <= s2.inclusion.zero);
    }
  }
}

// Area of the lens two discs of radius R share at distance r, and the
// volume two spheres share.
double disc_overlap(double R, double r) {
  if (r >= 2.0 * R) return 0.0;
  return 2.0 * R * R * std::acos(r / (2.0 * R)) -
         0.5 * r * std::sqrt(4.0 * R * R - r * r);
}
double sphere_overlap(double R, double r) {
  if (r >= 2.0 * R) return 0.0;
  return std::numbers::pi / 12.0 * (4.0 * R + r) * (2.0 * R - r) *
         (2.0 * R - r);
}

void test_single_disc_and_sphere() {
  {   // disc: the box is large enough that no periodic image comes near
    const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
    const double R = 0.2;
    shape_vector shapes;
    shapes.push_back(make<rvegen::circle<double>>(0.47, 0.52, R));
    for (auto boundary : {rvegen::two_point_boundary::periodic,
                          rvegen::two_point_boundary::bounded}) {
      const auto s2 =
          rvegen::two_point_correlation_2d(shapes, box, 256, 256, 32, boundary);
      double worst = 0.0;
      for (auto const& s : s2) {
        // Bounded: a lag (lx, ly) has (1 - |lx|)(1 - |ly|) of the pairs a
        // periodic one has; averaged over the directions that is
        // 1 - 4r/pi + r^2/pi.
        const double pairs =
            boundary == rvegen::two_point_boundary::periodic
                ? 1.0
                : 1.0 - 4.0 * s.r / std::numbers::pi +
                      s.r * s.r / std::numbers::pi;
        worst = std::max(worst, std::abs(s.s2 - disc_overlap(R, s.r) / pairs));
      }
      std::cout << "  disc, worst deviation from the lens area: " << worst
                << '\n';
      REQUIRE(worst < 4e-3);   // phi = 0.126
    }
  }
  {   // sphere
    const rvegen::domain<double> box{{1.0, 1.0, 1.0}};
    const double R = 0.22;
    shape_vector shapes;
    shapes.push_back(make<rvegen::sphere<double>>(0.5, 0.48, 0.51, R));
    const auto all = rvegen::two_point_correlations(shapes, box, 64, 64, 64, 32);
    const auto s2 = all.radial();
    double worst = 0.0, worst_axis = 0.0;
    for (std::size_t b = 0; b < s2.size(); ++b) {
      const double exact = sphere_overlap(R, s2[b].r);
      worst = std::max(worst, std::abs(s2[b].s2 - exact));
      for (std::size_t ax = 0; ax < 3; ++ax)
        if (!std::isnan(all.inclusion.axis[ax][b]))
          worst_axis = std::max(worst_axis,
                                std::abs(all.inclusion.axis[ax][b] - exact));
    }
    std::cout << "  sphere, worst deviation: radial " << worst << ", axes "
              << worst_axis << '\n';
    REQUIRE(worst < 2e-3);        // phi = 0.0446
    REQUIRE(worst_axis < 4e-3);   // a sphere looks the same along every axis
    REQUIRE(std::abs(all.inclusion.zero -
                     4.0 / 3.0 * std::numbers::pi * R * R * R) < 5e-4);
  }
}

// Random spheres that do not overlap, periodic: placed with the pipeline.
rvegen::pipeline_result periodic_spheres() {
  const json config = json::parse(R"({
    "domain_box": [1.0, 1.0, 1.0], "seed": 11,
    "distributions": {
      "any": {"type": "uniform_real", "a": 0.0, "b": 1.0},
      "radius": {"type": "constant", "value": 0.07}},
    "shapes": [{"type": "sphere_input", "pos_x_dist": "any",
                "pos_y_dist": "any", "pos_z_dist": "any",
                "radius_dist": "radius"}],
    "generator": {"type": "periodic", "max_attempts": 200000},
    "termination": {"type": "volume_fraction", "target_fraction": 0.2}
  })");
  rvegen::pipeline_options opts;
  opts.run_postprocess = false;
  opts.info = [](std::string const&) {};
  opts.warn = [](std::string const&) {};
  return rvegen::run_pipeline(config, opts);
}

void test_isotropic_packing_and_aligned_fibres() {
  {
    const auto result = periodic_spheres();
    const auto s2 = rvegen::two_point_correlations(
        result.shapes, result.domain_box, 64, 64, 64, 16);
    const double phi = s2.inclusion.zero;
    REQUIRE(std::abs(phi - 0.2) < 0.01);
    // Beyond two diameters the structure is forgotten: S2 -> phi^2.
    double worst_tail = 0.0, worst_axis = 0.0;
    for (std::size_t b = 10; b < 16; ++b)
      worst_tail = std::max(worst_tail,
                            std::abs(s2.inclusion.radial[b] - phi * phi));
    // The first bin is next to phi, and the curve falls from there.
    REQUIRE(s2.inclusion.radial[0] > 0.7 * phi);
    REQUIRE(s2.inclusion.radial[0] < phi);
    // Isotropic: the profiles along the axes follow the radial average.
    for (std::size_t b = 0; b < 16; ++b)
      for (std::size_t ax = 0; ax < 3; ++ax)
        worst_axis = std::max(worst_axis, std::abs(s2.inclusion.axis[ax][b] -
                                                   s2.inclusion.radial[b]));
    std::cout << "  spheres: phi " << phi << ", tail off phi^2 by "
              << worst_tail << ", axes off the radial average by "
              << worst_axis << '\n';
    REQUIRE(worst_tail < 3e-3);
    REQUIRE(worst_axis < 0.02);
  }
  {
    // Fibres along z through the whole (periodic) box, on a jittered
    // lattice in the plane.
    const rvegen::domain<double> box{{1.0, 1.0, 1.0}};
    shape_vector shapes;
    std::mt19937 engine{5};
    std::uniform_real_distribution<double> u{-0.03, 0.03};
    for (int i = 0; i < 5; ++i)
      for (int j = 0; j < 5; ++j) {
        const double x = 0.1 + 0.2 * i + u(engine);
        const double y = 0.1 + 0.2 * j + u(engine);
        shapes.push_back(make<rvegen::polyline_tube<double>>(
            std::vector<std::array<double, 3>>{{x, y, -0.5}, {x, y, 1.5}},
            0.05));
      }
    const auto s2 = rvegen::two_point_correlations(shapes, box, 64, 64, 64, 16);
    const double phi = s2.inclusion.zero;
    REQUIRE(std::abs(phi - 25.0 * std::numbers::pi * 0.0025) < 0.01);
    for (std::size_t b = 0; b < 16; ++b) {
      // Along the fibres nothing changes: a point in a fibre stays in it.
      REQUIRE(s2.inclusion.axis[2][b] == phi);
      // Across, a diameter away (r = 0.1), a point of a fibre has left it
      // and has not reached the next one.
      if (std::abs(s2.r(b) - 0.1) < 0.02) {
        REQUIRE(s2.inclusion.axis[0][b] < 0.25 * phi);
        REQUIRE(s2.inclusion.axis[1][b] < 0.25 * phi);
      }
    }
  }
}

void test_phases() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");
  phases.add("filler");
  phases.set_matrix("matrix");
  shape_vector shapes;
  std::mt19937 engine{9};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  for (int s = 0; s < 24; ++s) {
    auto c = make<rvegen::circle<double>>(u(engine), u(engine),
                                          0.03 + 0.05 * u(engine));
    c->set_phase_name(s % 3 == 0 ? "filler" : "fibre");
    shapes.push_back(std::move(c));
  }
  const std::size_t n = 48, nbins = 12;
  const auto ids = rvegen::sample_voxel_grid(shapes, box, n, n, 1, phases);
  auto id_of = [&](std::string const& name) {
    return static_cast<phase_id>(phases.id_of(name));
  };
  for (auto boundary : {rvegen::two_point_boundary::periodic,
                        rvegen::two_point_boundary::bounded}) {
    const bool periodic = boundary == rvegen::two_point_boundary::periodic;
    const auto s2 = rvegen::two_point_correlations(shapes, box, n, n, 1, nbins,
                                                   boundary, &phases);
    // Three phases: three of their own and three pairs, by id.
    REQUIRE(s2.phases.size() == 6);
    const std::vector<std::pair<std::string, std::string>> order{
        {"matrix", "matrix"}, {"matrix", "fibre"}, {"matrix", "filler"},
        {"fibre", "fibre"},   {"fibre", "filler"}, {"filler", "filler"}};
    if (s2.phases.size() != 6) continue;
    for (std::size_t c = 0; c < 6; ++c) {
      REQUIRE(s2.phases[c].a == order[c].first);
      REQUIRE(s2.phases[c].b == order[c].second);
      REQUIRE(same(s2.phases[c].profile,
                   brute_force(ids, {n, n, 1}, {1.0, 1.0, 0.0}, periodic, 0.5,
                               nbins, id_of(order[c].first),
                               id_of(order[c].second))));
    }
    // The inclusion columns: every voxel a shape covers, whatever its phase.
    auto covered = ids;
    for (auto& v : covered) v = v == id_of("matrix") ? 0 : 1;
    REQUIRE(same(s2.inclusion, brute_force(covered, {n, n, 1}, {1.0, 1.0, 0.0},
                                           periodic, 0.5, nbins, 1, 1)));
    if (periodic) {
      // Wherever the second point lands it is in some phase: the fibre's
      // own S2 and its pairs with the others add up to its volume fraction.
      const double fibre = s2.phases[3].profile.zero;
      for (std::size_t b = 0; b < nbins; ++b) {
        const double sum = s2.phases[3].profile.radial[b] +
                           s2.phases[1].profile.radial[b] +
                           s2.phases[4].profile.radial[b];
        REQUIRE(std::abs(sum - fibre) < 1e-12);
      }
    }
  }
  // Without a matrix phase, what no shape covers belongs to no phase.
  rvegen::phase_collection<double> bare;
  bare.add("fibre");
  bare.add("filler");
  const auto s2 = rvegen::two_point_correlations(
      shapes, box, n, n, 1, nbins, rvegen::two_point_boundary::periodic, &bare);
  REQUIRE(s2.phases.size() == 6);
  if (s2.phases.size() == 6) {
    REQUIRE(s2.phases[0].a == "unassigned" && s2.phases[0].b == "unassigned");
    REQUIRE(s2.phases[5].a == "filler");
    double total = 0.0;
    for (std::size_t c : {std::size_t{0}, std::size_t{3}, std::size_t{5}})
      total += s2.phases[c].profile.zero;
    REQUIRE(std::abs(total - 1.0) < 1e-12);
  }
}

void test_bad_arguments() {
  auto rejects = [](auto&& f) {
    try { f(); } catch (std::invalid_argument const&) { return true; }
    return false;
  };
  const std::array<double, 3> unit{1.0, 1.0, 1.0};
  REQUIRE(rejects([&] {
    rvegen::voxel_correlator c{{8, 8, 1}, unit, {true, true, true}, 0.5, 0};
  }));
  REQUIRE(rejects([&] {
    rvegen::voxel_correlator c{{8, 8, 1}, unit, {true, true, true}, 0.0, 4};
  }));
  REQUIRE(rejects([&] {
    rvegen::voxel_correlator c{{0, 8, 1}, unit, {true, true, true}, 0.5, 4};
  }));
  // A bounded box is padded: 1500^3 voxels are allowed, padded they are
  // more than the limit.
  REQUIRE(rejects([&] {
    rvegen::voxel_correlator c{{1500, 1500, 1500}, unit, {false, false, false},
                               0.5, 4};
  }));
  const rvegen::voxel_correlator c{{8, 8, 1}, unit, {true, true, true}, 0.5, 4};
  REQUIRE((c.padded_shape() == std::array<std::size_t, 3>{8, 8, 1}));
  const rvegen::voxel_correlator padded{
      {8, 8, 1}, unit, {false, false, false}, 0.5, 4};
  // Lags up to 4 voxels: 12 samples per axis, and 12 = 2^2 * 3 is kept.
  REQUIRE((padded.padded_shape() == std::array<std::size_t, 3>{12, 12, 1}));
  REQUIRE(rejects([&] {
    (void)c.transform(std::vector<phase_id>(63, 0),
                      [](phase_id v) { return v != 0; });
  }));
  const auto spectrum = c.transform(std::vector<phase_id>(64, 1),
                                    [](phase_id v) { return v != 0; });
  REQUIRE(rejects([&] { (void)c.correlate(spectrum, {}); }));
  // Everything filled: S2 is 1 at every lag.
  const auto full = c.correlate(spectrum, spectrum);
  REQUIRE(full.zero == 1.0);
  for (double v : full.radial) REQUIRE(std::isnan(v) || v == 1.0);
  // A cancel request ends the work.
  bool cancelled = false;
  const rvegen::cancel_fn stop = [] { return true; };
  try {
    (void)c.transform(std::vector<phase_id>(64, 1),
                      [](phase_id v) { return v != 0; }, &stop);
  } catch (...) {
    cancelled = true;
  }
  REQUIRE(cancelled);
}

std::string slurp(std::string const& path) {
  std::ifstream in{path};
  std::stringstream s;
  s << in.rdbuf();
  return s.str();
}

std::vector<std::vector<std::string>> rows_of(std::string const& text) {
  std::vector<std::vector<std::string>> rows;
  std::istringstream in{text};
  for (std::string line; std::getline(in, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::vector<std::string> fields;
    std::size_t from = 0;
    for (;;) {
      const auto comma = line.find(',', from);
      fields.push_back(line.substr(from, comma - from));
      if (comma == std::string::npos) break;
      from = comma + 1;
    }
    rows.push_back(std::move(fields));
  }
  return rows;
}

void test_post_process() {
  const std::string path = "two_point_test.csv";
  json config = json::parse(R"({
    "domain_box": [1.0, 1.0, 0.0], "seed": 4,
    "distributions": {
      "any": {"type": "uniform_real", "a": 0.0, "b": 1.0},
      "radius": {"type": "constant", "value": 0.04}},
    "shapes": [{"type": "circle_input", "pos_x_dist": "any",
                "pos_y_dist": "any", "radius_dist": "radius"}],
    "generator": {"type": "periodic", "max_attempts": 200000},
    "termination": {"type": "volume_fraction", "target_fraction": 0.3},
    "postprocess": [{"type": "statistics", "output_path": "two_point_test.csv",
                     "nx": 96, "ny": 96, "nbins": 16}]
  })");
  rvegen::pipeline_options opts;
  opts.info = [](std::string const&) {};
  opts.warn = [](std::string const&) {};
  auto run = [&] {
    std::remove(path.c_str());
    auto result = rvegen::run_pipeline(config, opts);
    return std::pair{slurp(path), std::move(result)};
  };
  {
    const auto [text, result] = run();
    const auto s2 = rvegen::two_point_correlations(
        result.shapes, result.domain_box, 96, 96, 1, 16);
    // A periodic generator: lags wrap around.
    REQUIRE(text.find("# S2 = pair counts of the 96 x 96 voxel grid, exact "
                      "(FFT); periodic box, lags wrap around\n") !=
            std::string::npos);
    REQUIRE(text.find("# columns: r, S2(r), g(r), S2_x(r), S2_y(r)\n") !=
            std::string::npos);
    const auto rows = rows_of(text);
    REQUIRE(rows.size() == 17);   // r = 0 and 16 bins
    bool five = true;
    for (auto const& row : rows) five = five && row.size() == 5;
    REQUIRE(five);
    if (rows.size() == 17 && five) {
      // r = 0: the volume fraction of the grid in every S2 column, no g.
      REQUIRE(std::stod(rows[0][0]) == 0.0);
      REQUIRE(std::abs(std::stod(rows[0][1]) - s2.inclusion.zero) < 1e-11);
      REQUIRE(rows[0][2].empty());
      REQUIRE(rows[0][3] == rows[0][1] && rows[0][4] == rows[0][1]);
      REQUIRE(text.find("# voxel_volume_fraction = " + rows[0][1] + "\n") !=
              std::string::npos);
      for (std::size_t b = 0; b < 16; ++b) {
        REQUIRE(std::abs(std::stod(rows[b + 1][0]) - s2.r(b)) < 1e-11);
        REQUIRE(std::abs(std::stod(rows[b + 1][1]) - s2.inclusion.radial[b]) <
                1e-11);
        REQUIRE(!rows[b + 1][2].empty());
        REQUIRE(std::abs(std::stod(rows[b + 1][3]) - s2.inclusion.axis[0][b]) <
                1e-11);
        REQUIRE(std::abs(std::stod(rows[b + 1][4]) - s2.inclusion.axis[1][b]) <
                1e-11);
      }
    }
    // The same run again: the same file, byte for byte.
    REQUIRE(run().first == text);

    // Said otherwise in the block: pairs inside the box only.
    config["postprocess"][0]["periodic"] = false;
    const auto bounded_text = run().first;
    REQUIRE(bounded_text.find("bounded box, pairs inside the box only") !=
            std::string::npos);
    const auto bounded = rvegen::two_point_correlations(
        result.shapes, result.domain_box, 96, 96, 1, 16,
        rvegen::two_point_boundary::bounded);
    const auto bounded_rows = rows_of(bounded_text);
    REQUIRE(bounded_rows.size() == 17);
    if (bounded_rows.size() == 17) {
      REQUIRE(std::abs(std::stod(bounded_rows[9][1]) -
                       bounded.inclusion.radial[8]) < 1e-11);
    }
    REQUIRE(bounded.inclusion.radial[8] != s2.inclusion.radial[8]);
    config["postprocess"][0].erase("periodic");
  }
  {
    // A generator that is not periodic: bounded without being told, and
    // periodic when the block says so.
    config["generator"]["type"] = "only_inside";
    REQUIRE(run().first.find("bounded box") != std::string::npos);
    config["postprocess"][0]["periodic"] = true;
    REQUIRE(run().first.find("periodic box") != std::string::npos);
    config["postprocess"][0].erase("periodic");
    config["generator"]["type"] = "periodic";
  }
  {
    // More bins than lags: fields without a lag are empty, along the axes
    // first.
    config["postprocess"][0]["nx"] = 8;
    config["postprocess"][0]["ny"] = 8;
    const auto rows = rows_of(run().first);
    REQUIRE(rows.size() == 17);
    std::size_t empty_axis = 0, empty_radial = 0;
    for (std::size_t b = 1; b < rows.size(); ++b) {
      empty_radial += rows[b][1].empty();
      empty_axis += rows[b][3].empty();
    }
    REQUIRE(empty_axis >= 12);   // 3 lags along x within r < 0.5
    REQUIRE(empty_radial < empty_axis);
    config["postprocess"][0]["nx"] = 96;
    config["postprocess"][0]["ny"] = 96;
  }
  {
    // Phases: four columns per phase and pair of phases (three in 2D).
    config["phases"] =
        json::parse(R"([{"name": "matrix"}, {"name": "fibre"}])");
    config["phase_assignment"] = json::parse(
        R"({"matrix": "matrix", "by_shape_type": {"circle": "fibre"}})");
    const auto text = run().first;
    REQUIRE(text.find("# columns: r, S2(r), g(r), S2_x(r), S2_y(r), "
                      "S2[matrix](r), S2_x[matrix](r), S2_y[matrix](r), "
                      "S2[matrix|fibre](r), S2_x[matrix|fibre](r), "
                      "S2_y[matrix|fibre](r), S2[fibre](r), S2_x[fibre](r), "
                      "S2_y[fibre](r)\n") != std::string::npos);
    const auto rows = rows_of(text);
    bool wide = !rows.empty();
    for (auto const& row : rows) wide = wide && row.size() == 14;
    REQUIRE(wide);
    if (wide) {
      // Every shape is fibre: the fibre columns are the inclusion columns.
      for (auto const& row : rows) {
        REQUIRE(row[11] == row[1]);
        REQUIRE(row[12] == row[3] && row[13] == row[4]);
      }
      // matrix + fibre fill the box.
      REQUIRE(std::abs(std::stod(rows[0][5]) + std::stod(rows[0][11]) - 1.0) <
              1e-11);
      REQUIRE(std::stod(rows[0][8]) == 0.0);
    }
    // Switched off: the five columns alone.
    config["postprocess"][0]["phase_correlations"] = false;
    for (auto const& row : rows_of(run().first)) REQUIRE(row.size() == 5);
  }
  std::remove(path.c_str());
}

void test_post_process_3d() {
  const std::string path = "two_point_test_3d.csv";
  rvegen::statistics_post_process<double> stats;
  stats.set_output_path(path);
  stats.set_grid(24, 24, 12);
  stats.set_nbins(6);
  const rvegen::domain<double> box{{1.0, 1.0, 0.5}};
  shape_vector shapes;
  shapes.push_back(make<rvegen::sphere<double>>(0.3, 0.3, 0.25, 0.15));
  shapes.push_back(make<rvegen::sphere<double>>(0.7, 0.6, 0.25, 0.12));
  stats.run(shapes, box);
  const auto text = slurp(path);
  // Outside the pipeline nothing says the box is periodic.
  REQUIRE(text.find("24 x 24 x 12 voxel grid, exact (FFT); bounded box") !=
          std::string::npos);
  REQUIRE(text.find("# columns: r, S2(r), g(r), S2_x(r), S2_y(r), S2_z(r)\n") !=
          std::string::npos);
  const auto rows = rows_of(text);
  REQUIRE(rows.size() == 7);
  for (auto const& row : rows) REQUIRE(row.size() == 6);
  // Bins up to half the smallest edge, 0.25.
  if (rows.size() == 7)
    REQUIRE(std::abs(std::stod(rows[6][0]) - 0.25 * 5.5 / 6.0) < 1e-11);
  stats.set_periodic(true);
  stats.run(shapes, box);
  REQUIRE(slurp(path).find("periodic box") != std::string::npos);
  // A grid that cannot be used is refused before the file is touched.
  std::remove(path.c_str());
  stats.set_grid(std::size_t{1} << 20, std::size_t{1} << 20, 1 << 10);
  bool threw = false;
  try { stats.run(shapes, box); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
  REQUIRE(!std::ifstream{path}.good());
}

} // namespace

int main() {
  test_against_every_pair();
  test_zero_lag_is_the_voxel_fraction();
  test_single_disc_and_sphere();
  test_isotropic_packing_and_aligned_fibres();
  test_phases();
  test_bad_arguments();
  test_post_process();
  test_post_process_3d();
  // A program's own pocketfft and rvegen's, side by side: the first bin of
  // the transform of 1, 2, 3, 4 is 10, from each of the two.
  REQUIRE(fft_isolation_first() == 20.0);
  REQUIRE(fft_isolation_second() == 20.0);
  std::remove("pipeline_stdout.txt");
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "two_point_fft_test: all passed\n";
  return 0;
}
