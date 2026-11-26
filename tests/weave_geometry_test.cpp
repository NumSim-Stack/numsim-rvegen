// Weave geometry regression test (#78).
//
// 1. Plain weave: at every (warp i, weft j) crossing the two
//    centrelines are ≥ 2·yarn_radius apart in z, with the over/under
//    sign (−1)^(i+j) alternating in both i and j; the full warp–weft
//    centreline clearance (GTE segment–segment distance, including
//    periodic images) is ≥ 2·yarn_radius; the yarns are periodic.
// 2. Validation: odd counts, amplitude < yarn_radius, yarns leaving
//    [0, Lz], too-steep crimp all throw.
// 3. Tow: fibre centrelines in a bundle are ≥ 2·fibre_radius apart,
//    stay inside the yarn envelope, a partial outer ring is spread
//    evenly, and an infeasible fibre_radius_factor throws.
// 4. generate_periodic_wraps: a box sticking out of both x faces gets
//    wraps on both sides; single-side corner cases keep their counts.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <vector>

#include <Mathematics/DistPointSegment.h>
#include <Mathematics/DistSegmentSegment.h>
#include <Mathematics/Segment.h>

#include "rvegen/rvegen.h"
#include "rvegen/shapes/periodic_wraps.h"

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

using tube_t = rvegen::polyline_tube<double>;
using vec3 = gte::Vector3<double>;

tube_t const& as_tube(std::unique_ptr<rvegen::shape_base<double>> const& s) {
  auto const* t = dynamic_cast<tube_t const*>(s.get());
  if (t == nullptr) throw std::logic_error{"expected a polyline_tube"};
  return *t;
}

// z of a polyline at coordinate `t` along axis `axis` (linear interp).
double polyline_z_at(std::vector<vec3> const& cl, int axis, double t) {
  for (std::size_t k = 1; k < cl.size(); ++k) {
    const double a = cl[k - 1][axis];
    const double b = cl[k][axis];
    if (t >= a && t <= b) {
      const double u = (b > a) ? (t - a) / (b - a) : 0.0;
      return cl[k - 1][2] + u * (cl[k][2] - cl[k - 1][2]);
    }
  }
  throw std::logic_error{"coordinate outside the polyline"};
}

// Minimum GTE segment–segment distance between two polylines, the
// second translated by `shift`.
double polyline_distance(std::vector<vec3> const& a, std::vector<vec3> const& b,
                         vec3 const& shift = vec3{0.0, 0.0, 0.0}) {
  gte::DCPQuery<double, gte::Segment3<double>, gte::Segment3<double>> query;
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < a.size(); ++i) {
    const gte::Segment3<double> sa{a[i - 1], a[i]};
    for (std::size_t j = 1; j < b.size(); ++j) {
      const gte::Segment3<double> sb{b[j - 1] + shift, b[j] + shift};
      best = std::min(best, query(sa, sb).distance);
    }
  }
  return best;
}

// Max over a's control points of the GTE distance to polyline b.
double max_node_distance_to(std::vector<vec3> const& a,
                            std::vector<vec3> const& b) {
  gte::DCPQuery<double, vec3, gte::Segment3<double>> query;
  double worst = 0.0;
  for (auto const& p : a) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t j = 1; j < b.size(); ++j) {
      best = std::min(best,
                      query(p, gte::Segment3<double>{b[j - 1], b[j]}).distance);
    }
    worst = std::max(worst, best);
  }
  return worst;
}

template <typename F>
bool throws(F&& f) {
  try { f(); } catch (std::runtime_error const&) { return true; }
  return false;
}

struct weave_cfg {
  std::array<double, 3> box;
  std::size_t n_warp, n_weft;
  double r, A;
  std::size_t n_seg;
};

// ----------------------------------------------------------------------------
// 1. Plain weave geometry.
// ----------------------------------------------------------------------------
void check_weave(weave_cfg const& c) {
  rvegen::weave_generator<double> gen{c.box, c.n_warp, c.n_weft, c.r, c.A,
                                      c.n_seg};
  auto shapes = gen.build();
  REQUIRE(shapes.size() == c.n_warp + c.n_weft);
  const double Lx = c.box[0], Ly = c.box[1];
  const double sx = Lx / static_cast<double>(c.n_weft);
  const double sy = Ly / static_cast<double>(c.n_warp);
  const double tol = 1e-12;

  // Crossing clearance + over/under sign.
  std::vector<std::vector<int>> sign(c.n_warp, std::vector<int>(c.n_weft, 0));
  for (std::size_t i = 0; i < c.n_warp; ++i) {
    auto const& warp = as_tube(shapes[i]).centerline();
    const double y_i = (static_cast<double>(i) + 0.5) * sy;
    for (std::size_t j = 0; j < c.n_weft; ++j) {
      auto const& weft = as_tube(shapes[c.n_warp + j]).centerline();
      const double x_j = (static_cast<double>(j) + 0.5) * sx;
      const double dz = polyline_z_at(warp, 0, x_j) - polyline_z_at(weft, 1, y_i);
      REQUIRE(std::abs(dz) >= 2.0 * c.r - tol);
      // The crossing is a control point: separation is exactly 2A.
      REQUIRE(std::abs(std::abs(dz) - 2.0 * c.A) < tol);
      sign[i][j] = dz > 0.0 ? +1 : -1;
      REQUIRE(sign[i][j] == (((i + j) % 2 == 0) ? +1 : -1));
    }
  }
  // Alternation in i and j, including across the periodic seam.
  for (std::size_t i = 0; i < c.n_warp; ++i) {
    for (std::size_t j = 0; j < c.n_weft; ++j) {
      REQUIRE(sign[i][j] == -sign[(i + 1) % c.n_warp][j]);
      REQUIRE(sign[i][j] == -sign[i][(j + 1) % c.n_weft]);
    }
  }

  // Full centreline clearance between every warp and weft, including
  // the weft's ±Lx images and the warp's ±Ly images. The analytic
  // centrelines are ≥ 2r apart; each polyline deviates from its curve
  // by at most the chord sagitta h²·max|z''|/8 = A·π²·h²/(8·s²), so
  // the polylines may come that much closer per yarn.
  const auto sagitta = [&](std::size_t n_nodes, double L, double s) {
    const double h = L / static_cast<double>(n_nodes - 1);
    const double pi = std::numbers::pi;
    return c.A * pi * pi * h * h / (8.0 * s * s);
  };
  const double chord_slack =
      sagitta(as_tube(shapes[0]).centerline().size(), Lx, sx)
      + sagitta(as_tube(shapes[c.n_warp]).centerline().size(), Ly, sy);
  double min_dist = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < c.n_warp; ++i) {
    auto const& warp = as_tube(shapes[i]).centerline();
    for (std::size_t j = 0; j < c.n_weft; ++j) {
      auto const& weft = as_tube(shapes[c.n_warp + j]).centerline();
      for (double dx : {-Lx, 0.0, Lx}) {
        for (double dy : {-Ly, 0.0, Ly}) {
          min_dist = std::min(min_dist,
                              polyline_distance(warp, weft, vec3{dx, dy, 0.0}));
        }
      }
    }
  }
  REQUIRE(min_dist >= 2.0 * c.r - chord_slack - 1e-12);

  // Periodicity: every yarn starts and ends at the same z with the
  // same end-segment slope.
  for (auto const& s : shapes) {
    auto const& cl = as_tube(s).centerline();
    const std::size_t n = cl.size();
    REQUIRE(std::abs(cl.front()[2] - cl.back()[2]) < tol);
    REQUIRE(std::abs((cl[1][2] - cl[0][2]) - (cl[n - 1][2] - cl[n - 2][2]))
            < tol);
  }

  // Everything stays inside [0, Lz].
  for (auto const& s : shapes) {
    for (auto const& p : as_tube(s).centerline()) {
      REQUIRE(p[2] - c.r >= -tol);
      REQUIRE(p[2] + c.r <= c.box[2] + tol);
    }
  }
}

void test_weave_crossings_clear_and_alternate() {
  check_weave({{1.0, 1.0, 0.3}, 4, 4, 0.04, 0.05, 32});
  // Non-square, unequal counts, segment count not a multiple of the
  // crossing count (rounded up internally so crossings are nodes).
  check_weave({{2.0, 1.0, 0.4}, 2, 6, 0.05, 0.08, 20});
  // Exactly on the A == r and A + r == Lz/2 bounds.
  check_weave({{1.0, 1.0, 0.2}, 2, 2, 0.05, 0.05, 16});
}

// ----------------------------------------------------------------------------
// 2. Validation.
// ----------------------------------------------------------------------------
void test_weave_validation() {
  using G = rvegen::weave_generator<double>;
  // Odd counts cannot tile a periodic plain weave.
  REQUIRE(throws([] { G{{1, 1, 0.3}, 3, 2, 0.04, 0.05}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 1, 0.04, 0.05}; }));
  // Amplitude below yarn radius → yarns interpenetrate at crossings.
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 2, 0.05, 0.03}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 2, 0.05, 0.0}; }));
  // z_centre ± (A + r) leaves [0, Lz].
  REQUIRE(throws([] { G{{1, 1, 0.2}, 2, 2, 0.05, 0.06}; }));
  // Yarn too thick for the spacing: r > s/(π√2) (s = 0.25 → 0.0563).
  REQUIRE(throws([] { G{{1, 1, 1.0}, 4, 4, 0.06, 0.06}; }));
  // Crimp radius of curvature s²/(π²A) below r (s = 0.25, A = 0.3).
  REQUIRE(throws([] { G{{1, 1, 1.0}, 4, 4, 0.05, 0.3}; }));
  // Valid configs still construct; flat 2D stays allowed.
  REQUIRE(!throws([] { G{{1, 1, 0.3}, 4, 4, 0.04, 0.05}; }));
  REQUIRE(!throws([] { G{{1, 1, 0.0}, 2, 2, 0.05, 0.0}; }));
}

// ----------------------------------------------------------------------------
// 3. Tow fibre layout.
// ----------------------------------------------------------------------------
void check_tow(std::size_t n_fibres, double factor) {
  const double r_yarn = 0.05;
  rvegen::weave_generator<double> base{{1.0, 1.0, 0.3}, 2, 2, r_yarn, 0.06, 16};
  rvegen::tow_weave_generator<double> gen{base, n_fibres, factor};
  auto yarns = base.build();
  auto fibres = gen.build();
  REQUIRE(fibres.size() == yarns.size() * n_fibres);
  const double r_f = r_yarn * factor;

  for (std::size_t y = 0; y < yarns.size(); ++y) {
    auto const& yarn_cl = as_tube(yarns[y]).centerline();
    for (std::size_t a = 0; a < n_fibres; ++a) {
      auto const& fa = as_tube(fibres[y * n_fibres + a]);
      REQUIRE(std::abs(fa.radius() - r_f) < 1e-15);
      // Fibre tube stays inside the yarn envelope.
      REQUIRE(max_node_distance_to(fa.centerline(), yarn_cl)
              <= r_yarn - r_f + 1e-12);
      for (std::size_t b = a + 1; b < n_fibres; ++b) {
        auto const& fb = as_tube(fibres[y * n_fibres + b]);
        REQUIRE(polyline_distance(fa.centerline(), fb.centerline())
                >= 2.0 * r_f * (1.0 - 1e-9));
      }
    }
    // Ring pattern (incl. a partial outer ring) is balanced around the
    // yarn axis: the fibre centroid at each node is the yarn node.
    for (std::size_t k = 0; k < yarn_cl.size(); ++k) {
      vec3 sum{0.0, 0.0, 0.0};
      for (std::size_t a = 0; a < n_fibres; ++a) {
        sum += as_tube(fibres[y * n_fibres + a]).centerline()[k];
      }
      sum /= static_cast<double>(n_fibres);
      REQUIRE(gte::Length(sum - yarn_cl[k]) < 1e-12);
    }
  }
}

void test_tow_fibres_do_not_overlap() {
  check_tow(7, 0.2);      // default factor, 1 ring
  check_tow(3, 0.2);      // partial ring 1 (2 of 6 sites)
  check_tow(25, 0.125);   // 3 rings, partial ring 3 (6 of 18 sites)
}

void test_tow_infeasible_factor_throws() {
  rvegen::weave_generator<double> base{{1.0, 1.0, 0.3}, 2, 2, 0.05, 0.06};
  using G = rvegen::tow_weave_generator<double>;
  // Old documented default: 25 fibres (3 rings) at 0.2 > 1/7.
  REQUIRE(throws([&] { G{base, 25, 0.2}; }));
  REQUIRE(throws([&] { G{base, 25}; }));
  // Ring bounds: 7 fibres ≤ 1/3, 19 ≤ 1/5, 37 ≤ 1/7, 38 → 1/9.
  REQUIRE(!throws([&] { G{base, 7, 1.0 / 3.0}; }));
  REQUIRE(throws([&] { G{base, 8, 1.0 / 3.0}; }));
  REQUIRE(!throws([&] { G{base, 19, 0.2}; }));
  REQUIRE(!throws([&] { G{base, 37, 1.0 / 7.0}; }));
  REQUIRE(throws([&] { G{base, 38, 1.0 / 7.0}; }));
  // A single fibre has no neighbours: any factor in (0, 1) is fine.
  REQUIRE(!throws([&] { G{base, 1, 0.9}; }));
}

// ----------------------------------------------------------------------------
// 4. Periodic wraps.
// ----------------------------------------------------------------------------
std::size_t wrap_count(std::vector<std::array<double, 3>> const& cl, double r,
                       std::array<double, 3> const& box) {
  tube_t t{cl, r};
  t.make_bounding_box();
  return rvegen::generate_periodic_wraps<double>(t, box).size();
}

void test_periodic_wraps_both_faces() {
  const std::array<double, 3> box{1.0, 1.0, 1.0};
  // Tube along x spanning [−r, 1 + r] — sticks out of both x faces.
  tube_t yarn{std::vector<std::array<double, 3>>{{0.0, 0.5, 0.5},
                                                 {1.0, 0.5, 0.5}},
              0.05};
  yarn.make_bounding_box();
  auto wraps = rvegen::generate_periodic_wraps<double>(yarn, box);
  REQUIRE(wraps.size() == 2);
  std::vector<double> dx;
  for (auto const& w : wraps) {
    dx.push_back(w->get_middle_point()[0] - yarn.get_middle_point()[0]);
    REQUIRE(std::abs(w->get_middle_point()[1] - 0.5) < 1e-12);
    REQUIRE(std::abs(w->get_middle_point()[2] - 0.5) < 1e-12);
  }
  std::sort(dx.begin(), dx.end());
  REQUIRE(std::abs(dx[0] + 1.0) < 1e-12);
  REQUIRE(std::abs(dx[1] - 1.0) < 1e-12);

  // Both x faces + low y face: 3·2 − 1 = 5 wraps.
  REQUIRE(wrap_count({{0.0, 0.02, 0.5}, {1.0, 0.02, 0.5}}, 0.05, box) == 5);
  // Both x faces in 2D (z ignored).
  REQUIRE(wrap_count({{0.0, 0.5, 0.0}, {1.0, 0.5, 0.0}}, 0.05,
                     {1.0, 1.0, 0.0}) == 2);
}

void test_periodic_wraps_single_side_counts_unchanged() {
  const std::array<double, 3> box{1.0, 1.0, 1.0};
  const auto pt = [](double x, double y, double z) {
    return std::vector<std::array<double, 3>>{{x, y, z}, {x, y, z + 0.01}};
  };
  REQUIRE(wrap_count(pt(0.5, 0.5, 0.5), 0.05, box) == 0);    // inside
  REQUIRE(wrap_count(pt(0.98, 0.5, 0.5), 0.05, box) == 1);   // +x
  REQUIRE(wrap_count(pt(0.02, 0.5, 0.5), 0.05, box) == 1);   // −x
  REQUIRE(wrap_count(pt(0.98, 0.98, 0.5), 0.05, box) == 3);  // +x +y
  REQUIRE(wrap_count(pt(0.02, 0.98, 0.5), 0.05, box) == 3);  // −x +y
  REQUIRE(wrap_count(pt(0.98, 0.98, 0.97), 0.05, box) == 7); // +x +y +z
  // 2D domain ignores z.
  REQUIRE(wrap_count(pt(0.98, 0.98, 0.0), 0.05, {1.0, 1.0, 0.0}) == 3);

  // Shift directions match the old single-side behaviour: +x poke
  // wraps to −L, −x poke wraps to +L.
  tube_t t{pt(0.98, 0.5, 0.5), 0.05};
  t.make_bounding_box();
  auto w = rvegen::generate_periodic_wraps<double>(t, box);
  REQUIRE(w.size() == 1);
  REQUIRE(std::abs(w[0]->get_middle_point()[0] - (0.98 - 1.0)) < 1e-12);
}

} // namespace

int main() {
  test_weave_crossings_clear_and_alternate();
  test_weave_validation();
  test_tow_fibres_do_not_overlap();
  test_tow_infeasible_factor_throws();
  test_periodic_wraps_both_faces();
  test_periodic_wraps_single_side_counts_unchanged();

  if (failures == 0) {
    std::cout << "weave_geometry_test: all tests passed\n";
    return EXIT_SUCCESS;
  }
  std::cerr << "weave_geometry_test: " << failures << " failure(s)\n";
  return EXIT_FAILURE;
}
