// Weave geometry regression test (#78, #3).
//
// 1. Plain / twill 2/2 / satin 5: at every (warp i, weft j) crossing
//    the two centrelines are 2·amplitude ≥ 2·yarn_radius apart in z,
//    with the over/under sign matching the pattern's interlacing
//    matrix (hard-coded here, independent of the generator); the full
//    warp–weft clearance of the emitted polylines (GTE segment–segment
//    distance, including periodic images) is ≥ 2·yarn_radius; the
//    yarns are periodic.
// 2. Validation: counts not a multiple of the repeat, amplitude ≤
//    yarn_radius, yarns leaving [0, Lz], too-steep crimp all throw.
// 5. Acceptance (#3): a 4×4 plain weave at fibre volume fraction 0.30
//    voxelizes to the analytic yarn volume within the voxel-size
//    error bound.
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
#include <utility>
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

// Axis-aligned box of a point set, and the distance between two such boxes
// (0 when they overlap): a lower bound of the distance between anything
// inside them.
struct aabb {
  vec3 lo, hi;
  void add(vec3 const& p) {
    for (int k = 0; k < 3; ++k) {
      lo[k] = std::min(lo[k], p[k]);
      hi[k] = std::max(hi[k], p[k]);
    }
  }
};
aabb box_of(vec3 const& p, vec3 const& q) {
  aabb r{p, p};
  r.add(q);
  return r;
}
double box_gap(aabb const& a, aabb const& b) {
  double sq = 0.0;
  for (int k = 0; k < 3; ++k) {
    const double g = std::max({0.0, a.lo[k] - b.hi[k], b.lo[k] - a.hi[k]});
    sq += g * g;
  }
  return std::sqrt(sq);
}

// min(distance between polyline a and polyline b + shift, cap), by GTE
// segment–segment distances. A segment (or segment pair) whose box is
// farther from the other side's box than the running minimum cannot lower
// it and is skipped, so the result is the exact minimum; `cap` lets a caller
// that minimises over many polyline pairs pass its running minimum in.
double polyline_distance(std::vector<vec3> const& a, std::vector<vec3> const& b,
                         vec3 const& shift = vec3{0.0, 0.0, 0.0},
                         double cap = std::numeric_limits<double>::infinity()) {
  if (a.size() < 2 || b.size() < 2) return cap;
  std::vector<vec3> bs(b.size());
  aabb b_all{b[0] + shift, b[0] + shift};
  for (std::size_t j = 0; j < b.size(); ++j) {
    bs[j] = b[j] + shift;
    b_all.add(bs[j]);
  }
  gte::DCPQuery<double, gte::Segment3<double>, gte::Segment3<double>> query;
  double best = cap;
  for (std::size_t i = 1; i < a.size(); ++i) {
    const aabb box_a = box_of(a[i - 1], a[i]);
    if (box_gap(box_a, b_all) > best) continue;
    const gte::Segment3<double> sa{a[i - 1], a[i]};
    for (std::size_t j = 1; j < bs.size(); ++j) {
      if (box_gap(box_a, box_of(bs[j - 1], bs[j])) > best) continue;
      best = std::min(best,
                      query(sa, gte::Segment3<double>{bs[j - 1], bs[j]}).distance);
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

// Largest 1/cos(theta/2) over the interior nodes of `cl`, theta being the
// angle between the in-plane normals of the two adjacent segments.
double max_mitre(std::vector<vec3> const& cl) {
  double worst = 1.0;
  for (std::size_t k = 1; k + 1 < cl.size(); ++k) {
    vec3 t0 = cl[k] - cl[k - 1], t1 = cl[k + 1] - cl[k];
    gte::Normalize(t0);
    gte::Normalize(t1);
    const double c = gte::Dot(t0, t1);                 // cos(theta)
    worst = std::max(worst, 1.0 / std::sqrt(0.5 * (1.0 + c)));
  }
  return worst;
}

template <typename F>
bool throws(F&& f) {
  // runtime_error from the generator's own checks, invalid_argument from
  // the domain type (bad box extents).
  try { f(); } catch (std::runtime_error const&) { return true; }
  catch (std::invalid_argument const&) { return true; }
  return false;
}

struct weave_cfg {
  std::array<double, 3> box;
  std::size_t n_warp, n_weft;
  double r, A;
  std::size_t n_seg;
  rvegen::weave_pattern pattern = rvegen::weave_pattern::plain;
};

// Reference interlacing matrices, one repeat each ('+' = warp over).
// Row = warp i, column = weft j.
int expected_interlacing(rvegen::weave_pattern p, std::size_t i,
                         std::size_t j) {
  static const char* const twill[4] = {"++--", "-++-", "--++", "+--+"};
  static const char* const satin[5] = {"-++++", "++-++", "++++-", "+-+++",
                                       "+++-+"};
  switch (p) {
    case rvegen::weave_pattern::twill_2_2:
      return twill[i % 4][j % 4] == '+' ? +1 : -1;
    case rvegen::weave_pattern::satin_5:
      return satin[i % 5][j % 5] == '+' ? +1 : -1;
    case rvegen::weave_pattern::plain: break;
  }
  return ((i + j) % 2 == 0) ? +1 : -1;
}

// ----------------------------------------------------------------------------
// 1. Plain weave geometry.
// ----------------------------------------------------------------------------
// Returns the measured minimum warp–weft polyline distance.
double check_weave(weave_cfg const& c) {
  rvegen::weave_generator<double> gen{c.box, c.n_warp, c.n_weft, c.r, c.A,
                                      c.n_seg, c.pattern};
  REQUIRE(gen.pattern() == c.pattern);
  auto shapes = gen.build();
  REQUIRE(shapes.size() == c.n_warp + c.n_weft);
  // Emitted segment counts: reported correctly, ≥ the request, and a
  // multiple of 2·(crossings) so crossings are nodes.
  REQUIRE(as_tube(shapes[0]).centerline().size() == gen.n_segments_warp() + 1);
  REQUIRE(as_tube(shapes[c.n_warp]).centerline().size()
          == gen.n_segments_weft() + 1);
  REQUIRE(gen.n_segments_warp() >= c.n_seg);
  REQUIRE(gen.n_segments_weft() >= c.n_seg);
  REQUIRE(gen.n_segments_warp() % (2 * c.n_weft) == 0);
  REQUIRE(gen.n_segments_weft() % (2 * c.n_warp) == 0);
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
      REQUIRE(sign[i][j] == expected_interlacing(c.pattern, i, j));
      REQUIRE(gen.interlacing(i, j) == sign[i][j]);
    }
  }
  // Plain weave: alternation in i and j, including across the seam.
  if (c.pattern == rvegen::weave_pattern::plain) {
    for (std::size_t i = 0; i < c.n_warp; ++i) {
      for (std::size_t j = 0; j < c.n_weft; ++j) {
        REQUIRE(sign[i][j] == -sign[(i + 1) % c.n_warp][j]);
        REQUIRE(sign[i][j] == -sign[i][(j + 1) % c.n_weft]);
      }
    }
  }

  // Clearance of the EMITTED polylines between every warp and weft,
  // including the weft's ±Lx and ±Ly images: ≥ 2r up to rounding.
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
  REQUIRE(min_dist >= 2.0 * c.r - 1e-12);

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
  return min_dist;
}

void test_weave_crossings_clear_and_alternate() {
  check_weave({{1.0, 1.0, 0.3}, 4, 4, 0.04, 0.05, 32});
  // Non-square, unequal counts, segment count not a multiple of the
  // crossing count; A beyond s/(π√2), so the flanks set the clearance.
  check_weave({{2.0, 1.0, 0.4}, 2, 6, 0.05, 0.08, 20});
  // Generous: A = 1.3r, dense 8×8, coarsest request.
  check_weave({{1.0, 1.0, 0.3}, 8, 8, 0.02, 0.026, 2});
  // On the A + r == Lz/2 bound.
  check_weave({{1.0, 1.0, 0.22}, 2, 2, 0.05, 0.06, 16});
  // Near the review's failing configs (A = r there; see validation):
  // a sliver of margin forces automatic refinement from n_seg = 2.
  for (std::size_t n_seg : {2u, 8u, 32u}) {
    check_weave({{1.0, 1.0, 1.0}, 4, 4, 0.056, 0.0562, n_seg});
  }
  for (std::size_t n_seg : {2u, 16u, 32u}) {
    check_weave({{1.0, 1.0, 1.0}, 2, 2, 0.11, 0.112, n_seg});
  }
}

void test_weave_patterns_clear_and_match_matrix() {
  using P = rvegen::weave_pattern;
  // Twill 2/2: square, non-square with unequal spacings, coarse
  // request with a thin margin (forces refinement).
  check_weave({{1.0, 1.0, 0.3}, 4, 4, 0.04, 0.05, 32, P::twill_2_2});
  check_weave({{2.0, 1.0, 0.3}, 8, 4, 0.02, 0.025, 8, P::twill_2_2});
  check_weave({{1.0, 1.0, 1.0}, 4, 4, 0.056, 0.0562, 2, P::twill_2_2});
  // Satin 5: odd repeat, long floats.
  check_weave({{1.0, 1.0, 0.3}, 5, 5, 0.035, 0.04, 32, P::satin_5});
  check_weave({{2.0, 1.0, 0.3}, 10, 5, 0.02, 0.026, 2, P::satin_5});
  check_weave({{1.0, 1.0, 1.0}, 5, 5, 0.045, 0.0451, 2, P::satin_5});
  // Plain via the explicit parameter matches the default.
  rvegen::weave_generator<double> a{{1, 1, 0.3}, 4, 4, 0.04, 0.05, 16};
  rvegen::weave_generator<double> b{{1, 1, 0.3}, 4, 4, 0.04, 0.05, 16,
                                    P::plain};
  auto sa = a.build();
  auto sb = b.build();
  for (std::size_t y = 0; y < sa.size(); ++y) {
    auto const& ca = as_tube(sa[y]).centerline();
    auto const& cb = as_tube(sb[y]).centerline();
    REQUIRE(ca.size() == cb.size());
    for (std::size_t k = 0; k < ca.size(); ++k) {
      REQUIRE(gte::Length(ca[k] - cb[k]) == 0.0);
    }
  }
}

// ----------------------------------------------------------------------------
// 2. Validation.
// ----------------------------------------------------------------------------
void test_weave_validation() {
  using G = rvegen::weave_generator<double>;
  // A pattern value outside the enum is rejected, not treated as plain.
  const auto bad = static_cast<rvegen::weave_pattern>(7);
  const auto throws_invalid = [](auto&& f) {
    try { f(); } catch (std::invalid_argument const&) { return true; }
    catch (...) {}
    return false;
  };
  REQUIRE(throws_invalid([&] { (void)rvegen::weave_pattern_repeat(bad); }));
  REQUIRE(throws_invalid([&] { G{{1, 1, 0.3}, 2, 2, 0.04, 0.05, 32, bad}; }));
  // Odd counts cannot tile a periodic plain weave.
  REQUIRE(throws([] { G{{1, 1, 0.3}, 3, 2, 0.04, 0.05}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 1, 0.04, 0.05}; }));
  // Amplitude below yarn radius → yarns interpenetrate at crossings.
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 2, 0.05, 0.03}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 2, 0.05, 0.0}; }));
  // A == r: an inscribed polyline always dips below the crimp peak next
  // to the crossing node, so no segment count reaches 2r. These are
  // the review's configs (15% overlap at the coarsest sampling before).
  for (std::size_t n_seg : {2u, 8u, 32u}) {
    REQUIRE(throws([&] { G{{1, 1, 1}, 4, 4, 0.056, 0.056, n_seg}; }));
  }
  for (std::size_t n_seg : {2u, 16u, 32u}) {
    REQUIRE(throws([&] { G{{1, 1, 1}, 2, 2, 0.11, 0.11, n_seg}; }));
  }
  // Margin so thin the polyline would need > max_segments_per_yarn.
  REQUIRE(throws([] { G{{1, 1, 1}, 2, 2, 0.1, 0.1 * (1 + 1e-9)}; }));
  // Segment request above the cap (also guards segments_for overflow).
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 2, 0.04, 0.05, G::max_segments_per_yarn + 1}; }));
  REQUIRE(throws([] {
    G{{1, 1, 0.3}, 2, 2, 0.04, 0.05, std::numeric_limits<std::size_t>::max()};
  }));
  REQUIRE(!throws([] { G{{1, 1, 0.3}, 2, 2, 0.04, 0.05, G::max_segments_per_yarn}; }));
  // Pattern repeats: twill 2/2 needs multiples of 4, satin 5 of 5.
  using P = rvegen::weave_pattern;
  REQUIRE(throws([] { G{{1, 1, 0.3}, 2, 4, 0.02, 0.03, 32, P::twill_2_2}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 4, 6, 0.02, 0.03, 32, P::twill_2_2}; }));
  REQUIRE(!throws([] { G{{1, 1, 0.3}, 8, 4, 0.02, 0.03, 32, P::twill_2_2}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 4, 5, 0.02, 0.03, 32, P::satin_5}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 5, 6, 0.02, 0.03, 32, P::satin_5}; }));
  REQUIRE(!throws([] { G{{1, 1, 0.3}, 5, 10, 0.02, 0.03, 32, P::satin_5}; }));
  // The clearance checks apply to every pattern.
  REQUIRE(throws([] { G{{1, 1, 0.3}, 5, 5, 0.035, 0.035, 32, P::satin_5}; }));
  REQUIRE(throws([] { G{{1, 1, 0.3}, 4, 4, 0.06, 0.07, 32, P::twill_2_2}; }));
  // z_centre ± (A + r) leaves [0, Lz].
  REQUIRE(throws([] { G{{1, 1, 0.2}, 2, 2, 0.05, 0.06}; }));
  // Yarn too thick for the spacing: r > s/(π√2) (s = 0.25 → 0.0563).
  REQUIRE(throws([] { G{{1, 1, 1.0}, 4, 4, 0.06, 0.06}; }));
  // Crimp radius of curvature s²/(π²A) below r (s = 0.25, A = 0.3).
  REQUIRE(throws([] { G{{1, 1, 1.0}, 4, 4, 0.05, 0.3}; }));
  // Valid configs still construct; flat 2D stays allowed.
  REQUIRE(!throws([] { G{{1, 1, 0.3}, 4, 4, 0.04, 0.05}; }));
  REQUIRE(!throws([] { G{{1, 1, 0.0}, 2, 2, 0.05, 0.0}; }));
  // 2D keeps the in-plane checks: 2x2 in a unit box has spacing 0.5, so
  // r = 0.4 makes parallel yarns overlap and leave the box.
  REQUIRE(throws([] { G{{1, 1, 0.0}, 2, 2, 0.4, 0.0}; }));
  REQUIRE(throws([] { G{{2, 1, 0.0}, 2, 2, 0.3, 0.0}; }));   // y spacing 0.5
  REQUIRE(!throws([] { G{{1, 1, 0.0}, 2, 2, 0.25, 0.0}; }));  // r = s/2
  // NaN / inf must be rejected, not slip through `x <= 0` checks.
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (double lz : {0.0, 0.3}) {
    REQUIRE(throws([&] { G{{1, 1, lz}, 2, 2, nan, 0.0}; }));
    REQUIRE(throws([&] { G{{1, 1, lz}, 2, 2, inf, 0.0}; }));
    REQUIRE(throws([&] { G{{nan, 1, lz}, 2, 2, 0.05, 0.0}; }));
    REQUIRE(throws([&] { G{{1, nan, lz}, 2, 2, 0.05, 0.0}; }));
    REQUIRE(throws([&] { G{{inf, 1, lz}, 2, 2, 0.05, 0.0}; }));
  }
  REQUIRE(throws([&] { G{{1, 1, nan}, 2, 2, 0.05, 0.0}; }));
  REQUIRE(throws([&] { G{{1, 1, 0.0}, 2, 2, 0.05, nan}; }));
  REQUIRE(throws([&] { G{{1, 1, 0.3}, 2, 2, 0.04, nan}; }));
  // Tow generator: NaN fibre factor.
  G base{{1, 1, 0.3}, 2, 2, 0.05, 0.06};
  REQUIRE(throws([&] {
    rvegen::tow_weave_generator<double>{base, 7, nan};
  }));
  // Tow over a 2D weave inherits the in-plane rejection.
  REQUIRE(throws([&] {
    rvegen::tow_weave_generator<double>{G{{1, 1, 0.0}, 2, 2, 0.4, 0.0}, 7};
  }));
}

// ----------------------------------------------------------------------------
// 3. Tow fibre layout.
// ----------------------------------------------------------------------------
void check_tow(std::size_t n_fibres, double factor,
               std::size_t n_seg = 16) {
  const double r_yarn = 0.05;
  rvegen::weave_generator<double> base{{1.0, 1.0, 0.3}, 2, 2, r_yarn, 0.06,
                                       n_seg};
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
      // Fibre nodes stay within the yarn envelope. Offsets are mitred, so
      // a node on the convex side of a kink sits (R - r_f)/cos(theta/2)
      // from the yarn node; bound that by the yarn's sharpest kink.
      REQUIRE(max_node_distance_to(fa.centerline(), yarn_cl)
              <= (r_yarn - r_f) * max_mitre(yarn_cl) + 1e-12);
      for (std::size_t b = a + 1; b < n_fibres; ++b) {
        auto const& fb = as_tube(fibres[y * n_fibres + b]);
        REQUIRE(polyline_distance(fa.centerline(), fb.centerline())
                >= 2.0 * r_f * (1.0 - 1e-9));
      }
    }
    // Periodic seam: each fibre's end node is its start node translated by
    // the yarn's own end-to-start vector (end normals use the end
    // segments, whose slopes the base weave makes equal).
    {
      const vec3 span = yarn_cl.back() - yarn_cl.front();
      for (std::size_t a = 0; a < n_fibres; ++a) {
        auto const& c = as_tube(fibres[y * n_fibres + a]).centerline();
        REQUIRE(gte::Length((c.back() - c.front()) - span) < 1e-12);
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


// Fibres of DIFFERENT yarns (including periodic images) must also clear
// 2 r_f: mitred offsets can sit slightly outside the yarn's polyline
// envelope on the convex side of a kink. Returns min distance / (2 r_f).
double check_tow_cross(rvegen::weave_generator<double> const& base,
                       std::array<double, 3> const& box,
                       std::size_t n_fibres, double factor) {
  rvegen::tow_weave_generator<double> gen{base, n_fibres, factor};
  auto yarns = base.build();
  auto fibres = gen.build();
  const double r_yarn = as_tube(yarns[0]).radius();
  const double r_f = r_yarn * factor;
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t ya = 0; ya < yarns.size(); ++ya) {
    for (std::size_t yb = ya; yb < yarns.size(); ++yb) {
      for (double dx : {-box[0], 0.0, box[0]}) {
        for (double dy : {-box[1], 0.0, box[1]}) {
          // A yarn and its own image shifted along its axis only share the
          // periodic seam point; skip that (and the identity).
          const bool warp = ya < base.n_warp_yarns();
          if (ya == yb && (warp ? dx != 0.0 : dy != 0.0)) continue;
          if (ya == yb && dx == 0.0 && dy == 0.0) continue;
          const vec3 shift{dx, dy, 0.0};
          if (polyline_distance(as_tube(yarns[ya]).centerline(),
                                as_tube(yarns[yb]).centerline(), shift)
              > 3.0 * r_yarn) continue;
          for (std::size_t a = 0; a < n_fibres; ++a) {
            for (std::size_t b = 0; b < n_fibres; ++b) {
              best = polyline_distance(
                  as_tube(fibres[ya * n_fibres + a]).centerline(),
                  as_tube(fibres[yb * n_fibres + b]).centerline(), shift,
                  best);
            }
          }
        }
      }
    }
  }
  return best / (2.0 * r_f);
}

void test_tow_cross_yarn_clearance() {
  using G = rvegen::weave_generator<double>;
  using P = rvegen::weave_pattern;
  for (std::size_t n_seg : {8u, 16u, 32u}) {
    for (auto const& pc :
         {std::pair{P::twill_2_2, weave_cfg{{1.0, 1.0, 0.3}, 4, 4, 0.04, 0.05, n_seg}},
          std::pair{P::twill_2_2, weave_cfg{{1.0, 1.0, 1.0}, 4, 4, 0.056, 0.0562, n_seg}},
          std::pair{P::satin_5, weave_cfg{{1.0, 1.0, 0.3}, 5, 5, 0.035, 0.04, n_seg}},
          std::pair{P::satin_5, weave_cfg{{1.0, 1.0, 1.0}, 5, 5, 0.045, 0.0451, n_seg}}}) {
      auto const& c = pc.second;
      G base{c.box, c.n_warp, c.n_weft, c.r, c.A, c.n_seg, pc.first};
      REQUIRE(check_tow_cross(base, c.box, 7, 1.0 / 3.0) >= 1.0 - 1e-9);
      REQUIRE(check_tow_cross(base, c.box, 19, 1.0 / 5.0) >= 1.0 - 1e-9);
    }
  }
  for (std::size_t n_seg : {8u, 16u, 32u}) {
    for (auto const& c : {weave_cfg{{1.0, 1.0, 0.3}, 4, 4, 0.04, 0.05, n_seg},
                          weave_cfg{{1.0, 1.0, 1.0}, 4, 4, 0.056, 0.0562, n_seg},
                          weave_cfg{{1.0, 1.0, 1.0}, 2, 2, 0.11, 0.112, n_seg}}) {
      G base{c.box, c.n_warp, c.n_weft, c.r, c.A, c.n_seg};
      const double m1 = check_tow_cross(base, c.box, 7, 1.0 / 3.0);
      const double m2 = check_tow_cross(base, c.box, 19, 1.0 / 5.0);
      REQUIRE(m1 >= 1.0 - 1e-9);
      REQUIRE(m2 >= 1.0 - 1e-9);
    }
  }
}

void test_tow_fibres_do_not_overlap() {
  check_tow(7, 0.2);      // default factor, 1 ring
  check_tow(2, 0.2);      // ring 1 with one site → centre empty, ±d
  check_tow(3, 0.2);      // partial ring 1 (2 of 6 sites)
  check_tow(4, 0.2);      // partial ring 1 (3 of 6 sites)
  check_tow(5, 0.2);      // partial ring 1 (4 of 6 sites)
  check_tow(8, 0.15);     // ring 2 with one site → centre empty
  check_tow(25, 0.125);   // 3 rings, partial ring 3 (6 of 18 sites)
  // Exactly at the documented bound factor = 1/(2K+1): fibres touch but
  // must not overlap anywhere along the polyline, not only at the nodes
  // (per-node normals gave 0.977 * 2 r_f with 8 segments). check_tow
  // measures every segment pair.
  for (std::size_t n_seg : {8u, 16u, 32u}) {
    check_tow(7, 1.0 / 3.0, n_seg);     // K = 1
    check_tow(19, 1.0 / 5.0, n_seg);    // K = 2
    check_tow(37, 1.0 / 7.0, n_seg);    // K = 3
  }
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
// 5. Acceptance (#3): 4×4 plain weave at fibre volume fraction 0.30.
// ----------------------------------------------------------------------------
// Voxel fraction of the yarns (plus their periodic wraps, so the cell
// holds exactly one period of every yarn) on an n_xy × n_xy × n_z grid.
double voxel_fraction(rvegen::weave_generator<double>::shape_vector const& yarns,
                      std::array<double, 3> const& box, std::size_t n_xy,
                      std::size_t n_z) {
  rvegen::weave_generator<double>::shape_vector all;
  for (auto const& y : yarns) {
    y->make_bounding_box();
    for (auto& w : rvegen::generate_periodic_wraps<double>(*y, box)) {
      all.push_back(std::move(w));
    }
    all.push_back(y->clone());
  }
  auto grid = rvegen::sample_voxel_grid(all, box, n_xy, n_xy, n_z);
  const auto filled = std::count_if(grid.begin(), grid.end(),
                                    [](auto id) { return id != 0; });
  return static_cast<double>(filled) / static_cast<double>(grid.size());
}

// What this proves: the yarns do not overlap and none is missing, since
// the voxel fill fraction must equal the summed tube volume. It does NOT
// prove the over/under topology: any non-overlapping pattern passes. The
// interlacing is covered by test_weave_patterns_clear_and_match_matrix.
void check_weave_voxel_fraction(rvegen::weave_pattern pattern, std::size_t n,
                                double r, double A, double target) {
  using G = rvegen::weave_generator<double>;
  // Yarn lengths don't depend on Lz, so size the cell from a trial
  // build: Lz = V_yarns / (target · Lx · Ly).
  double V = 0.0;
  for (auto const& s : G{{1, 1, 1}, n, n, r, A, 32, pattern}.build()) {
    V += s->volume();   // π r² · polyline length
  }
  const double Lz = V / target;
  const std::array<double, 3> box{1.0, 1.0, Lz};
  G gen{box, n, n, r, A, 32, pattern};
  auto yarns = gen.build();
  double V_check = 0.0;
  for (auto const& s : yarns) V_check += s->volume();
  const double vf_analytic = V_check / Lz;
  REQUIRE(std::abs(vf_analytic - target) < 1e-12);

  // The yarns never overlap (clearance ≥ 2r), so the union's volume is
  // the sum of the tube volumes; the polyline-joint correction is
  // O(r³θ³) per node (θ ≈ 0.27 rad here), < 0.1 % in total.
  //
  // Tolerance from the voxel size. Near-cubic voxels of edge dx; a
  // yarn cross-section is a disc of R = r/dx voxels whose boundary
  // ring (~2πR voxels against πR² inside) gives at most ~2/R relative
  // error per cross-section. The undulation moves the disc relative
  // to the lattice from one cross-section to the next, so the
  // N = n_yarns·L/dx cross-sections contribute errors of independent
  // sign: σ ≈ (2/R)/√N. We allow 3σ. (Measured over n_xy = 32…192:
  // |rel. error| 0.03 %…1.9 %, each ≤ 1.5σ. An interpenetrating weave
  // would lose up to 16r³/3 per crossing (Steinmetz solid), ≈ 15 %
  // here, far outside.)
  const std::size_t n_xy = 128;
  const auto n_z = static_cast<std::size_t>(std::lround(Lz * n_xy));
  const double dx = 1.0 / static_cast<double>(n_xy);
  const double R = r / dx;
  const double N = static_cast<double>(yarns.size()) / dx;
  const double tol = 3.0 * (2.0 / R) / std::sqrt(N);
  const double vf = voxel_fraction(yarns, box, n_xy, n_z);
  REQUIRE(std::abs(vf - vf_analytic) / vf_analytic <= tol);
  REQUIRE(tol < 0.05);   // keeps the test meaningful if parameters change
}

void test_plain_weave_voxel_fraction() {
  check_weave_voxel_fraction(rvegen::weave_pattern::plain, 4, 0.05, 0.055, 0.30);
}

// Twill 2/2 and satin 5 have long floats and different crossing sets, so
// check that they too voxelize to the analytic (non-interpenetrating) volume.
void test_pattern_weave_voxel_fraction() {
  check_weave_voxel_fraction(rvegen::weave_pattern::twill_2_2, 4, 0.04, 0.05, 0.10);
  check_weave_voxel_fraction(rvegen::weave_pattern::satin_5, 5, 0.035, 0.04, 0.10);
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

// build() output must carry bounding boxes so generate_periodic_wraps
// works on it directly (it returns nothing for shapes without one).
void test_build_output_has_bounding_box_for_wraps() {
  const std::array<double, 3> box{1.0, 1.0, 0.3};
  rvegen::weave_generator<double> w{box, 4, 4, 0.04, 0.05, 16};
  const auto count = [&](auto const& shapes) {
    std::size_t n = 0;
    for (auto const& s : shapes) {
      n += rvegen::generate_periodic_wraps<double>(*s, box).size();
    }
    return n;
  };
  const auto yarns = w.build();
  REQUIRE(count(yarns) == 16);
  rvegen::tow_weave_generator<double> tows{w, 7};
  const auto fibres = tows.build();
  REQUIRE(fibres.size() == 7 * yarns.size());
  REQUIRE(count(fibres) > 0);
  for (auto const& f : fibres) REQUIRE(f->bounding_box() != nullptr);
}

} // namespace

int main() {
  test_weave_crossings_clear_and_alternate();
  test_weave_patterns_clear_and_match_matrix();
  test_weave_validation();
  test_plain_weave_voxel_fraction();
  test_pattern_weave_voxel_fraction();
  test_tow_fibres_do_not_overlap();
  test_tow_cross_yarn_clearance();
  test_tow_infeasible_factor_throws();
  test_periodic_wraps_both_faces();
  test_periodic_wraps_single_side_counts_unchanged();
  test_build_output_has_bounding_box_for_wraps();

  if (failures == 0) {
    std::cout << "weave_geometry_test: all tests passed\n";
    return EXIT_SUCCESS;
  }
  std::cerr << "weave_geometry_test: " << failures << " failure(s)\n";
  return EXIT_FAILURE;
}
