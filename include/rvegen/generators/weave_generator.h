#pragma once

// Deterministic woven-fabric generator (the #3 / #109 headline).
//
// Produces a woven layout: N_warp yarns running along +x at
// evenly-spaced y, interlacing with N_weft yarns running along +y
// at evenly-spaced x. Each yarn is a `polyline_tube` whose
// centerline is sampled along its primary axis with a z-undulation
// (crimp). With s_x = Lx/N_weft, s_y = Ly/N_warp the yarn spacings,
// z_c the mid-plane and crossings at x_j = (j+½)·s_x, y_i = (i+½)·s_y:
//
// Interlacing. The pattern is a matrix P(i, j) ∈ {+1, −1} with
// P = +1 when warp i passes OVER weft j, periodic with repeat R in
// both indices (`weave_pattern`):
//   * plain      R = 2  P = +1 iff (i + j) mod 2 = 0      (1 over 1)
//   * twill_2_2  R = 4  P = +1 iff (j − i) mod 4 ∈ {0, 1} (2 over 2,
//                       shifted one weft per warp → diagonal rib)
//   * satin_5    R = 5  P = −1 iff j ≡ 2i (mod 5)         (warp-faced:
//                       over 4, under 1; interlacing points never
//                       adjacent, step 2)
// At crossing (i, j) the warp sits at z_c + A·P(i, j) and the weft at
// z_c − A·P(i, j), so the centrelines are 2A apart and the topology is
// exactly P.
//
// Profile. Between two consecutive crossings (local t ∈ [0, 1]) a yarn
// with crossing values p0, p1 follows
//   z = z_c + A·[ (p0 + p1)/2 + (p0 − p1)/2 · cos(π t) ],
// i.e. a half-cosine transition when the sign changes and a flat
// float when it doesn't. For plain weave this IS the sinusoid
// warp_i(x) = z_c + A·(−1)^i·sin(π x/s_x) (weft: minus sign, j, y).
// The profile is C¹ (zero slope at every crossing) with
// |z''| ≤ A·π²/s² everywhere — the same bound as the plain sinusoid,
// because floats only add zero-curvature stretches.
//
// Periodicity: P has repeat R, so each yarn count must be a multiple
// of R (even for plain, 4 for twill_2_2, 5 for satin_5). For plain
// weave this is topological, not a choice of phase: travelling once
// across a periodic cell a warp flips over/under at every crossing,
// so it returns to its start state only if the count is even.
//
// This is a deterministic tessellation, NOT a sampling generator —
// hence it does NOT inherit `rve_generator_base` (which takes
// inputs + termination + an engine and runs an iterative placement
// loop). The output type matches: a `shape_vector` that any
// downstream writer (voxel_writer, gmsh_geo_writer, vtk) can
// consume just like the output of `random_generator` etc.
//
// Geometry conventions:
//   * Domain box `[0, Lx] × [0, Ly] × [0, Lz]`. The weave is laid
//     in the (x, y) plane centred at z = Lz/2 (for 3D) or z = 0
//     (for 2D — amplitude must be 0 then; a flat 2D weave is a
//     projection and its yarns overlap at the crossings by design).
//   * Yarn radii are uniform across both warp and weft.
//   * `n_segments_per_yarn` is the MINIMUM number of segments per
//     yarn centerline. Each yarn family uses 2·N·q segments (N = the
//     number of yarns it crosses), so every crossing (crimp peak) and
//     every mid-point (z = z_c) is an exact control point; q is the
//     smallest value that meets the request AND the polyline
//     clearance below. The result is capped at `max_segments_per_yarn`
//     (the request too); configs needing more throw.
//
// Feasibility (3D, checked in the ctor, with r = yarn_radius,
// A = amplitude, s = min(s_x, s_y), m = min(A, s/(π√2))):
//   * Analytic clearance. Warp and weft are 2A apart at each
//     crossing. The crossing is their closest approach only while
//     A ≤ s/(π√2); beyond that the steep flanks come closer than 2A.
//     The closest approach grows monotonically with A, so the analytic
//     centreline clearance is ≥ 2m for every (i, j) pair, including
//     periodic images. (m > r also implies s > 2r, so parallel
//     neighbours never touch.)
//     The proof holds for every pattern. Write the warp near crossing
//     (i, j) as z_c + A·P·g(u), u = π(x − x_j)/s_x, and the weft as
//     z_c − A·P·h(v). Then g(0) = h(0) = 1, and the fastest any
//     profile leaves ±A is the half-cosine, so 1 − g(u) ≤ 1 − cos u
//     ≤ u²/2 on |u| ≤ π, and 1 − g ≤ 2 ≤ u²/2 beyond. With
//     c = (g + h)/2: 4A² − A²(g + h)² = 4A²(1 − c)(1 + c)
//     ≤ 8A²(1 − c) ≤ 2A²(u² + v²) ≤ (s/π)²(u² + v²) when A ≤ s/(π√2),
//     so the squared distance is ≥ 4A². Floats never reduce clearance.
//   * Polyline clearance. The emitted polylines are inscribed in the
//     curves, and linear interpolation deviates from a curve by at
//     most h²·max|z''|/8 = A·π²/(32·q²) (h = s/(2q) per segment). Each
//     yarn polyline is therefore within e = A·π²/(32q²) of its curve,
//     and the polylines are ≥ 2m − 2e apart. q is raised until
//     e ≤ m − r, which guarantees clearance ≥ 2r for the polylines that
//     are actually emitted. This needs A > r and r < s/(π√2) STRICTLY:
//     at A = r an inscribed polyline always dips below the crimp peak
//     next to the crossing node, so no finite segment count reaches 2r.
//   * r·A·π² ≤ s² — the minimum radius of curvature of the crimp,
//     s²/(π²A) at the peaks, must be ≥ r or the tube folds over itself.
//   * A + r ≤ Lz/2 — z_c ± (A + r) must stay inside [0, Lz].
//
// What this header lands today (phase 1 of #3):
//   * Direct ctor with all weave parameters as floats.
//   * `build()` returning `shape_vector`.
//   * Per-yarn phase tagging via `set_warp_phase_name` /
//     `set_weft_phase_name` so downstream writers can emit
//     separate Physical groups per yarn family.
//
// Out of scope here, ships in follow-up PRs:
//   * JSON-driven registration in a generator registry (today the
//     generator registry only carries iterative-sampling
//     generators; weave is deterministic and doesn't fit the
//     `compute(inputs, termination, box)` shape — needs a separate
//     registry or a generator-base relaxation).
//   * Other patterns (twill 3/1, satin 8, basket weave) — add a case
//     to `interlacing()`; the clearance argument covers any ±1 matrix.
//   * Non-circular (lenticular) yarn cross-sections, which are what
//     let real fabrics pack tighter than the r ≤ s/(π√2) bound.
//   * Tow + yarn hierarchy (each warp/weft as a bundle of fibres
//     rather than a single tube).
//   * Variable yarn radius along the centerline (yarn compression
//     at intersections).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../shapes/polyline_tube.h"
#include "../shapes/shape_base.h"

namespace rvegen {

// Interlacing pattern of a `weave_generator` (see the header comment
// for the matrices and their repeats).
enum class weave_pattern { plain, twill_2_2, satin_5 };

// Repeat of a pattern: the yarn counts must be multiples of it.
[[nodiscard]] constexpr std::size_t
weave_pattern_repeat(weave_pattern p) noexcept {
  switch (p) {
    case weave_pattern::twill_2_2: return 4;
    case weave_pattern::satin_5: return 5;
    case weave_pattern::plain: break;
  }
  return 2;
}

template <typename T = double>
class weave_generator {
public:
  using value_type = T;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  // Construct a weave on the given domain (plain by default).
  // `amplitude` is the peak-to-zero z-displacement of each yarn's
  // centerline (so total z-extent is 2·amplitude + 2·yarn_radius).
  // Throws std::runtime_error when a yarn count is not a multiple of
  // the pattern repeat, or for a configuration that would make yarns
  // interpenetrate or leave the domain (see the feasibility list in
  // the header comment).
  weave_generator(std::array<value_type, 3> const& domain_box,
                  std::size_t n_warp_yarns,
                  std::size_t n_weft_yarns,
                  value_type yarn_radius,
                  value_type amplitude,
                  std::size_t n_segments_per_yarn = 32,
                  weave_pattern pattern = weave_pattern::plain)
      : _domain{domain_box},
        _n_warp{n_warp_yarns},
        _n_weft{n_weft_yarns},
        _yarn_radius{yarn_radius},
        _amplitude{amplitude},
        _n_segments{n_segments_per_yarn},
        _n_seg_warp{0},
        _n_seg_weft{0},
        _pattern{pattern} {
    if (n_warp_yarns == 0 || n_weft_yarns == 0) {
      throw std::runtime_error{
          "weave_generator: n_warp_yarns and n_weft_yarns must both be ≥ 1"};
    }
    if (!(yarn_radius > value_type{0}) || !std::isfinite(yarn_radius)) {
      throw std::runtime_error{
          "weave_generator: yarn_radius must be positive and finite"};
    }
    if (n_segments_per_yarn < 2) {
      throw std::runtime_error{
          "weave_generator: n_segments_per_yarn must be ≥ 2 (the "
          "polyline_tube needs at least two centerline points)"};
    }
    if (n_segments_per_yarn > max_segments_per_yarn) {
      throw std::runtime_error{
          "weave_generator: n_segments_per_yarn must be ≤ "
          + std::to_string(max_segments_per_yarn)};
    }
    if (!(_domain[0] > value_type{0}) || !(_domain[1] > value_type{0})
        || !std::isfinite(_domain[0]) || !std::isfinite(_domain[1])) {
      throw std::runtime_error{
          "weave_generator: domain Lx and Ly must both be positive and "
          "finite"};
    }
    if (!(_domain[2] >= value_type{0}) || !std::isfinite(_domain[2])) {
      throw std::runtime_error{
          "weave_generator: domain Lz must be ≥ 0 and finite"};
    }
    // 2D domains (Lz == 0) physically can't carry undulation; the
    // yarns would land at z = ±amplitude, outside a zero-extent domain.
    // Reject rather than silently producing geometrically meaningless
    // shapes — voxelization of a 2D domain collapses to a single
    // z-slice that won't capture undulation anyway.
    if (_domain[2] <= value_type{0} && !(amplitude == value_type{0})) {
      throw std::runtime_error{
          "weave_generator: 2D domain (Lz == 0) requires amplitude = 0; "
          "for an undulating weave use a 3D domain with positive Lz"};
    }
    const std::size_t repeat = weave_pattern_repeat(pattern);
    if (n_warp_yarns % repeat != 0 || n_weft_yarns % repeat != 0) {
      throw std::runtime_error{
          "weave_generator: n_warp_yarns and n_weft_yarns must both be "
          "multiples of the pattern repeat ("
          + std::to_string(repeat) + " for "
          + (pattern == weave_pattern::twill_2_2 ? "twill_2_2"
             : pattern == weave_pattern::satin_5 ? "satin_5" : "plain")
          + ") so the interlacing tiles the periodic cell"};
    }
    // Segment multiplier q needed for polyline clearance (3D only;
    // a flat 2D weave has no crimp to resolve, but still needs the
    // in-plane spacing / box checks).
    std::size_t q_clearance = 1;
    if (_domain[2] > value_type{0}) q_clearance = validate_clearance();
    else validate_in_plane();
    _n_seg_warp = segments_for(_n_weft, q_clearance);
    _n_seg_weft = segments_for(_n_warp, q_clearance);
  }

  // Upper limit on segments per yarn centerline, for both the request
  // and the automatic refinement.
  static constexpr std::size_t max_segments_per_yarn = 4096;

  // Optional phase-name tags applied to every produced yarn. By
  // default both are empty (untagged); writers attach a
  // `phase_collection` and the empty-name shapes fall through to
  // gmsh's default group / get id 0 in the voxel grid. Setting
  // these lets a downstream pipeline emit separate Physical Surfaces
  // per yarn family ("warp", "weft", or whatever names the user picks).
  void set_warp_phase_name(std::string name) { _warp_phase = std::move(name); }
  void set_weft_phase_name(std::string name) { _weft_phase = std::move(name); }

  [[nodiscard]] std::string const& warp_phase_name() const noexcept {
    return _warp_phase;
  }
  [[nodiscard]] std::string const& weft_phase_name() const noexcept {
    return _weft_phase;
  }

  [[nodiscard]] std::size_t n_warp_yarns() const noexcept { return _n_warp; }
  [[nodiscard]] std::size_t n_weft_yarns() const noexcept { return _n_weft; }
  [[nodiscard]] weave_pattern pattern() const noexcept { return _pattern; }

  // Interlacing matrix: +1 when warp i passes OVER weft j, −1 when it
  // passes under. Periodic in i and j with the pattern repeat.
  [[nodiscard]] int interlacing(std::size_t i, std::size_t j) const noexcept {
    switch (_pattern) {
      case weave_pattern::twill_2_2:
        return ((j + 4 - i % 4) % 4 < 2) ? +1 : -1;
      case weave_pattern::satin_5:
        return (j % 5 == (2 * i) % 5) ? -1 : +1;
      case weave_pattern::plain: break;
    }
    return ((i + j) % 2 == 0) ? +1 : -1;
  }

  // Segments actually emitted per warp / weft centerline (≥ the
  // requested n_segments_per_yarn; see the header comment).
  [[nodiscard]] std::size_t n_segments_warp() const noexcept {
    return _n_seg_warp;
  }
  [[nodiscard]] std::size_t n_segments_weft() const noexcept {
    return _n_seg_weft;
  }

  // Build the woven layout. Returns one polyline_tube per yarn,
  // first the warp yarns (along +x) then the weft yarns (along +y).
  [[nodiscard]] shape_vector build() const {
    shape_vector out;
    out.reserve(_n_warp + _n_weft);

    const value_type Lx = _domain[0];
    const value_type Ly = _domain[1];
    const value_type zc = z_centre();

    // Warp yarns run along +x at evenly-spaced y. The first yarn sits
    // at y = Ly/(2·N_warp); successive yarns are at uniform spacing
    // Ly/N_warp; the last is at y = Ly·(2·N_warp - 1)/(2·N_warp). The
    // boundary-most yarns are set back by half-spacing from the domain
    // edges, which is what uniform-interior-spacing requires (any
    // edge-flush layout would leave the interior non-uniform). Weft
    // uses the same convention along x, so the crossings are at
    // ((j+½)·s_x, (i+½)·s_y).
    const std::size_t n_seg_warp = _n_seg_warp;
    for (std::size_t i = 0; i < _n_warp; ++i) {
      const value_type y = yarn_position(i, _n_warp, Ly);
      std::vector<std::array<value_type, 3>> centerline;
      centerline.reserve(n_seg_warp + 1);
      for (std::size_t k = 0; k <= n_seg_warp; ++k) {
        const value_type x = Lx * static_cast<value_type>(k)
                              / static_cast<value_type>(n_seg_warp);
        centerline.push_back({x, y, zc + crimp(i, x, Lx, _n_weft, true)});
      }
      auto tube = std::make_unique<polyline_tube<value_type>>(
          centerline, _yarn_radius);
      tube->make_bounding_box();
      if (!_warp_phase.empty()) tube->set_phase_name(_warp_phase);
      out.push_back(std::move(tube));
    }

    // Weft yarns run along +y at evenly-spaced x, with the opposite
    // crimp sign so warp and weft sit on opposite sides of z_centre at
    // every crossing.
    const std::size_t n_seg_weft = _n_seg_weft;
    for (std::size_t j = 0; j < _n_weft; ++j) {
      const value_type x = yarn_position(j, _n_weft, Lx);
      std::vector<std::array<value_type, 3>> centerline;
      centerline.reserve(n_seg_weft + 1);
      for (std::size_t k = 0; k <= n_seg_weft; ++k) {
        const value_type y = Ly * static_cast<value_type>(k)
                              / static_cast<value_type>(n_seg_weft);
        centerline.push_back({x, y, zc - crimp(j, y, Ly, _n_warp, false)});
      }
      auto tube = std::make_unique<polyline_tube<value_type>>(
          centerline, _yarn_radius);
      tube->make_bounding_box();
      if (!_weft_phase.empty()) tube->set_phase_name(_weft_phase);
      out.push_back(std::move(tube));
    }

    return out;
  }

private:
  std::array<value_type, 3> _domain;
  std::size_t _n_warp;
  std::size_t _n_weft;
  value_type _yarn_radius;
  value_type _amplitude;
  std::size_t _n_segments;
  std::size_t _n_seg_warp;
  std::size_t _n_seg_weft;
  weave_pattern _pattern;
  std::string _warp_phase;
  std::string _weft_phase;

  [[nodiscard]] value_type z_centre() const noexcept {
    return (_domain[2] > value_type{0}) ? _domain[2] * value_type{0.5}
                                        : value_type{0};
  }

  // Centre of yarn `idx` of `n` evenly spaced across length `L`.
  [[nodiscard]] static value_type yarn_position(std::size_t idx,
                                                std::size_t n,
                                                value_type L) noexcept {
    return L * (value_type{2} * static_cast<value_type>(idx) + value_type{1})
           / (value_type{2} * static_cast<value_type>(n));
  }

  // Signed crimp offset of yarn `idx` at coordinate `t` along its axis
  // (warp: x, weft: y), where s = L / n_crossing is the spacing of the
  // yarns it crosses. Returns A·P along the warp's row of the
  // interlacing matrix (warp) or its column (weft), half-cosine
  // between crossings of opposite sign, flat between equal ones.
  [[nodiscard]] value_type crimp(std::size_t idx, value_type t, value_type L,
                                 std::size_t n_crossing,
                                 bool is_warp) const noexcept {
    using std::numbers::pi_v;
    const value_type spacing = L / static_cast<value_type>(n_crossing);
    // Crossings at integer u; u ∈ [−½, n_crossing − ½] over [0, L].
    const value_type u = t / spacing - value_type{0.5};
    const value_type k_floor = std::floor(u);
    const value_type frac = u - k_floor;
    const auto n = static_cast<long long>(n_crossing);
    const long long k = static_cast<long long>(k_floor);
    const auto wrap = [n](long long c) {
      return static_cast<std::size_t>(((c % n) + n) % n);
    };
    const std::size_t c0 = wrap(k);
    const std::size_t c1 = wrap(k + 1);
    const value_type p0 = static_cast<value_type>(
        is_warp ? interlacing(idx, c0) : interlacing(c0, idx));
    const value_type p1 = static_cast<value_type>(
        is_warp ? interlacing(idx, c1) : interlacing(c1, idx));
    return _amplitude * ((p0 + p1) * value_type{0.5}
                         + (p0 - p1) * value_type{0.5}
                               * std::cos(pi_v<value_type> * frac));
  }

  // Segments for a yarn that crosses `n_crossing` others: 2·n_crossing·q
  // with q ≥ q_min and 2·n_crossing·q ≥ the requested count, so each
  // crossing (j+½)·s and each mid-point j·s lands on a control point.
  // Throws if that exceeds max_segments_per_yarn. Overflow-safe: the
  // request is already ≤ the cap.
  [[nodiscard]] std::size_t segments_for(std::size_t n_crossing,
                                         std::size_t q_min) const {
    if (n_crossing > max_segments_per_yarn / 2) {
      throw std::runtime_error{
          "weave_generator: too many yarns — each yarn needs 2 segments "
          "per crossing and at most "
          + std::to_string(max_segments_per_yarn) + " segments are allowed"};
    }
    const std::size_t step = 2 * n_crossing;
    const std::size_t q = std::max((_n_segments + step - 1) / step, q_min);
    if (q > max_segments_per_yarn / step) {
      throw std::runtime_error{
          "weave_generator: the polyline needs more than "
          + std::to_string(max_segments_per_yarn) + " segments per yarn to "
          "keep warp and weft 2·yarn_radius apart — increase amplitude "
          "and/or the yarn spacing, or reduce yarn_radius, to widen the "
          "margin min(amplitude, spacing/(π·√2)) − yarn_radius"};
    }
    return step * q;
  }

  // 2D (Lz == 0) checks: parallel yarns must not overlap and the
  // outermost yarns must stay inside the box, i.e. 2·r ≤ the yarn
  // spacing in both directions (edge yarns sit half a spacing from the
  // faces). Warp/weft overlap at crossings is by design in 2D.
  void validate_in_plane() const {
    const value_type s = std::min(
        _domain[0] / static_cast<value_type>(_n_weft),
        _domain[1] / static_cast<value_type>(_n_warp));
    const value_type tol = value_type{1}
        + value_type{64} * std::numeric_limits<value_type>::epsilon();
    if (!(value_type{2} * _yarn_radius <= s * tol)) {
      throw std::runtime_error{
          "weave_generator: yarn_radius must be ≤ spacing/2, with "
          "spacing = min(Lx/n_weft_yarns, Ly/n_warp_yarns) — otherwise "
          "parallel yarns overlap and the outer yarns leave the box"};
    }
  }

  // Checks the 3D feasibility conditions and returns the segment
  // multiplier q needed for the polyline clearance (see header).
  [[nodiscard]] std::size_t validate_clearance() const {
    using std::numbers::pi_v;
    using std::numbers::sqrt2_v;
    const value_type r = _yarn_radius;
    const value_type A = _amplitude;
    const value_type s = std::min(
        _domain[0] / static_cast<value_type>(_n_weft),
        _domain[1] / static_cast<value_type>(_n_warp));
    // Relative slack so configurations exactly on a bound (e.g.
    // A + r == Lz/2) are not rejected by rounding.
    const value_type tol = value_type{1}
        + value_type{64} * std::numeric_limits<value_type>::epsilon();
    const value_type a_flank = s / (pi_v<value_type> * sqrt2_v<value_type>);
    if (!(A > r)) {
      throw std::runtime_error{
          "weave_generator: amplitude must be > yarn_radius — warp and "
          "weft centrelines are 2·amplitude apart at each crossing, and "
          "the sampled polyline dips below the crimp peak next to it, so "
          "amplitude ≤ yarn_radius makes the yarns interpenetrate"};
    }
    if (!(r < a_flank)) {
      throw std::runtime_error{
          "weave_generator: yarn_radius must be < spacing/(π·√2), with "
          "spacing = min(Lx/n_weft_yarns, Ly/n_warp_yarns) — otherwise "
          "the crimp flanks bring warp and weft closer than 2·yarn_radius "
          "next to the crossings"};
    }
    if (r * A * pi_v<value_type> * pi_v<value_type> > s * s * tol) {
      throw std::runtime_error{
          "weave_generator: yarn_radius·amplitude·π² must be ≤ spacing² — "
          "the crimp's radius of curvature at the peaks is smaller than "
          "yarn_radius, so the tube would fold over itself"};
    }
    if (A + r > z_centre() * tol) {
      throw std::runtime_error{
          "weave_generator: amplitude + yarn_radius must be ≤ Lz/2 so "
          "the yarns stay inside [0, Lz]"};
    }
    // Polyline clearance: smallest q with A·π²/(32q²) ≤ m − r.
    const value_type margin = std::min(A, a_flank) - r;
    const value_type q_needed =
        pi_v<value_type> * std::sqrt(A / (value_type{32} * margin));
    if (!(q_needed < static_cast<value_type>(max_segments_per_yarn))) {
      return max_segments_per_yarn + 1;   // segments_for() throws
    }
    return std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(q_needed)));
  }
};

} // namespace rvegen
