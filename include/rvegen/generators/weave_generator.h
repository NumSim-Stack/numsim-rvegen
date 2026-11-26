#pragma once

// Deterministic woven-fabric generator (the #3 / #109 headline).
//
// Produces a plain-weave layout: N_warp yarns running along +x at
// evenly-spaced y, interlacing with N_weft yarns running along +y
// at evenly-spaced x. Each yarn is a `polyline_tube` whose
// centerline is sampled along its primary axis with a sinusoidal
// z-undulation (crimp). With s_x = Lx/N_weft, s_y = Ly/N_warp the yarn
// spacings and z_c the mid-plane,
//
//   warp_i(x) = z_c + A·(−1)^i · sin(π x / s_x)
//   weft_j(y) = z_c − A·(−1)^j · sin(π y / s_y)
//
// Crossings sit at x_j = (j+½)·s_x, y_i = (i+½)·s_y, where
// warp_i(x_j) = z_c + A·(−1)^(i+j) and weft_j(y_i) = z_c − A·(−1)^(i+j):
// warp i passes OVER weft j when (i+j) is even and UNDER otherwise, the
// centrelines are 2A apart in z at every crossing, and the over/under
// sign alternates along both i and j (1×1 plain-weave topology).
//
// Periodicity: each profile has period 2s, so it tiles the domain only
// when the yarn count across it is even. This is topological, not a
// choice of phase: travelling once across a periodic cell a warp passes
// N_weft crossings and flips over/under at each, so it returns to its
// start state only if N_weft is even (same for wefts and N_warp). Both
// counts must therefore be even (≥ 2); odd counts are rejected rather
// than producing a seam where the alternation breaks.
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
//   * `n_segments_per_yarn` controls how finely each yarn's
//     centerline is sampled. It is rounded UP to a multiple of
//     2·(number of crossings along the yarn) so every crossing (crimp
//     peak) and every mid-point (z = z_c) is an exact control point.
//     16 segments is plenty for visualisation, 32–64 for fine voxel
//     grids.
//
// Feasibility (3D, checked in the ctor, with r = yarn_radius,
// A = amplitude, s = min(s_x, s_y)):
//   * A ≥ r — warp and weft are 2A apart at each crossing, so this is
//     what keeps the tubes from interpenetrating there.
//   * r ≤ s / (π√2) — the crossing is the closest approach between
//     warp i and weft j only while A ≤ s/(π√2); beyond that the steep
//     flanks of the crimp come closer than 2A. Since the closest
//     approach grows monotonically with A, the centreline clearance is
//     ≥ 2·min(A, s/(π√2)) for every (i, j) pair, including periodic
//     images, and these two bounds together give clearance ≥ 2r. (This
//     also implies s > 2r, so parallel neighbours never touch.)
//   * r·A·π² ≤ s² — the minimum radius of curvature of the crimp,
//     s²/(π²A) at the peaks, must be ≥ r or the tube folds over itself.
//   * A + r ≤ Lz/2 — z_c ± (A + r) must stay inside [0, Lz].
//   These are guarantees for the analytic centreline. The polyline
//   samples it exactly at every crossing; between nodes each chord
//   deviates by at most A·π²·h²/(8·s²) (h = segment length), which
//   is how much closer two polyline yarns may come at the bounds.
//   Raise n_segments_per_yarn to shrink that error quadratically.
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
//   * Twill / satin weaves (different over/under patterns at
//     intersection points).
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

template <typename T = double>
class weave_generator {
public:
  using value_type = T;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  // Construct a plain weave on the given domain. `amplitude` is the
  // peak-to-zero z-displacement of each yarn's sinusoidal
  // centerline (so total z-extent is 2·amplitude + 2·yarn_radius).
  // Throws std::runtime_error on odd yarn counts or a configuration
  // that would make yarns interpenetrate or leave the domain (see the
  // feasibility list in the header comment).
  weave_generator(std::array<value_type, 3> const& domain_box,
                  std::size_t n_warp_yarns,
                  std::size_t n_weft_yarns,
                  value_type yarn_radius,
                  value_type amplitude,
                  std::size_t n_segments_per_yarn = 32)
      : _domain{domain_box},
        _n_warp{n_warp_yarns},
        _n_weft{n_weft_yarns},
        _yarn_radius{yarn_radius},
        _amplitude{amplitude},
        _n_segments{n_segments_per_yarn} {
    if (n_warp_yarns == 0 || n_weft_yarns == 0) {
      throw std::runtime_error{
          "weave_generator: n_warp_yarns and n_weft_yarns must both be ≥ 1"};
    }
    if (yarn_radius <= value_type{0}) {
      throw std::runtime_error{
          "weave_generator: yarn_radius must be positive"};
    }
    if (n_segments_per_yarn < 2) {
      throw std::runtime_error{
          "weave_generator: n_segments_per_yarn must be ≥ 2 (the "
          "polyline_tube needs at least two centerline points)"};
    }
    if (_domain[0] <= value_type{0} || _domain[1] <= value_type{0}) {
      throw std::runtime_error{
          "weave_generator: domain Lx and Ly must both be positive"};
    }
    // 2D domains (Lz == 0) physically can't carry undulation; the
    // yarns would land at z = ±amplitude, outside a zero-extent domain.
    // Reject rather than silently producing geometrically meaningless
    // shapes — voxelization of a 2D domain collapses to a single
    // z-slice that won't capture undulation anyway.
    if (_domain[2] <= value_type{0} && amplitude != value_type{0}) {
      throw std::runtime_error{
          "weave_generator: 2D domain (Lz == 0) requires amplitude = 0; "
          "for an undulating weave use a 3D domain with positive Lz"};
    }
    if (n_warp_yarns % 2 != 0 || n_weft_yarns % 2 != 0) {
      throw std::runtime_error{
          "weave_generator: n_warp_yarns and n_weft_yarns must both be "
          "even — a periodic plain weave flips over/under at every "
          "crossing, so an odd count breaks the alternation at the "
          "periodic seam"};
    }
    if (_domain[2] > value_type{0}) validate_clearance();
  }

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
    const std::size_t n_seg_warp = segments_for(_n_weft);
    for (std::size_t i = 0; i < _n_warp; ++i) {
      const value_type y = yarn_position(i, _n_warp, Ly);
      std::vector<std::array<value_type, 3>> centerline;
      centerline.reserve(n_seg_warp + 1);
      for (std::size_t k = 0; k <= n_seg_warp; ++k) {
        const value_type x = Lx * static_cast<value_type>(k)
                              / static_cast<value_type>(n_seg_warp);
        centerline.push_back({x, y, zc + crimp(i, x, Lx, _n_weft)});
      }
      auto tube = std::make_unique<polyline_tube<value_type>>(
          centerline, _yarn_radius);
      if (!_warp_phase.empty()) tube->set_phase_name(_warp_phase);
      out.push_back(std::move(tube));
    }

    // Weft yarns run along +y at evenly-spaced x, with the opposite
    // crimp sign so warp and weft sit on opposite sides of z_centre at
    // every crossing.
    const std::size_t n_seg_weft = segments_for(_n_warp);
    for (std::size_t j = 0; j < _n_weft; ++j) {
      const value_type x = yarn_position(j, _n_weft, Lx);
      std::vector<std::array<value_type, 3>> centerline;
      centerline.reserve(n_seg_weft + 1);
      for (std::size_t k = 0; k <= n_seg_weft; ++k) {
        const value_type y = Ly * static_cast<value_type>(k)
                              / static_cast<value_type>(n_seg_weft);
        centerline.push_back({x, y, zc - crimp(j, y, Ly, _n_warp)});
      }
      auto tube = std::make_unique<polyline_tube<value_type>>(
          centerline, _yarn_radius);
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

  // Signed crimp offset A·(−1)^idx·sin(π t / s) of yarn `idx` at
  // coordinate `t` along its axis, where s = L / n_crossing is the
  // spacing of the yarns it crosses.
  [[nodiscard]] value_type crimp(std::size_t idx, value_type t, value_type L,
                                 std::size_t n_crossing) const noexcept {
    using std::numbers::pi_v;
    const value_type spacing = L / static_cast<value_type>(n_crossing);
    const value_type sign = (idx % 2 == 0) ? value_type{1} : value_type{-1};
    return sign * _amplitude * std::sin(pi_v<value_type> * t / spacing);
  }

  // Segments for a yarn that crosses `n_crossing` others: the requested
  // count rounded up to a multiple of 2·n_crossing, so each crossing
  // (j+½)·s and each mid-point j·s lands on a control point.
  [[nodiscard]] std::size_t segments_for(std::size_t n_crossing) const noexcept {
    const std::size_t step = 2 * n_crossing;
    return ((_n_segments + step - 1) / step) * step;
  }

  void validate_clearance() const {
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
    if (A * tol < r) {
      throw std::runtime_error{
          "weave_generator: amplitude must be ≥ yarn_radius — warp and "
          "weft centrelines are 2·amplitude apart at each crossing, so a "
          "smaller amplitude makes the yarns interpenetrate"};
    }
    if (r * pi_v<value_type> * sqrt2_v<value_type> > s * tol) {
      throw std::runtime_error{
          "weave_generator: yarn_radius must be ≤ spacing/(π·√2), with "
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
  }
};

} // namespace rvegen
