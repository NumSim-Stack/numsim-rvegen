#pragma once

// Deterministic woven-fabric generator (the #3 / #109 headline).
//
// Produces a plain-weave layout: N_warp yarns running along +x at
// evenly-spaced y, interlacing with N_weft yarns running along +y
// at evenly-spaced x. Each yarn is a `polyline_tube` whose
// centerline is sampled at K control points along its primary axis,
// with a sinusoidal z-undulation. Warp and weft are 180° out of
// phase so they over-/under-cross at each intersection, giving the
// characteristic textile structure.
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
//     (for 2D — undulation amplitude must be ≤ 0 then or the
//     yarns leave the plane).
//   * Yarn radii are uniform across both warp and weft.
//   * `n_segments_per_yarn` controls how finely each yarn's
//     centerline is sampled. The sinusoid is smoothest at high
//     resolutions; 16 segments is plenty for visualisation, 32–64
//     for fine voxel grids.
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
//   * Tow + yarn hierarchy (each warp/weft as a bundle of fibres
//     rather than a single tube).
//   * Variable yarn radius along the centerline (yarn compression
//     at intersections).

#include <array>
#include <cmath>
#include <cstddef>
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
    const value_type Lz = _domain[2];
    // Centre the weave on z = Lz/2 (or z=0 for 2D — amplitude must
    // then be 0 or negative for the layout to make sense, but we
    // don't refuse it: caller may want a flat 2D weave for testing).
    const value_type z_centre = (Lz > value_type{0}) ? Lz * value_type{0.5}
                                                     : value_type{0};

    // Warp yarns run along +x at evenly-spaced y. The first yarn sits
    // at y = Ly/(2·N_warp); successive yarns are at uniform spacing
    // Ly/N_warp; the last is at y = Ly·(2·N_warp - 1)/(2·N_warp). The
    // boundary-most yarns are set back by half-spacing from the domain
    // edges, which is what uniform-interior-spacing requires (any
    // edge-flush layout would leave the interior non-uniform). Weft
    // uses the same convention along x.
    //
    // z(x) = z_centre + amplitude · sin(2π · x / period_x), where
    // period_x equals the weft yarn spacing — one wavelength per
    // intersection-period, matching the textile-physics meaning.
    const value_type period_x =
        (_n_weft > 0) ? Lx / static_cast<value_type>(_n_weft) : Lx;
    const value_type period_y =
        (_n_warp > 0) ? Ly / static_cast<value_type>(_n_warp) : Ly;

    using std::numbers::pi_v;
    const value_type two_pi = value_type{2} * pi_v<value_type>;

    // Warp yarns: vary x ∈ [0, Lx], hold y fixed.
    for (std::size_t i = 0; i < _n_warp; ++i) {
      const value_type y = Ly * (value_type{2} * static_cast<value_type>(i)
                                  + value_type{1})
                            / (value_type{2} * static_cast<value_type>(_n_warp));
      std::vector<std::array<value_type, 3>> centerline;
      centerline.reserve(_n_segments + 1);
      for (std::size_t k = 0; k <= _n_segments; ++k) {
        const value_type x = Lx * static_cast<value_type>(k)
                              / static_cast<value_type>(_n_segments);
        const value_type z = z_centre +
            _amplitude * std::sin(two_pi * x / period_x);
        centerline.push_back({x, y, z});
      }
      auto tube = std::make_unique<polyline_tube<value_type>>(
          centerline, _yarn_radius);
      if (!_warp_phase.empty()) tube->set_phase_name(_warp_phase);
      out.push_back(std::move(tube));
    }

    // Weft yarns: vary y ∈ [0, Ly], hold x fixed. z is 180° out of
    // phase with the warp (minus sign on the sinusoid) so the two
    // yarn families interlace.
    for (std::size_t j = 0; j < _n_weft; ++j) {
      const value_type x = Lx * (value_type{2} * static_cast<value_type>(j)
                                  + value_type{1})
                            / (value_type{2} * static_cast<value_type>(_n_weft));
      std::vector<std::array<value_type, 3>> centerline;
      centerline.reserve(_n_segments + 1);
      for (std::size_t k = 0; k <= _n_segments; ++k) {
        const value_type y = Ly * static_cast<value_type>(k)
                              / static_cast<value_type>(_n_segments);
        const value_type z = z_centre -
            _amplitude * std::sin(two_pi * y / period_y);
        centerline.push_back({x, y, z});
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
};

} // namespace rvegen
