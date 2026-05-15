#pragma once

// Tow-level weave generator (#109 follow-up to #57).
//
// Wraps `weave_generator` and replaces each macroscopic yarn with N
// finer parallel fibres arranged in a deterministic hexagonal pack
// within the yarn's circular cross-section. The fibre-bundle
// representation matches the real-textile-composite "tow" concept
// where each yarn is a bundle of monofilaments rather than a single
// solid cylinder — which matters for FFT homogenization of fibre
// composites where the intra-tow stiffness is the dominant
// micromechanical parameter.
//
// Geometry:
//   * Each fibre is a `polyline_tube` with radius
//     `yarn_radius · fibre_radius_factor` (default 0.2 → 5×5 = 25
//     fibres pack visibly without overlap at most realistic
//     cross-section fill fractions).
//   * Fibres are laid in a hexagonal pattern in the yarn's
//     cross-section plane, then translated rigidly per centerline
//     control point — they all share the yarn's undulation. The
//     pattern is centred on the yarn axis.
//   * The cross-section plane is the yarn's primary perpendicular
//     plane: warp yarns offset in (y, z), weft in (x, z). The
//     undulation is in z; the in-plane offset is the lateral
//     spread.
//
// This is a deliberately-simple phase 1. Full Frenet-frame tow
// twisting (each fibre helically twisting around the tow axis) is
// out of scope; the parallel-transport frame would give a more
// physically faithful result.

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../shapes/polyline_tube.h"
#include "../shapes/shape_base.h"
#include "weave_generator.h"

namespace rvegen {

template <typename T = double>
class tow_weave_generator {
public:
  using value_type = T;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  // Construct from a base weave + the per-tow fibre count and
  // radius factor. `fibre_radius_factor` is the ratio of each
  // fibre's radius to the macroscopic yarn radius — small enough
  // (< ~0.3) that the hex-packed fibres fit in the yarn's
  // cross-section without overlap.
  tow_weave_generator(weave_generator<T> base,
                      std::size_t n_fibres_per_tow,
                      T fibre_radius_factor = T{0.2})
      : _base{std::move(base)},
        _n_fibres{n_fibres_per_tow},
        _fibre_radius_factor{fibre_radius_factor} {
    if (n_fibres_per_tow == 0) {
      throw std::runtime_error{
          "tow_weave_generator: n_fibres_per_tow must be ≥ 1"};
    }
    if (fibre_radius_factor <= T{0} || fibre_radius_factor >= T{1}) {
      throw std::runtime_error{
          "tow_weave_generator: fibre_radius_factor must lie in (0, 1)"};
    }
  }

  [[nodiscard]] std::size_t n_fibres_per_tow() const noexcept {
    return _n_fibres;
  }
  [[nodiscard]] T fibre_radius_factor() const noexcept {
    return _fibre_radius_factor;
  }

  // Build the tow-level RVE. Calls the base weave generator, then
  // expands every yarn into `_n_fibres` parallel fibres.
  [[nodiscard]] shape_vector build() const {
    auto yarns = _base.build();
    shape_vector out;
    out.reserve(yarns.size() * _n_fibres);

    const T fibre_r = _yarn_radius_of_base() * _fibre_radius_factor;
    const auto offsets = hex_pack_offsets(
        _n_fibres, _yarn_radius_of_base() - fibre_r);

    for (std::size_t y = 0; y < yarns.size(); ++y) {
      auto* yarn = dynamic_cast<polyline_tube<T>*>(yarns[y].get());
      if (!yarn) continue;   // base may produce empty / unsupported
      const bool is_warp = y < _base.n_warp_yarns();
      auto const& yarn_phase_name =
          is_warp ? _base.warp_phase_name() : _base.weft_phase_name();
      for (auto const& off : offsets) {
        std::vector<std::array<T, 3>> centerline;
        centerline.reserve(yarn->centerline().size());
        for (auto const& p : yarn->centerline()) {
          // Warp yarns (primary axis +x) → spread in (y, z).
          // Weft yarns (primary axis +y) → spread in (x, z).
          if (is_warp) {
            centerline.push_back({p[0], p[1] + off[0], p[2] + off[1]});
          } else {
            centerline.push_back({p[0] + off[0], p[1], p[2] + off[1]});
          }
        }
        auto fibre = std::make_unique<polyline_tube<T>>(centerline, fibre_r);
        if (!yarn_phase_name.empty()) fibre->set_phase_name(yarn_phase_name);
        out.push_back(std::move(fibre));
      }
    }
    return out;
  }

private:
  weave_generator<T> _base;
  std::size_t _n_fibres;
  T _fibre_radius_factor;

  // Recover the base weave's yarn radius. The base doesn't expose
  // it directly (it's stored internally), so we sniff one produced
  // yarn — cheap because the base's build() is deterministic at
  // small problem sizes, but slightly indirect. A future
  // weave_generator could expose a yarn_radius() accessor and we'd
  // skip this.
  [[nodiscard]] T _yarn_radius_of_base() const {
    auto probe = _base.build();
    if (probe.empty()) return T{0};
    auto* yarn = dynamic_cast<polyline_tube<T>*>(probe[0].get());
    return yarn ? yarn->radius() : T{0};
  }

  // Hex-packed 2D offsets in a disk of given enclosing radius.
  // Deterministic, centred on origin, scaled so all N points fit
  // inside the disk. For N = 1, returns the single offset {0, 0}.
  static std::vector<std::array<T, 2>> hex_pack_offsets(
      std::size_t n_points, T enclosing_radius) {
    std::vector<std::array<T, 2>> pts;
    if (n_points == 0) return pts;
    pts.push_back({T{0}, T{0}});
    if (n_points == 1) return pts;
    // Walk hexagonal rings outward until we have ≥ N points; trim.
    // Ring k (k ≥ 1) has 6·k points at radius k·d, where d is the
    // ring spacing. We don't know d until we know how many rings
    // we'll need — so build rings unit-spaced, then rescale at the
    // end so the outermost point sits on `enclosing_radius`.
    std::size_t ring = 1;
    while (pts.size() < n_points) {
      const T rk = static_cast<T>(ring);
      const std::size_t in_ring = 6 * ring;
      for (std::size_t i = 0; i < in_ring && pts.size() < n_points; ++i) {
        const T theta = static_cast<T>(2) * static_cast<T>(M_PI)
                      * static_cast<T>(i) / static_cast<T>(in_ring);
        pts.push_back({rk * std::cos(theta), rk * std::sin(theta)});
      }
      ++ring;
    }
    // Rescale so the outermost ring sits at enclosing_radius.
    const T outermost = static_cast<T>(ring - 1);
    if (outermost > T{0}) {
      const T scale = enclosing_radius / outermost;
      for (auto& p : pts) { p[0] *= scale; p[1] *= scale; }
    }
    return pts;
  }
};

} // namespace rvegen
