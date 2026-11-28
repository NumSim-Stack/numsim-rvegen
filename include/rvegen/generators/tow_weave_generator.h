#pragma once

// Tow-level weave generator (#109 follow-up to #57).
//
// Wraps `weave_generator` and replaces each macroscopic yarn with N
// finer parallel fibres arranged in a deterministic hexagonal pack
// within the yarn's circular cross-section. The fibre-bundle
// representation matches the real-textile-composite "tow" concept
// where each yarn is a bundle of monofilaments rather than a single
// solid cylinder.
//
// Geometry:
//   * Each fibre is a `polyline_tube` with radius
//     r_f = `yarn_radius · fibre_radius_factor`.
//   * Fibres are laid in rings around the yarn axis: one at the
//     centre, then ring k (k = 1..K) with up to 6k fibres at radius
//     k·d, d = (yarn_radius − r_f)/K, so the outermost fibres touch
//     the yarn surface from inside. A partially filled outer ring is
//     spread evenly around the full circle; if it would hold a single
//     fibre (n = 2, 8, 20, ...) the centre is left empty and the outer
//     ring gets two opposite fibres, so the bundle stays centred.
//   * Non-overlap: the closest fibre pairs are radial neighbours (gap
//     d) and adjacent fibres on ring 1 (chord 2d·sin 30° = d); every
//     other in-ring chord 2kd·sin(π/6k) is ≥ d. So fibres are disjoint
//     iff d ≥ 2r_f, i.e. iff fibre_radius_factor ≤ 1/(2K+1). The ctor
//     THROWS when that fails rather than shrinking the fibres: r_f
//     sets the intra-tow fibre volume fraction, the parameter this
//     generator exists to control, so it must not change silently.
//     The default factor 0.2 fits up to 7 fibres with clearance (K = 1,
//     bound 1/3) and up to 19 touching (K = 2, bound 1/5); 20–37
//     fibres (K = 3) need ≤ 1/7, e.g. 25 fibres at 0.125.
//   * The ring pattern is placed in the yarn's local normal plane at
//     each centerline control point, spanned by the lateral axis
//     (y for warp, x for weft) and the in-plane normal of the crimp
//     (≈ z). Following the crimp's normal rather than translating
//     rigidly in z keeps fibres d apart on the sloped flanks too. The
//     offsets are mitred at the nodes so every fibre segment is a
//     translate of the yarn's own segment: clearance is d along the whole
//     polyline (not just at the nodes) and the fibres stay inside the
//     yarn's polyline envelope.
//
// This is a deliberately-simple phase 1. Full Frenet-frame tow
// twisting (each fibre helically twisting around the tow axis) is
// out of scope; the parallel-transport frame would give a more
// physically faithful result.

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

#include <Mathematics/Vector3.h>

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
  // fibre's radius to the macroscopic yarn radius. Throws
  // std::runtime_error unless it is ≤ 1/(2K+1), K the number of rings
  // needed for `n_fibres_per_tow` (see the header comment).
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
    if (!(fibre_radius_factor > T{0}) || !(fibre_radius_factor < T{1})) {
      throw std::runtime_error{
          "tow_weave_generator: fibre_radius_factor must lie in (0, 1)"};
    }
    const std::size_t K = rings_for(n_fibres_per_tow);
    const T max_factor = T{1} / static_cast<T>(2 * K + 1);
    const T tol = T{1} + T{64} * std::numeric_limits<T>::epsilon();
    if (K > 0 && fibre_radius_factor > max_factor * tol) {
      throw std::runtime_error{
          "tow_weave_generator: fibre_radius_factor too large for "
          + std::to_string(n_fibres_per_tow) + " fibres per tow ("
          + std::to_string(K) + " rings) — fibres would overlap; it must "
          "be ≤ 1/(2·rings + 1) = " + std::to_string(max_factor)};
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
  [[nodiscard]] shape_vector build(cancel_fn const* cancel = nullptr) const {
    auto yarns = _base.build(cancel);
    shape_vector out;
    out.reserve(yarns.size() * _n_fibres);

    // Yarn radius is sniffed once from the first produced yarn — the
    // base generator uses a single radius across warp + weft, so
    // looking at any one yarn is sufficient. Avoids calling
    // _base.build() a second time.
    T yarn_radius{0};
    if (!yarns.empty()) {
      if (auto* tube0 = dynamic_cast<polyline_tube<T>*>(yarns[0].get())) {
        yarn_radius = tube0->radius();
      }
    }
    const T fibre_r = yarn_radius * _fibre_radius_factor;
    const auto offsets = hex_pack_offsets(
        _n_fibres, yarn_radius - fibre_r);

    for (std::size_t y = 0; y < yarns.size(); ++y) {
      poll_cancel(cancel);
      auto* yarn = dynamic_cast<polyline_tube<T>*>(yarns[y].get());
      if (!yarn) continue;   // base may produce empty / unsupported
      const bool is_warp = y < _base.n_warp_yarns();
      auto const& yarn_phase_name =
          is_warp ? _base.warp_phase_name() : _base.weft_phase_name();
      // Mitred offset frame. N_j is the in-plane unit normal of segment j
      // (perpendicular to the lateral axis and the segment tangent). A
      // fibre displaced by `off[1]` along the normal must lie on the
      // line parallel to each adjacent segment at distance off[1], so at
      // a node the displacement is the mitre vector
      //   m = (N_{j-1} + N_j) / (1 + N_{j-1}·N_j),   m·N_{j-1} = m·N_j = 1,
      // and at the ends simply N_0 / N_{last}. Every fibre segment is then
      // a rigid translation of the yarn's own segment, so two fibres stay
      // exactly their ring-offset apart along the whole segment, not only
      // at the nodes (a per-node normal alone shrinks the gap by
      // ~cos(theta/2) between nodes).
      auto const& cl = yarn->centerline();
      const gte::Vector3<T> lateral = is_warp
          ? gte::Vector3<T>{T{0}, T{1}, T{0}}
          : gte::Vector3<T>{T{1}, T{0}, T{0}};
      const std::size_t nseg = cl.size() > 0 ? cl.size() - 1 : 0;
      std::vector<gte::Vector3<T>> seg_normal;
      seg_normal.reserve(nseg);
      for (std::size_t j = 0; j < nseg; ++j) {
        const gte::Vector3<T> tangent = cl[j + 1] - cl[j];
        auto n = is_warp ? gte::UnitCross(tangent, lateral)
                         : gte::UnitCross(lateral, tangent);
        if (gte::Length(n) == T{0} && !seg_normal.empty()) {
          n = seg_normal.back();   // zero-length segment
        }
        seg_normal.push_back(n);
      }
      std::vector<gte::Vector3<T>> normals;   // mitre vector per node
      normals.reserve(cl.size());
      for (std::size_t k = 0; k < cl.size(); ++k) {
        if (nseg == 0) {
          normals.push_back(gte::Vector3<T>{T{0}, T{0}, T{1}});
        } else if (k == 0) {
          normals.push_back(seg_normal.front());
        } else if (k == nseg) {
          normals.push_back(seg_normal.back());
        } else {
          auto const& n0 = seg_normal[k - 1];
          auto const& n1 = seg_normal[k];
          const T denom = std::max(T{1} + gte::Dot(n0, n1), T{1e-3});
          normals.push_back((n0 + n1) / denom);
        }
      }
      for (auto const& off : offsets) {
        std::vector<gte::Vector3<T>> centerline;
        centerline.reserve(cl.size());
        for (std::size_t k = 0; k < cl.size(); ++k) {
          centerline.push_back(cl[k] + off[0] * lateral + off[1] * normals[k]);
        }
        auto fibre = std::make_unique<polyline_tube<T>>(centerline, fibre_r);
        fibre->make_bounding_box();
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

  // Number of rings K needed for n points: centre + 6k per ring k,
  // i.e. the smallest K with 1 + 3K(K+1) ≥ n.
  static std::size_t rings_for(std::size_t n_points) noexcept {
    std::size_t K = 0;
    while (1 + 3 * K * (K + 1) < n_points) ++K;
    return K;
  }

  // Ring-packed 2D offsets in a disk of given enclosing radius.
  // Deterministic, centred on origin (centroid exactly 0), scaled so
  // all N points fit inside the disk. For N = 1, returns the single
  // offset {0, 0}.
  static std::vector<std::array<T, 2>> hex_pack_offsets(
      std::size_t n_points, T enclosing_radius) {
    using std::numbers::pi_v;
    std::vector<std::array<T, 2>> pts;
    if (n_points == 0) return pts;
    if (n_points == 1) return {{T{0}, T{0}}};
    // A last ring holding a single point would pull the centroid off
    // the yarn axis. In that case (n = 2, 8, 20, ...) leave the centre
    // empty and put two diametrically opposite points on the last
    // ring instead; spacing only grows, so non-overlap is unaffected.
    const std::size_t K = rings_for(n_points);
    const bool skip_centre = n_points - (1 + 3 * (K - 1) * K) == 1;
    if (!skip_centre) pts.push_back({T{0}, T{0}});
    // Walk rings outward until we have N points. Ring k (k ≥ 1) has
    // 6·k points at radius k·d, where d is the ring spacing; a
    // partially filled last ring spreads its points evenly over the
    // full circle. We don't know d until we know how many rings
    // we'll need — so build rings unit-spaced, then rescale at the
    // end so the outermost point sits on `enclosing_radius`.
    std::size_t ring = 1;
    while (pts.size() < n_points) {
      const T rk = static_cast<T>(ring);
      const std::size_t in_ring = std::min(6 * ring, n_points - pts.size());
      for (std::size_t i = 0; i < in_ring; ++i) {
        const T theta = T{2} * pi_v<T>
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
