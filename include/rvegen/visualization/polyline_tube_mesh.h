#pragma once

// to_mesh(polyline_tube) — sweeps a constant-radius tube along the
// tube's centerline by generating a ring of vertices per centerline
// point and stitching consecutive rings.
//
// Why parallel-transport frame (not raw Frenet):
//   Frenet uses the principal normal `N = dT/ds` which is undefined
//   when the curve is straight (T is constant ⇒ dT/ds = 0). Polyline
//   tubes very commonly contain straight segments. Parallel transport
//   propagates the frame from one centerline point to the next by
//   rotating the previous normal about the rotation axis (T_prev ×
//   T_next). On a straight segment T doesn't change, the rotation is
//   identity, and the frame coasts through — no special case needed.
//
// What this header lands today (closes #26):
//   * `triangle_mesh<T> to_mesh(polyline_tube<T> const&,
//     std::size_t segments_per_ring = 16)`.
//   * End caps: triangle fan from the centerline's first/last vertex
//     to the corresponding ring. Closes the tube so volumetric
//     visualisers don't see-through the ends.
//   * Per-vertex outward radial normals for smooth shading.
//
// Out of scope here (separate concerns, tracked elsewhere):
//   * Precise `collision_details(polyline_tube, X)` overloads.
//   * `polyline_tube_input` JSON schema (PR #21 already shipped).
//   * `weave_generator` (#3 headline).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>

#include <Mathematics/Vector3.h>

#include "../shapes/polyline_tube.h"
#include "triangle_mesh.h"

namespace rvegen {

namespace detail {

// Returns a unit vector not parallel to `axis`. Used to seed the initial
// frame at the first centerline vertex. Picks the world-axis with the
// smallest |dot product| against `axis` — guarantees the cross product
// is well-conditioned (≥ √(2/3) magnitude before normalisation).
template <typename T>
[[nodiscard]] inline gte::Vector<3, T>
pick_aux_axis(gte::Vector<3, T> const& axis) {
  const T ax = std::abs(axis[0]);
  const T ay = std::abs(axis[1]);
  const T az = std::abs(axis[2]);
  gte::Vector<3, T> aux;
  aux[0] = T{0}; aux[1] = T{0}; aux[2] = T{0};
  if (ax <= ay && ax <= az)      aux[0] = T{1};
  else if (ay <= az)             aux[1] = T{1};
  else                           aux[2] = T{1};
  return aux;
}

template <typename T>
[[nodiscard]] inline T vec_len(gte::Vector<3, T> const& v) {
  return std::sqrt(gte::Dot(v, v));
}

// Rotate `v` to align `from` with `to` using Rodrigues. Both `from`
// and `to` are unit vectors. Parallel: `v` is returned unchanged.
// Anti-parallel (a 180-degree reversal): rotate by pi about an axis
// perpendicular to `from`, so the frame stays orthonormal (no zero/NaN).
template <typename T>
[[nodiscard]] inline gte::Vector<3, T>
rotate_to_align(gte::Vector<3, T> const& v,
                gte::Vector<3, T> const& from,
                gte::Vector<3, T> const& to) {
  const T cos_theta = gte::Dot(from, to);
  if (cos_theta > T{1} - T{1e-9}) return v;
  if (cos_theta < T{-1} + T{1e-9}) {
    auto a = gte::Cross(from, pick_aux_axis(from));
    a /= vec_len(a);
    const T d = gte::Dot(a, v);
    gte::Vector<3, T> r;
    for (int k = 0; k < 3; ++k) r[k] = T{2} * d * a[k] - v[k];
    return r;
  }
  const auto axis_raw = gte::Cross(from, to);
  const T sin_theta = vec_len(axis_raw);
  gte::Vector<3, T> axis = axis_raw;
  axis /= sin_theta;
  // Rodrigues: v_rot = v cos θ + (axis × v) sin θ + axis (axis · v)(1 − cos θ).
  const auto cross = gte::Cross(axis, v);
  const T dot = gte::Dot(axis, v);
  gte::Vector<3, T> result;
  for (int k = 0; k < 3; ++k) {
    result[k] = v[k] * cos_theta + cross[k] * sin_theta +
                axis[k] * dot * (T{1} - cos_theta);
  }
  return result;
}

} // namespace detail

template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(polyline_tube<T> const& tube,
        std::size_t segments_per_ring = 16) {
  triangle_mesh<T> m;
  auto const& centerline = tube.centerline();
  if (centerline.size() < 2 || segments_per_ring < 3) return m;

  const T radius = tube.radius();
  const std::size_t N = segments_per_ring;

  // Drop consecutive centreline points closer than a relative epsilon
  // (zero-length segments have no tangent and would collapse the tube).
  std::vector<gte::Vector<3, T>> pts;
  pts.reserve(centerline.size());
  {
    gte::Vector<3, T> lo = centerline[0], hi = centerline[0];
    for (auto const& p : centerline)
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], p[k]);
        hi[k] = std::max(hi[k], p[k]);
      }
    const T eps = T{1e-9} * std::max(detail::vec_len(hi - lo), radius);
    for (auto const& p : centerline) {
      if (pts.empty() || detail::vec_len(p - pts.back()) > eps) pts.push_back(p);
    }
  }
  // Fewer than two distinct points: nothing to sweep; the caller reports
  // the empty mesh as "shape not rendered".
  if (pts.size() < 2) return m;
  const std::size_t M = pts.size();

  // Pre-compute per-segment tangents (M-1 of them). At each centerline
  // point we use the forward edge's tangent; at the last point we
  // reuse the previous edge.
  std::vector<gte::Vector<3, T>> tangents(M);
  for (std::size_t i = 0; i + 1 < M; ++i) {
    auto d = pts[i + 1] - pts[i];
    d /= detail::vec_len(d);
    tangents[i] = d;
  }
  tangents[M - 1] = tangents[M - 2];

  // Initial frame at vertex 0: normal = tangent x aux (normalised),
  // binormal = tangent x normal (right-handed).
  gte::Vector<3, T> normal;
  {
    const auto aux = detail::pick_aux_axis(tangents[0]);
    auto n_raw = gte::Cross(tangents[0], aux);
    const T nlen = detail::vec_len(n_raw);
    if (nlen > T{0}) n_raw /= nlen;
    normal = n_raw;
  }
  auto binormal = gte::Cross(tangents[0], normal);

  // Reserve: M rings x N vertices, plus 2 cap-centre vertices.
  m.verts.reserve(M * N + 2);
  m.normals.reserve(M * N + 2);
  m.tris.reserve((M - 1) * 2 * N + 2 * N);   // side quads + 2 end caps

  // Largest mitre scale 1/cos(theta/2); bounds the ring for very sharp bends.
  constexpr T max_mitre = T{4};

  // Ring generation + frame parallel transport.
  constexpr T two_pi = T{2} * std::numbers::pi_v<T>;
  for (std::size_t i = 0; i < M; ++i) {
    // Interior joint: mitre. The ring lies in the plane perpendicular to
    // the bisector of the incoming/outgoing tangents, with the in-bend-plane
    // component of each radial scaled by 1/cos(theta/2), so it sits on both
    // adjacent cylinders. End rings are plain (perpendicular to the tangent).
    gte::Vector<3, T> ring_n = normal, ring_b = binormal;  // ring frame
    gte::Vector<3, T> bend_dir;                            // in-plane, scaled
    T mitre = T{1};
    gte::Vector<3, T> next_normal = normal, next_binormal = binormal;
    if (i > 0) {
      auto const& t_in  = tangents[i - 1];
      auto const& t_out = tangents[i];
      if (i + 1 < M) {
        auto bis = t_in + t_out;
        const T blen = detail::vec_len(bis);
        if (blen > T{1e-9}) {
          bis /= blen;
          ring_n = detail::rotate_to_align(normal,   t_in, bis);
          ring_b = detail::rotate_to_align(binormal, t_in, bis);
          const T cos_half = std::max(gte::Dot(t_in, bis), T{1} / max_mitre);
          auto diff = t_out - t_in;
          const T dlen = detail::vec_len(diff);
          if (dlen > T{1e-12}) {
            bend_dir = diff / dlen;
            mitre = T{1} / cos_half;
          }
          next_normal   = detail::rotate_to_align(ring_n, bis, t_out);
          next_binormal = detail::rotate_to_align(ring_b, bis, t_out);
        } else {  // 180-degree reversal: plain ring, frame flips by pi
          next_normal   = detail::rotate_to_align(normal,   t_in, t_out);
          next_binormal = detail::rotate_to_align(binormal, t_in, t_out);
        }
      } else {
        ring_n = next_normal   = detail::rotate_to_align(normal,   t_in, t_out);
        ring_b = next_binormal = detail::rotate_to_align(binormal, t_in, t_out);
      }
    }
    auto const& c = pts[i];
    for (std::size_t k = 0; k < N; ++k) {
      const T theta = (static_cast<T>(k) / static_cast<T>(N)) * two_pi;
      const T cs = std::cos(theta);
      const T sn = std::sin(theta);
      // radial direction at this ring vertex — already unit length.
      const std::array<T, 3> radial{
          cs * ring_n[0] + sn * ring_b[0],
          cs * ring_n[1] + sn * ring_b[1],
          cs * ring_n[2] + sn * ring_b[2]};
      std::array<T, 3> off = radial;
      if (mitre != T{1}) {
        const T along = (mitre - T{1}) *
            (radial[0] * bend_dir[0] + radial[1] * bend_dir[1] +
             radial[2] * bend_dir[2]);
        for (int q = 0; q < 3; ++q) off[q] += along * bend_dir[q];
      }
      m.verts.push_back({c[0] + radius * off[0],
                         c[1] + radius * off[1],
                         c[2] + radius * off[2]});
      m.normals.push_back(radial);
    }
    normal = next_normal;
    binormal = next_binormal;
  }

  // Side quads: between ring i (offset i*N) and ring i+1 (offset (i+1)*N).
  for (std::size_t i = 0; i + 1 < M; ++i) {
    const std::size_t base_a = i * N;
    const std::size_t base_b = (i + 1) * N;
    for (std::size_t k = 0; k < N; ++k) {
      const std::size_t k1 = (k + 1) % N;
      // Two triangles per quad, CCW seen from outside (ring runs CCW
      // about the tangent, so (a_k, a_k1, b_k) has an outward normal):
      // (a_k, a_k1, b_k) and (b_k, a_k1, b_k1).
      m.tris.push_back({base_a + k,  base_a + k1, base_b + k});
      m.tris.push_back({base_b + k,  base_a + k1, base_b + k1});
    }
  }

  // End caps: fan from a centre vertex at centerline[0] / centerline[M−1].
  const std::size_t cap_start = m.verts.size();
  {
    auto const& c0 = pts[0];
    m.verts.push_back({c0[0], c0[1], c0[2]});
    m.normals.push_back({-tangents[0][0], -tangents[0][1], -tangents[0][2]});
    for (std::size_t k = 0; k < N; ++k) {
      const std::size_t k1 = (k + 1) % N;
      // Inward winding so the cap normal points along -tangent (outward
      // from the tube along the first ring).
      m.tris.push_back({cap_start, k1, k});
    }
  }
  const std::size_t cap_end = m.verts.size();
  {
    auto const& cN = pts[M - 1];
    m.verts.push_back({cN[0], cN[1], cN[2]});
    m.normals.push_back({tangents[M - 1][0], tangents[M - 1][1], tangents[M - 1][2]});
    const std::size_t base = (M - 1) * N;
    for (std::size_t k = 0; k < N; ++k) {
      const std::size_t k1 = (k + 1) % N;
      m.tris.push_back({cap_end, base + k, base + k1});
    }
  }

  return m;
}

} // namespace rvegen
