#pragma once

// 2D bounded Voronoi tessellation by seed points (#114 phase 1).
//
// Given a rectangular domain and N seed points in it, produces N
// convex polygons — one per seed — that partition the domain. Each
// cell is the set of points closer to its seed than to any other.
// The standard textbook Voronoi diagram, clipped to a bounding box.
//
// Why not use Voro++ / CGAL?
//   For 2D Voronoi in a rectangle the algorithm is well-bounded:
//   Sutherland-Hodgman half-plane clipping. Each cell starts as
//   the bounding rectangle and is clipped against the perpendicular
//   bisector of the seed against every other seed (keeping the
//   "closer-to-our-seed" side). ~80 lines, no external dependency,
//   robust to all non-degenerate inputs.
//
//   The 3D counterpart lives in voronoi_generator_3d.h.
//
// Periodic mode (#8): `periodic = true` tessellates the rectangle as a
// 2-torus. Each cell is the Voronoi cell of its seed against all 3x3 image
// seeds (shifted by -L, 0, +L per axis), starting from the rectangle of
// size (Lx, Ly) centred on the seed and with NO clipping to the domain. A
// cell crossing a box edge is returned as ONE whole convex polygon that
// sticks out of [0,Lx]x[0,Ly] (the same convention as periodic inclusions
// with wraps); areas still sum to Lx*Ly and opposite edges match up to a
// translation by L.
//
// Weights (#8, Laguerre / power diagram): optional per-seed weights w_i
// [length^2] replace |x-p_i|^2 by |x-p_i|^2 - w_i. A weighted cell may be
// empty (< 3 vertices) and need not contain its seed.
//
// Output type:
//   `build()` returns `std::vector<polygon_type>` where
//   `polygon_type = std::vector<std::array<T, 2>>` — one CCW vertex
//   list per cell, in seed-order (cell `i` corresponds to seed `i`).
//   Empty polygons (numerically collapsed cells) keep their slot in
//   the output so index == seed index always holds. Wrapping
//   the polygons into a shape_base subclass (a future 2D
//   `convex_polygon` shape, or extruded 3D voronoi_cell prisms) is
//   deliberately separate — this header is the algorithm primitive.
//
// Numerical considerations:
//   * `T` should be a floating-point type. Integer / fixed-point
//     specialisations would need an exact-arithmetic kernel.
//   * Scale-invariant (#79): bisectors are normalised to unit normals,
//     so the clip test is a signed distance and every tolerance is
//     relative to the domain size. Works unchanged from 1e-6 to 1e3.
//   * Seeds on the bounding-box edge are valid (their cells stay
//     non-degenerate).
//   * Coincident seeds (closer than 1e-9 of the domain size) and seeds
//     outside the domain are rejected with std::runtime_error.
//   * Edge-on-bisector intersections are handled by the half-plane
//     test `dist ≤ 0`: a point exactly on the bisector lands on BOTH
//     sides' boundary and is kept in both cells.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "voronoi_seed_validation.h"

namespace rvegen {

template <typename T = double>
class voronoi_generator_2d {
public:
  using value_type = T;
  using point_type = std::array<T, 2>;
  using polygon_type = std::vector<point_type>;

  // Domain is the rectangle `[0, Lx] × [0, Ly]`. Seeds must lie in
  // the closed domain (boundary seeds are fine) and must be pairwise
  // distinct; violations throw std::runtime_error.
  voronoi_generator_2d(T Lx, T Ly, std::vector<point_type> seeds,
                       bool periodic = false, std::vector<T> weights = {})
      : _Lx{Lx}, _Ly{Ly}, _seeds{std::move(seeds)}, _periodic{periodic},
        _weights{std::move(weights)} {
    if (!_weights.empty() && _weights.size() != _seeds.size()) {
      throw std::runtime_error{
          "voronoi_generator_2d: weights must be empty or one per seed"};
    }
    for (auto const& [name, v] :
         std::array<std::pair<char const*, T>, 2>{{{"Lx", Lx}, {"Ly", Ly}}}) {
      if (!std::isfinite(v) || !(v > T{0})) {
        throw std::invalid_argument{
            std::string{"voronoi_generator_2d: "} + name +
            " must be finite and positive (got " + std::to_string(v) + ")"};
      }
    }
    if (_seeds.empty()) {
      throw std::runtime_error{
          "voronoi_generator_2d: at least one seed required"};
    }
    for (std::size_t i = 0; i < _seeds.size(); ++i) {
      if (!std::isfinite(_seeds[i][0]) || !std::isfinite(_seeds[i][1])) {
        throw std::invalid_argument{
            "voronoi_generator_2d: seeds[" + std::to_string(i) +
            "] has a non-finite coordinate"};
      }
    }
    for (std::size_t i = 0; i < _weights.size(); ++i) {
      if (!std::isfinite(_weights[i])) {
        throw std::invalid_argument{"voronoi_generator_2d: weights[" +
                                    std::to_string(i) + "] is not finite"};
      }
    }
    detail::validate_voronoi_seeds<T, 2>({Lx, Ly}, _seeds,
                                         "voronoi_generator_2d", periodic);
  }

  [[nodiscard]] bool periodic() const noexcept { return _periodic; }

  [[nodiscard]] std::array<T, 3> box() const noexcept {
    return {_Lx, _Ly, T{0}};
  }

  [[nodiscard]] std::size_t seed_count() const noexcept {
    return _seeds.size();
  }

  // Build all cells. Cell `i` corresponds to seed `i`. CCW vertex
  // order in each cell. For one seed, the cell is the full bounding
  // rectangle.
  [[nodiscard]] std::vector<polygon_type> build() const {
    std::vector<polygon_type> out;
    out.reserve(_seeds.size());
    polygon_type bbox = make_bounding_rectangle();
    const int range = _periodic ? 1 : 0;
    const T merge_tol =
        std::numeric_limits<T>::epsilon() * T{16} * std::min(_Lx, _Ly);

    for (std::size_t i = 0; i < _seeds.size(); ++i) {
      polygon_type cell = bbox;
      auto const& p_i = _seeds[i];
      if (_periodic) {   // rectangle of one period centred on the seed
        cell = {{p_i[0] - _Lx / 2, p_i[1] - _Ly / 2},
                {p_i[0] + _Lx / 2, p_i[1] - _Ly / 2},
                {p_i[0] + _Lx / 2, p_i[1] + _Ly / 2},
                {p_i[0] - _Lx / 2, p_i[1] + _Ly / 2}};
      }
      const T w_i = _weights.empty() ? T{0} : _weights[i];
      for (std::size_t j = 0; j < _seeds.size() && cell.size() >= 3; ++j)
      for (int sx = -range; sx <= range && cell.size() >= 3; ++sx)
      for (int sy = -range; sy <= range && cell.size() >= 3; ++sy) {
        if (i == j && sx == 0 && sy == 0) continue;
        const point_type p_j{_seeds[j][0] + sx * _Lx, _seeds[j][1] + sy * _Ly};
        const T w_j = _weights.empty() ? T{0} : _weights[j];
        // Perpendicular-bisector half-plane, unit normal n toward p_j:
        //   n · x ≤ n · (p_i + p_j) / 2
        const T dx = p_j[0] - p_i[0];
        const T dy = p_j[1] - p_i[1];
        const T len = std::sqrt(dx * dx + dy * dy);
        const T A = dx / len;
        const T B = dy / len;
        const T C = A * (p_i[0] + p_j[0]) * T{0.5} +
                    B * (p_i[1] + p_j[1]) * T{0.5} -
                    (w_j - w_i) / (T{2} * len);   // power-diagram shift
        cell = clip_polygon(cell, A, B, C, merge_tol);
      }
      out.push_back(std::move(cell));
    }
    return out;
  }

private:
  T _Lx;
  T _Ly;
  std::vector<point_type> _seeds;
  bool _periodic;
  std::vector<T> _weights;   // empty = plain Voronoi

  // Bounding rectangle vertices in CCW order, starting bottom-left.
  [[nodiscard]] polygon_type make_bounding_rectangle() const {
    return {{T{0}, T{0}},
            {_Lx,  T{0}},
            {_Lx,  _Ly },
            {T{0}, _Ly }};
  }

  // Sutherland-Hodgman clipping by the half-plane `A·x + B·y ≤ C`
  // with (A, B) a UNIT normal, so `A·x + B·y - C` is a signed
  // distance. Walks the polygon edges and for each edge keeps:
  //   * The destination vertex if it's inside.
  //   * The intersection point if the edge crosses the boundary.
  // Boundary points count as inside (≤, not <). Consecutive vertices
  // closer than `merge_tol` are merged so on-bisector vertices are not
  // duplicated.
  static polygon_type clip_polygon(polygon_type const& polygon,
                                    T A, T B, T C, T merge_tol) {
    polygon_type out;
    out.reserve(polygon.size() + 1);
    const auto n = polygon.size();
    if (n == 0) return out;

    auto dist = [A, B, C](point_type const& p) {
      return A * p[0] + B * p[1] - C;
    };
    auto push = [&out, merge_tol](point_type const& p) {
      if (!out.empty() &&
          std::abs(out.back()[0] - p[0]) <= merge_tol &&
          std::abs(out.back()[1] - p[1]) <= merge_tol) {
        return;
      }
      out.push_back(p);
    };

    for (std::size_t k = 0; k < n; ++k) {
      auto const& start = polygon[k];
      auto const& end   = polygon[(k + 1) % n];
      const T ds = dist(start);
      const T de = dist(end);
      const bool start_in = ds <= T{0};
      const bool end_in   = de <= T{0};
      if (start_in != end_in) {
        // Endpoints are on strictly opposite sides (or one exactly on
        // the line), so ds - de != 0 and t lies in [0, 1].
        const T t = ds / (ds - de);
        push(point_type{start[0] + t * (end[0] - start[0]),
                        start[1] + t * (end[1] - start[1])});
      }
      if (end_in) push(end);
      // else (both outside): drop both.
    }
    if (out.size() > 1 &&
        std::abs(out.front()[0] - out.back()[0]) <= merge_tol &&
        std::abs(out.front()[1] - out.back()[1]) <= merge_tol) {
      out.pop_back();
    }
    return out;
  }
};

// Convenience helper: shoelace formula for the signed area of a
// CCW polygon (returns positive area for a correctly-wound cell).
// Exposed so tests and callers can verify the area-covers-domain
// invariant without redoing the math.
template <typename T>
[[nodiscard]] T polygon_area(std::vector<std::array<T, 2>> const& poly) {
  const auto n = poly.size();
  if (n < 3) return T{0};
  T sum{0};
  for (std::size_t k = 0; k < n; ++k) {
    auto const& a = poly[k];
    auto const& b = poly[(k + 1) % n];
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return T{0.5} * std::abs(sum);
}

} // namespace rvegen

// One-line bridge from voronoi_generator_2d into the rvegen shape
// pipeline: build the cells, wrap each as a convex_polygon, return
// the shape_vector that the standard writers (voxel, gmsh, svg)
// consume. The first seed_count() entries are the cells in seed order (#79):
// a numerically collapsed (< 3 vertex) cell becomes an empty
// convex_polygon placeholder (area 0, contains nothing) so downstream
// per-grain data indexed by seed stays aligned.
//
// For a periodic generator a cell that sticks out of the box is built whole,
// so its wrapped pieces (translated copies, generate_periodic_wraps) are
// appended after the primaries: without them part of the box has no owner
// (about 27 % of the voxels in a typical 8-seed case). Every piece carries
// info "grain_id" = its seed index, so all pieces of one grain share the
// grain's id (sample_voxel_grid honours it).
//
// Lives in its own header so callers can pull in the helper without
// dragging in shape_base / convex_polygon transitively when they
// only want the raw polygon output.

#include "../shapes/convex_polygon.h"
#include "../shapes/periodic_wraps.h"
#include "../shapes/shape_base.h"

namespace rvegen {

template <typename T = double>
[[nodiscard]] inline std::vector<std::unique_ptr<shape_base<T>>>
voronoi_to_shapes(voronoi_generator_2d<T> const& gen) {
  auto cells = gen.build();
  std::vector<std::unique_ptr<shape_base<T>>> out;
  out.reserve(cells.size());
  for (std::size_t i = 0; i < cells.size(); ++i) {
    auto& v = cells[i];
    if (v.size() < 3) {
      out.push_back(std::make_unique<convex_polygon<T>>());   // placeholder
    } else {
      out.push_back(std::make_unique<convex_polygon<T>>(std::move(v)));
    }
    out.back()->info().set("grain_id", i);
  }
  if (gen.periodic()) {
    const std::size_t n = out.size();
    for (std::size_t i = 0; i < n; ++i) {
      auto wraps = generate_periodic_wraps<T>(*out[i], gen.box());
      for (auto& w : wraps) out.push_back(std::move(w));
    }
  }
  return out;
}

} // namespace rvegen
