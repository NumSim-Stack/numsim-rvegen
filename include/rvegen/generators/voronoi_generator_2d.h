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

#include "cancel_poll.h"
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
  //
  // A cell is its start rectangle clipped by the bisector of every other
  // seed (and, periodic, of its eight images), in the order of the seed
  // list. Doing exactly that costs the square of the number of seeds
  // (#154), although nearly all of those clips cut nothing: after the first
  // few seeds the cell is small, and only seeds near it can cut it. So the
  // clips that cannot cut are not computed:
  //   * the first `direct_seeds` seeds are clipped one by one;
  //   * the remaining seeds are taken in index ranges that double in size
  //     ("levels"). Before a level, the region in which a seed could still
  //     cut the cell is computed (seeds_that_may_cut); the seeds of the
  //     level in that region come from a grid over the level, are sorted
  //     back into list order and clipped; the others are skipped.
  // A clip that cuts nothing still rotates the vertex list by one and
  // merges vertices closer than the tolerance, so the skipped clips are
  // applied as that (skip_clips). The cells are therefore the ones the full
  // sequence produces, vertex for vertex and bit for bit, with the same
  // start vertex.
  [[nodiscard]] std::vector<polygon_type> build(
      cancel_fn const* cancel = nullptr) const {
    std::vector<polygon_type> out;
    const std::size_t n = _seeds.size();
    out.reserve(n);
    polygon_type bbox = make_bounding_rectangle();
    const std::size_t images = _periodic ? 9 : 1;
    const T merge_tol =
        std::numeric_limits<T>::epsilon() * T{16} * std::min(_Lx, _Ly);
    const auto levels = make_levels();
    std::vector<std::size_t> candidates;
    polygon_type scratch;
    std::vector<disc> discs;

    for (std::size_t i = 0; i < n; ++i) {
      poll_cancel(cancel);
      polygon_type cell = bbox;
      auto const& p_i = _seeds[i];
      if (_periodic) {   // rectangle of one period centred on the seed
        cell = {{p_i[0] - _Lx / 2, p_i[1] - _Ly / 2},
                {p_i[0] + _Lx / 2, p_i[1] - _Ly / 2},
                {p_i[0] + _Lx / 2, p_i[1] + _Ly / 2},
                {p_i[0] - _Lx / 2, p_i[1] + _Ly / 2}};
      }
      const T w_i = _weights.empty() ? T{0} : _weights[i];
      // Position of a clip in the full sequence: seed index times the
      // number of images plus the image ((sx + 1) * 3 + (sy + 1), 0 when not
      // periodic). The seed itself (image 0, 0) is not a clip.
      const std::size_t self = i * images + (_periodic ? 4 : 0);
      std::size_t done = 0;   // clips of the full sequence dealt with
      auto clip_at = [&](std::size_t position) {
        if (position == self || cell.size() < 3) return;
        const std::size_t index = position - (position > self ? 1 : 0);
        skip_clips(cell, index - done, merge_tol, scratch);
        done = index;
        if (cell.size() < 3) return;
        const std::size_t j = position / images;
        const int image = static_cast<int>(position % images);
        const int sx = _periodic ? image / 3 - 1 : 0;
        const int sy = _periodic ? image % 3 - 1 : 0;
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
        clip_polygon(cell, A, B, C, merge_tol, scratch);
        cell.swap(scratch);
        ++done;
      };

      const std::size_t direct = std::min(n, direct_seeds) * images;
      for (std::size_t position = 0; position < direct && cell.size() >= 3;
           ++position) {
        clip_at(position);
      }
      for (auto const& level : levels) {
        if (cell.size() < 3) break;
        candidates.clear();
        seeds_that_may_cut(level, cell, p_i, w_i, discs, candidates);
        std::sort(candidates.begin(), candidates.end());
        for (auto position : candidates) {
          if (cell.size() < 3) break;
          clip_at(position);
        }
      }
      if (cell.size() >= 3) {
        skip_clips(cell, n * images - 1 - done, merge_tol, scratch);
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

  // Seeds clipped one by one before the levels start.
  static constexpr std::size_t direct_seeds = 16;

  // The seeds with index in [first, last), binned on a grid over the box
  // (cell of a seed: its coordinates divided by the cell size, rounded
  // down, the last cell also holding seeds on the far edge).
  struct level {
    std::size_t first{0};
    std::size_t last{0};
    std::size_t nx{1};
    std::size_t ny{1};
    T hx{1};
    T hy{1};
    T max_weight{0};
    std::vector<std::size_t> start;   // per grid cell, into `seeds`
    std::vector<std::size_t> seeds;   // seed indices, grouped by grid cell
  };

  [[nodiscard]] static std::size_t grid_index(T x, T h, std::size_t n) {
    const T q = std::floor(x / h);
    if (!(q > T{0})) return 0;
    return std::min(static_cast<std::size_t>(q), n - 1);
  }

  [[nodiscard]] std::vector<level> make_levels() const {
    std::vector<level> levels;
    const std::size_t n = _seeds.size();
    for (std::size_t first = direct_seeds; first < n; first *= 2) {
      level lv;
      lv.first = first;
      lv.last = std::min(n, 2 * first);
      const T count = static_cast<T>(lv.last - lv.first);
      // About two seeds per grid cell, cells about square.
      const T cells = std::max(T{1}, count / T{2});
      lv.nx = static_cast<std::size_t>(std::clamp(
          std::round(std::sqrt(cells * _Lx / _Ly)), T{1}, T{4096}));
      lv.ny = static_cast<std::size_t>(std::clamp(
          std::round(std::sqrt(cells * _Ly / _Lx)), T{1}, T{4096}));
      lv.hx = _Lx / static_cast<T>(lv.nx);
      lv.hy = _Ly / static_cast<T>(lv.ny);
      auto cell_of = [&](std::size_t j) {
        return grid_index(_seeds[j][1], lv.hy, lv.ny) * lv.nx +
               grid_index(_seeds[j][0], lv.hx, lv.nx);
      };
      lv.start.assign(lv.nx * lv.ny + 1, 0);
      for (std::size_t j = lv.first; j < lv.last; ++j) ++lv.start[cell_of(j) + 1];
      for (std::size_t c = 0; c + 1 < lv.start.size(); ++c)
        lv.start[c + 1] += lv.start[c];
      lv.seeds.resize(lv.last - lv.first);
      auto next = lv.start;
      for (std::size_t j = lv.first; j < lv.last; ++j)
        lv.seeds[next[cell_of(j)]++] = j;
      if (!_weights.empty()) {
        lv.max_weight = *std::max_element(
            _weights.begin() + static_cast<std::ptrdiff_t>(lv.first),
            _weights.begin() + static_cast<std::ptrdiff_t>(lv.last));
      }
      levels.push_back(std::move(lv));
    }
    return levels;
  }

  // Appends the positions (seed index * images + image) of all seeds of
  // the level, images included, that may cut `cell` or any part of it.
  //
  // A seed q with weight w_q cuts the cell of p with weight w_p exactly when
  // some point x of the cell is nearer to q in the power distance:
  // |x-q|^2 - w_q < |x-p|^2 - w_p. The difference of the two sides is
  // linear in x, so it is largest at a vertex, and a part of the cell has no
  // point outside the cell: it is enough to look at the vertices of the
  // cell as it is now. For a vertex v that leaves q inside the disc around
  // v with radius^2 |v-p|^2 - w_p + w_q, at most the radius^2 below with
  // the largest weight of the level. The discs are widened by a rounding
  // margin; the box around them is looked up in the grid, one row and
  // column more on each side because a seed and its shifted image need not
  // round into the same grid cell, and a seed found there is kept when it
  // lies in one of the discs.
  struct disc {
    point_type centre;
    T radius2;
  };

  void seeds_that_may_cut(level const& lv, polygon_type const& cell,
                          point_type const& p, T w_p,
                          std::vector<disc>& discs,
                          std::vector<std::size_t>& out) const {
    T lo_x = std::numeric_limits<T>::max(), lo_y = lo_x;
    T hi_x = std::numeric_limits<T>::lowest(), hi_y = hi_x;
    discs.clear();
    const T slack = T{1e-9} * std::max(_Lx, _Ly);
    for (auto const& v : cell) {
      const T dx = v[0] - p[0];
      const T dy = v[1] - p[1];
      const T r2 = dx * dx + dy * dy - w_p + lv.max_weight;
      if (!(r2 > T{0})) continue;
      const T r = std::sqrt(r2) * (T{1} + T{1e-9}) + slack;
      lo_x = std::min(lo_x, v[0] - r);
      hi_x = std::max(hi_x, v[0] + r);
      lo_y = std::min(lo_y, v[1] - r);
      hi_y = std::max(hi_y, v[1] + r);
      discs.push_back({v, r * r});
    }
    if (discs.empty()) return;
    // Grid rows and columns of the box, in the numbering that continues
    // across the periods: column c + s * nx is column c of the image
    // shifted by s periods. Only the images -1, 0, 1 exist.
    auto range = [this](T lo, T hi, T h, std::size_t n) {
      const auto count = static_cast<long long>(n);
      const long long least = _periodic ? -count : 0;
      const long long most = _periodic ? 2 * count - 1 : count - 1;
      const T a = std::floor(lo / h) - T{1};
      const T b = std::floor(hi / h) + T{1};
      const long long first = a <= static_cast<T>(least)
                                  ? least
                                  : static_cast<long long>(std::min(
                                        a, static_cast<T>(most + 1)));
      const long long last = b >= static_cast<T>(most)
                                 ? most
                                 : static_cast<long long>(std::max(
                                       b, static_cast<T>(least - 1)));
      return std::pair<long long, long long>{first, last};
    };
    const auto [x0, x1] = range(lo_x, hi_x, lv.hx, lv.nx);
    const auto [y0, y1] = range(lo_y, hi_y, lv.hy, lv.ny);
    const auto nx = static_cast<long long>(lv.nx);
    const auto ny = static_cast<long long>(lv.ny);
    const std::size_t images = _periodic ? 9 : 1;
    for (long long iy = y0; iy <= y1; ++iy) {
      const long long sy = iy < 0 ? -1 : (iy >= ny ? 1 : 0);
      const long long cy = iy - sy * ny;
      for (long long ix = x0; ix <= x1; ++ix) {
        const long long sx = ix < 0 ? -1 : (ix >= nx ? 1 : 0);
        const long long cx = ix - sx * nx;
        const auto c = static_cast<std::size_t>(cy * nx + cx);
        const std::size_t image =
            _periodic ? static_cast<std::size_t>((sx + 1) * 3 + (sy + 1)) : 0;
        for (std::size_t k = lv.start[c]; k < lv.start[c + 1]; ++k) {
          const std::size_t j = lv.seeds[k];
          const T qx = _seeds[j][0] + static_cast<T>(sx) * _Lx;
          const T qy = _seeds[j][1] + static_cast<T>(sy) * _Ly;
          for (auto const& d : discs) {
            const T dx = qx - d.centre[0];
            const T dy = qy - d.centre[1];
            if (dx * dx + dy * dy <= d.radius2) {
              out.push_back(j * images + image);
              break;
            }
          }
        }
      }
    }
  }

  // `count` clips that cut nothing. Each one leaves every vertex in place,
  // moves the first vertex to the end and merges neighbours (cyclically)
  // closer than the tolerance. With no such neighbours, which is the rule,
  // all of them together are one rotation of the list; otherwise clips that
  // keep everything are applied until one of them merges nothing.
  static void skip_clips(polygon_type& cell, std::size_t count, T merge_tol,
                         polygon_type& scratch) {
    while (count > 0 && cell.size() >= 3) {
      const std::size_t size = cell.size();
      bool close = false;
      for (std::size_t k = 0; k < size && !close; ++k) {
        auto const& a = cell[k];
        auto const& b = cell[(k + 1) % size];
        close = std::abs(a[0] - b[0]) <= merge_tol &&
                std::abs(a[1] - b[1]) <= merge_tol;
      }
      if (!close) {
        std::rotate(cell.begin(),
                    cell.begin() + static_cast<std::ptrdiff_t>(count % size),
                    cell.end());
        return;
      }
      clip_polygon(cell, T{0}, T{0}, T{1}, merge_tol, scratch);   // keeps all
      cell.swap(scratch);
      --count;
    }
  }

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
    clip_polygon(polygon, A, B, C, merge_tol, out);
    return out;
  }

  // The same into a buffer the caller reuses (`out` must not be `polygon`).
  static void clip_polygon(polygon_type const& polygon, T A, T B, T C,
                           T merge_tol, polygon_type& out) {
    out.clear();
    out.reserve(polygon.size() + 1);
    const auto n = polygon.size();
    if (n == 0) return;

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
voronoi_to_shapes(voronoi_generator_2d<T> const& gen,
                  cancel_fn const* cancel = nullptr) {
  auto cells = gen.build(cancel);
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
