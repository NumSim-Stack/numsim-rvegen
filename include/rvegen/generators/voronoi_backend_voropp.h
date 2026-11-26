#pragma once

// Voro++ backend for voronoi_generator_3d (#70). Only compiled when the
// build defines RVEGEN_WITH_VORO_PP (CMake option RVEGEN_WITH_VORO_PP=ON,
// which fetches Voro++ v0.4.6, BSD-3, and links it). Without the macro this
// header is empty, keeping rvegen header-only with no extra dependency.
//
// Voro++ computes each cell by incremental plane cutting inside a block
// grid, so it scales roughly linearly in the seed count (vs O(N^5) for the
// in-house enumeration). Output is converted to the unchanged public
// `voronoi_cell` shape; the voronoi_cell ctor re-derives outward normals and
// face winding, so Voro++'s own face orientation convention does not matter.
//
// The container is the closed box [0,Lx]x[0,Ly]x[0,Lz], non-periodic. Seeds
// have already been validated (inside the closed domain, pairwise distinct)
// by voronoi_generator_3d. Voro++ uses absolute internal tolerances, so the
// problem is non-dimensionalised first: coordinates are divided by the
// longest box edge, tessellated in a box of size <= 1, and the cell
// vertices are scaled back (same scale-invariance as the in-house backend).

#ifdef RVEGEN_WITH_VORO_PP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include <voro++.hh>

#include "../shapes/shape_base.h"
#include "../shapes/voronoi_cell.h"

namespace rvegen::detail {

template <typename T>
[[nodiscard]] std::vector<std::unique_ptr<shape_base<T>>>
build_voronoi_cells_voro_pp(T Lx, T Ly, T Lz,
                            std::vector<std::array<T, 3>> const& seeds) {
  const std::size_t n = seeds.size();

  // Block grid: aim for ~6 seeds per block (Voro++'s recommended density),
  // distributed according to the box aspect ratio.
  const double sc = static_cast<double>(std::max({Lx, Ly, Lz}));
  const double ux = Lx / sc, uy = Ly / sc, uz = Lz / sc;
  const double vol = ux * uy * uz;
  const double h = std::cbrt(vol * 6.0 / static_cast<double>(n));
  auto blocks = [h](double L) {
    return std::clamp(static_cast<int>(L / h), 1, 64);
  };
  voro::container con(0.0, ux, 0.0, uy, 0.0, uz, blocks(ux), blocks(uy),
                      blocks(uz), false, false, false, 8);

  // Voro++ silently drops points with coordinate == upper bound, so seeds
  // on the upper faces are moved inward by one ulp (a ~1e-16 relative shift).
  auto inward = [](double x, double hi) {
    return x >= hi ? std::nextafter(hi, 0.0) : x;
  };
  for (std::size_t i = 0; i < n; ++i) {
    con.put(static_cast<int>(i), inward(seeds[i][0] / sc, ux),
            inward(seeds[i][1] / sc, uy), inward(seeds[i][2] / sc, uz));
  }

  std::vector<std::unique_ptr<shape_base<T>>> out(n);
  voro::c_loop_all loop(con);
  voro::voronoicell cell;
  std::vector<double> verts;
  std::vector<int> face_v;
  std::size_t produced = 0;
  if (loop.start()) {
    do {
      if (!con.compute_cell(cell, loop)) continue;   // cell fully clipped
      const auto id = static_cast<std::size_t>(loop.pid());
      double x, y, z;
      loop.pos(x, y, z);
      cell.vertices(x, y, z, verts);
      cell.face_vertices(face_v);

      std::vector<gte::Vector<3, T>> gv(verts.size() / 3);
      for (std::size_t k = 0; k < gv.size(); ++k) {
        gv[k][0] = static_cast<T>(verts[3 * k] * sc);
        gv[k][1] = static_cast<T>(verts[3 * k + 1] * sc);
        gv[k][2] = static_cast<T>(verts[3 * k + 2] * sc);
      }
      // face_vertices layout: n, v_0 .. v_{n-1}, n, ...
      std::vector<std::vector<std::size_t>> faces;
      for (std::size_t k = 0; k < face_v.size();) {
        const auto cnt = static_cast<std::size_t>(face_v[k]);
        faces.emplace_back(face_v.begin() + static_cast<std::ptrdiff_t>(k + 1),
                           face_v.begin() +
                               static_cast<std::ptrdiff_t>(k + 1 + cnt));
        k += cnt + 1;
      }
      out[id] = std::make_unique<voronoi_cell<T>>(std::move(gv),
                                                   std::move(faces));
      ++produced;
    } while (loop.inc());
  }
  if (produced != n) {
    throw std::runtime_error{
        "voronoi_generator_3d (voro++): backend dropped " +
        std::to_string(n - produced) + " seed(s)"};
  }
  return out;
}

} // namespace rvegen::detail

#endif // RVEGEN_WITH_VORO_PP
