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
//
// Periodic mode uses the container's native x/y/z periodicity. Cells are
// returned around their ORIGINAL seed position (unclipped, may stick out of
// the box), matching the in-house periodic backend. Weighted (Laguerre)
// mode uses voro::container_poly with radius r_i = sqrt(w_i - min w), which
// is the same power diagram (weights are only defined up to a constant).
// A weighted seed whose cell vanishes yields an empty voronoi_cell.

#ifdef RVEGEN_WITH_VORO_PP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "voronoi_voropp_exit_shim.h"   // defines the `exit` redirect
#include <voro++.hh>
#undef exit   // restore `exit` for the rest of the TU (shim macro)

#include "../shapes/shape_base.h"
#include "../shapes/voronoi_cell.h"

namespace rvegen::detail {

template <typename T>
[[nodiscard]] std::vector<std::unique_ptr<shape_base<T>>>
build_voronoi_cells_voro_pp(T Lx, T Ly, T Lz,
                            std::vector<std::array<T, 3>> const& seeds,
                            bool periodic = false,
                            std::vector<T> const& weights = {}) {
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
  const bool weighted = !weights.empty();
  const int bx = blocks(ux), by = blocks(uy), bz = blocks(uz);
  voro::container mono(0.0, ux, 0.0, uy, 0.0, uz, bx, by, bz, periodic,
                       periodic, periodic, 8);
  voro::container_poly poly(0.0, ux, 0.0, uy, 0.0, uz, bx, by, bz, periodic,
                            periodic, periodic, 8);

  // Voro++ silently drops points whose block index rounds to (or past) the
  // container edge: seeds on the upper faces, and seeds a hair outside the
  // lower faces (validation tolerates ~1e-9 slack). Clamp every coordinate
  // into [0, u*(1 - 1e-12)] in non-dimensional units: the margin is many
  // ulps of the block-index arithmetic, so it is robust at any box size and
  // aspect ratio, yet moves a seed by only ~1e-12 of its own axis extent.
  // The final seed-count check below still guards against any other drop.
  constexpr double margin = 1e-12;
  auto inside = [](double x, double hi) {
    return std::clamp(x, 0.0, hi * (1.0 - margin));
  };
  double wmin = 0.0;
  if (weighted) wmin = static_cast<double>(*std::min_element(weights.begin(), weights.end()));
  for (std::size_t i = 0; i < n; ++i) {
    const double px = inside(seeds[i][0] / sc, ux), py = inside(seeds[i][1] / sc, uy),
                 pz = inside(seeds[i][2] / sc, uz);
    if (weighted) {
      const double r = std::sqrt((static_cast<double>(weights[i]) - wmin) / (sc * sc));
      poly.put(static_cast<int>(i), px, py, pz, r);
    } else {
      mono.put(static_cast<int>(i), px, py, pz);
    }
  }

  std::vector<std::unique_ptr<shape_base<T>>> out(n);
  voro::voronoicell cell;
  std::vector<double> verts;
  std::vector<int> face_v;
  std::size_t produced = 0;
  auto run = [&](auto& con) {
    voro::c_loop_all loop(con);
    if (!loop.start()) return;
    do {
      if (!con.compute_cell(cell, loop)) continue;   // cell fully clipped
      const auto id = static_cast<std::size_t>(loop.pid());
      double x, y, z;
      loop.pos(x, y, z);
      if (periodic) {   // report the cell around the original seed
        x = seeds[id][0] / sc; y = seeds[id][1] / sc; z = seeds[id][2] / sc;
      }
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
  };
  if (weighted) run(poly); else run(mono);
  if (weighted) {
    for (auto& c : out) {
      if (!c) c = std::make_unique<voronoi_cell<T>>();   // vanished cell
    }
  } else if (produced != n) {
    throw std::runtime_error{
        "voronoi_generator_3d (voro++): backend dropped " +
        std::to_string(n - produced) + " seed(s)"};
  }
  return out;
}

} // namespace rvegen::detail

#endif // RVEGEN_WITH_VORO_PP
