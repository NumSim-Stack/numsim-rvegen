#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace rvegen {

// Lightweight triangle-mesh POD produced by `to_mesh(shape)` and consumed by
// out-of-tree visualizers (e.g. Tessera's VTK renderer). The library itself
// stays renderer-agnostic — this struct is the boundary type.
//
// `verts`   — vertex positions in world coordinates.
// `tris`    — triangle indices into `verts`. Each entry is (i0, i1, i2);
//             counter-clockwise when viewed from outside (right-hand rule
//             gives the outward normal).
// `normals` — optional per-vertex normals. If empty, the consumer is free
//             to compute flat or smoothed normals from the face geometry.
//             Sphere meshes populate this (smooth shading wins on curved
//             surfaces); box meshes leave it empty (flat shading is the
//             expected look).
template <typename T>
struct triangle_mesh {
  std::vector<std::array<T, 3>>           verts;
  std::vector<std::array<std::size_t, 3>> tris;
  std::vector<std::array<T, 3>>           normals;

  [[nodiscard]] bool empty() const noexcept {
    return verts.empty() || tris.empty();
  }
};

} // namespace rvegen
