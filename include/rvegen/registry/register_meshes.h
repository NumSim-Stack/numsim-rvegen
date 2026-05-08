#pragma once

#include "../shapes/box.h"
#include "../shapes/sphere.h"
#include "../visualization/box_mesh.h"
#include "../visualization/mesh_dispatcher.h"
#include "../visualization/sphere_mesh.h"

namespace rvegen {

// Populate the mesh_dispatcher singleton with the library's built-in 3D
// shapes. Out-of-tree shapes register themselves the same way:
//   mesh_dispatcher<T>::instance().register_shape<my_shape<T>>();
//
// 2D shapes (circle, rectangle, ellipse) intentionally skip registration:
// they have no 3D mesh, and the dispatcher returns an empty mesh on miss
// rather than throwing — a renderer asking for one of them gets a no-op.
template <typename T = double>
inline void register_all_meshes() {
  auto& d = mesh_dispatcher<T>::instance();
  d.template register_shape<sphere<T>>();
  d.template register_shape<box<T>>();
}

} // namespace rvegen
