#pragma once

#include "../shapes/box.h"
#include "../shapes/circle.h"
#include "../shapes/ellipse.h"
#include "../shapes/rectangle.h"
#include "../shapes/sphere.h"
#include "../visualization/box_mesh.h"
#include "../visualization/circle_mesh.h"
#include "../visualization/ellipse_mesh.h"
#include "../visualization/mesh_dispatcher.h"
#include "../visualization/rectangle_mesh.h"
#include "../visualization/sphere_mesh.h"

namespace rvegen {

// Populate the mesh_dispatcher singleton with every built-in shape.
// Out-of-tree shapes register themselves the same way:
//   mesh_dispatcher<T>::instance().register_shape<my_shape<T>>();
//
// 2D shapes (circle, rectangle, ellipse) get flat triangle meshes in the
// z=0 plane — what a 3D renderer wants when asked to display a 2D RVE.
// 3D shapes (sphere, box) get full surface meshes.
template <typename T = double>
inline void register_all_meshes() {
  auto& d = mesh_dispatcher<T>::instance();
  d.template register_shape<sphere<T>>();
  d.template register_shape<box<T>>();
  d.template register_shape<circle<T>>();
  d.template register_shape<rectangle<T>>();
  d.template register_shape<ellipse<T>>();
}

} // namespace rvegen
