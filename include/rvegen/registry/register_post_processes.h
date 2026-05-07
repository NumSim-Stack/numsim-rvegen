#pragma once

#include <numsim-core/object_registry.h>

#include "../postprocess/gmsh_geo_writer.h"
#include "../postprocess/post_process_base.h"
#include "../postprocess/svg_3d_writer.h"
#include "../postprocess/svg_writer.h"
#include "../postprocess/three_js_writer.h"
#include "../postprocess/voxel_writer.h"
#include "../postprocess/vtk_legacy_writer.h"
#include "../types.h"

namespace rvegen {

// Post-process registry — JSON-driven instantiation of post-processing
// stages that run after generation. All output writers (gmsh_geo, voxel,
// vtk_legacy, svg, svg_3d, three_js) are post-processes; each owns its
// destination via output_path in its schema. A pipeline can run any
// number of them, in order.
template <typename T = double>
using post_process_registry_t = numsim_core::object_registry<
    post_process_base<T>,
    parameter_controller_t,
    parameter_handler_t const>;

template <typename T = double>
inline void register_gmsh_geo_writer() {
  post_process_registry_t<T>::instance()
      .template register_type<gmsh_geo_writer<T>>("gmsh_geo");
}

template <typename T = double>
inline void register_voxel_writer() {
  post_process_registry_t<T>::instance()
      .template register_type<voxel_writer<T>>("voxel");
}

template <typename T = double>
inline void register_vtk_legacy_writer() {
  post_process_registry_t<T>::instance()
      .template register_type<vtk_legacy_writer<T>>("vtk_legacy");
}

template <typename T = double>
inline void register_svg_writer() {
  post_process_registry_t<T>::instance()
      .template register_type<svg_writer<T>>("svg");
}

template <typename T = double>
inline void register_three_js_writer() {
  post_process_registry_t<T>::instance()
      .template register_type<three_js_writer<T>>("three_js");
}

template <typename T = double>
inline void register_svg_3d_writer() {
  post_process_registry_t<T>::instance()
      .template register_type<svg_3d_writer<T>>("svg_3d");
}

template <typename T = double>
inline void register_all_post_processes() {
  register_gmsh_geo_writer<T>();
  register_voxel_writer<T>();
  register_vtk_legacy_writer<T>();
  register_svg_writer<T>();
  register_three_js_writer<T>();
  register_svg_3d_writer<T>();
}

} // namespace rvegen
