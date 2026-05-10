#pragma once

// JSON-driven input that loads a triangle mesh from an STL file once
// at construction time, then on each new_shape() clones the mesh and
// places it at a sampled position.
//
// Why load-once-clone-many:
//   STL parsing is the costly step; cloning a triangle vector is a
//   memcpy. The shape's own translation logic (set_middle_point) shifts
//   the cached centroid + face triangles. Each placement is therefore
//   O(N_triangles) memory and time — fine for moderate meshes, worth
//   sharing the triangle list via shared_ptr for very large meshes
//   (tracked as follow-up).
//
// What this header lands today (phase 2 of #9, after the
// mesh_inclusion shape + STL reader in PR #23):
//   * `mesh_inclusion_input<T>` — schema-driven (`stl_path`,
//     `position_x_dist`, `position_y_dist`, `position_z_dist`).
//   * Registered as `"mesh_inclusion_input"` in `register_inputs.h`.
//   * STL is read once at input construction; `new_shape()` clones the
//     cached triangle list and translates the clone to the sampled
//     position.
//
// Out of scope here, ships in follow-up PRs against #9:
//   * Per-shape rotation distributions (axis-angle or Bingham
//     orientation). Needs rotation logic in mesh_inclusion (the
//     primitive only supports translation today via set_middle_point).
//   * Per-shape uniform scaling. Same shape mutation gap.
//   * Binary STL reader hookup once it lands.
//   * shared_ptr-backed triangle storage to avoid the per-shape
//     memcpy on heavy meshes.

#include <memory>
#include <string>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../io/stl_reader.h"
#include "../shapes/mesh_inclusion.h"
#include "../types.h"
#include "circle_input.h"   // distribution_map_t alias
#include "shape_input_base.h"

namespace rvegen {

template <typename T = double>
class mesh_inclusion_input final : public shape_input_base<T> {
public:
  using value_type = T;
  using triangle_type = typename mesh_inclusion<T>::triangle_type;

  mesh_inclusion_input(std::string stl_path,
                       distribution_base<value_type>& px,
                       distribution_base<value_type>& py,
                       distribution_base<value_type>& pz)
      : _stl_path{std::move(stl_path)},
        _triangles{read_stl_ascii_file<value_type>(_stl_path)},
        _px{px}, _py{py}, _pz{pz} {}

  mesh_inclusion_input(parameter_handler_t const& handler,
                       distribution_map_t<value_type> const& d)
      : mesh_inclusion_input(
            handler.template get<std::string>("stl_path"),
            *d.at(handler.template get<std::string>("position_x_dist")),
            *d.at(handler.template get<std::string>("position_y_dist")),
            *d.at(handler.template get<std::string>("position_z_dist"))) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("stl_path")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"path to an ASCII STL file describing the mesh inclusion (read once at input construction)">>();
    s.template insert<std::string>("position_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the placed mesh's centre x-coordinate">>();
    s.template insert<std::string>("position_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the placed mesh's centre y-coordinate">>();
    s.template insert<std::string>("position_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the placed mesh's centre z-coordinate">>();
    return s;
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    auto mesh = std::make_unique<mesh_inclusion<value_type>>(_triangles);
    mesh->set_middle_point({_px(), _py(), _pz()});
    return mesh;
  }

  [[nodiscard]] std::string const& stl_path() const noexcept {
    return _stl_path;
  }

private:
  std::string _stl_path;
  std::vector<triangle_type> _triangles;
  distribution_base<value_type>& _px;
  distribution_base<value_type>& _py;
  distribution_base<value_type>& _pz;
};

} // namespace rvegen
