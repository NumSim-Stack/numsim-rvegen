#pragma once

// JSON-driven input that loads a triangle mesh from an STL file once
// at construction time, then on each new_shape() hands a shared
// reference to the same triangle list to the produced mesh_inclusion
// (no per-shape triangle copy) and stores a per-shape translation
// offset.
//
// Why load-once-share-many:
//   STL parsing is the costly step. The triangle list is wrapped in a
//   `std::shared_ptr<vector<triangle_type> const>` and handed to every
//   placed `mesh_inclusion` by reference (refcount bump only). Each
//   placement carries only its own translation offset — set_middle_point
//   is O(1) instead of O(N_triangles). For 1000 placements of a 10k-tri
//   mesh: one triangle allocation, not 1000.
//
// What this header lands today (phase 2 of #9, after the
// mesh_inclusion shape + STL reader in PR #23):
//   * `mesh_inclusion_input<T>` — schema-driven (`stl_path`,
//     `position_x_dist`, `position_y_dist`, `position_z_dist`).
//   * Registered as `"mesh_inclusion_input"` in `register_inputs.h`.
//   * STL is read once at input construction and wrapped in a
//     `std::shared_ptr<vector<triangle_type> const>`. `new_shape()`
//     constructs a `mesh_inclusion` that captures the same shared_ptr
//     (refcount bump only) and applies the sampled position via the
//     primitive's O(1) offset — no triangle data is copied per shape.
//
// Auto-detect ASCII vs binary STL:
//   The input loads via `read_stl_file()`, which sniffs the file header
//   and dispatches to either `read_stl_ascii_file` or `read_stl_binary_file`.
//   Consumers don't need to know which format their STL is in.
//
// Out of scope here, ships in follow-up PRs against #9:
//   * Per-shape rotation distributions (axis-angle or Bingham
//     orientation). Needs rotation logic in mesh_inclusion (today the
//     primitive's offset only encodes translation).
//   * Per-shape uniform scaling. Same primitive-side gap.

#include <memory>
#include <stdexcept>
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
        _shared_triangles{std::make_shared<std::vector<triangle_type> const>(
            read_stl_file<value_type>(_stl_path))},
        _px{px}, _py{py}, _pz{pz} {
    // An empty triangle list yields a `mesh_inclusion` whose
    // `is_inside` is always false. The generator's volume_fraction
    // termination would then never satisfy and the run would loop
    // until `max_attempts` (silent timeout) — fail fast instead.
    if (!_shared_triangles || _shared_triangles->empty()) {
      throw std::runtime_error{
          "mesh_inclusion_input: STL file '" + _stl_path +
          "' contained no triangles; refusing to construct an "
          "always-empty inclusion."};
    }
  }

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

  // Zero-copy placement: the shared_ptr is captured by `mesh_inclusion`,
  // so 1000 placements share one triangle vector instead of cloning it
  // 1000 times. set_middle_point only updates the per-shape offset.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    auto mesh = std::make_unique<mesh_inclusion<value_type>>(_shared_triangles);
    mesh->set_middle_point({_px(), _py(), _pz()});
    return mesh;
  }

  [[nodiscard]] std::string const& stl_path() const noexcept {
    return _stl_path;
  }

private:
  std::string _stl_path;
  std::shared_ptr<std::vector<triangle_type> const> _shared_triangles;
  distribution_base<value_type>& _px;
  distribution_base<value_type>& _py;
  distribution_base<value_type>& _pz;
};

} // namespace rvegen
