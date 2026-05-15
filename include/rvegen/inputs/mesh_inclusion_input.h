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
// What this header provides:
//   * `mesh_inclusion_input<T>` — schema-driven (`mesh_path`,
//     `position_x_dist`, `position_y_dist`, `position_z_dist`).
//   * Registered as `"mesh_inclusion_input"` in `register_inputs.h`.
//   * The mesh file is read once at input construction and wrapped in a
//     `std::shared_ptr<vector<triangle_type> const>`. `new_shape()`
//     constructs a `mesh_inclusion` that captures the same shared_ptr
//     (refcount bump only) and applies the sampled position via the
//     primitive's O(1) offset — no triangle data is copied per shape.
//
// Mesh-file format dispatch:
//   The input loads via `read_mesh_file()`, which picks STL vs PLY by
//   the path's extension and then auto-detects ASCII vs binary for
//   each. Both `.stl` and `.ply` consumer files work transparently.
//
// JSON field name:
//   The canonical field is `mesh_path`. `stl_path` is kept as a
//   deprecated alias for back-compat with configs written when STL
//   was the only supported format. Exactly one of the two must be
//   present; supplying both is rejected as ambiguous.
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
#include "../io/mesh_reader.h"
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

  mesh_inclusion_input(std::string mesh_path,
                       distribution_base<value_type>& px,
                       distribution_base<value_type>& py,
                       distribution_base<value_type>& pz)
      : _mesh_path{std::move(mesh_path)},
        _shared_triangles{std::make_shared<std::vector<triangle_type> const>(
            read_mesh_file<value_type>(_mesh_path))},
        _px{px}, _py{py}, _pz{pz} {
    // An empty triangle list yields a `mesh_inclusion` whose
    // `is_inside` is always false. The generator's volume_fraction
    // termination would then never satisfy and the run would loop
    // until `max_attempts` (silent timeout) — fail fast instead.
    if (!_shared_triangles || _shared_triangles->empty()) {
      throw std::runtime_error{
          "mesh_inclusion_input: mesh file '" + _mesh_path +
          "' contained no triangles; refusing to construct an "
          "always-empty inclusion."};
    }
  }

  mesh_inclusion_input(parameter_handler_t const& handler,
                       distribution_map_t<value_type> const& d)
      : mesh_inclusion_input(
            resolve_mesh_path(handler),
            *d.at(handler.template get<std::string>("position_x_dist")),
            *d.at(handler.template get<std::string>("position_y_dist")),
            *d.at(handler.template get<std::string>("position_z_dist"))) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    // `mesh_path` is the canonical name. `stl_path` is the deprecated
    // alias kept for back-compat — exactly one of the two must be
    // present (enforced at ctor time, not by the schema layer, since
    // a "one of N" group constraint isn't expressible there).
    s.template insert<std::string>("mesh_path")
        .template add<numsim_core::description_label<"path to an STL (.stl, ASCII or binary) or PLY (.ply, ASCII or binary) file describing the mesh inclusion; format dispatched by extension (canonical; supersedes the legacy stl_path alias)">>();
    s.template insert<std::string>("stl_path")
        .template add<numsim_core::description_label<"deprecated alias for mesh_path; kept for back-compat with configs written before PLY support landed. Use mesh_path in new configs.">>();
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

  [[nodiscard]] std::string const& mesh_path() const noexcept {
    return _mesh_path;
  }

private:
  // Pick mesh_path (canonical) or stl_path (deprecated alias) from
  // the handler; exactly one of the two must be present. Supplying
  // neither or both is rejected with a clear error.
  static std::string resolve_mesh_path(parameter_handler_t const& h) {
    const bool has_mesh = h.contains("mesh_path");
    const bool has_stl  = h.contains("stl_path");
    if (has_mesh && has_stl) {
      throw std::runtime_error{
          "mesh_inclusion_input: both 'mesh_path' and the deprecated "
          "'stl_path' alias are set in the JSON config — please use "
          "only 'mesh_path'."};
    }
    if (!has_mesh && !has_stl) {
      throw std::runtime_error{
          "mesh_inclusion_input: 'mesh_path' is required (or the "
          "deprecated 'stl_path' alias for back-compat)."};
    }
    return h.template get<std::string>(has_mesh ? "mesh_path" : "stl_path");
  }

  std::string _mesh_path;
  std::shared_ptr<std::vector<triangle_type> const> _shared_triangles;
  distribution_base<value_type>& _px;
  distribution_base<value_type>& _py;
  distribution_base<value_type>& _pz;
};

} // namespace rvegen
