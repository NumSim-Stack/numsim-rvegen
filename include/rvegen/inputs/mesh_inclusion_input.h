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
//   * The mesh file is read once at input construction into a single
//     prototype `mesh_inclusion` — that is where the mesh is validated
//     (closed, manifold, finite), re-oriented and its AABB / volume
//     cached, so a broken file fails at config load. `new_shape()`
//     copies the prototype (shared_ptr refcount bump + cached arrays)
//     and applies the sampled position via the primitive's O(1)
//     offset — no triangle data is copied or re-validated per shape.
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
// Optional orientation and scale (all fields optional; omitted means
// identity rotation and unit scale, i.e. the previous behaviour):
//   * `orientation`: "none" (default), "uniform" or "euler_zyz".
//       - "uniform": isotropic random orientation (Shoemake uniform
//         quaternion). `orientation_dist` names a distribution that
//         samples U(0, 1) (e.g. {"type": "uniform_real", "a": 0,
//         "b": 1}); three draws per placed shape.
//       - "euler_zyz": R = Rz(alpha)·Ry(beta)·Rz(gamma) with the angles
//         (radians) drawn from `euler_alpha_dist`, `euler_beta_dist`,
//         `euler_gamma_dist`. NB: uniform Euler angles are not
//         isotropic — use "uniform" for random orientation.
//   * `scale_dist`: distribution sampled for the uniform scale factor
//     (must be > 0) about the placed mesh's centre.
// C++ users set the same via `set_uniform_orientation`,
// `set_euler_zyz_orientation` and `set_scale_distribution`.

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "../distributions/distribution_base.h"
#include "../distributions/uniform_rotation.h"
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
                       distribution_base<value_type>& pz,
                       ply_read_limits limits = {})
      : _mesh_path{std::move(mesh_path)},
        _prototype{load_prototype(_mesh_path, limits)},
        _px{px}, _py{py}, _pz{pz} {}

  mesh_inclusion_input(parameter_handler_t const& handler,
                       distribution_map_t<value_type> const& d)
      : mesh_inclusion_input(
            resolve_mesh_path(handler),
            lookup_distribution(d, handler, "position_x_dist"),
            lookup_distribution(d, handler, "position_y_dist"),
            lookup_distribution(d, handler, "position_z_dist"),
            read_limits(handler)) {
    apply_transform_options(handler, d);
  }

  // ---- optional orientation / scale (C++ API) ---------------------------
  // Distributions are borrowed by reference, like the position ones.
  void set_uniform_orientation(distribution_base<value_type>& unit_uniform) {
    _orientation = orientation_mode::uniform;
    _rot_dists = {&unit_uniform, nullptr, nullptr};
  }
  void set_euler_zyz_orientation(distribution_base<value_type>& alpha,
                                 distribution_base<value_type>& beta,
                                 distribution_base<value_type>& gamma) {
    _orientation = orientation_mode::euler_zyz;
    _rot_dists = {&alpha, &beta, &gamma};
  }
  void set_scale_distribution(distribution_base<value_type>& scale) {
    _scale_dist = &scale;
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    // `mesh_path` is the canonical name. `stl_path` is the deprecated
    // alias kept for back-compat — exactly one of the two must be
    // present (enforced at ctor time, not by the schema layer, since
    // a "one of N" group constraint isn't expressible there).
    s.template insert<std::string>("mesh_path")
        .template add<numsim_core::description_label<"path to an STL (.stl, ASCII or binary) or PLY (.ply, ASCII or binary) file describing the mesh inclusion; format dispatched by extension (canonical; supersedes the legacy stl_path alias). A relative path is resolved against the process working directory, not against the JSON file">>();
    s.template insert<std::string>("stl_path")
        .template add<numsim_core::description_label<"deprecated alias for mesh_path; kept for back-compat with configs written before PLY support landed. Use mesh_path in new configs.">>();
    s.template insert<std::size_t>("max_triangles")
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::unit_label<"triangles">>()
        .template add<numsim_core::description_label<"optional cap on the triangle count read from a PLY file (fan-triangulated output; default 5000000). Raise it for genuinely huge meshes">>();
    s.template insert<std::string>("position_x_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the placed mesh's centre x-coordinate">>();
    s.template insert<std::string>("position_y_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the placed mesh's centre y-coordinate">>();
    s.template insert<std::string>("position_z_dist")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"name of a distribution sampled for the placed mesh's centre z-coordinate">>();
    s.template insert<std::string>("orientation")
        .template add<numsim_core::description_label<"optional orientation mode: 'none' (default), 'uniform' (isotropic random rotation via Shoemake uniform quaternion; needs orientation_dist) or 'euler_zyz' (needs euler_alpha_dist, euler_beta_dist, euler_gamma_dist)">>();
    s.template insert<std::string>("orientation_dist")
        .template add<numsim_core::description_label<"for orientation 'uniform': name of a distribution sampling U(0, 1); three draws per placed shape">>();
    s.template insert<std::string>("euler_alpha_dist")
        .template add<numsim_core::description_label<"for orientation 'euler_zyz': name of a distribution sampled for the first z rotation angle (radians)">>();
    s.template insert<std::string>("euler_beta_dist")
        .template add<numsim_core::description_label<"for orientation 'euler_zyz': name of a distribution sampled for the y rotation angle (radians)">>();
    s.template insert<std::string>("euler_gamma_dist")
        .template add<numsim_core::description_label<"for orientation 'euler_zyz': name of a distribution sampled for the second z rotation angle (radians)">>();
    s.template insert<std::string>("scale_dist")
        .template add<numsim_core::description_label<"optional: name of a distribution sampled for the uniform scale factor (> 0) of each placed mesh; default 1">>();
    return s;
  }

  // Zero-copy placement: copying the prototype shares its triangle
  // vector, so 1000 placements share one triangle vector instead of
  // cloning it 1000 times. set_middle_point only updates the per-shape
  // offset.
  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    auto mesh = std::make_unique<mesh_inclusion<value_type>>(_prototype);
    mesh->set_middle_point({_px(), _py(), _pz()});
    if (_orientation == orientation_mode::uniform) {
      const value_type u1 = (*_rot_dists[0])();
      const value_type u2 = (*_rot_dists[0])();
      const value_type u3 = (*_rot_dists[0])();
      mesh->set_rotation(uniform_random_rotation(u1, u2, u3));
    } else if (_orientation == orientation_mode::euler_zyz) {
      const value_type a = (*_rot_dists[0])();
      const value_type b = (*_rot_dists[1])();
      const value_type g = (*_rot_dists[2])();
      mesh->set_rotation_euler_zyz(a, b, g);
    }
    if (_scale_dist) mesh->set_scale((*_scale_dist)());
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

  // Read + validate the mesh once. An empty triangle list yields a
  // `mesh_inclusion` whose `is_inside` is always false; the
  // generator's volume_fraction termination would then never satisfy
  // and the run would loop until `max_attempts` (silent timeout) —
  // fail fast instead.
  static ply_read_limits read_limits(parameter_handler_t const& h) {
    ply_read_limits l;
    if (h.contains("max_triangles")) {
      l.max_triangles = h.template get<std::size_t>("max_triangles");
    }
    return l;
  }

  static mesh_inclusion<value_type> load_prototype(
      std::string const& path, ply_read_limits const& limits) {
    auto triangles = read_mesh_file<value_type>(path, limits);
    if (triangles.empty()) {
      throw std::runtime_error{
          "mesh_inclusion_input: mesh file '" + path +
          "' contained no triangles; refusing to construct an "
          "always-empty inclusion."};
    }
    try {
      return mesh_inclusion<value_type>{std::move(triangles)};
    } catch (std::runtime_error const& e) {
      throw std::runtime_error{"mesh_inclusion_input: mesh file '" + path +
                               "' is invalid: " + e.what()};
    }
  }

  // Resolve the optional orientation / scale fields of the JSON config.
  void apply_transform_options(parameter_handler_t const& h,
                               distribution_map_t<value_type> const& d) {
    auto lookup = [&](char const* key) -> distribution_base<value_type>& {
      if (!h.contains(key)) {
        throw std::runtime_error{
            std::string{"mesh_inclusion_input: '"} + key +
            "' is required for the selected orientation mode"};
      }
      return lookup_distribution(d, h, key);
    };
    const std::string mode =
        h.contains("orientation") ? h.template get<std::string>("orientation")
                                  : std::string{"none"};
    // Orientation keys that the active mode does not consume would be
    // silently ignored (e.g. a mistyped or missing 'orientation'): reject.
    struct key_mode { char const* key; char const* used_by; };
    for (auto const& [key, used_by] : {key_mode{"orientation_dist", "uniform"},
                                       key_mode{"euler_alpha_dist", "euler_zyz"},
                                       key_mode{"euler_beta_dist", "euler_zyz"},
                                       key_mode{"euler_gamma_dist", "euler_zyz"}}) {
      if (h.contains(key) && mode != used_by) {
        throw std::runtime_error{
            std::string{"mesh_inclusion_input: '"} + key +
            "' is only used with orientation '" + used_by +
            "' but the active orientation mode is '" + mode +
            "' (set 'orientation' accordingly or remove the key)"};
      }
    }
    if (mode == "uniform") {
      set_uniform_orientation(lookup("orientation_dist"));
    } else if (mode == "euler_zyz") {
      set_euler_zyz_orientation(lookup("euler_alpha_dist"),
                                lookup("euler_beta_dist"),
                                lookup("euler_gamma_dist"));
    } else if (mode != "none") {
      throw std::runtime_error{
          "mesh_inclusion_input: unknown orientation '" + mode +
          "' (expected 'none', 'uniform' or 'euler_zyz')"};
    }
    if (h.contains("scale_dist")) set_scale_distribution(lookup("scale_dist"));
  }

  enum class orientation_mode { none, uniform, euler_zyz };

  std::string _mesh_path;
  mesh_inclusion<value_type> _prototype;
  orientation_mode _orientation{orientation_mode::none};
  std::array<distribution_base<value_type>*, 3> _rot_dists{};
  distribution_base<value_type>* _scale_dist{nullptr};
  distribution_base<value_type>& _px;
  distribution_base<value_type>& _py;
  distribution_base<value_type>& _pz;
};

} // namespace rvegen
