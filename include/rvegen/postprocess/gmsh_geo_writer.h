#pragma once

#include <array>
#include <cstddef>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <map>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
#include "../phase/phase.h"
#include "../shapes/box.h"
#include "../shapes/circle.h"
#include "../shapes/convex_polygon.h"
#include "../shapes/rectangle.h"
#include "../shapes/sphere.h"
#include "../types.h"

namespace rvegen {

// Writes an RVE as a gmsh .geo script. Gmsh consumes this file and produces
// the FEM mesh.
//
// Output structure:
//   * 2D RVE: a Rectangle entity for the domain, then per-shape 2D entities
//   * 3D RVE: a Box entity for the domain, then per-shape 3D entities
//
// Shape dispatch uses dynamic_cast — output is one-shot, not on the inner
// generation loop, so the cost is negligible and the dispatch is localised.
//
// Supported shapes: circle, sphere, rectangle, box, convex_polygon.
// Adding a shape: extend the dispatch chain in write().
//
// Gmsh version requirement:
//   The phase-aware path (set_phases attached) emits
//   `Physical Surface("name", id) = {...};` — the string-named form
//   that gmsh added in 4.0 alongside the OpenCASCADE kernel.
//   Pre-4.0 gmsh consumes only `Physical Surface(id) = {...};` (no
//   name). If you target an older toolchain, leave set_phases empty.
template <typename T = double>
class gmsh_geo_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  // Default ctor leaves _output_path empty — only stream-based write() is
  // usable in that state. Kept so tests / library users can capture the
  // emitted text into a stringstream without needing a file.
  gmsh_geo_writer() = default;

  explicit gmsh_geo_writer(std::string output_path, bool periodic = false)
      : _output_path{std::move(output_path)}, _periodic{periodic} {}

  explicit gmsh_geo_writer(parameter_handler_t const& handler)
      : _output_path{handler.template get<std::string>("output_path")},
        _periodic{handler.contains("periodic")
                      ? handler.template get<bool>("periodic")
                      : false} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("output_path").template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"destination path for the gmsh .geo script">>();
    // Optional — default false. Schema does NOT mark it required so existing
    // configs keep working unchanged.
    s.template insert<bool>("periodic")
        .template add<numsim_core::description_label<"emit Periodic Curve/Surface directives + Coherence; for FE-RVE workflows that need matching nodes on opposite faces (default false)">>();
    return s;
  }

  [[nodiscard]] bool periodic() const noexcept { return _periodic; }
  void set_periodic(bool v) noexcept { _periodic = v; }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  // Attach a phase_collection so each emitted inclusion entity is added
  // to a gmsh Physical Surface (2D) / Physical Volume (3D) named after
  // its `phase_name()`. The numeric tag is the collection's `id_of(name)`.
  // Untagged shapes (empty phase_name) are not added to any Physical
  // group — the standard gmsh convention is for them to fall into the
  // matrix domain via Coherence.
  //
  // Caller retains ownership; the pointer must outlive the next run() /
  // write() call. Passing nullptr clears the attachment.
  void set_phases(phase_collection<value_type> const* phases) noexcept {
    _phases = phases;
  }
  [[nodiscard]] phase_collection<value_type> const* phases() const noexcept {
    return _phases;
  }

  // Strict mode: when set_phases is attached AND strict==true, any
  // shape carrying an empty `phase_name` makes `write()` throw before
  // emitting geometry. Default (strict==false) silently drops
  // untagged shapes from the Physical groups, which is the intended
  // fallback for "treat as matrix" but also masks the bug-of-omission
  // when a user *meant* to tag a shape and forgot. Use strict mode
  // when your generation pipeline guarantees every shape gets tagged.
  //
  // When no `phase_collection` is attached, strict mode is silently a
  // no-op — the writer has no phase grouping to refuse against.
  void set_phase_strict(bool strict) noexcept { _phase_strict = strict; }
  [[nodiscard]] bool phase_strict() const noexcept { return _phase_strict; }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "gmsh_geo_writer: output_path not set; cannot run() as post-process"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "gmsh_geo_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  // Stream-based emit. Kept public so tests and embedders can capture the
  // output to a stringstream without round-tripping through a file.
  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& box) const {
    // Validate phase tags BEFORE writing anything. If a shape carries a
    // phase_name not present in the collection, we throw here rather than
    // mid-emit — which would otherwise leave a partial .geo on disk with
    // entities but no Physical groups. Untagged shapes (empty name) are
    // intentionally allowed; they fall into gmsh's default region.
    if (_phases) {
      for (auto const& shape : shapes) {
        const auto name = shape->phase_name();
        if (name.empty()) {
          if (_phase_strict) {
            throw std::runtime_error{
                "gmsh_geo_writer: shape has empty phase_name and "
                "strict mode is enabled (set_phase_strict(true)). "
                "Tag every shape's phase, or disable strict mode."};
          }
          continue;
        }
        if (!_phases->contains(name)) {
          throw std::runtime_error{
              "gmsh_geo_writer: shape carries phase_name '" + name +
              "' which is not declared in the attached phase_collection"};
        }
      }
    }

    const bool is_3d = box[2] > value_type{0};
    if (is_3d) {
      write_3d(out, shapes, box);
    } else {
      write_2d(out, shapes, box);
    }
  }

private:
  static void write_header(std::ostream& out) {
    out << "// rvegen — generated gmsh .geo script\n"
        << "// Mesh.CharacteristicLengthMin = 0.05;\n"
        << "// Mesh.CharacteristicLengthMax = 0.1;\n\n";
  }

  void write_2d(std::ostream& out,
                shape_vector const& shapes,
                std::array<value_type, 3> const& domain_box) const {
    write_header(out);
    out << "Rectangle(1) = {0, 0, 0, "
        << domain_box[0] << ", " << domain_box[1] << ", 0};\n\n";

    // Track (entity_id, phase_name) per emitted inclusion so we can
    // later group them into gmsh Physical Surface directives.
    std::vector<std::pair<std::size_t, std::string>> tagged_entities;
    std::size_t entity_id = 2;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      std::size_t this_id = 0;
      if (auto const* c = dynamic_cast<circle<value_type> const*>(raw); c) {
        // gmsh: Circle(id) = {x, y, z, r}; via Disk surface for filled inclusion.
        out << "Disk(" << entity_id << ") = {"
            << (*c)(0) << ", " << (*c)(1) << ", 0, "
            << c->radius << ", " << c->radius << "};\n";
        this_id = entity_id++;
      } else if (auto const* r = dynamic_cast<rectangle<value_type> const*>(raw); r) {
        const auto half_w = r->width()  * value_type{0.5};
        const auto half_h = r->height() * value_type{0.5};
        out << "Rectangle(" << entity_id << ") = {"
            << (*r)(0) - half_w << ", " << (*r)(1) - half_h << ", 0, "
            << r->width() << ", " << r->height() << ", 0};\n";
        this_id = entity_id++;
      } else if (auto const* p = dynamic_cast<convex_polygon<value_type> const*>(raw); p) {
        // Custom polygon: emit N Points + N Lines + 1 Line Loop + 1
        // Plane Surface. The Plane Surface id is the entity that
        // ends up in any Physical group. Total ids consumed = 2N + 2.
        auto const& verts = p->vertices();
        const std::size_t n_verts = verts.size();
        const std::size_t first_point_id = entity_id;
        for (auto const& v : verts) {
          out << "Point(" << entity_id << ") = {"
              << v[0] << ", " << v[1] << ", 0, 1.0};\n";
          ++entity_id;
        }
        const std::size_t first_line_id = entity_id;
        for (std::size_t k = 0; k < n_verts; ++k) {
          const std::size_t p_from = first_point_id + k;
          const std::size_t p_to   = first_point_id + ((k + 1) % n_verts);
          out << "Line(" << entity_id << ") = {"
              << p_from << ", " << p_to << "};\n";
          ++entity_id;
        }
        out << "Line Loop(" << entity_id << ") = {";
        for (std::size_t k = 0; k < n_verts; ++k) {
          if (k > 0) out << ", ";
          out << (first_line_id + k);
        }
        out << "};\n";
        const std::size_t loop_id = entity_id++;
        out << "Plane Surface(" << entity_id << ") = {" << loop_id << "};\n";
        this_id = entity_id++;
      } else {
        out << "// (unsupported 2D shape skipped)\n";
        continue;
      }
      tagged_entities.emplace_back(this_id, shape->phase_name());
    }
    if (_periodic) write_periodic_2d(out, domain_box);
    if (_phases) write_physical_groups(out, tagged_entities, "Surface");
  }

  void write_3d(std::ostream& out,
                shape_vector const& shapes,
                std::array<value_type, 3> const& domain_box) const {
    write_header(out);
    out << "Box(1) = {0, 0, 0, "
        << domain_box[0] << ", " << domain_box[1] << ", " << domain_box[2]
        << "};\n\n";

    std::vector<std::pair<std::size_t, std::string>> tagged_entities;
    std::size_t entity_id = 2;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      std::size_t this_id = 0;
      if (auto const* s = dynamic_cast<sphere<value_type> const*>(raw); s) {
        out << "Sphere(" << entity_id << ") = {"
            << (*s)(0) << ", " << (*s)(1) << ", " << (*s)(2) << ", "
            << s->radius << "};\n";
        this_id = entity_id++;
      } else if (auto const* b = dynamic_cast<box<value_type> const*>(raw); b) {
        const auto hx = b->width()  * value_type{0.5};
        const auto hy = b->height() * value_type{0.5};
        const auto hz = b->depth()  * value_type{0.5};
        out << "Box(" << entity_id << ") = {"
            << (*b)(0) - hx << ", " << (*b)(1) - hy << ", " << (*b)(2) - hz << ", "
            << b->width() << ", " << b->height() << ", " << b->depth() << "};\n";
        this_id = entity_id++;
      } else {
        out << "// (unsupported 3D shape skipped)\n";
        continue;
      }
      tagged_entities.emplace_back(this_id, shape->phase_name());
    }
    if (_periodic) write_periodic_3d(out, domain_box);
    if (_phases) write_physical_groups(out, tagged_entities, "Volume");
  }

  // Group emitted inclusion entities by phase_name and emit one gmsh
  // `Physical Surface(id, "name") = {e1, e2, ...};` (2D) or
  // `Physical Volume(id, "name") = {e1, e2, ...};` (3D) directive per
  // phase that appears in the shape list. Untagged shapes (empty
  // phase_name) are skipped — they fall through to gmsh's default
  // unassigned region which downstream solvers typically treat as the
  // matrix.
  //
  // `kind` is the gmsh entity kind word ("Surface" for 2D, "Volume" for
  // 3D) so we can reuse this routine across write_2d/write_3d.
  //
  // Output ordering: `Physical` directives are emitted in alphabetical
  // order of phase_name (the `std::map` key order). Inside each group,
  // entity IDs are in the shape-vector's insertion order. Gmsh itself
  // is order-insensitive for these directives, but downstream
  // .geo-diffing tools will see name-keyed sorted output, which is the
  // deterministic invariant pinned by the regression tests.
  void write_physical_groups(
      std::ostream& out,
      std::vector<std::pair<std::size_t, std::string>> const& tagged,
      char const* kind) const {
    std::map<std::string, std::vector<std::size_t>> by_phase;
    for (auto const& [id, name] : tagged) {
      if (name.empty()) continue;
      by_phase[name].push_back(id);
    }
    if (by_phase.empty()) return;
    out << "\n// Physical groups — one per phase, keyed off shape.phase_name().\n";
    for (auto const& [name, ids] : by_phase) {
      const auto pid = _phases->id_of(name);
      out << "Physical " << kind << "(\"" << name << "\", " << pid << ") = {";
      for (std::size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) out << ", ";
        out << ids[i];
      }
      out << "};\n";
    }
  }

  // Emit gmsh's `Periodic Curve` directives pairing the four boundary
  // edges of a 2D unit cell, plus `Coherence;` to deduplicate inclusion
  // wrap-copies into a clean periodic geometry.
  //
  // Edge-numbering convention (gmsh built-in for a Rectangle entity):
  //     curve 1 = bottom (y=0)
  //     curve 2 = right  (x=Lx)
  //     curve 3 = top    (y=Ly)
  //     curve 4 = left   (x=0)
  static void write_periodic_2d(std::ostream& out,
                                 std::array<value_type, 3> const& dom) {
    out << "\n// Periodic boundary conditions — opposite edges identified.\n"
        << "Periodic Curve { 3 } = { 1 } Translate { 0, "
        << dom[1] << ", 0 };\n"
        << "Periodic Curve { 2 } = { 4 } Translate { "
        << dom[0] << ", 0, 0 };\n"
        << "\n// Boolean fragment + coherence — splits inclusions across\n"
        << "// the boundary and merges duplicate geometry from wrap copies.\n"
        << "Coherence;\n";
  }

  // Same for a 3D unit cube. Surface numbering for a Box entity in
  // gmsh ≥ 4.10:
  //     surface 1 = -x face       surface 2 = +x face
  //     surface 3 = -y face       surface 4 = +y face
  //     surface 5 = -z face       surface 6 = +z face
  static void write_periodic_3d(std::ostream& out,
                                 std::array<value_type, 3> const& dom) {
    out << "\n// Periodic boundary conditions — opposite faces identified.\n"
        << "Periodic Surface { 2 } = { 1 } Translate { "
        << dom[0] << ", 0, 0 };\n"
        << "Periodic Surface { 4 } = { 3 } Translate { 0, "
        << dom[1] << ", 0 };\n"
        << "Periodic Surface { 6 } = { 5 } Translate { 0, 0, "
        << dom[2] << " };\n"
        << "\nCoherence;\n";
  }

  std::string _output_path;
  bool _periodic{false};
  phase_collection<value_type> const* _phases{nullptr};
  bool _phase_strict{false};
};

} // namespace rvegen
