#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
#include "../intersection/aabb_overlap.h"
#include "../phase/phase.h"
#include "../shapes/box.h"
#include "../shapes/circle.h"
#include "../shapes/convex_polygon.h"
#include "../shapes/ellipse.h"
#include "../shapes/rectangle.h"
#include "../shapes/sphere.h"
#include "../types.h"

namespace rvegen {

// Writes an RVE as a gmsh .geo script for the OpenCASCADE kernel. Gmsh
// consumes this file and produces a conforming FEM mesh.
//
// Output structure (2D shown; 3D uses Box / Sphere / Volume):
//   1. `SetFactory("OpenCASCADE");` + optional Mesh.* directives.
//   2. `Rectangle(1)` for the domain, then one entity per inclusion with
//      explicit tags 2, 3, ... (Disk / Rectangle / Plane Surface).
//   3. Inclusions whose bounding box leaves the domain (periodic wrap
//      copies, random-generator overhang) are clipped to the domain with
//      `BooleanIntersection`; the result is captured in `_inc_<id>()`.
//   4. `BooleanFragments` of the domain against all inclusions — the
//      result is one conforming geometry: the matrix (a new tag) plus
//      the inclusions, which keep their tags because they are disjoint
//      and fully inside the domain after step 3.
//   5. periodic=true: every boundary curve (2D) / face (3D) on the low
//      side of each axis is paired with its translated image on the
//      high side by bounding-box matching (the pattern from gmsh's
//      tutorial t18), *after* fragmenting so the constraints reference
//      the final entities.
//   6. With a phase_collection attached: one Physical group per phase
//      (alphabetical), plus a "matrix" group holding everything that is
//      not a tagged inclusion. gmsh only saves elements that belong to a
//      Physical group once any group exists, so the matrix group is what
//      keeps the matrix in the .msh.
//
// All numbers are written in their shortest round-trip decimal form, so
// the geometry gmsh sees is exactly the geometry rvegen generated.
//
// Supported shapes: circle, rectangle, ellipse, convex_polygon (2D);
// sphere, box (3D). Any other shape makes write() throw before anything
// is emitted — dropping an inclusion silently would produce a wrong RVE.
//
// Gmsh version requirement: ≥ 4.0 (OpenCASCADE kernel, string-named
// Physical groups). Tested against 4.12.
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
                      : false},
        _char_length_min{handler.contains("char_length_min")
                             ? handler.template get<value_type>("char_length_min")
                             : value_type{0}},
        _char_length_max{handler.contains("char_length_max")
                             ? handler.template get<value_type>("char_length_max")
                             : value_type{0}},
        _element_order{handler.contains("element_order")
                           ? handler.template get<int>("element_order")
                           : 0},
        _algorithm{handler.contains("algorithm")
                       ? handler.template get<int>("algorithm")
                       : 0},
        _recombine_quads{handler.contains("recombine_quads")
                             ? handler.template get<bool>("recombine_quads")
                             : false} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("output_path").template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"destination path for the gmsh .geo script">>();
    // Optional — default false. Schema does NOT mark it required so existing
    // configs keep working unchanged.
    s.template insert<bool>("periodic")
        .template add<numsim_core::description_label<"emit Periodic Curve/Surface constraints pairing opposite domain faces; for FE-RVE workflows that need matching nodes on opposite faces (default false)">>();
    // Optional mesh-size knobs. When > 0, the writer emits
    // `Mesh.CharacteristicLengthMin/Max = X;` directives so the
    // generated .geo runs through gmsh with the requested element
    // size without manual editing. Default 0 keeps the original
    // commented-placeholder behaviour.
    s.template insert<value_type>("char_length_min")
        .template add<numsim_core::description_label<"if > 0, emits Mesh.CharacteristicLengthMin = <value> in the .geo header (controls minimum element size). Default 0 = no directive emitted.">>();
    s.template insert<value_type>("char_length_max")
        .template add<numsim_core::description_label<"if > 0, emits Mesh.CharacteristicLengthMax = <value> in the .geo header (controls maximum element size). Default 0 = no directive emitted.">>();
    s.template insert<int>("element_order")
        .template add<numsim_core::description_label<"if > 0, emits Mesh.ElementOrder = <value> in the .geo header — 1 for linear elements (gmsh default), 2 for quadratic (better accuracy on elliptic FE problems). Default 0 = no directive emitted.">>()
        .template add<numsim_core::range<0, 5>>();
    s.template insert<int>("algorithm")
        .template add<numsim_core::description_label<"if > 0, emits Mesh.Algorithm = <value> in the .geo header — gmsh 2D meshing algorithm choice (1=MeshAdapt, 2=Auto, 3=Initial only, 5=Delaunay, 6=Frontal-Delaunay, 7=BAMG, 8=Frontal-Delaunay for quads, 9=Packing of parallelograms). Default 0 = no directive emitted (gmsh defaults to 6/Frontal-Delaunay). Note: gmsh 4.12 segfaults with algorithm 8 when periodic constraints are present; with periodic=true use the default plus recombine_quads to get quads.">>()
        .template add<numsim_core::range<0, 10>>();
    s.template insert<bool>("recombine_quads")
        .template add<numsim_core::description_label<"if true, emits Mesh.RecombineAll = 1 in the header — gmsh recombines the triangular mesh into quadrilateral / hexahedral elements. Useful for FE-RVE workflows that need structured-style elements. Default false.">>();
    return s;
  }

  [[nodiscard]] bool periodic() const noexcept { return _periodic; }
  void set_periodic(bool v) noexcept { _periodic = v; }

  // Optional gmsh mesh-size knobs. Setting either or both to a
  // positive value makes the writer emit `Mesh.CharacteristicLengthMin
  // = X;` and/or `Mesh.CharacteristicLengthMax = Y;` in the .geo
  // header — so the FE mesher uses the requested size without a
  // manual edit. Pass 0 (or leave default) to skip the directive.
  void set_characteristic_length_min(value_type v) noexcept {
    _char_length_min = v;
  }
  void set_characteristic_length_max(value_type v) noexcept {
    _char_length_max = v;
  }
  [[nodiscard]] value_type characteristic_length_min() const noexcept {
    return _char_length_min;
  }
  [[nodiscard]] value_type characteristic_length_max() const noexcept {
    return _char_length_max;
  }

  // Optional gmsh element-order knob. 0 (default) → no directive
  // emitted, gmsh uses its built-in default (linear). 1 → linear
  // elements (triangles / tetrahedra). 2 → quadratic elements
  // (6-node triangles / 10-node tetrahedra), which converge faster
  // on elliptic FE problems at the cost of more DOFs per element.
  void set_element_order(int order) noexcept { _element_order = order; }
  [[nodiscard]] int element_order() const noexcept { return _element_order; }

  // Gmsh 2D meshing-algorithm choice. 0 (default) → no directive
  // emitted, gmsh uses Frontal-Delaunay (algorithm 6). Other useful
  // values: 8 (Frontal-Delaunay-for-quads, pairs well with
  // recombine_quads), 9 (Packing of Parallelograms). See gmsh manual
  // §7.4.1 for the full list.
  void set_algorithm(int algorithm) noexcept { _algorithm = algorithm; }
  [[nodiscard]] int algorithm() const noexcept { return _algorithm; }

  // Recombine triangle meshes into quadrilateral elements (or tet
  // meshes into hexahedra for 3D). Useful for FE-RVE workflows that
  // need structured-style element shapes — e.g. some FFT solvers
  // expect quad/hex meshes.
  void set_recombine_quads(bool recombine) noexcept {
    _recombine_quads = recombine;
  }
  [[nodiscard]] bool recombine_quads() const noexcept { return _recombine_quads; }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  // Attach a phase_collection so each emitted inclusion entity is added
  // to a gmsh Physical Surface (2D) / Physical Volume (3D) named after
  // its `phase_name()`. The numeric tag is the collection's `id_of(name)`.
  //
  // Everything that is not a tagged inclusion — the matrix region plus
  // any untagged inclusion — goes into a "matrix" group. If the
  // collection declares a phase named "matrix" that group uses its id;
  // otherwise gmsh assigns the next free tag.
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
  // emitting geometry. Default (strict==false) puts untagged shapes in
  // the "matrix" group, which is the intended fallback for "treat as
  // matrix" but also masks the bug-of-omission when a user *meant* to
  // tag a shape and forgot. Use strict mode when your generation
  // pipeline guarantees every shape gets tagged.
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
    const bool is_3d = box[2] > value_type{0};

    // Validate everything BEFORE writing anything, so a throw never
    // leaves a partial .geo behind.
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      if (!is_supported(*shapes[i], is_3d)) {
        throw std::runtime_error{
            "gmsh_geo_writer: shape #" + std::to_string(i) +
            " is not supported in a " + (is_3d ? "3D" : "2D") +
            " domain (supported: " +
            (is_3d ? "sphere, box" : "circle, rectangle, ellipse, convex_polygon") +
            ")"};
      }
    }
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

    write_header(out);
    const auto inclusions = is_3d ? write_entities_3d(out, shapes, box)
                                  : write_entities_2d(out, shapes, box);
    const char* kind = is_3d ? "Volume" : "Surface";
    write_fragments(out, inclusions, kind);
    if (_periodic) write_periodic(out, box, is_3d);
    if (_phases) write_physical_groups(out, inclusions, kind);
  }

private:
  // Shortest decimal form that round-trips to the same value
  // (std::to_chars without a precision) — `0.05` stays `0.05`, while
  // values that need 17 digits get them. Independent of the stream's
  // precision / locale settings.
  static std::string num(value_type v) {
    std::array<char, 64> buf{};
    auto const res = std::to_chars(buf.data(), buf.data() + buf.size(), v);
    return std::string(buf.data(), res.ptr);
  }

  // One emitted inclusion: `ref` is the .geo expression for its final
  // entity tag(s) — a plain id ("5") or a clip-result list ("_inc_5()").
  struct inclusion_ref {
    std::string ref;
    std::string phase_name;
  };

  static bool is_supported(shape_base<value_type> const& s, bool is_3d) {
    if (is_3d) {
      return dynamic_cast<sphere<value_type> const*>(&s) ||
             dynamic_cast<box<value_type> const*>(&s);
    }
    return dynamic_cast<circle<value_type> const*>(&s) ||
           dynamic_cast<rectangle<value_type> const*>(&s) ||
           dynamic_cast<ellipse<value_type> const*>(&s) ||
           dynamic_cast<convex_polygon<value_type> const*>(&s);
  }

  // A shape needs clipping when it is not contained in the domain. Uses
  // the bounding box when one has been computed, otherwise the centre ±
  // max_expansion() extent. Over-estimating is harmless (intersecting a
  // shape that is already inside is a no-op in gmsh); under-estimating
  // would leave geometry outside the RVE.
  static bool needs_clip(shape_base<value_type> const& s,
                         std::array<value_type, 3> const& box) {
    if (auto const* bb = s.bounding_box(); bb != nullptr) {
      return !aabb_inside(*bb, box[0], box[1], box[2]);
    }
    const auto mid = s.get_middle_point();
    const auto ext = s.max_expansion();
    for (std::size_t k = 0; k < 3; ++k) {
      if (mid[k] - ext[k] < value_type{0} || mid[k] + ext[k] > box[k]) {
        return true;
      }
    }
    return false;
  }

  void write_header(std::ostream& out) const {
    out << "// rvegen — generated gmsh .geo script\n"
        << "SetFactory(\"OpenCASCADE\");\n";
    if (_periodic) {
      // Tight (triangulation-based) bounding boxes; OCC's default boxes
      // are inflated for curved entities, which breaks the bounding-box
      // matching of periodic boundary entities below.
      out << "Geometry.OCCBoundsUseStl = 1;\n";
    }
    // Emit active mesh-size directives (positive value → uncommented
    // directive; otherwise keep the placeholder so users still see
    // the syntax they need to enable manually).
    if (_char_length_min > value_type{0}) {
      out << "Mesh.CharacteristicLengthMin = " << num(_char_length_min) << ";\n";
    } else {
      out << "// Mesh.CharacteristicLengthMin = 0.05;\n";
    }
    if (_char_length_max > value_type{0}) {
      out << "Mesh.CharacteristicLengthMax = " << num(_char_length_max) << ";\n";
    } else {
      out << "// Mesh.CharacteristicLengthMax = 0.1;\n";
    }
    if (_element_order > 0) {
      out << "Mesh.ElementOrder = " << _element_order << ";\n";
    }
    if (_algorithm > 0) {
      if (_periodic && _algorithm == 8) {
        out << "// WARNING: gmsh 4.12 crashes meshing Algorithm 8 with periodic\n"
               "// constraints; drop this line if gmsh segfaults.\n";
      }
      out << "Mesh.Algorithm = " << _algorithm << ";\n";
    }
    if (_recombine_quads) {
      out << "Mesh.RecombineAll = 1;\n";
    }
    out << "\n";
  }

  std::vector<inclusion_ref> write_entities_2d(
      std::ostream& out, shape_vector const& shapes,
      std::array<value_type, 3> const& domain_box) const {
    out << "Rectangle(1) = {0, 0, 0, "
        << num(domain_box[0]) << ", " << num(domain_box[1]) << ", 0};\n\n";

    std::vector<std::size_t> ids;
    std::size_t entity_id = 2;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      if (auto const* c = dynamic_cast<circle<value_type> const*>(raw); c) {
        out << "Disk(" << entity_id << ") = {"
            << num((*c)(0)) << ", " << num((*c)(1)) << ", 0, "
            << num(c->radius) << ", " << num(c->radius) << "};\n";
      } else if (auto const* r = dynamic_cast<rectangle<value_type> const*>(raw); r) {
        const auto half_w = r->width()  * value_type{0.5};
        const auto half_h = r->height() * value_type{0.5};
        out << "Rectangle(" << entity_id << ") = {"
            << num((*r)(0) - half_w) << ", " << num((*r)(1) - half_h) << ", 0, "
            << num(r->width()) << ", " << num(r->height()) << ", 0};\n";
      } else if (auto const* e = dynamic_cast<ellipse<value_type> const*>(raw); e) {
        write_ellipse(out, *e, entity_id);
      } else if (auto const* p = dynamic_cast<convex_polygon<value_type> const*>(raw); p) {
        write_polygon(out, *p, entity_id);
      }
      ids.push_back(entity_id++);
    }
    return write_clips(out, shapes, ids, domain_box, "Surface");
  }

  std::vector<inclusion_ref> write_entities_3d(
      std::ostream& out, shape_vector const& shapes,
      std::array<value_type, 3> const& domain_box) const {
    out << "Box(1) = {0, 0, 0, "
        << num(domain_box[0]) << ", " << num(domain_box[1]) << ", " << num(domain_box[2])
        << "};\n\n";

    std::vector<std::size_t> ids;
    std::size_t entity_id = 2;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      if (auto const* s = dynamic_cast<sphere<value_type> const*>(raw); s) {
        out << "Sphere(" << entity_id << ") = {"
            << num((*s)(0)) << ", " << num((*s)(1)) << ", " << num((*s)(2)) << ", "
            << num(s->radius) << "};\n";
      } else if (auto const* b = dynamic_cast<box<value_type> const*>(raw); b) {
        const auto hx = b->width()  * value_type{0.5};
        const auto hy = b->height() * value_type{0.5};
        const auto hz = b->depth()  * value_type{0.5};
        out << "Box(" << entity_id << ") = {"
            << num((*b)(0) - hx) << ", " << num((*b)(1) - hy) << ", " << num((*b)(2) - hz) << ", "
            << num(b->width()) << ", " << num(b->height()) << ", " << num(b->depth()) << "};\n";
      }
      ids.push_back(entity_id++);
    }
    return write_clips(out, shapes, ids, domain_box, "Volume");
  }

  // gmsh's OCC Disk needs rx >= ry; swap the semi-axes and rotate a
  // further 90° when the ellipse's first axis is the minor one.
  static void write_ellipse(std::ostream& out, ellipse<value_type> const& e,
                            std::size_t id) {
    value_type rx = e.extent[0];
    value_type ry = e.extent[1];
    value_type angle = std::atan2(e.axis[0][1], e.axis[0][0]);
    if (rx < ry) {
      std::swap(rx, ry);
      angle += std::numbers::pi_v<value_type> / value_type{2};
    }
    out << "Disk(" << id << ") = {"
        << num(e.center[0]) << ", " << num(e.center[1]) << ", 0, "
        << num(rx) << ", " << num(ry) << "};\n";
    if (angle != value_type{0}) {
      out << "Rotate {{0, 0, 1}, {" << num(e.center[0]) << ", " << num(e.center[1])
          << ", 0}, " << num(angle) << "} { Surface{" << id << "}; }\n";
    }
  }

  // Points / lines / loop get fresh tags from gmsh (newp / newl / newll)
  // so they never collide with the tags OCC auto-assigns to the
  // boundaries of Rectangle / Disk entities. Only the Plane Surface uses
  // the explicit inclusion id.
  static void write_polygon(std::ostream& out, convex_polygon<value_type> const& p,
                            std::size_t id) {
    auto const& verts = p.vertices();
    const std::size_t n = verts.size();
    for (std::size_t k = 0; k < n; ++k) {
      out << "_p" << k << " = newp; Point(_p" << k << ") = {"
          << num(verts[k][0]) << ", " << num(verts[k][1]) << ", 0};\n";
    }
    for (std::size_t k = 0; k < n; ++k) {
      out << "_l" << k << " = newl; Line(_l" << k << ") = {_p" << k
          << ", _p" << ((k + 1) % n) << "};\n";
    }
    out << "_ll = newll; Line Loop(_ll) = {";
    for (std::size_t k = 0; k < n; ++k) {
      if (k > 0) out << ", ";
      out << "_l" << k;
    }
    out << "};\n"
        << "Plane Surface(" << id << ") = {_ll};\n";
  }

  // Clip every inclusion that leaves the domain. Runs after all
  // primitives are emitted so the tags of new boolean results can never
  // collide with a later explicit inclusion id.
  std::vector<inclusion_ref> write_clips(
      std::ostream& out, shape_vector const& shapes,
      std::vector<std::size_t> const& ids,
      std::array<value_type, 3> const& domain_box, char const* kind) const {
    std::vector<inclusion_ref> refs;
    refs.reserve(shapes.size());
    bool header_done = false;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      const auto id = std::to_string(ids[i]);
      if (needs_clip(*shapes[i], domain_box)) {
        if (!header_done) {
          out << "\n// Clip inclusions that leave the domain (periodic wrap copies,\n"
                 "// overhanging shapes) so the mesh stays inside the RVE.\n";
          header_done = true;
        }
        out << "_inc_" << id << "() = BooleanIntersection{ " << kind << "{"
            << id << "}; Delete; }{ " << kind << "{1}; };\n";
        refs.push_back({"_inc_" + id + "()", shapes[i]->phase_name()});
      } else {
        refs.push_back({id, shapes[i]->phase_name()});
      }
    }
    return refs;
  }

  static void write_ref_list(std::ostream& out,
                             std::vector<std::string> const& refs) {
    out << "{";
    for (std::size_t i = 0; i < refs.size(); ++i) {
      if (i > 0) out << ", ";
      out << refs[i];
    }
    out << "}";
  }

  // Fragment the domain against all inclusions → one conforming
  // geometry. The inclusions are disjoint and inside the domain, so OCC
  // keeps their tags; the matrix becomes a new entity.
  static void write_fragments(std::ostream& out,
                              std::vector<inclusion_ref> const& inclusions,
                              char const* kind) {
    std::vector<std::string> refs;
    refs.reserve(inclusions.size());
    for (auto const& inc : inclusions) refs.push_back(inc.ref);
    out << "\n// Fragment the domain against the inclusions: one conforming\n"
           "// geometry with shared interfaces (matrix + inclusions).\n"
        << "_inc() = ";
    write_ref_list(out, refs);
    out << ";\n"
        << "If (#_inc() > 0)\n"
        << "  BooleanFragments{ " << kind << "{1}; Delete; }{ " << kind
        << "{_inc()}; Delete; }\n"
        << "EndIf\n";
  }

  // Pair every boundary curve (2D) / face (3D) on the low side of each
  // axis with its translated image on the high side, matching by
  // bounding box (gmsh tutorial t18). Emitted after fragmenting so the
  // constraints reference the final, split boundary entities.
  static void write_periodic(std::ostream& out,
                             std::array<value_type, 3> const& dom,
                             bool is_3d) {
    const char* ent = is_3d ? "Surface" : "Curve";
    const int n_axes = is_3d ? 3 : 2;
    const value_type lmax = std::max({dom[0], dom[1], dom[2]});
    out << "\n// Periodic boundary conditions: pair each low-side boundary "
        << (is_3d ? "face" : "curve") << "\n"
        << "// with its translated image on the opposite side.\n"
        << "_pe = " << num(lmax * value_type{1e-6}) << ";\n";
    for (int a = 0; a < n_axes; ++a) {
      std::array<value_type, 3> shift{value_type{0}, value_type{0}, value_type{0}};
      shift[a] = dom[a];
      // Thin slab around the low face of axis `a`, spanning the domain
      // on the other axes.
      out << "_lo() = " << ent << " In BoundingBox{-_pe, -_pe, -_pe, ";
      for (int k = 0; k < 3; ++k) {
        if (k > 0) out << ", ";
        if (k == a || (!is_3d && k == 2)) out << "_pe";
        else out << num(dom[k]) << " + _pe";
      }
      out << "};\n"
          << "For _i In {0:#_lo()-1}\n"
          << "  _b() = BoundingBox " << ent << "{_lo(_i)};\n"
          << "  _hi() = " << ent << " In BoundingBox{";
      for (int m = 0; m < 6; ++m) {
        if (m > 0) out << ", ";
        out << "_b(" << m << ") " << (m < 3 ? "- " : "+ ") << "_pe + "
            << num(shift[m % 3]);
      }
      out << "};\n"
          << "  For _j In {0:#_hi()-1}\n"
          << "    _c() = BoundingBox " << ent << "{_hi(_j)};\n"
          << "    If (";
      for (int m = 0; m < 6; ++m) {
        if (m > 0) out << " && ";
        out << "Fabs(_c(" << m << ") - " << num(shift[m % 3]) << " - _b(" << m
            << ")) < _pe";
      }
      out << ")\n"
          << "      Periodic " << ent << "{_hi(_j)} = {_lo(_i)} Translate{"
          << num(shift[0]) << ", " << num(shift[1]) << ", " << num(shift[2]) << "};\n"
          << "    EndIf\n"
          << "  EndFor\n"
          << "EndFor\n";
    }
  }

  // One `Physical <kind>("name", id)` per phase that appears on a shape,
  // in alphabetical order of phase_name (std::map key order), plus the
  // "matrix" group: every entity that is not a tagged inclusion
  // (matrix region + untagged inclusions). Inside each group, entities
  // are in shape-vector insertion order.
  void write_physical_groups(std::ostream& out,
                             std::vector<inclusion_ref> const& inclusions,
                             char const* kind) const {
    static constexpr char const* matrix_name = "matrix";
    std::map<std::string, std::vector<std::string>> by_phase;
    std::vector<std::string> tagged;
    for (auto const& inc : inclusions) {
      if (inc.phase_name.empty()) continue;
      by_phase[inc.phase_name].push_back(inc.ref);
      tagged.push_back(inc.ref);
    }

    out << "\n// Physical groups — one per phase, keyed off shape.phase_name(),\n"
           "// plus \"matrix\" for everything that is not a tagged inclusion.\n"
        << "_matrix() = " << kind << "{:};\n";
    if (!tagged.empty()) {
      out << "_matrix() -= ";
      write_ref_list(out, tagged);
      out << ";\n";
    }
    by_phase[matrix_name].push_back("_matrix()");

    const bool matrix_declared = _phases->contains(matrix_name);
    for (auto const& [name, refs] : by_phase) {
      // An undeclared "matrix" phase gets gmsh's next free tag; emit it
      // after the declared ids so auto-assignment can't collide.
      if (name == matrix_name && !matrix_declared) continue;
      out << "Physical " << kind << "(\"" << name << "\", "
          << _phases->id_of(name) << ") = ";
      write_ref_list(out, refs);
      out << ";\n";
    }
    if (!matrix_declared) {
      out << "Physical " << kind << "(\"" << matrix_name << "\") = ";
      write_ref_list(out, by_phase[matrix_name]);
      out << ";\n";
    }
  }

  std::string _output_path;
  bool _periodic{false};
  phase_collection<value_type> const* _phases{nullptr};
  bool _phase_strict{false};
  value_type _char_length_min{0};
  value_type _char_length_max{0};
  int _element_order{0};
  int _algorithm{0};
  bool _recombine_quads{false};
};

} // namespace rvegen
