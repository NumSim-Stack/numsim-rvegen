#pragma once

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <map>
#include <memory>
#include <numbers>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <typeinfo>
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
//   0. Scale: when the largest box extent L is not 1, every coordinate,
//      radius and mesh size is written divided by L (OpenCASCADE's
//      absolute tolerances make RVEs of 1e-5 or 1e5 units lose or fail
//      their boolean clips) and `Mesh.ScalingFactor = L` makes gmsh save
//      the mesh in the original units. Unit boxes are written unchanged.
//   3. Inclusions whose bounding box leaves the domain (periodic wrap
//      copies, random-generator overhang) are clipped to the domain with
//      `BooleanIntersection`; the result is captured in `_inc_<id>()`.
//      The result list is filtered: gmsh 4.14.0 answers an empty
//      intersection with the tool's tag (the domain), which is dropped.
//      Circles / spheres that provably miss the domain (centre-to-box
//      distance >= r, typical for periodic wrap images) are not emitted.
//      A shape that clearly overlaps the domain but clips to nothing
//      makes the script call gmsh `Error(...)`. With periodic=true a
//      circle / sphere within 1e-5 L of touching a periodic plane is
//      written 1e-5 L short of it (a one-sided point contact cannot be
//      paired across the periodic faces).
//   3b. First shape wins (like voxel_grid): an inclusion whose bounding
//      box overlaps an earlier one becomes `BooleanDifference` of itself
//      minus those earlier inclusions, so the final pieces are disjoint
//      and every piece belongs to exactly one phase. Overlapping /
//      nested / full-domain inputs therefore get the right phases;
//      shapes whose boxes do not overlap an earlier one stay untouched.
//      With periodic=true the wrap copies of one particle share the
//      priority of its earliest copy, so the result does not depend on
//      where the copies sit in the list, and spheres cut by a periodic
//      face get their poles moved off planes that pass near them (gmsh
//      cannot mesh such caps consistently on both sides).
//      The pieces of every inclusion are tracked in `_inc_<id>()` lists
//      (an inclusion that becomes empty is simply an empty list).
//   4. `BooleanFragments` of the domain against all inclusions — the
//      result is one conforming geometry: the matrix (a new tag) plus
//      the (now disjoint, inside-the-domain) inclusion pieces, which
//      keep their tags.
//   5. periodic=true: every boundary curve (2D) / face (3D) on the low
//      side of each axis is paired with its translated image on the
//      high side by bounding-box matching (the pattern from gmsh's
//      tutorial t18), *after* fragmenting so the constraints reference
//      the final entities. A low-side entity without a partner (e.g. a
//      boundary-cut inclusion with no image on the opposite face) makes
//      the script call gmsh `Error(...)` instead of silently producing a
//      non-periodic mesh.
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
// Physical groups). Tested against 4.12 and 4.14.
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
    // Build the whole script first: a rejected shape must not truncate
    // an existing file.
    std::ostringstream buf;
    write(buf, shapes, domain_box);
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "gmsh_geo_writer: cannot open '" + _output_path + "' for writing"};
    }
    out << buf.str();
  }

  // Stream-based emit. Kept public so tests and embedders can capture the
  // output to a stringstream without round-tripping through a file.
  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& box) const {
    const bool is_3d = box[2] > value_type{0};

    for (auto const v : box) require_finite(v, "domain size");
    if (!(box[0] > value_type{0}) || !(box[1] > value_type{0}) || box[2] < value_type{0}) {
      throw std::runtime_error{
          "gmsh_geo_writer: domain size must be > 0 in x and y and >= 0 in z"};
    }
    for (auto const v : {_char_length_min, _char_length_max}) {
      require_finite(v, "mesh size");
    }

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

    // Emit into a local buffer (num() rejects NaN/inf by throwing), so
    // `out` only ever receives a complete script.
    set_length_scale(box);
    std::ostringstream buf;
    write_header(buf);
    // Degenerate polygons (empty / collapsed / zero-area Voronoi cells)
    // would produce an invalid `Line Loop{}`; drop them before any entity
    // or clip id is assigned so ids stay index-aligned with `drawable`.
    shape_vector drawable;
    drawable.reserve(shapes.size());
    for (auto const& s : shapes) {
      if (!is_3d) {
        if (auto const* p = dynamic_cast<convex_polygon<value_type> const*>(s.get()); p) {
          // Clip to the domain: periodic wrap images may poke out of it,
          // and a polygon with < 3 clipped vertices (empty / collapsed /
          // sliver) would give an invalid `Line Loop{}`.
          if (clipped_vertices(*p, box).size() < 3) {
            buf << "// (degenerate polygon skipped)\n";
            continue;
          }
        }
      }
      drawable.push_back(s->clone());
    }
    const auto inclusions = is_3d ? write_entities_3d(buf, drawable, box)
                                  : write_entities_2d(buf, drawable, box);
    const char* kind = is_3d ? "Volume" : "Surface";
    write_fragments(buf, inclusions, kind);
    if (_periodic) write_periodic(buf, box, is_3d);
    if (_phases) write_physical_groups(buf, inclusions, kind);
    out << buf.str();
  }

private:
  // OpenCASCADE booleans work with absolute tolerances (~1e-7), so an RVE
  // that is tiny or huge in absolute units loses or fails its clips. The
  // script therefore describes the geometry in box-relative units: every
  // coordinate, radius and mesh size is divided by the largest box extent
  // L, and `Mesh.ScalingFactor = L` makes gmsh save the mesh in the
  // original units. Boxes with L == 1 are written unchanged.
  // Set at the start of every write() (the writer is not thread-safe for
  // concurrent write() calls on one object).
  void set_length_scale(std::array<value_type, 3> const& box) const {
    _length_scale = std::max({box[0], box[1], box[2]});
  }

  // Shortest decimal form that round-trips to the same value
  // (std::to_chars without a precision) — `0.05` stays `0.05`, while
  // values that need 17 digits get them. Independent of the stream's
  // precision / locale settings.
  static void require_finite(value_type v, char const* what) {
    if (!std::isfinite(v)) {
      throw std::runtime_error{std::string{"gmsh_geo_writer: non-finite "} +
                               what + " (NaN or inf)"};
    }
  }

  static std::string raw_num(value_type v) {
    require_finite(v, "coordinate or size");
    std::array<char, 64> buf{};
    auto const res = std::to_chars(buf.data(), buf.data() + buf.size(), v);
    return std::string(buf.data(), res.ptr);
  }

  // A length (coordinate, radius, mesh size) in the units of the script:
  // divided by the characteristic length of the current write() call, which
  // is 1 for unit-sized boxes (output unchanged).
  std::string num(value_type v) const {
    require_finite(v, "coordinate or size");
    return raw_num(_length_scale == value_type{1} ? v : v / _length_scale);
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

  // Plain extent box so aabb_overlap() also works for shapes that have no
  // bounding box computed (centre ± max_expansion()).
  struct extent_box final : bounding_box_base<value_type> {
    std::array<value_type, 3> top, bottom;
    [[nodiscard]] std::array<value_type, 3> const& top_point() const override { return top; }
    [[nodiscard]] std::array<value_type, 3> const& bottom_point() const override { return bottom; }
    [[nodiscard]] std::unique_ptr<bounding_box_base<value_type>> clone() const override {
      return std::make_unique<extent_box>(*this);
    }
  };

  static extent_box extent_of(shape_base<value_type> const& s) {
    extent_box e;
    if (auto const* bb = s.bounding_box(); bb != nullptr) {
      e.top = bb->top_point();
      e.bottom = bb->bottom_point();
    } else {
      const auto mid = s.get_middle_point();
      const auto ext = s.max_expansion();
      for (std::size_t k = 0; k < 3; ++k) {
        e.top[k] = mid[k] + ext[k];
        e.bottom[k] = mid[k] - ext[k];
      }
    }
    return e;
  }

  // Distance from `p` to the domain box (0 when inside), over `dim` axes.
  static value_type distance_to_box(std::array<value_type, 3> const& p,
                                    std::array<value_type, 3> const& box,
                                    std::size_t dim) {
    value_type d2{0};
    for (std::size_t k = 0; k < dim; ++k) {
      const value_type d = std::max({value_type{0}, -p[k], p[k] - box[k]});
      d2 += d * d;
    }
    return std::sqrt(d2);
  }

  // Radius written for a circle / sphere. In a periodic RVE a circle that
  // touches a periodic plane (|r - distance of the centre to the nearest
  // lattice plane| below 1e-5 L, from either side) leaves a one-sided
  // point contact: the plane carries a split vertex on the low face but
  // not on the high face, and gmsh cannot pair the two faces. Such a
  // circle is written 1e-5 L short of the plane instead. The decision only
  // depends on the centre modulo the box, so a shape and its periodic
  // images are treated identically. Everything else keeps its radius.
  value_type emitted_radius(std::array<value_type, 3> const& centre,
                            value_type r,
                            std::array<value_type, 3> const& box) const {
    if (!_periodic) return r;
    const value_type gap = value_type{1e-5} * _length_scale;
    if (!(r > value_type{10} * gap)) return r;
    value_type out = r;
    for (std::size_t k = 0; k < dim_of(box); ++k) {
      const value_type t = centre[k] / box[k];
      const value_type d = std::abs((t - std::round(t)) * box[k]);
      if (std::abs(r - d) < gap) out = std::min(out, d - gap);
    }
    return out;
  }

  static std::size_t dim_of(std::array<value_type, 3> const& box) {
    return box[2] > value_type{0} ? 3 : 2;
  }

  // True when the shape provably has no volume inside the domain: it is
  // not emitted at all (exact and cheap, and it keeps OCC away from
  // zero-measure contacts). Circle / sphere: centre-to-box distance >= r;
  // every shape: the extent box lies outside / only touches the domain.
  static bool misses_domain(shape_base<value_type> const& s,
                            std::array<value_type, 3> const& box) {
    const std::size_t dim = dim_of(box);
    if (auto const* c = dynamic_cast<circle<value_type> const*>(&s)) {
      if (distance_to_box(c->get_middle_point(), box, dim) >= c->radius) return true;
    } else if (auto const* sp = dynamic_cast<sphere<value_type> const*>(&s)) {
      if (distance_to_box(sp->get_middle_point(), box, dim) >= sp->radius) return true;
    }
    const auto e = extent_of(s);
    for (std::size_t k = 0; k < dim; ++k) {
      if (e.top[k] <= value_type{0} || e.bottom[k] >= box[k]) return true;
    }
    return false;
  }

  // True when the shape clearly overlaps the domain (by more than a small
  // fraction of the box), so an empty clip result cannot be "vanished" but
  // is a failed boolean. Conservative: only cases that are exact for the
  // shape — centre well inside the domain (every supported shape contains
  // its centre), circle / sphere penetration depth, rectangle / box extent
  // overlap. Anything else is not claimed to overlap.
  static bool clearly_overlaps(shape_base<value_type> const& s,
                               std::array<value_type, 3> const& box) {
    const std::size_t dim = dim_of(box);
    const value_type margin =
        value_type{1e-4} * std::max({box[0], box[1], box[2]});
    const auto mid = s.get_middle_point();
    bool centre_inside = true;
    for (std::size_t k = 0; k < dim; ++k) {
      if (!(mid[k] > margin && mid[k] < box[k] - margin)) centre_inside = false;
    }
    if (centre_inside) return true;
    if (auto const* c = dynamic_cast<circle<value_type> const*>(&s)) {
      return distance_to_box(mid, box, dim) < c->radius - margin;
    }
    if (auto const* sp = dynamic_cast<sphere<value_type> const*>(&s)) {
      return distance_to_box(mid, box, dim) < sp->radius - margin;
    }
    if (dynamic_cast<rectangle<value_type> const*>(&s) ||
        dynamic_cast<rvegen::box<value_type> const*>(&s)) {
      const auto e = extent_of(s);
      for (std::size_t k = 0; k < dim; ++k) {
        if (!(e.top[k] > margin && e.bottom[k] < box[k] - margin)) return false;
      }
      return true;
    }
    return false;
  }

  void write_header(std::ostream& out) const {
    out << "// rvegen — generated gmsh .geo script\n";
    if (_length_scale != value_type{1}) {
      out << "// Lengths are in box-relative units (divided by the largest box\n"
             "// extent L = " << raw_num(_length_scale) << "); Mesh.ScalingFactor = L below makes gmsh\n"
             "// save the mesh in the original units. This keeps OpenCASCADE's\n"
             "// absolute tolerances independent of the RVE size.\n";
    }
    out << "SetFactory(\"OpenCASCADE\");\n";
    if (_length_scale != value_type{1}) {
      out << "Mesh.ScalingFactor = " << raw_num(_length_scale) << ";\n";
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
      if (misses_domain(*raw, domain_box)) {
        ids.push_back(entity_id++);  // keeps tags stable; nothing emitted
        continue;
      }
      if (auto const* c = dynamic_cast<circle<value_type> const*>(raw); c) {
        out << "Disk(" << entity_id << ") = {"
            << num((*c)(0)) << ", " << num((*c)(1)) << ", 0, "
            << num(emitted_radius(c->get_middle_point(), c->radius, domain_box)) << ", "
            << num(emitted_radius(c->get_middle_point(), c->radius, domain_box)) << "};\n";
      } else if (auto const* r = dynamic_cast<rectangle<value_type> const*>(raw); r) {
        const auto half_w = r->width()  * value_type{0.5};
        const auto half_h = r->height() * value_type{0.5};
        out << "Rectangle(" << entity_id << ") = {"
            << num((*r)(0) - half_w) << ", " << num((*r)(1) - half_h) << ", 0, "
            << num(r->width()) << ", " << num(r->height()) << ", 0};\n";
      } else if (auto const* e = dynamic_cast<ellipse<value_type> const*>(raw); e) {
        write_ellipse(out, *e, entity_id);
      } else if (auto const* p = dynamic_cast<convex_polygon<value_type> const*>(raw); p) {
        write_polygon(out, clipped_vertices(*p, domain_box), entity_id);
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
      if (misses_domain(*raw, domain_box)) {
        ids.push_back(entity_id++);  // keeps tags stable; nothing emitted
        continue;
      }
      if (auto const* s = dynamic_cast<sphere<value_type> const*>(raw); s) {
        out << "Sphere(" << entity_id << ") = {"
            << num((*s)(0)) << ", " << num((*s)(1)) << ", " << num((*s)(2)) << ", "
            << num(emitted_radius(s->get_middle_point(), s->radius, domain_box))
            << "};\n";
        if (_periodic) write_pole_rotation(out, *s, entity_id, domain_box);
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

  // gmsh's OCC sphere has its poles on +-z. A sphere cut by a periodic
  // face whose plane passes close to a pole (|offset| << r) gets a cap
  // boundary that nearly touches the parametric singularity; gmsh's
  // surface mesher then fails, "splits those edges and tries again", and
  // thereby changes the 1D mesh of one boundary curve but not of its
  // periodic partner ("curve N has 5 vertices, whereas correspondant M
  // has 2"). Moving the poles to a better axis avoids it. The choice only
  // depends on the centre modulo the box (distance to the nearest lattice
  // plane), so a sphere and its periodic images are rotated identically
  // and their cut curves keep matching seams. z (gmsh's default) is kept
  // unless another axis is clearly better.
  static int pole_axis(sphere<value_type> const& s,
                       std::array<value_type, 3> const& box) {
    std::array<value_type, 3> off{};  // signed offset to nearest lattice plane
    std::array<bool, 3> cut{};
    const value_type r = s.radius;
    for (std::size_t k = 0; k < 3; ++k) {
      if (!(box[k] > value_type{0})) continue;
      const value_type t = s(k) / box[k];
      off[k] = (t - std::round(t)) * box[k];
      cut[k] = std::abs(off[k]) < r;
    }
    auto score = [&](std::size_t pole) {
      value_type m = r;
      for (std::size_t k = 0; k < 3; ++k) {
        if (!cut[k]) continue;
        // distance of the nearest pole to the plane along axis k
        m = std::min(m, k == pole ? r - std::abs(off[k]) : std::abs(off[k]));
      }
      return m;
    };
    int best = 2;
    value_type best_score = score(2);
    if (best_score >= value_type{0.5} * r) return best;
    for (int a : {0, 1}) {
      const value_type sc = score(static_cast<std::size_t>(a));
      if (sc > best_score * value_type{1.0001}) {
        best = a;
        best_score = sc;
      }
    }
    return best;
  }

  void write_pole_rotation(std::ostream& out, sphere<value_type> const& s,
                                  std::size_t id,
                                  std::array<value_type, 3> const& box) const {
    const int axis = pole_axis(s, box);
    if (axis == 2) return;
    // z -> x : rotate about y; z -> -y : rotate about x.
    out << "Rotate {{" << (axis == 0 ? "0, 1, 0" : "1, 0, 0") << "}, {"
        << num(s(0)) << ", " << num(s(1)) << ", " << num(s(2))
        << "}, Pi/2} { Volume{" << id << "}; }\n";
  }

  // gmsh's OCC Disk needs rx >= ry; swap the semi-axes and rotate a
  // further 90° when the ellipse's first axis is the minor one.
  void write_ellipse(std::ostream& out, ellipse<value_type> const& e,
                            std::size_t id) const {
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
          << ", 0}, " << raw_num(angle) << "} { Surface{" << id << "}; }\n";
    }
  }

  // Sutherland-Hodgman clip of a (convex) polygon against [0,Lx]x[0,Ly];
  // consecutive near-duplicate vertices (a vertex lying on a clip edge) are
  // merged so no zero-length Line is emitted. A polygon fully inside is
  // returned unchanged. Returns fewer than 3 vertices when nothing (or only
  // a sliver) remains.
  static std::vector<std::array<value_type, 2>>
  clip_polygon_to_rect(std::vector<std::array<value_type, 2>> poly,
                       value_type Lx, value_type Ly) {
    const value_type eps = std::min(Lx, Ly) * value_type{1e-12};
    // axis, bound, keep-side sign: keep points with sign*(p[axis]-bound) <= 0
    struct edge { int axis; value_type bound; value_type sign; };
    const edge edges[4] = {{0, value_type{0}, value_type{-1}},
                           {0, Lx, value_type{1}},
                           {1, value_type{0}, value_type{-1}},
                           {1, Ly, value_type{1}}};
    for (auto const& e : edges) {
      if (poly.size() < 3) return {};
      std::vector<std::array<value_type, 2>> out;
      const std::size_t n = poly.size();
      for (std::size_t i = 0; i < n; ++i) {
        auto const& a = poly[i];
        auto const& b = poly[(i + 1) % n];
        const value_type da = e.sign * (a[e.axis] - e.bound);
        const value_type db = e.sign * (b[e.axis] - e.bound);
        const bool ain = da <= value_type{0}, bin = db <= value_type{0};
        if (ain) out.push_back(a);
        if (ain != bin) {
          const value_type t = da / (da - db);
          std::array<value_type, 2> q{a[0] + t * (b[0] - a[0]),
                                      a[1] + t * (b[1] - a[1])};
          q[e.axis] = e.bound;   // exactly on the boundary
          out.push_back(q);
        }
      }
      poly = std::move(out);
    }
    std::vector<std::array<value_type, 2>> res;
    for (auto const& v : poly) {
      if (!res.empty() && std::abs(v[0] - res.back()[0]) <= eps &&
          std::abs(v[1] - res.back()[1]) <= eps) continue;
      res.push_back(v);
    }
    while (res.size() > 1 && std::abs(res.front()[0] - res.back()[0]) <= eps &&
           std::abs(res.front()[1] - res.back()[1]) <= eps) res.pop_back();
    // Drop slivers with (numerically) zero area.
    value_type area2{0};
    for (std::size_t i = 0; i < res.size(); ++i) {
      auto const& a = res[i];
      auto const& b = res[(i + 1) % res.size()];
      area2 += a[0] * b[1] - a[1] * b[0];
    }
    if (res.size() < 3 || std::abs(area2) <= eps * std::max(Lx, Ly)) return {};
    return res;
  }

  static std::vector<std::array<value_type, 2>>
  clipped_vertices(convex_polygon<value_type> const& p, std::array<value_type, 3> const& box) {
    std::vector<std::array<value_type, 2>> poly;
    for (auto const& v : p.vertices()) poly.push_back({v[0], v[1]});
    return clip_polygon_to_rect(std::move(poly), box[0], box[1]);
  }

  // Points / lines / loop get fresh tags from gmsh (newp / newl / newll)
  // so they never collide with the tags OCC auto-assigns to the
  // boundaries of Rectangle / Disk entities. Only the Plane Surface uses
  // the explicit inclusion id.
  void write_polygon(std::ostream& out,
                     std::vector<std::array<value_type, 2>> const& verts,
                     std::size_t id) const {
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

  // First-wins priority of every shape. Normally the shape index. With
  // periodic=true the wrap copies of one particle (same type and size,
  // centres differing by whole box lengths) share the priority of the
  // earliest copy: otherwise the order A, B, A', B' would let A beat B on
  // one side of the box but B' beat A' on the other, and the two faces
  // could no longer be paired.
  std::vector<std::size_t> priorities(
      shape_vector const& shapes, std::vector<extent_box> const& extents,
      std::array<value_type, 3> const& box) const {
    std::vector<std::size_t> prio(shapes.size());
    for (std::size_t i = 0; i < shapes.size(); ++i) prio[i] = i;
    if (!_periodic) return prio;
    const value_type tol = value_type{1e-9} * std::max({box[0], box[1], box[2]});
    auto images = [&](std::size_t i, std::size_t j) {
      if (typeid(*shapes[i]) != typeid(*shapes[j])) return false;
      const auto mi = shapes[i]->get_middle_point();
      const auto mj = shapes[j]->get_middle_point();
      for (std::size_t k = 0; k < 3; ++k) {
        const value_type size_i = extents[i].top[k] - extents[i].bottom[k];
        const value_type size_j = extents[j].top[k] - extents[j].bottom[k];
        if (std::abs(size_i - size_j) > tol) return false;
        if (!(box[k] > value_type{0})) continue;
        const value_type d = (mi[k] - mj[k]) / box[k];
        if (std::abs(d - std::round(d)) * box[k] > tol) return false;
      }
      return true;
    };
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      for (std::size_t j = 0; j < i; ++j) {
        if (prio[j] == j && images(i, j)) { prio[i] = j; break; }
      }
    }
    return prio;
  }

  // Reduce every inclusion to its final pieces, in shape order:
  //   1. clip to the domain if it leaves it (BooleanIntersection);
  //   2. first shape wins: subtract all earlier inclusions whose
  //      bounding box overlaps (BooleanDifference), so the pieces are
  //      disjoint and exactly one phase owns every point.
  // Runs after all primitives are emitted so the tags of new boolean
  // results can never collide with a later explicit inclusion id.
  // Inclusions that need neither step keep their plain id.
  std::vector<inclusion_ref> write_clips(
      std::ostream& out, shape_vector const& shapes,
      std::vector<std::size_t> const& ids,
      std::array<value_type, 3> const& domain_box, char const* kind) const {
    std::vector<inclusion_ref> refs(shapes.size());
    bool header_done = false;
    std::vector<extent_box> extents;
    extents.reserve(shapes.size());
    for (auto const& sh : shapes) extents.push_back(extent_of(*sh));
    const auto prio = priorities(shapes, extents, domain_box);
    std::vector<std::size_t> order(shapes.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t a, std::size_t b) { return prio[a] < prio[b]; });
    for (std::size_t i : order) {
      const auto id = std::to_string(ids[i]);
      const auto var = "_inc_" + id + "()";
      const bool clip = needs_clip(*shapes[i], domain_box);

      const bool absent = misses_domain(*shapes[i], domain_box);
      std::vector<std::string> tools;
      for (std::size_t j = 0; j < shapes.size() && !absent; ++j) {
        const bool before = prio[j] < prio[i] || (prio[j] == prio[i] && j < i);
        if (before && !misses_domain(*shapes[j], domain_box) &&
            aabb_overlap(extents[i], extents[j])) {
          tools.push_back(refs[j].ref);
        }
      }

      if (!absent && !clip && tools.empty()) {
        refs[i] = {id, shapes[i]->phase_name()};
        continue;
      }
      if (!header_done) {
        out << "\n// Clip inclusions that leave the domain (periodic wrap copies,\n"
               "// overhanging shapes) and make overlapping inclusions disjoint\n"
               "// (first shape wins). `_inc_<id>()` holds the final pieces.\n";
        header_done = true;
      }
      if (absent) {
        out << "// shape #" << i << " does not reach the domain: not emitted\n"
            << var << " = {};\n";
        refs[i] = {var, shapes[i]->phase_name()};
        continue;
      }
      if (clip) {
        // The result list of an intersection that can be empty is not
        // trusted as it is: gmsh 4.14.0 answers an empty intersection with
        // the tag of the tool (the domain, tag 1), which would delete the
        // matrix later. Tag 1 is never a legitimate piece, so drop it.
        out << "_r() = BooleanIntersection{ " << kind << "{" << id
            << "}; Delete; }{ " << kind << "{1}; };\n"
            << var << " = {};\n"
            << "For _k In {0:#_r()-1}\n"
            << "  If (_r(_k) != 1)\n"
            << "    " << var << " += _r(_k);\n"
            << "  EndIf\n"
            << "EndFor\n";
        if (clearly_overlaps(*shapes[i], domain_box)) {
          out << "If (#" << var << " == 0)\n"
              << "  Error(\"gmsh_geo_writer: shape #" << i
              << " clearly overlaps the domain but its clip came out empty "
                 "(OpenCASCADE boolean failed)\");\n"
              << "EndIf\n";
        }
      } else {
        out << var << " = {" << id << "};\n";
      }
      if (!tools.empty()) {
        out << "_tool() = ";
        write_ref_list(out, tools);
        out << ";\n"
            << "If (#" << var << " > 0 && #_tool() > 0)\n"
            << "  " << var << " = BooleanDifference{ " << kind << "{" << var
            << "}; Delete; }{ " << kind << "{_tool()}; };\n"
            << "EndIf\n";
      }
      refs[i] = {var, shapes[i]->phase_name()};
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

  // Build the matrix explicitly as domain minus all inclusion pieces,
  // then fragment matrix and pieces together so they share interfaces.
  // The pieces are disjoint (see write_clips) and the matrix is disjoint
  // from them, so OCC keeps every tag; fragmenting the *domain* against
  // the inclusions instead would let OCC re-tag coincident areas (e.g.
  // an inclusion equal to the domain) and silently swap phases.
  // `_matrix()` is empty when the inclusions cover the whole domain.
  static void write_fragments(std::ostream& out,
                              std::vector<inclusion_ref> const& inclusions,
                              char const* kind) {
    std::vector<std::string> refs;
    refs.reserve(inclusions.size());
    for (auto const& inc : inclusions) refs.push_back(inc.ref);
    out << "\n// Matrix = domain minus inclusions; fragment everything into one\n"
           "// conforming geometry with shared interfaces.\n"
        << "_inc() = ";
    write_ref_list(out, refs);
    out << ";\n"
        << "_matrix() = {1};\n"
        << "If (#_inc() > 0)\n"
        << "  _matrix() = BooleanDifference{ " << kind << "{1}; Delete; }{ " << kind
        << "{_inc()}; };\n"
        << "  BooleanFragments{ " << kind << "{_matrix(), _inc()}; Delete; }{}\n"
        << "EndIf\n";
  }

  // Pair every boundary curve (2D) / face (3D) on the low side of each
  // axis with its translated image on the high side, matching by
  // bounding box (gmsh tutorial t18). Emitted after fragmenting so the
  // constraints reference the final, split boundary entities.
  void write_periodic(std::ostream& out,
                             std::array<value_type, 3> const& dom,
                             bool is_3d) const {
    const char* ent = is_3d ? "Surface" : "Curve";
    const int n_axes = is_3d ? 3 : 2;
    const value_type lmax = std::max({dom[0], dom[1], dom[2]});
    out << "\n// Periodic boundary conditions: pair each low-side boundary "
        << (is_3d ? "face" : "curve") << "\n"
        << "// with its translated image on the opposite side.\n"
        << "_pe = " << num(lmax * value_type{1e-6}) << ";\n";
    if (is_3d) {
      // Tight (triangulation-based) bounding boxes; OCC's default boxes
      // are inflated for curved faces, which breaks the bounding-box
      // matching in 3D. Not needed (and ~16x slower) in 2D.
      out << "Geometry.OCCBoundsUseStl = 1;\n";
    }
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
          << "  _found = 0;\n"
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
          << "      _found = 1;\n"
          << "    EndIf\n"
          << "  EndFor\n"
          << "  If (!_found)\n"
          << "    Error(\"gmsh_geo_writer: " << ent
          << " on the low side of axis " << a
          << " has no periodic partner on the opposite side\");\n"
          << "  EndIf\n"
          << "EndFor\n";
    }
    if (is_3d) out << "Geometry.OCCBoundsUseStl = 0;\n";
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
    // Untagged inclusions are treated as matrix.
    by_phase[matrix_name].push_back("_matrix()");
    for (auto const& inc : inclusions) {
      by_phase[inc.phase_name.empty() ? matrix_name : inc.phase_name]
          .push_back(inc.ref);
    }

    out << "\n// Physical groups — one per phase, keyed off shape.phase_name(),\n"
           "// plus \"matrix\" for everything that is not a tagged inclusion.\n";

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
  mutable value_type _length_scale{1};
  int _element_order{0};
  int _algorithm{0};
  bool _recombine_quads{false};
};

} // namespace rvegen
