// pybind11 bindings for rvegen — phase-2 shape surface.
//
// Scope of this binding:
//   * Shape primitives (circle, sphere, rectangle, box, ellipse,
//     polyline_tube, mesh_inclusion, voronoi_cell) — ctors, accessors,
//     `is_inside`, `volume` / `area`.
//   * These are pure value types: own their data, no engine reference,
//     no callbacks. The safe subset before distributions / generators
//     are bound.
//
// Out of scope; comes in subsequent PRs:
//   * Distributions, generators, terminations — these all need engine
//     references at construction; the Python binding for that needs
//     careful lifetime + GIL handling. Specifically: an engine must
//     outlive the distribution from Python's perspective (use
//     `py::keep_alive`); `operator()` mutates engine state, so two
//     Python threads sampling from the same distribution concurrently
//     under a released GIL is UB.
//   * JSON-driven config path — once the registry types are bound,
//     Python users will be able to call `rvegen.run_from_json(cfg_str)`.
//   * Output writers (gmsh, voxel grid, VTK) — they take a vector of
//     `unique_ptr<shape_base<T>>`; binding `shape_base` polymorphically
//     in pybind11 needs a holder type and explicit registration of
//     each derived shape.
//   * Wheels via cibuildwheel — separate from the conda/vcpkg
//     packaging in the packaging PR.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rvegen/generators/voronoi_generator_2d.h"
#include "rvegen/generators/weave_generator.h"
#include "rvegen/shapes/box.h"
#include "rvegen/shapes/circle.h"
#include "rvegen/shapes/convex_polygon.h"
#include "rvegen/shapes/ellipse.h"
#include "rvegen/shapes/mesh_inclusion.h"
#include "rvegen/shapes/polyline_tube.h"
#include "rvegen/shapes/rectangle.h"
#include "rvegen/shapes/sphere.h"
#include "rvegen/shapes/voronoi_cell.h"
#include "rvegen/io/stl_reader.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "rvegen — Representative Volume Element generator (Python bindings, "
      "phase-2 shape surface: Circle, Sphere, Rectangle, Box, Ellipse, "
      "PolylineTube, MeshInclusion, VoronoiCell).";

  py::class_<rvegen::circle<double>>(m, "Circle")
      .def(py::init<double, double, double>(), py::arg("x"), py::arg("y"),
           py::arg("radius"))
      .def_property_readonly("x",
                             [](rvegen::circle<double> const& c) { return c(0); })
      .def_property_readonly("y",
                             [](rvegen::circle<double> const& c) { return c(1); })
      .def_property_readonly(
          "radius", [](rvegen::circle<double> const& c) { return c.radius; })
      .def("is_inside", [](rvegen::circle<double> const& c, double x,
                           double y) { return c.is_inside({x, y, 0.0}); })
      .def("area", &rvegen::circle<double>::area);

  py::class_<rvegen::sphere<double>>(m, "Sphere")
      .def(py::init<double, double, double, double>(), py::arg("x"),
           py::arg("y"), py::arg("z"), py::arg("radius"))
      .def_property_readonly("x",
                             [](rvegen::sphere<double> const& s) { return s.center[0]; })
      .def_property_readonly("y",
                             [](rvegen::sphere<double> const& s) { return s.center[1]; })
      .def_property_readonly("z",
                             [](rvegen::sphere<double> const& s) { return s.center[2]; })
      .def_property_readonly(
          "radius", [](rvegen::sphere<double> const& s) { return s.radius; })
      .def("is_inside",
           [](rvegen::sphere<double> const& s, double x, double y, double z) {
             return s.is_inside({x, y, z});
           })
      .def("volume", &rvegen::sphere<double>::volume);

  py::class_<rvegen::rectangle<double>>(m, "Rectangle")
      .def(py::init<double, double, double, double>(), py::arg("x"),
           py::arg("y"), py::arg("width"), py::arg("height"))
      .def_property_readonly(
          "x", [](rvegen::rectangle<double> const& r) { return r(0); })
      .def_property_readonly(
          "y", [](rvegen::rectangle<double> const& r) { return r(1); })
      .def_property_readonly(
          "width", [](rvegen::rectangle<double> const& r) { return r.width(); })
      .def_property_readonly(
          "height",
          [](rvegen::rectangle<double> const& r) { return r.height(); })
      .def("is_inside",
           [](rvegen::rectangle<double> const& r, double x, double y) {
             return r.is_inside({x, y, 0.0});
           })
      .def("area", &rvegen::rectangle<double>::area);

  py::class_<rvegen::box<double>>(m, "Box")
      .def(py::init<double, double, double, double, double, double>(),
           py::arg("x"), py::arg("y"), py::arg("z"), py::arg("width"),
           py::arg("height"), py::arg("depth"))
      .def_property_readonly(
          "x", [](rvegen::box<double> const& b) { return b(0); })
      .def_property_readonly(
          "y", [](rvegen::box<double> const& b) { return b(1); })
      .def_property_readonly(
          "z", [](rvegen::box<double> const& b) { return b(2); })
      .def_property_readonly(
          "width", [](rvegen::box<double> const& b) { return b.width(); })
      .def_property_readonly(
          "height", [](rvegen::box<double> const& b) { return b.height(); })
      .def_property_readonly(
          "depth", [](rvegen::box<double> const& b) { return b.depth(); })
      .def("is_inside",
           [](rvegen::box<double> const& b, double x, double y, double z) {
             return b.is_inside({x, y, z});
           })
      .def("volume", &rvegen::box<double>::volume);

  py::class_<rvegen::ellipse<double>>(m, "Ellipse")
      .def(py::init<double, double, double, double, double>(),
           py::arg("x"), py::arg("y"), py::arg("radius_a"),
           py::arg("radius_b"), py::arg("rotation_rad"))
      .def_property_readonly(
          "x", [](rvegen::ellipse<double> const& e) { return e.center[0]; })
      .def_property_readonly(
          "y", [](rvegen::ellipse<double> const& e) { return e.center[1]; })
      .def_property_readonly(
          "radius_a",
          [](rvegen::ellipse<double> const& e) { return e.extent[0]; })
      .def_property_readonly(
          "radius_b",
          [](rvegen::ellipse<double> const& e) { return e.extent[1]; })
      .def("is_inside",
           [](rvegen::ellipse<double> const& e, double x, double y) {
             return e.is_inside({x, y, 0.0});
           })
      .def("area", &rvegen::ellipse<double>::area);

  py::class_<rvegen::polyline_tube<double>>(m, "PolylineTube")
      // Centerline as a Python list of (x, y, z) tuples — converted to
      // std::vector<std::array<double, 3>> via pybind11/stl.h.
      .def(py::init<std::vector<std::array<double, 3>> const&, double>(),
           py::arg("centerline"), py::arg("radius"))
      .def_property_readonly(
          "radius",
          [](rvegen::polyline_tube<double> const& t) { return t.radius(); })
      .def_property_readonly(
          "centerline",
          [](rvegen::polyline_tube<double> const& t) {
            std::vector<std::array<double, 3>> out;
            out.reserve(t.centerline().size());
            for (auto const& v : t.centerline()) {
              out.push_back({v[0], v[1], v[2]});
            }
            return out;
          })
      .def("is_inside",
           [](rvegen::polyline_tube<double> const& t, double x, double y,
              double z) { return t.is_inside({x, y, z}); })
      .def("volume", &rvegen::polyline_tube<double>::volume);

  py::class_<rvegen::mesh_inclusion<double>>(m, "MeshInclusion")
      // `from_stl_file(path)` factory — the natural Python entry point.
      // Direct ctor from a Python triangle list is more painful (would
      // need to bind gte::Triangle3) and offers no win over loading via
      // STL. Skipped for phase-2.
      .def_static("from_stl_file",
                  [](std::string const& path) {
                    return rvegen::mesh_inclusion<double>{
                        rvegen::read_stl_ascii_file<double>(path)};
                  },
                  py::arg("path"))
      .def_property_readonly(
          "triangle_count",
          [](rvegen::mesh_inclusion<double> const& m) {
            return m.triangle_count();
          })
      .def("is_inside",
           [](rvegen::mesh_inclusion<double> const& m, double x, double y,
              double z) { return m.is_inside({x, y, z}); })
      .def("volume", &rvegen::mesh_inclusion<double>::volume)
      .def("set_middle_point",
           [](rvegen::mesh_inclusion<double>& m, double x, double y,
              double z) { m.set_middle_point({x, y, z}); });

  py::class_<rvegen::voronoi_cell<double>>(m, "VoronoiCell")
      // Vertices: list of (x, y, z) tuples.
      // Faces: list of vertex-index lists per face.
      .def(py::init([](std::vector<std::array<double, 3>> const& verts,
                       std::vector<std::vector<std::size_t>> const& faces) {
             std::vector<gte::Vector<3, double>> v;
             v.reserve(verts.size());
             for (auto const& a : verts) {
               gte::Vector<3, double> g;
               g[0] = a[0]; g[1] = a[1]; g[2] = a[2];
               v.push_back(g);
             }
             return rvegen::voronoi_cell<double>{std::move(v), faces};
           }),
           py::arg("vertices"), py::arg("faces"))
      .def_property_readonly(
          "vertices",
          [](rvegen::voronoi_cell<double> const& c) {
            std::vector<std::array<double, 3>> out;
            out.reserve(c.vertices().size());
            for (auto const& v : c.vertices()) {
              out.push_back({v[0], v[1], v[2]});
            }
            return out;
          })
      .def_property_readonly(
          "faces",
          [](rvegen::voronoi_cell<double> const& c) { return c.faces(); })
      .def("is_inside",
           [](rvegen::voronoi_cell<double> const& c, double x, double y,
              double z) { return c.is_inside({x, y, z}); })
      .def("volume", &rvegen::voronoi_cell<double>::volume);

  // 2D convex polygon — Voronoi-cell-ready, full shape_base contract
  // on the C++ side.
  py::class_<rvegen::convex_polygon<double>>(m, "ConvexPolygon")
      .def(py::init<std::vector<std::array<double, 2>>>(),
           py::arg("vertices"),
           "Construct from a CCW-ordered list of (x, y) tuples; ≥ 3 "
           "vertices required.")
      .def_property_readonly(
          "vertices",
          [](rvegen::convex_polygon<double> const& p) { return p.vertices(); })
      .def_property_readonly(
          "vertex_count",
          [](rvegen::convex_polygon<double> const& p) {
            return p.vertex_count();
          })
      .def("area", &rvegen::convex_polygon<double>::area)
      .def("is_inside",
           [](rvegen::convex_polygon<double> const& p, double x, double y) {
             return p.is_inside({x, y, 0.0});
           })
      .def("get_middle_point",
           [](rvegen::convex_polygon<double> const& p) {
             return p.get_middle_point();
           })
      .def("set_middle_point",
           [](rvegen::convex_polygon<double>& p, double x, double y) {
             p.set_middle_point({x, y, 0.0});
           });

  // Bounded 2D Voronoi tessellation by seed points (#114 phase 1).
  // build_polygons() returns the raw vertex lists; build_cells()
  // returns them as ConvexPolygon objects ready for downstream use.
  py::class_<rvegen::voronoi_generator_2d<double>>(m, "VoronoiGenerator2d")
      .def(py::init<double, double, std::vector<std::array<double, 2>>>(),
           py::arg("Lx"), py::arg("Ly"), py::arg("seeds"))
      .def_property_readonly(
          "seed_count",
          [](rvegen::voronoi_generator_2d<double> const& g) {
            return g.seed_count();
          })
      .def("build_polygons",
           [](rvegen::voronoi_generator_2d<double> const& g) {
             return g.build();
           },
           "Returns a list-of-list-of-(x,y) — one CCW polygon per seed.")
      .def("build_cells",
           [](rvegen::voronoi_generator_2d<double> const& g) {
             // Iterate the raw polygons and wrap each as a value
             // ConvexPolygon. Degenerate (<3 vertex) cells are skipped
             // — same convention as voronoi_to_shapes on the C++ side.
             auto polys = g.build();
             std::vector<rvegen::convex_polygon<double>> out;
             out.reserve(polys.size());
             for (auto& v : polys) {
               if (v.size() < 3) continue;
               out.emplace_back(std::move(v));
             }
             return out;
           },
           "Returns a list of ConvexPolygon — Voronoi cells ready for "
           "downstream is_inside / area queries.");

  // Deterministic plain-weave generator (#3 / #109). build() returns
  // a Python list of PolylineTube — every yarn the generator produces
  // IS a polyline_tube by construction, so the down-cast is type-safe.
  // Each tube in the list owns its own data (deep-copied on the way
  // out so Python lifetime is independent of the generator).
  py::class_<rvegen::weave_generator<double>>(m, "WeaveGenerator")
      .def(py::init<std::array<double, 3> const&, std::size_t, std::size_t,
                    double, double, std::size_t>(),
           py::arg("domain_box"),
           py::arg("n_warp_yarns"),
           py::arg("n_weft_yarns"),
           py::arg("yarn_radius"),
           py::arg("amplitude"),
           py::arg("n_segments_per_yarn") = std::size_t{32})
      .def_property_readonly(
          "n_warp_yarns",
          [](rvegen::weave_generator<double> const& g) {
            return g.n_warp_yarns();
          })
      .def_property_readonly(
          "n_weft_yarns",
          [](rvegen::weave_generator<double> const& g) {
            return g.n_weft_yarns();
          })
      .def("set_warp_phase_name", &rvegen::weave_generator<double>::set_warp_phase_name,
           py::arg("name"))
      .def("set_weft_phase_name", &rvegen::weave_generator<double>::set_weft_phase_name,
           py::arg("name"))
      .def_property_readonly(
          "warp_phase_name",
          [](rvegen::weave_generator<double> const& g) {
            return g.warp_phase_name();
          })
      .def_property_readonly(
          "weft_phase_name",
          [](rvegen::weave_generator<double> const& g) {
            return g.weft_phase_name();
          })
      .def("build",
           [](rvegen::weave_generator<double> const& g) {
             auto raw = g.build();
             std::vector<rvegen::polyline_tube<double>> out;
             out.reserve(raw.size());
             for (auto& s : raw) {
               auto* tube =
                   dynamic_cast<rvegen::polyline_tube<double>*>(s.get());
               if (!tube) {
                 throw std::runtime_error{
                     "WeaveGenerator.build: produced a non-polyline_tube "
                     "shape (impossible by construction — binding bug)"};
               }
               // Copy the tube out by value so the Python list entries
               // outlive the unique_ptr<shape_base> from build().
               out.push_back(*tube);
             }
             return out;
           });
}
