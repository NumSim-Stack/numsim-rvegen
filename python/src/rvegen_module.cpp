// pybind11 bindings for rvegen — phase 1 surface.
//
// Scope of this binding (foundation for #4):
//   * Shape primitives (circle, sphere, rectangle, box) — ctors,
//     accessors, `is_inside`, `volume` / `area`.
//   * These are pure value types: own their data, no engine reference,
//     no callbacks. The safe subset for a phase-1 binding.
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

#include "rvegen/shapes/box.h"
#include "rvegen/shapes/circle.h"
#include "rvegen/shapes/rectangle.h"
#include "rvegen/shapes/sphere.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "rvegen — Representative Volume Element generator (Python bindings, "
      "phase 1 — shape primitives: Circle, Sphere, Rectangle, Box).";

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
}
