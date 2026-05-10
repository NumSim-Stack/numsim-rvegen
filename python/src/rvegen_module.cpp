// pybind11 bindings for rvegen — phase 1 surface.
//
// Scope of this binding (foundation for #4):
//   * Shape primitives (circle, sphere, rectangle, box) — ctors,
//     accessors, `is_inside`, `volume` / `area`.
//
// Out of scope; comes in subsequent PRs:
//   * Distributions, generators, terminations — these all need engine
//     references at construction; the Python binding for that needs
//     careful lifetime + GIL handling.
//   * JSON-driven config path — once the registry types are bound,
//     Python users will be able to call `rvegen.run_from_json(cfg_str)`.
//   * Output writers (gmsh, voxel grid, VTK) — they take a vector of
//     `unique_ptr<shape_base<T>>`; binding `shape_base` polymorphically
//     in pybind11 needs a holder type and explicit registration of
//     each derived shape.
//   * Wheels via cibuildwheel — handled by the packaging PR.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "rvegen/shapes/circle.h"
#include "rvegen/shapes/sphere.h"
#include "rvegen/shapes/rectangle.h"
#include "rvegen/shapes/box.h"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "rvegen — Representative Volume Element generator (Python bindings, "
      "phase 1 — shape primitives only).";

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
}
