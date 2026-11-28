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

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <limits>
#include <optional>
#include <random>

#include "rvegen/generators/grain_size_control.h"
#include "rvegen/phase/phase_assignment.h"
#include "rvegen/pipeline.h"
#include "rvegen/postprocess/voxel_grid.h"
#include "rvegen/schema/schema_to_json.h"
#include "rvegen/generators/voronoi_generator_2d.h"
#include "rvegen/generators/voronoi_generator_3d.h"
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

namespace {

using shape_ptr = std::unique_ptr<rvegen::shape_base<double>>;

// ---- JSON <-> Python helpers (via the stdlib json module; no extra dep) ----
nlohmann::json py_to_json(py::handle obj) {
  // numpy scalars / arrays are converted via .tolist(); NaN / inf are
  // rejected up front (they are not JSON and would otherwise surface as an
  // opaque parse error).
  auto fallback = py::cpp_function([](py::handle o) -> py::object {
    if (py::hasattr(o, "tolist")) return o.attr("tolist")();
    throw py::type_error("rvegen config: value of type '" +
                         py::str(py::type::handle_of(o).attr("__name__"))
                             .cast<std::string>() +
                         "' is not JSON-serialisable");
  });
  std::string text;
  try {
    text = py::module_::import("json")
               .attr("dumps")(obj, py::arg("default") = fallback,
                              py::arg("allow_nan") = false)
               .cast<std::string>();
  } catch (py::error_already_set& e) {
    if (e.matches(PyExc_ValueError))
      throw py::value_error(
          "rvegen config contains NaN or infinity, which is not valid JSON");
    throw;
  }
  return nlohmann::json::parse(text);
}

py::object json_to_py(nlohmann::json const& j) {
  return py::module_::import("json").attr("loads")(j.dump());
}

// A config is a dict (JSON-style) or a path (str / os.PathLike) to a JSON file.
// A config and, when it was read from a file, that file's directory (empty
// for a dict): relative input paths in the config are resolved against it.
struct loaded_config {
  nlohmann::json json;
  std::filesystem::path dir;
};

loaded_config load_config(py::object const& cfg) {
  if (py::isinstance<py::dict>(cfg)) return {py_to_json(cfg), {}};
  if (py::isinstance<py::str>(cfg) || py::hasattr(cfg, "__fspath__")) {
    auto const path =
        py::module_::import("os").attr("fspath")(cfg).cast<std::string>();
    std::ifstream in{path};
    if (!in)
      throw std::runtime_error{"rvegen.run: cannot open config file: " + path};
    try {
      nlohmann::json j;
      in >> j;
      auto dir = std::filesystem::path{path}.parent_path();
      if (dir.empty()) dir = ".";
      return {std::move(j), std::move(dir)};
    } catch (std::exception const& e) {
      throw std::runtime_error{"rvegen.run: parse error in " + path + ": " +
                               e.what()};
    }
  }
  throw std::invalid_argument{
      "rvegen.run: config must be a dict or a path to a JSON file"};
}

// Ctrl-C support for long C++ calls running with the GIL released. The
// cancel hook is polled by the generators / voxel sampler; at most every
// ~50 ms of wall time it re-acquires the GIL and runs PyErr_CheckSignals().
// If that reports an exception (KeyboardInterrupt), the exception is parked
// here, the hook keeps returning true so the C++ side unwinds, and the
// binding rethrows it with rethrow_if_interrupted().
class interrupt_watch {
public:
  [[nodiscard]] std::function<bool()> hook() {
    return [this]() -> bool {
      if (_error) return true;
      if ((++_calls & 0xF) != 0) return false;   // keep the clock off the hot path
      auto const now = std::chrono::steady_clock::now();
      if (now - _last < std::chrono::milliseconds{50}) return false;
      _last = now;
      py::gil_scoped_acquire acquire;
      if (PyErr_CheckSignals() != 0) {
        _error.emplace();   // fetches + clears the Python error indicator
        return true;
      }
      return false;
    };
  }
  // Call with the GIL held.
  void rethrow_if_interrupted() {
    if (_error) {
      auto e = std::move(*_error);
      _error.reset();
      throw e;
    }
  }

private:
  std::optional<py::error_already_set> _error;
  std::uint32_t _calls{0};
  std::chrono::steady_clock::time_point _last{std::chrono::steady_clock::now()};
};

// Runs f(cancel_fn const*) with Ctrl-C support, for bound builders that
// keep the GIL (so Ctrl-C raises KeyboardInterrupt instead of waiting).
template <typename F>
auto interruptible(F&& f) {
  interrupt_watch watch;
  rvegen::cancel_fn const cancel = watch.hook();
  try {
    return f(&cancel);
  } catch (rvegen::generation_cancelled const&) {
    watch.rethrow_if_interrupted();
    throw;
  }
}

// The `log` callable of rvegen.run() / rvegen.check(): gets each line as it
// is produced. The pipeline may run with the GIL released, so every call
// takes it. An exception raised by the callable is parked here (it must
// not travel through the C++ pipeline, which would read it without the
// GIL), failed() makes the run stop at its next cancel poll, and the
// binding rethrows it with rethrow_if_failed().
class log_sink {
public:
  explicit log_sink(py::object callable) : _log{std::move(callable)} {}

  // Routes the pipeline's lines to the callable (no-op for log=None).
  void attach(rvegen::pipeline_options& opts, bool verbose) {
    if (_log.is_none()) return;
    auto emit = [this](std::string const& line) {
      if (_error) return;
      py::gil_scoped_acquire acquire;
      try {
        _log(line);
      } catch (py::error_already_set& e) {
        _error.emplace(std::move(e));
      }
    };
    opts.info = emit;
    if (verbose) opts.debug = emit;
  }
  [[nodiscard]] bool failed() const noexcept { return _error.has_value(); }
  // Call with the GIL held.
  void rethrow_if_failed() {
    if (_error) {
      auto e = std::move(*_error);
      _error.reset();
      throw e;
    }
  }

private:
  py::object _log;
  std::optional<py::error_already_set> _error;
};

// input_dir / output_dir of run() and check(): None, a str or a path-like.
// A config read from a file resolves its input paths against that file's
// directory unless input_dir says otherwise.
void set_directories(rvegen::pipeline_options& opts, loaded_config const& cfg,
                     py::object const& input_dir, py::object const& output_dir) {
  auto to_path = [](py::object const& o) {
    return std::filesystem::path{
        py::module_::import("os").attr("fspath")(o).cast<std::string>()};
  };
  opts.input_dir = input_dir.is_none() ? cfg.dir : to_path(input_dir);
  if (!output_dir.is_none()) opts.output_dir = to_path(output_dir);
}

// Result of rvegen.run(): a sequence of shapes plus a summary dict.
struct run_result {
  py::list shapes;
  py::dict summary;
};

// Registry introspection submodule: keys() and schema(name).
template <typename Registry>
void bind_registry(py::module_& parent, char const* name) {
  auto sub = parent.def_submodule(name, "Registered types in this category.");
  sub.def("keys", []() {
    rvegen::register_all_types();
    auto keys = Registry::instance().registered_types();
    std::vector<std::string> out(keys.begin(), keys.end());
    std::sort(out.begin(), out.end());
    return out;
  }, "Sorted names of every registered type.");
  sub.def("schema", [](std::string const& type) {
    rvegen::register_all_types();
    auto& reg = Registry::instance();
    if (!reg.contains(type))
      throw py::key_error{"unknown type '" + type + "'"};
    return json_to_py(rvegen::schema_to_json(reg.schema(type)));
  }, py::arg("name"),
     "Parameter schema of a type: {field: {type, required, description, ...}}.");
}

} // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() =
      "rvegen — Representative Volume Element generator (Python bindings: "
      "shape primitives, registry introspection, JSON-driven run(), "
      "voxelize()).";
  rvegen::register_all_types();   // populate every registry once

  // Polymorphic base: lets rvegen.run() hand back shapes that pybind11
  // downcasts to their concrete bound type (Circle, Sphere, ...).
  py::class_<rvegen::shape_base<double>>(m, "Shape")
      .def_property_readonly("kind", [](rvegen::shape_base<double> const& s) {
        return rvegen::shape_type_name(s);
      })
      .def_property_readonly("center", [](rvegen::shape_base<double> const& s) {
        return s.get_middle_point();
      })
      .def("area", &rvegen::shape_base<double>::area)
      .def("volume", &rvegen::shape_base<double>::volume)
      .def("is_inside",
           [](rvegen::shape_base<double> const& s, double x, double y, double z) {
             return s.is_inside({x, y, z});
           },
           py::arg("x"), py::arg("y"), py::arg("z") = 0.0)
      .def_property("phase_name",
          [](rvegen::shape_base<double> const& s) { return s.phase_name(); },
          [](rvegen::shape_base<double>& s, std::string const& n) {
            s.set_phase_name(n);
          })
      .def_property_readonly("info", [](rvegen::shape_base<double> const& s) {
        return json_to_py(s.info().json());
      })
      .def_property_readonly("is_primary",
          &rvegen::shape_base<double>::is_primary,
          "False for a periodic copy of another shape (see image_shift).")
      .def_property_readonly("image_shift",
          [](rvegen::shape_base<double> const& s) {
            auto const& sh = s.image_shift();
            return py::make_tuple(sh[0], sh[1], sh[2]);
          },
          "(i, j, k): this shape is its primary translated by i*Lx, j*Ly, "
          "k*Lz. (0, 0, 0) for a primary.")
      .def_property_readonly("inclusion_id",
          [](rvegen::shape_base<double> const& s) -> py::object {
            if (auto const id = s.inclusion_id()) return py::int_(*id);
            return py::none();
          },
          "Id shared by a primary and its periodic copies; None for a shape "
          "without copies. Unique per inclusion, not consecutive.");

  py::class_<rvegen::circle<double>, rvegen::shape_base<double>>(m, "Circle")
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

  py::class_<rvegen::sphere<double>, rvegen::shape_base<double>>(m, "Sphere")
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

  py::class_<rvegen::rectangle<double>, rvegen::shape_base<double>>(m, "Rectangle")
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

  py::class_<rvegen::box<double>, rvegen::shape_base<double>>(m, "Box")
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

  py::class_<rvegen::ellipse<double>, rvegen::shape_base<double>>(m, "Ellipse")
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

  py::class_<rvegen::polyline_tube<double>, rvegen::shape_base<double>>(m, "PolylineTube")
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

  py::class_<rvegen::mesh_inclusion<double>, rvegen::shape_base<double>>(m, "MeshInclusion")
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

  py::class_<rvegen::voronoi_cell<double>, rvegen::shape_base<double>>(m, "VoronoiCell")
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
  py::class_<rvegen::convex_polygon<double>, rvegen::shape_base<double>>(m, "ConvexPolygon")
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
      .def(py::init<double, double, std::vector<std::array<double, 2>>, bool,
                    std::vector<double>>(),
           py::arg("Lx"), py::arg("Ly"), py::arg("seeds"),
           py::arg("periodic") = false,
           py::arg("weights") = std::vector<double>{})
      .def_property_readonly(
          "seed_count",
          [](rvegen::voronoi_generator_2d<double> const& g) {
            return g.seed_count();
          })
      .def("build_polygons",
           [](rvegen::voronoi_generator_2d<double> const& g) {
             return interruptible(
                 [&](rvegen::cancel_fn const* c) { return g.build(c); });
           },
           "Returns a list-of-list-of-(x,y) — one CCW polygon per seed.")
      .def("build_cells",
           [](rvegen::voronoi_generator_2d<double> const& g) {
             // Iterate the raw polygons and wrap each as a value
             // ConvexPolygon. Degenerate (<3 vertex) cells become empty
             // placeholders so cell i always matches seed i (same
             // convention as voronoi_to_shapes on the C++ side).
             auto polys = interruptible(
                 [&](rvegen::cancel_fn const* c) { return g.build(c); });
             std::vector<rvegen::convex_polygon<double>> out;
             out.reserve(polys.size());
             for (auto& v : polys) {
               if (v.size() < 3) {
                 out.emplace_back();
                 continue;
               }
               out.emplace_back(std::move(v));
             }
             return out;
           },
           "Returns a list of ConvexPolygon — Voronoi cells ready for "
           "downstream is_inside / area queries.");

  // Bounded or periodic 3D Voronoi / Laguerre tessellation (#8, #70).
  // build() returns one VoronoiCell per seed, in seed order. Periodic
  // cells are whole convex polytopes around their seed and may extend
  // outside the box.
  m.def("voro_pp_available", &rvegen::voro_pp_available,
        "True if rvegen was built with the Voro++ backend.");
  // "auto" (default) picks Voro++ when the module was built with it
  // (RVEGEN_WITH_VORO_PP; roughly linear in N) and otherwise falls back to
  // the in-house backend. The in-house backend is brute force: one bounded
  // tessellation costs ~N^4 (N=100 about 1 s, N=300 about 85 s), a
  // periodic one ~N^1.5 (N=3200 about 67 s); it refuses more than 300
  // (bounded) / 3000 (periodic) seeds with an error pointing to Voro++.
  // python/CMakeLists.txt does not enable Voro++, so a default Python
  // build resolves "auto" to the in-house backend.
  auto parse_backend = [](std::string const& name) {
    if (name == "auto") {
      return rvegen::voro_pp_available() ? rvegen::voronoi_backend::voro_pp
                                         : rvegen::voronoi_backend::in_house;
    }
    if (name == "in_house") return rvegen::voronoi_backend::in_house;
    if (name == "voro_pp") return rvegen::voronoi_backend::voro_pp;
    throw py::value_error("backend must be 'auto', 'in_house' or 'voro_pp'");
  };
  py::class_<rvegen::voronoi_generator_3d<double>>(m, "VoronoiGenerator3d")
      .def(py::init([parse_backend](double Lx, double Ly, double Lz,
                                    std::vector<std::array<double, 3>> seeds,
                                    bool periodic, std::string const& backend,
                                    std::vector<double> weights) {
             return rvegen::voronoi_generator_3d<double>{
                 Lx, Ly, Lz, std::move(seeds), parse_backend(backend),
                 periodic, std::move(weights)};
           }),
           py::arg("Lx"), py::arg("Ly"), py::arg("Lz"), py::arg("seeds"),
           py::arg("periodic") = false, py::arg("backend") = "auto",
           py::arg("weights") = std::vector<double>{})
      .def_property_readonly(
          "seed_count",
          [](rvegen::voronoi_generator_3d<double> const& g) {
            return g.seed_count();
          })
      .def_property_readonly(
          "periodic",
          [](rvegen::voronoi_generator_3d<double> const& g) {
            return g.periodic();
          })
      .def("build",
           [](rvegen::voronoi_generator_3d<double> const& g) {
             std::vector<rvegen::voronoi_cell<double>> out;
             std::vector<std::size_t> empty;
             {
               // Tessellation is pure C++ and can take long: release the GIL.
               py::gil_scoped_release release;
               auto cells = g.build();
               empty = g.empty_cell_indices(cells);
               for (auto const& sh : cells) {
                 out.push_back(
                     static_cast<rvegen::voronoi_cell<double> const&>(*sh));
               }
             }
             if (!empty.empty()) {
               // Weighted (power-diagram) seeds can end up with no cell.
               std::string msg = std::to_string(empty.size()) +
                                 " seed(s) have an empty cell (e.g. index " +
                                 std::to_string(empty.front()) +
                                 "); the weights are too spread out";
               PyErr_WarnEx(PyExc_RuntimeWarning, msg.c_str(), 1);
               if (PyErr_Occurred()) throw py::error_already_set();
             }
             return out;
           },
           "Returns a list of VoronoiCell, one per seed (index = seed index).");

  // Lognormal grain-size control: seeds + Laguerre weights (see
  // grain_size_control.h). Returns a dict.
  m.def("lognormal_grain_seeds_3d",
        [parse_backend](double L, double mean_diameter, double sigma_ln,
                        unsigned rng_seed, std::string const& backend,
                        bool periodic, double tolerance,
                        std::size_t max_iterations, std::size_t max_restarts) {
          const auto be = parse_backend(backend);   // may throw: GIL held
          rvegen::grain_size_result<double> r;
          {
            // Pure C++ from here (many tessellations): let other Python
            // threads run. Ctrl-C is NOT delivered until the call returns
            // (no PyErr_CheckSignals is wired to progress_options::cancel).
            py::gil_scoped_release release;
            std::mt19937 eng{rng_seed};
            r = rvegen::lognormal_grain_seeds_3d<double>(
                L, L, L, mean_diameter, sigma_ln, eng, be, periodic, tolerance,
                max_iterations, max_restarts);
          }
          if (!r.converged) {
            PyErr_WarnEx(
                PyExc_RuntimeWarning,
                "lognormal_grain_seeds_3d did not reach the requested volume "
                "tolerance; the grain-size distribution is approximate "
                "(see 'max_rel_volume_error' / 'converged')",
                1);
            if (PyErr_Occurred()) throw py::error_already_set();
          }
          py::dict d;
          d["seeds"] = r.seeds;
          d["weights"] = r.weights;
          d["target_diameters"] = r.target_diameters;
          d["diameters"] = r.diameters;
          d["max_rel_volume_error"] = r.max_rel_volume_error;
          d["converged"] = r.converged;
          d["empty_cells"] = r.empty_cells;
          d["iterations"] = r.iterations;
          d["restarts"] = r.restarts;
          return d;
        },
        py::arg("L"), py::arg("mean_diameter"), py::arg("sigma_ln"),
        py::arg("rng_seed") = 0, py::arg("backend") = "auto",
        py::arg("periodic") = true, py::arg("tolerance") = 0.01,
        py::arg("max_iterations") = 1000, py::arg("max_restarts") = 2);

  // Deterministic weave generator (plain / twill 2/2 / satin 5) (#3 / #109). build() returns
  // a Python list of PolylineTube — every yarn the generator produces
  // IS a polyline_tube by construction, so the down-cast is type-safe.
  // Each tube in the list owns its own data (deep-copied on the way
  // out so Python lifetime is independent of the generator).
  py::enum_<rvegen::weave_pattern>(m, "WeavePattern")
      .value("plain", rvegen::weave_pattern::plain)
      .value("twill_2_2", rvegen::weave_pattern::twill_2_2)
      .value("satin_5", rvegen::weave_pattern::satin_5);

  // `pattern` accepts a WeavePattern or one of the strings "plain",
  // "twill_2_2", "satin_5".
  py::class_<rvegen::weave_generator<double>>(m, "WeaveGenerator")
      .def(py::init([](std::array<double, 3> const& box, std::size_t n_warp,
                       std::size_t n_weft, double r, double a,
                       std::size_t n_seg, py::object const& pattern) {
             rvegen::weave_pattern p = rvegen::weave_pattern::plain;
             if (py::isinstance<py::str>(pattern)) {
               const auto name = pattern.cast<std::string>();
               if (name == "plain") p = rvegen::weave_pattern::plain;
               else if (name == "twill_2_2") p = rvegen::weave_pattern::twill_2_2;
               else if (name == "satin_5") p = rvegen::weave_pattern::satin_5;
               else throw py::value_error(
                   "WeaveGenerator: unknown pattern '" + name +
                   "' (expected plain, twill_2_2 or satin_5)");
             } else {
               p = pattern.cast<rvegen::weave_pattern>();
             }
             return rvegen::weave_generator<double>(box, n_warp, n_weft, r, a,
                                                    n_seg, p);
           }),
           py::arg("domain_box"),
           py::arg("n_warp_yarns"),
           py::arg("n_weft_yarns"),
           py::arg("yarn_radius"),
           py::arg("amplitude"),
           py::arg("n_segments_per_yarn") = std::size_t{32},
           py::arg("pattern") = rvegen::weave_pattern::plain)
      .def_property_readonly(
          "pattern",
          [](rvegen::weave_generator<double> const& g) {
            return g.pattern();
          })
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
             auto raw = interruptible(
                 [&](rvegen::cancel_fn const* c) { return g.build(c); });
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

  // ---- Registry introspection: rvegen.<category>.keys() / .schema(name) ----
  bind_registry<rvegen::distribution_registry_t<>>(m, "distributions");
  bind_registry<rvegen::direction_distribution_registry_t<>>(
      m, "direction_distributions");
  bind_registry<rvegen::input_registry_t<>>(m, "inputs");
  bind_registry<rvegen::generator_registry_t<>>(m, "generators");
  bind_registry<rvegen::termination_registry_t<>>(m, "terminations");
  bind_registry<rvegen::post_process_registry_t<>>(m, "post_processes");
  m.attr("shapes") = m.attr("inputs");   // "shapes" config section == shape inputs

  // ---- Pipeline ----------------------------------------------------------
  py::class_<run_result>(m, "RunResult",
      "Shapes produced by rvegen.run(): a sequence of Shape objects plus a "
      "`summary` dict (seed, n_shapes, domain_box, volume_fraction, phases, "
      "postprocess).")
      .def_readonly("shapes", &run_result::shapes)
      .def_readonly("summary", &run_result::summary)
      .def("__len__", [](run_result const& r) { return r.shapes.size(); })
      .def("__iter__", [](run_result const& r) { return py::iter(r.shapes); })
      .def("__getitem__", [](run_result const& r, py::object idx) {
        return r.shapes.attr("__getitem__")(idx);
      });

  m.def("check",
      [](py::object const& config, std::optional<std::uint64_t> seed,
         bool strict, py::object const& input_dir,
         py::object const& output_dir, py::object log, bool verbose) {
        auto const cfg = load_config(config);
        rvegen::pipeline_options opts;
        opts.seed = seed;
        opts.strict_keys = strict;
        set_directories(opts, cfg, input_dir, output_dir);
        log_sink sink{std::move(log)};
        sink.attach(opts, verbose);
        std::vector<std::string> warnings_out;
        opts.warn = [&warnings_out](std::string const& w) {
          warnings_out.push_back(w);
        };
        rvegen::check_config(cfg.json, opts);
        sink.rethrow_if_failed();
        for (auto const& w : warnings_out) {
          if (PyErr_WarnEx(PyExc_UserWarning, w.c_str(), 1) < 0)
            throw py::error_already_set();
        }
      },
      py::arg("config"), py::arg("seed") = py::none(),
      py::arg("strict") = true, py::kw_only(),
      py::arg("input_dir") = py::none(), py::arg("output_dir") = py::none(),
      py::arg("log") = py::none(), py::arg("verbose") = false,
      "Validate a config without generating anything: every block is built, "
      "phases are parsed and the output directories of the writers are "
      "checked, exactly as run() does before it places the first shape. No "
      "file is written. Raises on the first problem, returns None when the "
      "config is valid. strict=False turns an unknown top-level key from an "
      "error into a UserWarning. input_dir, output_dir, log and verbose are "
      "those of run().");

  m.def("run",
      [](py::object const& config, std::optional<std::uint64_t> seed,
         bool postprocess, bool strict, py::object const& input_dir,
         py::object const& output_dir, py::object log, bool verbose) {
        auto const cfg = load_config(config);
        rvegen::pipeline_options opts;
        opts.seed = seed;
        opts.run_postprocess = postprocess;
        opts.strict_keys = strict;
        set_directories(opts, cfg, input_dir, output_dir);
        log_sink sink{std::move(log)};
        sink.attach(opts, verbose);
        std::vector<std::string> warnings_out;   // surfaced as UserWarning below
        opts.warn = [&warnings_out](std::string const& w) {
          warnings_out.push_back(w);
        };
        interrupt_watch watch;
        // A log callable that raised stops the run like Ctrl-C does.
        opts.progress.cancel = [poll = watch.hook(), &sink]() {
          return sink.failed() || poll();
        };
        rvegen::pipeline_result res;
        try {
          py::gil_scoped_release release;   // pure C++ until the result is built
          res = rvegen::run_pipeline(cfg.json, opts);
        } catch (rvegen::pipeline_cancelled const&) {
          sink.rethrow_if_failed();         // the log callable's exception
          watch.rethrow_if_interrupted();   // KeyboardInterrupt from Ctrl-C
          throw;
        }
        sink.rethrow_if_failed();
        for (auto const& w : warnings_out) {
          // -W error turns the warning into an exception: propagate it.
          if (PyErr_WarnEx(PyExc_UserWarning, w.c_str(), 1) < 0)
            throw py::error_already_set();
        }
        run_result out;
        py::dict sm;
        sm["seed"] = res.seed;
        sm["n_shapes"] = res.shapes.size();
        sm["domain_box"] = res.domain_box;
        sm["volume_fraction"] =
            rvegen::current_volume_fraction<double>(res.shapes, res.domain_box);
        std::vector<std::string> phase_names;
        if (res.phases)
          for (auto const* p : res.phases->ordered())
            phase_names.push_back(p->name);
        sm["phases"] = phase_names;
        sm["postprocess"] = res.post_processes_run;
        for (auto& sh : res.shapes) out.shapes.append(py::cast(std::move(sh)));
        out.summary = sm;
        return out;
      },
      py::arg("config"), py::arg("seed") = py::none(),
      py::arg("postprocess") = true, py::arg("strict") = true, py::kw_only(),
      py::arg("input_dir") = py::none(), py::arg("output_dir") = py::none(),
      py::arg("log") = py::none(), py::arg("verbose") = false,
      "Run a full rvegen pipeline. `config` is a dict (same schema as the "
      "CLI JSON) or a path to a JSON file; `seed` overrides the config "
      "seed. Writers under \"postprocess\" run unless postprocess=False. "
      "The whole config is validated before the first shape is placed; an "
      "unknown top-level key is an error unless strict=False (then a "
      "UserWarning). Returns a RunResult (sequence of shapes + summary).\n\n"
      "Paths: a relative input path in the config (mesh_path) is resolved "
      "against input_dir; by default that is the directory of the config "
      "file, or the working directory for a dict. A relative output_path "
      "is resolved against output_dir (default: the working directory).\n\n"
      "Output: run() prints nothing. log, a callable taking one str "
      "(log=print, a logger's .info), receives each progress line as it is "
      "produced; verbose=True adds the detail lines (defaults used per "
      "block, resolved paths, the seed). Warnings are always raised as "
      "UserWarning.");

  // ---- Voxelisation ---------------------------------------------------------
  m.def("voxelize",
      [](std::vector<rvegen::shape_base<double> const*> const& shapes,
         std::array<double, 3> domain, std::size_t nx, std::size_t ny,
         std::size_t nz) {
        if (nx == 0 || ny == 0 || nz == 0)
          throw py::value_error("rvegen.voxelize: nx, ny and nz must be >= 1");
        for (double d : domain)
          if (!std::isfinite(d))
            throw py::value_error("rvegen.voxelize: domain must be finite");
        if (!(domain[0] > 0.0) || !(domain[1] > 0.0) || domain[2] < 0.0)
          throw py::value_error(
              "rvegen.voxelize: domain needs x > 0, y > 0 and z >= 0");
        constexpr auto lim = std::numeric_limits<std::size_t>::max() / 4;
        if (nx > lim / ny || nx * ny > lim / nz)
          throw py::value_error("rvegen.voxelize: grid size overflows");
        // sample_voxel_grid wants unique_ptrs; clone() leaves the caller's
        // shapes untouched.
        std::vector<shape_ptr> owned;
        owned.reserve(shapes.size());
        for (auto const* s : shapes) owned.push_back(s->clone());
        std::vector<std::uint32_t> flat;
        interrupt_watch watch;
        rvegen::cancel_fn const cancel = watch.hook();
        try {
          py::gil_scoped_release release;
          flat = rvegen::sample_voxel_grid<double>(owned, domain, nx, ny, nz,
                                                   &cancel);
        } catch (rvegen::generation_cancelled const&) {
          watch.rethrow_if_interrupted();
          throw;
        }
        const std::size_t nz_eff = rvegen::effective_nz(domain, nz);
        // Core order is (k, j, i) with i fastest; expose it as an array
        // indexed [i, j, k] (shape (nx, ny, nz_eff)) without reshuffling.
        auto* heap = new std::vector<std::uint32_t>(std::move(flat));
        py::capsule owner{heap, [](void* p) {
          delete static_cast<std::vector<std::uint32_t>*>(p);
        }};
        constexpr auto sz = static_cast<py::ssize_t>(sizeof(std::uint32_t));
        return py::array_t<std::uint32_t>(
            {static_cast<py::ssize_t>(nx), static_cast<py::ssize_t>(ny),
             static_cast<py::ssize_t>(nz_eff)},
            {sz, sz * static_cast<py::ssize_t>(nx),
             sz * static_cast<py::ssize_t>(nx * ny)},
            heap->data(), owner);
      },
      py::arg("shapes"), py::arg("domain"), py::arg("nx"), py::arg("ny"),
      py::arg("nz") = 1,
      "Sample shapes onto an nx*ny*nz grid; returns a uint32 array indexed "
      "[i, j, k] with shape (nx, ny, nz_eff). The array is Fortran-ordered "
      "(F-contiguous, x fastest; zero-copy view of the sampled buffer): "
      "use numpy.ascontiguousarray() if you need C order. Voxel value = 1-based index of the first containing shape "
      "(0 = outside). 2D domains (z == 0) collapse to nz = 1.");
}
