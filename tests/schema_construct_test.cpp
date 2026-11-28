// Key-drift guard. Schemas are declared in parameters() and read again by
// name in each constructor, so a typo in either place is no longer caught at
// compile time. This test builds a minimal valid config from every registered
// type's own schema and constructs the type through build_from_json; a
// constructor reading a key its schema does not declare throws (missing
// parameter) and fails the test.
//
// Config generation, per schema key:
//   * keys carrying set_default are left out (the controller fills them);
//   * double / size_t / int values are placed inside the declared range
//     (midpoint of a closed range, min + 1 for a lower bound only,
//     0.5 / 4 when unconstrained);
//   * bool keys are set to true;
//   * string keys ending in "_dist" name a constant distribution "d" in the
//     distribution map handed to the inputs ("direction_dist" names the
//     direction distribution "dir"); "output_path" points into the temp
//     directory; other string keys are only set when required;
//   * optional list keys (voronoi "seeds") are left out.
//
// A few types need values a range hint cannot express; `adjust` sets them
// per type (orientation tensors with trace 1, weave yarn geometry that
// fits the box, a voronoi seed count); voronoi_3d is run a second time in
// grain-size mode, which reads its own keys.
//
// Covered: every type in the distribution, direction_distribution, generator,
// termination, input and post_process registries, plus the shapes circle,
// rectangle, ellipse, sphere and box (not registered; constructed directly
// from a handler built from Shape::parameters()).
//
// The deterministic generators (weave, tow_weave, voronoi_2d, voronoi_3d)
// keep the handler and read their keys in build(), so for them the test
// also runs compute() once.
//
// mesh_inclusion_input needs a real mesh file, so it is handled separately:
// the test writes a closed tetrahedron as an ASCII STL into the working
// directory, points mesh_path at it and constructs the input once per
// orientation mode ("uniform" with orientation_dist, "euler_zyz" with the
// three euler_*_dist keys; each mode rejects the other's keys). No
// registered type is skipped.
//
// Not exercised:
//   * Optional string keys that are not distribution references
//     (phase_name, metadata, stl_path) are not exercised.

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <typeindex>
#include <vector>

#include <nlohmann/json.hpp>

#include <numsim-core/input_parameter_controller.h>

#include "rvegen/rvegen.h"
#include "rvegen/shapes/box.h"
#include "rvegen/shapes/circle.h"
#include "rvegen/shapes/ellipse.h"
#include "rvegen/shapes/rectangle.h"
#include "rvegen/shapes/sphere.h"

namespace {

int failures = 0;
int constructed = 0;

using base_t = numsim_core::input_parameter_base<
    std::string, rvegen::parameter_handler_t>;
template <typename T>
using typed_t = numsim_core::input_parameter<
    T, std::string, rvegen::parameter_handler_t>;
using json = nlohmann::json;

template <typename T>
bool has_default(typed_t<T> const& p) {
  for (auto const& c : p.checks()) {
    if (dynamic_cast<numsim_core::set_default<
            T, std::string, rvegen::parameter_handler_t> const*>(c.get()))
      return true;
  }
  return false;
}

template <typename T>
bool is_req(typed_t<T> const& p) {
  for (auto const& c : p.checks()) {
    if (dynamic_cast<numsim_core::is_required<
            T, std::string, rvegen::parameter_handler_t> const*>(c.get()))
      return true;
  }
  return false;
}

// Picks a value inside the slot's range hint, if any.
template <typename T>
T pick_number(typed_t<T> const& p, T unconstrained) {
  for (auto const& c : p.checks()) {
    auto const* r = dynamic_cast<numsim_core::range_hint_base const*>(c.get());
    if (!r) continue;
    T const lo = std::any_cast<T>(r->min_value());
    T const hi = std::any_cast<T>(r->max_value());
    T const lim = std::numeric_limits<T>::max();
    if (hi >= lim / 2) return lo + T{1};
    return static_cast<T>(lo + (hi - lo) / T{2});
  }
  return unconstrained;
}

// Adds "key": value to `cfg` for one schema slot; returns false for an
// unknown value type (reported as a failure by the caller).
bool fill(json& cfg, std::string const& key, base_t const& base) {
  if (auto const* p = dynamic_cast<typed_t<double> const*>(&base)) {
    if (!has_default(*p)) cfg[key] = pick_number<double>(*p, 0.5);
    return true;
  }
  if (auto const* p = dynamic_cast<typed_t<std::size_t> const*>(&base)) {
    if (!has_default(*p)) cfg[key] = pick_number<std::size_t>(*p, 4);
    return true;
  }
  if (auto const* p = dynamic_cast<typed_t<int> const*>(&base)) {
    if (!has_default(*p)) cfg[key] = pick_number<int>(*p, 4);
    return true;
  }
  if (auto const* p = dynamic_cast<typed_t<bool> const*>(&base)) {
    if (!has_default(*p)) cfg[key] = true;
    return true;
  }
  if (auto const* p = dynamic_cast<typed_t<std::string> const*>(&base)) {
    if (has_default(*p)) return true;
    if (key == "direction_dist") {
      cfg[key] = "dir";
    } else if (key.size() > 5 && key.ends_with("_dist")) {
      cfg[key] = "d";
    } else if (key == "output_path") {
      cfg[key] = (std::filesystem::temp_directory_path() /
                  "rvegen_schema_construct_test.out").string();
    } else if (is_req(*p)) {
      cfg[key] = "x";
    }
    return true;
  }
  if (auto const* p = dynamic_cast<typed_t<std::uint64_t> const*>(&base)) {
    if (!has_default(*p)) cfg[key] = pick_number<std::uint64_t>(*p, 4);
    return true;
  }
  if (auto const* p = dynamic_cast<typed_t<std::vector<double>> const*>(&base)) {
    // Optional lists (explicit voronoi seeds) are left out.
    return !is_req(*p);
  }
  return false;
}

// Values a range hint cannot express.
void adjust(json& cfg, std::string const& type) {
  if (type == "bingham_a2") {   // axisymmetric, trace 1
    cfg["a11"] = 0.5; cfg["a22"] = 0.25; cfg["a33"] = 0.25;
    cfg["a12"] = 0.0; cfg["a13"] = 0.0; cfg["a23"] = 0.0;
  } else if (type == "von_mises_a2") {   // trace 1, a12^2 <= a11 a22
    cfg["a11"] = 0.7; cfg["a22"] = 0.3; cfg["a12"] = 0.2;
  } else if (type == "weave" || type == "tow_weave") {
    cfg["n_warp_yarns"] = 2; cfg["n_weft_yarns"] = 2;
    cfg["yarn_radius"] = 0.04; cfg["amplitude"] = 0.06;
    if (type == "tow_weave") cfg["n_fibres_per_tow"] = 3;
  } else if (type == "voronoi_2d" || type == "voronoi_3d") {
    cfg["n_seeds"] = 6;   // defaults to 0 = "use the explicit seeds list"
    // Grain-size control replaces n_seeds; it gets its own run below.
    cfg.erase("grain_diameter_mean");
  }
}

json make_config(rvegen::parameter_controller_t const& schema,
                 std::string const& type_name) {
  json cfg;
  cfg["type"] = type_name;
  for (auto const& [key, param] : schema) {
    if (!fill(cfg, key, *param)) {
      std::cerr << "FAIL " << type_name << ": no value rule for key '" << key
                << "'\n";
      ++failures;
    }
  }
  adjust(cfg, type_name);
  return cfg;
}

template <typename Fn>
void attempt(std::string const& registry, std::string const& type, Fn&& fn) {
  try {
    fn();
    ++constructed;
  } catch (std::exception const& e) {
    std::cerr << "FAIL " << registry << "/" << type << ": " << e.what()
              << "\n";
    ++failures;
  }
}

template <typename Registry, typename... Args>
void run_registry(Registry& reg, std::string const& name,
                  std::string const& skip, Args&... extra) {
  for (auto const& type : reg.registered_types()) {
    if (type == skip) continue;
    attempt(name, type, [&] {
      auto cfg = make_config(reg.schema(type), type);
      auto obj = rvegen::build_from_json(reg, cfg, extra...);
      if (!obj) throw std::runtime_error("null object");
    });
  }
}

template <typename Shape>
void run_shape(std::string const& type) {
  attempt("shape", type, [&] {
    auto const schema = Shape::parameters();
    auto cfg = make_config(schema, type);
    rvegen::parameter_handler_t handler;
    rvegen::parameter_visitor_nlohmann visitor{cfg};
    schema.accept(visitor, handler);
    Shape shape{handler};
    (void)shape;
  });
}

// Writes a small closed tetrahedron as an ASCII STL.
void write_tetrahedron_stl(std::filesystem::path const& path) {
  std::ofstream out{path};
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << "solid tetra\n";
  auto facet = [&](char const* n, char const* a, char const* b, char const* c) {
    out << "facet normal " << n << "\nouter loop\nvertex " << a
        << "\nvertex " << b << "\nvertex " << c << "\nendloop\nendfacet\n";
  };
  facet("0 0 -1", "0 0 0", "0 1 0", "1 0 0");
  facet("0 -1 0", "0 0 0", "1 0 0", "0 0 1");
  facet("-1 0 0", "0 0 0", "0 0 1", "0 1 0");
  facet("0.577 0.577 0.577", "1 0 0", "0 1 0", "0 0 1");
  out << "endsolid tetra\n";
}

} // namespace

int main() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_direction_distributions<>();
  rvegen::register_all_generators<>();
  rvegen::register_all_terminations<>();
  rvegen::register_all_inputs<>();
  rvegen::register_all_post_processes<>();

  std::mt19937 engine{42};
  run_registry(rvegen::distribution_registry_t<>::instance(), "distribution",
               "", engine);
  run_registry(rvegen::direction_distribution_registry_t<>::instance(),
               "direction_distribution", "", engine);
  run_registry(rvegen::generator_registry_t<>::instance(), "generator", "");
  // Deterministic generators read their keys in build(): run them once.
  {
    auto& reg = rvegen::generator_registry_t<>::instance();
    for (auto const& type : reg.registered_types()) {
      attempt("generator(compute)", type, [&] {
        auto gen = rvegen::build_from_json(reg, make_config(reg.schema(type), type));
        if (!dynamic_cast<rvegen::deterministic_generator<double>*>(gen.get()))
          return;
        rvegen::rve_generator_base<double>::input_vector none;
        rvegen::number_of_inclusions<double> one{1};
        std::array<double, 3> const dom{1.0, 1.0,
                                        type == "voronoi_2d" ? 0.0 : 0.4};
        if (gen->compute(none, one, dom).empty())
          throw std::runtime_error("no shapes built");
      });
    }
    // voronoi_3d in grain-size mode reads a different set of keys.
    attempt("generator(compute)", "voronoi_3d[grain_size]", [&] {
      auto cfg = make_config(reg.schema("voronoi_3d"), "voronoi_3d");
      cfg.erase("n_seeds");
      cfg["grain_diameter_mean"] = 0.5;
      auto gen = rvegen::build_from_json(reg, cfg);
      rvegen::rve_generator_base<double>::input_vector none;
      rvegen::number_of_inclusions<double> one{1};
      if (gen->compute(none, one, {1.0, 1.0, 1.0}).empty())
        throw std::runtime_error("no shapes built");
    });
  }
  std::array<double, 3> const box{1.0, 1.0, 1.0};
  run_registry(rvegen::termination_registry_t<>::instance(), "termination",
               "", box);
  run_registry(rvegen::post_process_registry_t<>::instance(), "post_process",
               "");

  // Inputs look distributions up by name; every "*_dist" key resolves to "d".
  rvegen::distribution_map_t<double> dists;
  dists["d"] = std::make_shared<
      rvegen::constant_distribution<double, std::mt19937>>(0.5, engine);
  dists["dir"] = std::make_shared<rvegen::direction_distribution_holder<double>>(
      std::shared_ptr<rvegen::direction_distribution_base<double>>{
          std::make_shared<rvegen::bingham_distribution<double, std::mt19937>>(
              std::array<double, 3>{0.0, 0.0, 1.0}, 2.0, engine)});
  run_registry(rvegen::input_registry_t<>::instance(), "input",
               "mesh_inclusion_input", dists);

  // mesh_inclusion_input: same schema-driven config, plus a real mesh file.
  {
    auto const stl = std::filesystem::current_path() /
                     "rvegen_schema_construct_test_tetra.stl";
    for (std::string const mode : {"uniform", "euler_zyz"}) {
      attempt("input", "mesh_inclusion_input[" + mode + "]", [&] {
        write_tetrahedron_stl(stl);
        auto& reg = rvegen::input_registry_t<>::instance();
        auto cfg = make_config(reg.schema("mesh_inclusion_input"),
                               "mesh_inclusion_input");
        cfg["mesh_path"] = stl.string();
        cfg["orientation"] = mode;
        if (mode == "uniform") {
          for (auto const* k : {"euler_alpha_dist", "euler_beta_dist",
                                "euler_gamma_dist"})
            cfg.erase(k);
        } else {
          cfg.erase("orientation_dist");
        }
        auto obj = rvegen::build_from_json(reg, cfg, dists);
        if (!obj) throw std::runtime_error("null object");
      });
    }
    std::error_code ec;
    std::filesystem::remove(stl, ec);
  }

  run_shape<rvegen::circle<>>("circle");
  run_shape<rvegen::rectangle<>>("rectangle");
  run_shape<rvegen::ellipse<>>("ellipse");
  run_shape<rvegen::sphere<>>("sphere");
  run_shape<rvegen::box<>>("box");

  if (failures != 0) {
    std::cerr << "schema_construct_test: " << failures << " failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "schema_construct_test: PASS (" << constructed
            << " types constructed)\n";
  return EXIT_SUCCESS;
}
