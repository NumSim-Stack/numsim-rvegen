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
//     distribution map handed to the inputs; "output_path" points into the
//     temp directory; other string keys are only set when required.
//
// Covered: every type in the distribution, direction_distribution, generator,
// termination, input and post_process registries, plus the shapes circle,
// rectangle, ellipse, sphere and box (not registered; constructed directly
// from a handler built from Shape::parameters()).
//
// Skipped:
//   * mesh_inclusion_input: needs a real mesh file on disk.
//   * Optional string keys that are not distribution references
//     (phase_name, metadata, stl_path) are not exercised.

#include <array>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <typeindex>

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
    if (key.size() > 5 && key.ends_with("_dist")) {
      cfg[key] = "d";
    } else if (key == "output_path") {
      cfg[key] = (std::filesystem::temp_directory_path() /
                  "rvegen_schema_construct_test.out").string();
    } else if (is_req(*p)) {
      cfg[key] = "x";
    }
    return true;
  }
  return false;
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
  std::array<double, 3> const box{1.0, 1.0, 1.0};
  run_registry(rvegen::termination_registry_t<>::instance(), "termination",
               "", box);
  run_registry(rvegen::post_process_registry_t<>::instance(), "post_process",
               "");

  // Inputs look distributions up by name; every "*_dist" key resolves to "d".
  rvegen::distribution_map_t<double> dists;
  dists["d"] = std::make_shared<
      rvegen::constant_distribution<double, std::mt19937>>(0.5, engine);
  run_registry(rvegen::input_registry_t<>::instance(), "input",
               "mesh_inclusion_input", dists);

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
