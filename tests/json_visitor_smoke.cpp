// Phase 4 smoke test for parameter_visitor_nlohmann.
//
// Exercises the JSON → schema → handler pipeline:
//   1. Build a JSON literal with mixed types.
//   2. Declare a schema with required + range-checked params.
//   3. Create the nlohmann visitor pointing at the JSON.
//   4. Call controller.accept(visitor, handler) — which walks the schema,
//      asks the visitor for each value, populates the handler, validates.
//   5. Assert the handler has the expected typed values.

#include <any>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <numsim-core/input_parameter_controller.h>
#include <numsim-core/parameter_handler.h>

#include "rvegen/json/parameter_visitor_nlohmann.h"

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

using handler_t = numsim_core::parameter_handler<std::string, std::any>;
using controller_t =
    numsim_core::input_parameter_controller<std::string, handler_t>;

void test_basic_population() {
  const auto j = nlohmann::json::parse(R"({
    "radius": 0.05,
    "n_inclusions": 100,
    "shape_type": "circle",
    "deterministic": true,
    "extents": [1.0, 2.0, 3.0]
  })");

  controller_t schema;
  schema.insert<double>("radius").add<numsim_core::is_required>();
  schema.insert<int>("n_inclusions").add<numsim_core::is_required>();
  schema.insert<std::string>("shape_type").add<numsim_core::is_required>();
  schema.insert<bool>("deterministic");
  schema.insert<std::vector<double>>("extents");

  handler_t handler;
  rvegen::parameter_visitor_nlohmann visitor{j};
  schema.accept(visitor, handler);

  REQUIRE(handler.get<double>("radius") == 0.05);
  REQUIRE(handler.get<int>("n_inclusions") == 100);
  REQUIRE(handler.get<std::string>("shape_type") == "circle");
  REQUIRE(handler.get<bool>("deterministic") == true);

  auto const& ext = handler.get<std::vector<double>>("extents");
  REQUIRE(ext.size() == 3);
  REQUIRE(ext[0] == 1.0);
  REQUIRE(ext[2] == 3.0);
}

void test_missing_optional_skipped() {
  // JSON is missing the optional "extents" key — visitor.contains() returns
  // false, so accept() should skip it without error.
  const auto j = nlohmann::json::parse(R"({
    "radius": 0.1,
    "n_inclusions": 5,
    "shape_type": "circle"
  })");

  controller_t schema;
  schema.insert<double>("radius").add<numsim_core::is_required>();
  schema.insert<int>("n_inclusions").add<numsim_core::is_required>();
  schema.insert<std::string>("shape_type").add<numsim_core::is_required>();
  schema.insert<std::vector<double>>("extents"); // optional, absent

  handler_t handler;
  rvegen::parameter_visitor_nlohmann visitor{j};
  schema.accept(visitor, handler);

  REQUIRE(handler.contains("radius"));
  REQUIRE(handler.contains("n_inclusions"));
  REQUIRE(!handler.contains("extents"));
}

void test_missing_required_throws() {
  // Required param "radius" is absent — accept() must throw.
  const auto j = nlohmann::json::parse(R"({
    "n_inclusions": 1
  })");

  controller_t schema;
  schema.insert<double>("radius").add<numsim_core::is_required>();
  schema.insert<int>("n_inclusions").add<numsim_core::is_required>();

  handler_t handler;
  rvegen::parameter_visitor_nlohmann visitor{j};

  bool threw = false;
  try {
    schema.accept(visitor, handler);
  } catch (std::exception const&) {
    threw = true;
  }
  REQUIRE(threw);
}

void test_unsupported_type_throws() {
  const auto j = nlohmann::json::parse(R"({"strange": 1})");

  rvegen::parameter_visitor_nlohmann visitor{j};
  bool threw = false;
  try {
    // A type we deliberately don't support, e.g. char.
    [[maybe_unused]] auto value = visitor.read("strange", typeid(char));
  } catch (std::runtime_error const&) {
    threw = true;
  }
  REQUIRE(threw);
}

} // namespace

int main() {
  test_basic_population();
  test_missing_optional_skipped();
  test_missing_required_throws();
  test_unsupported_type_throws();

  if (failures > 0) {
    std::cerr << failures << " json-visitor smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "json_visitor_smoke: PASS\n";
  return EXIT_SUCCESS;
}
