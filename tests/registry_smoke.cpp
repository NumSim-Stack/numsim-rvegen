// Phase 5 smoke test: end-to-end JSON-driven instantiation through the registry.
//
// This is the architectural lynchpin for "as flexible as possible." Adding a
// new distribution type = new file + register_X() call. JSON config picks it
// by name. Zero edits to existing code.
//
// Pipeline exercised here:
//   1. register_all_distributions() — populates the singleton registry
//   2. JSON config:  {"type": "uniform_real", "a": 0.0, "b": 1.0}
//   3. registry.schema("uniform_real") → parameter_controller_t (the schema)
//   4. parameter_visitor_nlohmann walks JSON → fills parameter_handler_t
//   5. registry.create("uniform_real", handler, engine) → unique_ptr<distribution_base>
//   6. (*dist)() — sample. Verify in [a, b].

#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"

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

void test_create_uniform_real_from_json() {
  // Once-per-process registration. Idempotent: register_type overwrites by
  // key, so calling twice is fine for tests.
  rvegen::register_all_distributions<>();

  auto& registry = rvegen::distribution_registry_t<>::instance();
  REQUIRE(registry.contains("uniform_real"));

  // Caller picks the type by name (typically from JSON's "type" field).
  const std::string type_name = "uniform_real";

  // Schema for that type — used to populate a parameter_handler.
  auto schema = registry.schema(type_name);

  // The user's JSON config for this distribution instance.
  const auto j = nlohmann::json::parse(R"({"a": 1.5, "b": 4.5})");

  rvegen::parameter_handler_t handler;
  rvegen::parameter_visitor_nlohmann visitor{j};
  schema.accept(visitor, handler);

  REQUIRE(handler.get<double>("a") == 1.5);
  REQUIRE(handler.get<double>("b") == 4.5);

  // Construct via the registry.
  std::mt19937 engine{2026};
  auto dist = registry.create(type_name, handler, engine);
  REQUIRE(dist != nullptr);

  // Sample and verify in range.
  for (int i = 0; i < 100; ++i) {
    const auto v = (*dist)();
    REQUIRE(v >= 1.5 && v <= 4.5);
  }
}

void test_unknown_type_throws() {
  rvegen::register_all_distributions<>();
  auto& registry = rvegen::distribution_registry_t<>::instance();

  rvegen::parameter_handler_t handler;
  std::mt19937 engine{0};

  bool threw = false;
  try {
    [[maybe_unused]] auto dist = registry.create("nonexistent_type", handler, engine);
  } catch (std::runtime_error const&) {
    threw = true;
  }
  REQUIRE(threw);
}

void test_missing_required_in_json_throws() {
  rvegen::register_all_distributions<>();
  auto& registry = rvegen::distribution_registry_t<>::instance();

  // JSON is missing required "b".
  const auto j = nlohmann::json::parse(R"({"a": 0.0})");

  auto schema = registry.schema("uniform_real");
  rvegen::parameter_handler_t handler;
  rvegen::parameter_visitor_nlohmann visitor{j};

  bool threw = false;
  try {
    schema.accept(visitor, handler);
  } catch (std::exception const&) {
    threw = true;
  }
  REQUIRE(threw);
}

} // namespace

int main() {
  test_create_uniform_real_from_json();
  test_unknown_type_throws();
  test_missing_required_in_json_throws();

  if (failures > 0) {
    std::cerr << failures << " registry smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "registry_smoke: PASS\n";
  return EXIT_SUCCESS;
}
