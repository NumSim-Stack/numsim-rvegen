// Full-stack smoke: drive the entire RVE pipeline from a single JSON config
// through every registry. Essentially what the CLI does, written as a test.
//
// Pipeline:
//   1. register_all_*()  for distributions, terminations, generators,
//                        inputs, post-processes.
//   2. Parse JSON config with sections {distributions, shapes, generator,
//                                       termination, postprocess}.
//   3. For each named distribution: schema → handler → registry.create →
//      shared_ptr stored in distribution_map.
//   4. For each shape entry: schema → handler → input_registry.create
//      (passing the distribution_map for cross-reference resolution).
//   5. Generator + termination + post-process: schema → handler →
//      registry.create.
//   6. Run generator.compute() → shapes.
//   7. Run post_process.run() → output file written; read it back to
//      verify content.
//   8. Assert: 8 shapes, no AABB overlap, voxel file has the expected
//      header, gmsh post-process emits Disk entities.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                               \
    }                                                                          \
  } while (0)

// build_from_json now lives in rvegen/registry/build.h, pulled in via
// rvegen/rvegen.h. Use rvegen::build_from_json directly.

void test_full_pipeline_from_json() {
  // Wire every registry. Idempotent.
  rvegen::register_all_distributions<>();
  rvegen::register_all_terminations<>();
  rvegen::register_all_post_processes<>();
  rvegen::register_all_generators<>();
  rvegen::register_all_inputs<>();

  // Single JSON config drives the whole pipeline.
  const auto j = nlohmann::json::parse(R"({
    "distributions": {
      "uniform_x":      { "type": "uniform_real", "a": 0.05, "b": 0.95 },
      "uniform_y":      { "type": "uniform_real", "a": 0.05, "b": 0.95 },
      "uniform_radius": { "type": "uniform_real", "a": 0.03, "b": 0.05 }
    },
    "shapes": [
      { "type": "circle_input",
        "pos_x_dist":  "uniform_x",
        "pos_y_dist":  "uniform_y",
        "radius_dist": "uniform_radius" }
    ],
    "generator":   { "type": "only_inside", "max_attempts": 100000 },
    "termination": { "type": "number_of_inclusions", "target": 8 },
    "postprocess": [
      { "type": "voxel", "nx": 32, "ny": 32, "nz": 1,
        "output_path": "full_registry_smoke.vox" }
    ]
  })");

  // ---- distributions section: build the name → distribution map. ---------
  std::mt19937 engine{2026};
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : j.at("distributions").items()) {
    auto dist = rvegen::build_from_json(rvegen::distribution_registry_t<>::instance(),
                                spec, engine);
    // upgrade unique_ptr → shared_ptr so circle_input can take stable refs.
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
        std::move(dist)});
  }
  REQUIRE(dists.size() == 3);

  // ---- shapes section: each entry is a shape_input. ----------------------
  rvegen::only_inside_generator<double>::input_vector inputs;
  for (auto const& spec : j.at("shapes")) {
    auto input = rvegen::build_from_json(rvegen::input_registry_t<>::instance(),
                                  spec, dists);
    inputs.push_back(std::move(input));
  }
  REQUIRE(inputs.size() == 1);

  // ---- generator + termination + post-process ---------------------------
  auto generator   = rvegen::build_from_json(rvegen::generator_registry_t<>::instance(),
                                     j.at("generator"));
  const std::array<double, 3> domain_box{1.0, 1.0, 0.0};
  auto termination = rvegen::build_from_json(rvegen::termination_registry_t<>::instance(),
                                     j.at("termination"), domain_box);
  auto post_proc   = rvegen::build_from_json(rvegen::post_process_registry_t<>::instance(),
                                     j.at("postprocess").at(0));

  REQUIRE(generator   != nullptr);
  REQUIRE(termination != nullptr);
  REQUIRE(post_proc   != nullptr);

  // ---- run the pipeline -------------------------------------------------
  auto shapes = generator->compute(inputs, *termination, domain_box);
  REQUIRE(shapes.size() == 8);

  for (auto const& s : shapes) {
    REQUIRE(rvegen::aabb_inside(*s->bounding_box(),
                                 domain_box[0], domain_box[1], domain_box[2]));
  }
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    for (std::size_t k = i + 1; k < shapes.size(); ++k) {
      REQUIRE(!rvegen::aabb_overlap(*shapes[i]->bounding_box(),
                                     *shapes[k]->bounding_box()));
    }
  }

  // ---- run post-process: writes file to disk ----------------------------
  post_proc->run(shapes, domain_box);
  std::ifstream voxel_file{"full_registry_smoke.vox"};
  REQUIRE(voxel_file.is_open());
  std::stringstream voxel_buf;
  voxel_buf << voxel_file.rdbuf();
  REQUIRE(voxel_buf.str().find("# rvegen voxel grid") != std::string::npos);
  std::remove("full_registry_smoke.vox");
}

void test_alternate_post_process() {
  // Same registries, swap post-process type to gmsh_geo.
  rvegen::register_all_distributions<>();
  rvegen::register_all_post_processes<>();

  const auto spec = nlohmann::json::parse(R"({
    "type": "gmsh_geo",
    "output_path": "full_registry_smoke_alt.geo"
  })");
  auto writer = rvegen::build_from_json(
      rvegen::post_process_registry_t<>::instance(), spec);
  REQUIRE(writer != nullptr);

  rvegen::post_process_base<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  writer->run(shapes, {1.0, 1.0, 0.0});
  std::ifstream geo_file{"full_registry_smoke_alt.geo"};
  REQUIRE(geo_file.is_open());
  std::stringstream geo_buf;
  geo_buf << geo_file.rdbuf();
  REQUIRE(geo_buf.str().find("Disk(") != std::string::npos);
  std::remove("full_registry_smoke_alt.geo");
}

void test_unknown_type_in_each_registry_throws() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_terminations<>();
  rvegen::register_all_post_processes<>();
  rvegen::register_all_generators<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine;
  rvegen::distribution_map_t<double> empty_map;
  rvegen::parameter_handler_t empty_handler;
  std::array<double, 3> empty_box{0.0, 0.0, 0.0};

  bool threw = false;
  try { rvegen::distribution_registry_t<>::instance().create("nope", empty_handler, engine); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);

  threw = false;
  try { rvegen::input_registry_t<>::instance().create("nope", empty_handler, empty_map); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);

  threw = false;
  try { rvegen::termination_registry_t<>::instance().create("nope", empty_handler, empty_box); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

} // namespace

int main() {
  test_full_pipeline_from_json();
  test_alternate_post_process();
  test_unknown_type_in_each_registry_throws();

  if (failures > 0) {
    std::cerr << failures << " full-registry smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "full_registry_smoke: PASS\n";
  return EXIT_SUCCESS;
}
