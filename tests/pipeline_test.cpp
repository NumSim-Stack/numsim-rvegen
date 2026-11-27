// rvegen::run_pipeline / generate (the library entry shared by the CLI and
// the Python bindings) and schema_to_json (#4).

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/pipeline.h"
#include "rvegen/schema/schema_to_json.h"

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

using json = nlohmann::json;

json circles_config() {
  return json::parse(R"({
    "domain_box": [1.0, 1.0, 0.0],
    "seed": 5,
    "distributions": {
      "x": {"type": "uniform_real", "a": 0.1, "b": 0.9},
      "r": {"type": "uniform_real", "a": 0.03, "b": 0.05}
    },
    "shapes": [{"type": "circle_input", "pos_x_dist": "x",
                "pos_y_dist": "x", "radius_dist": "r"}],
    "generator": {"type": "only_inside", "max_attempts": 100000},
    "termination": {"type": "number_of_inclusions", "target": 6},
    "phases": [{"name": "fibre", "material": {}}],
    "phase_assignment": {"all_inclusions": "fibre"}
  })");
}

void test_generate() {
  std::vector<std::string> lines;
  rvegen::pipeline_options opts;
  opts.info = [&](std::string const& l) { lines.push_back(l); };
  auto r = rvegen::run_pipeline(circles_config(), opts);
  REQUIRE(r.shapes.size() == 6);
  REQUIRE(r.seed == 5);
  REQUIRE(r.phases && r.phases->size() == 1);
  REQUIRE(r.shapes[0]->phase_name() == "fibre");
  REQUIRE(lines.size() >= 2);
  REQUIRE(lines[0] == "rvegen: generated 6 shapes");
  REQUIRE(lines[1] == "rvegen: phase_assignment tagged 6 shapes");

  // Seed override changes the layout; same seed reproduces it.
  rvegen::pipeline_options o2;
  o2.seed = 9;
  auto a = rvegen::generate(circles_config(), o2);
  auto b = rvegen::generate(circles_config(), o2);
  REQUIRE(a.seed == 9);
  REQUIRE(a.shapes[0]->get_middle_point() == b.shapes[0]->get_middle_point());
  REQUIRE(a.shapes[0]->get_middle_point() != r.shapes[0]->get_middle_point());
}

void test_postprocess_toggle() {
  auto cfg = circles_config();
  cfg["postprocess"] = json::array({json{{"type", "damask_material_config"},
                                         {"output_path", "pipeline_test.yaml"}}});
  cfg["phases"][0]["material"] = json{{"E", 1.0e9}, {"nu", 0.3}};
  cfg["phases"].push_back(json{{"name", "matrix"},
                               {"material", json{{"E", 1.0e8}, {"nu", 0.3}}}});
  rvegen::pipeline_options off;
  off.run_postprocess = false;
  REQUIRE(rvegen::run_pipeline(cfg, off).post_processes_run.empty());
  auto on = rvegen::run_pipeline(cfg);
  REQUIRE(on.post_processes_run.size() == 1);
  REQUIRE(on.post_processes_run[0] == "damask_material_config");
}

void test_seed_and_warnings() {
  auto cfg = circles_config();
  cfg["seed"] = 4294967296ULL;            // would silently truncate in mt19937
  bool threw = false;
  try { (void)rvegen::generate(cfg); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  cfg["seed"] = -3;
  threw = false;
  try { (void)rvegen::generate(cfg); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  rvegen::pipeline_options big;
  big.seed = 1ULL << 40;                  // override path is checked too
  threw = false;
  try { (void)rvegen::generate(circles_config(), big); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);

  // Deterministic generator + shapes/termination + unknown key -> warnings.
  auto det = circles_config();
  det["generator"] = json{{"type", "voronoi_2d"}, {"n_seeds", 3}, {"seed", 1}};
  det["bogus"] = 1;
  std::vector<std::string> warns;
  rvegen::pipeline_options o;
  o.warn = [&](std::string const& w) { warns.push_back(w); };
  (void)rvegen::generate(det, o);
  auto has = [&](char const* t) {
    for (auto const& w : warns) if (w.find(t) != std::string::npos) return true;
    return false;
  };
  REQUIRE(has("\"shapes\" is ignored"));
  REQUIRE(has("\"termination\" is ignored"));
  REQUIRE(has("unknown top-level key 'bogus'"));
}

void test_schema_to_json() {
  rvegen::register_all_types();
  auto j = rvegen::schema_to_json(
      rvegen::generator_registry_t<>::instance().schema("weave"));
  REQUIRE(j.at("n_warp_yarns").at("type") == "size_t");
  REQUIRE(j.at("n_warp_yarns").at("required") == true);
  REQUIRE(j.at("n_warp_yarns").at("min") == 1);
  REQUIRE(j.at("yarn_radius").at("type") == "double");
  REQUIRE(j.at("n_segments_per_yarn").at("has_default") == true);
  REQUIRE(j.at("n_segments_per_yarn").at("required") == false);
  REQUIRE(!j.at("amplitude").at("description").get<std::string>().empty());
}

} // namespace

int main() {
  test_generate();
  test_postprocess_toggle();
  test_seed_and_warnings();
  test_schema_to_json();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "pipeline_test OK\n";
  return EXIT_SUCCESS;
}
