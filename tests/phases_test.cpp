// JSON phases path (#7): phase_assignment forms, the generic
// post_process_base::set_phases hook, and phase-loading validation.

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
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

using json = nlohmann::json;
using shapes_t = std::vector<std::unique_ptr<rvegen::shape_base<double>>>;

shapes_t sample_shapes() {
  shapes_t v;
  v.push_back(std::make_unique<rvegen::circle<double>>(0.25, 0.25, 0.1));
  v.push_back(std::make_unique<rvegen::circle<double>>(0.75, 0.25, 0.1));
  v.push_back(std::make_unique<rvegen::circle<double>>(0.25, 0.75, 0.1));
  v.push_back(std::make_unique<rvegen::rectangle<double>>(0.75, 0.75, 0.2, 0.2));
  return v;
}

rvegen::phase_collection<double> make_phases() {
  return rvegen::load_phases_from_json<double>(json::parse(R"([
    {"name": "matrix"},
    {"name": "fibre"},
    {"name": "filler"}
  ])"));
}

void test_all_inclusions() {
  auto phases = make_phases();
  auto s = sample_shapes();
  auto n = rvegen::phase_assignment<double>::from_json(
               json{{"all_inclusions", "fibre"}}, phases).apply(s);
  REQUIRE(n == 4);
  for (auto const& sh : s) REQUIRE(sh->phase_name() == "fibre");
}

void test_by_shape_type_and_precedence() {
  auto phases = make_phases();
  auto s = sample_shapes();
  s[1]->info().set_phase_name("filler");   // input-set name must survive
  auto a = rvegen::phase_assignment<double>::from_json(
      json::parse(R"({"all_inclusions": "matrix",
                      "by_shape_type": {"rectangle": "filler"},
                      "by_index": {"0": "fibre"}})"), phases);
  REQUIRE(a.apply(s) == 3);
  REQUIRE(s[0]->phase_name() == "fibre");    // by_index beats all_inclusions
  REQUIRE(s[1]->phase_name() == "filler");   // pre-set, untouched
  REQUIRE(s[2]->phase_name() == "matrix");   // all_inclusions fallback
  REQUIRE(s[3]->phase_name() == "filler");   // by_shape_type
}

void test_by_index_ranges() {
  auto phases = make_phases();
  auto s = sample_shapes();
  rvegen::phase_assignment<double>::from_json(
      json::parse(R"({"by_index": {"0": "matrix", "1-2": "fibre", "3-": "filler"}})"),
      phases).apply(s);
  REQUIRE(s[0]->phase_name() == "matrix");
  REQUIRE(s[1]->phase_name() == "fibre");
  REQUIRE(s[2]->phase_name() == "fibre");
  REQUIRE(s[3]->phase_name() == "filler");
}

void test_assignment_errors() {
  auto phases = make_phases();
  auto bad = [&](char const* text) {
    try {
      (void)rvegen::phase_assignment<double>::from_json(json::parse(text), phases);
    } catch (std::exception const&) { return true; }
    return false;
  };
  REQUIRE(bad(R"({"all_inclusions": "nope"})"));
  REQUIRE(bad(R"({"by_shape_type": {"hexagon": "fibre"}})"));
  REQUIRE(bad(R"({"by_index": {"x": "fibre"}})"));
  REQUIRE(bad(R"({"by_index": {"5-2": "fibre"}})"));
  REQUIRE(bad(R"({"surprise": "fibre"})"));
}

void test_set_phases_hook_reaches_writers() {
  rvegen::register_all_post_processes<>();
  auto phases = make_phases();
  auto& reg = rvegen::post_process_registry_t<>::instance();
  REQUIRE(!reg.contains("damask_material_config"));   // geometry only

  auto pp = rvegen::build_from_json(
      reg, json{{"type", "voxel"}, {"nx", 4}, {"ny", 4}, {"nz", 1},
                {"output_path", "x.vox"}});
  pp->set_phases(&phases);   // through post_process_base
  auto* vw = dynamic_cast<rvegen::voxel_writer<double>*>(pp.get());
  REQUIRE(vw != nullptr);
  REQUIRE(vw->phases() == &phases);   // existing writer overrode the hook

  auto gm = rvegen::build_from_json(
      reg, json{{"type", "gmsh_geo"}, {"output_path", "x.geo"}});
  gm->set_phases(&phases);
  auto* gw = dynamic_cast<rvegen::gmsh_geo_writer<double>*>(gm.get());
  REQUIRE(gw != nullptr && gw->phases() == &phases);
}

// Phases are geometry labels: a "material" entry or any other key besides
// "name" is rejected at load time, naming the phase and the key.
void test_phases_reject_material_and_unknown_keys() {
  auto load_error = [](char const* text) -> std::string {
    try {
      (void)rvegen::load_phases_from_json<double>(json::parse(text));
    } catch (std::runtime_error const& e) {
      return e.what();
    }
    return {};
  };
  const auto mat = load_error(R"([{"name": "fibre", "material": {"K": 1.0}}])");
  REQUIRE(mat.find("fibre") != std::string::npos);
  REQUIRE(mat.find("material") != std::string::npos);
  const auto unk = load_error(R"([{"name": "fibre", "colour": "red"}])");
  REQUIRE(unk.find("colour") != std::string::npos);
  REQUIRE(load_error(R"([{"name": "matrix"}, {"name": "fibre"}])").empty());
}

void test_by_index_unique_inclusions() {
  auto phases = make_phases();
  // primary, wrap (centre outside), primary, primary
  shapes_t s;
  s.push_back(std::make_unique<rvegen::circle<double>>(0.95, 0.5, 0.1));
  s.push_back(std::make_unique<rvegen::circle<double>>(1.05, 0.5, 0.1));   // wrap of #0
  s.push_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));
  s.push_back(std::make_unique<rvegen::circle<double>>(0.2, 0.2, 0.05));
  rvegen::phase_assignment<double>::from_json(
      json::parse(R"({"by_index": {"0": "fibre", "1": "filler", "2": "matrix"}})"),
      phases).apply(s, std::array<double, 3>{1.0, 1.0, 0.0});
  REQUIRE(s[0]->phase_name() == "fibre");
  REQUIRE(s[1]->phase_name() == "fibre");    // wrap follows its primary
  REQUIRE(s[2]->phase_name() == "filler");
  REQUIRE(s[3]->phase_name() == "matrix");
}

void test_by_index_overlap_and_order() {
  auto phases = make_phases();
  auto bad = [&](char const* text) {
    try { (void)rvegen::phase_assignment<double>::from_json(json::parse(text), phases); }
    catch (std::exception const&) { return true; }
    return false;
  };
  REQUIRE(bad(R"({"by_index": {"1-3": "fibre", "2": "filler"}})"));
  REQUIRE(bad(R"({"by_index": {"4-": "fibre", "10-20": "filler"}})"));
  // overflow names the key
  try {
    (void)rvegen::phase_assignment<double>::from_json(
        json::parse(R"({"by_index": {"99999999999999999999999": "fibre"}})"), phases);
    REQUIRE(false);
  } catch (std::runtime_error const& e) {
    REQUIRE(std::string{e.what()}.find("99999999999999999999999") != std::string::npos);
  }
  // numeric (not lexicographic) disjoint ranges are fine in any key order
  shapes_t s;
  for (int i = 0; i < 12; ++i)
    s.push_back(std::make_unique<rvegen::circle<double>>(0.05 + 0.07 * i, 0.5, 0.01));
  rvegen::phase_assignment<double>::from_json(
      json::parse(R"({"by_index": {"10-20": "filler", "9": "fibre"}})"), phases)
      .apply(s);
  REQUIRE(s[9]->phase_name() == "fibre");
  REQUIRE(s[10]->phase_name() == "filler");
  REQUIRE(s[11]->phase_name() == "filler");
  REQUIRE(s[8]->phase_name().empty());
}

} // namespace

int main() {
  test_phases_reject_material_and_unknown_keys();
  test_by_index_unique_inclusions();
  test_by_index_overlap_and_order();
  test_all_inclusions();
  test_by_shape_type_and_precedence();
  test_by_index_ranges();
  test_assignment_errors();
  test_set_phases_hook_reaches_writers();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "phases_test OK\n";
  return EXIT_SUCCESS;
}
