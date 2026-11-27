// JSON phases path (#7): phase_assignment forms, the generic
// post_process_base::set_phases hook, and the DAMASK material.yaml writer.

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
    {"name": "matrix", "material": {"K": 3.0e9, "G": 1.0e9}},
    {"name": "fibre",  "material": {"E": 240.0e9, "nu": 0.25}},
    {"name": "filler", "material": {"damask": {"elastic": {"type": "Hooke"}}}}
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
  REQUIRE(reg.contains("damask_material_config"));

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

void test_damask_yaml() {
  rvegen::damask_material_config<double> w;
  auto phases = make_phases();
  w.set_phases(&phases);
  std::ostringstream out;
  w.write(out);
  auto const y = out.str();
  REQUIRE(y.find("\"matrix\":") != std::string::npos);
  REQUIRE(y.find("\"fibre\":") != std::string::npos);
  REQUIRE(y.find("N_constituents: 1") != std::string::npos);
  // matrix K=3e9, G=1e9 -> C_11 = K + 4G/3 = 4.333e9, C_44 = G.
  REQUIRE(y.find("\"C_44\":1000000000.0") != std::string::npos);
  REQUIRE(y.find("\"C_11\":4333333333.33") != std::string::npos);
  // E/nu path: G = E/(2(1+nu)) = 96e9.
  REQUIRE(y.find("\"C_44\":96000000000.0") != std::string::npos);
  // damask passthrough block.
  REQUIRE(y.find("{\"elastic\":{\"type\":\"Hooke\"}}") != std::string::npos);
  // one material entry per phase.
  std::size_t count = 0;
  for (auto pos = y.find("\n    constituents:"); pos != std::string::npos;
       pos = y.find("\n    constituents:", pos + 1)) ++count;
  REQUIRE(count == 4);   // matrix (voxel id 0) + 3 phases

  // No usable material -> throws, naming the phase.
  rvegen::phase_collection<double> bad;
  bad.add("matrix", json{{"K", 3e9}, {"G", 1e9}});
  bad.add("mystery", json{{"type", "void"}});
  rvegen::damask_material_config<double> w2;
  w2.set_phases(&bad);
  bool threw = false;
  try { std::ostringstream o; w2.write(o); }
  catch (std::runtime_error const& e) {
    threw = std::string{e.what()}.find("mystery") != std::string::npos;
  }
  REQUIRE(threw);

  // Without phases attached it refuses rather than writing an empty file.
  rvegen::damask_material_config<double> w3;
  threw = false;
  try { std::ostringstream o; w3.write(o); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

bool writes_throw(rvegen::phase_collection<double> const& ph,
                  std::string* msg = nullptr) {
  rvegen::damask_material_config<double> w;
  w.set_phases(&ph);
  try { std::ostringstream o; w.write(o); }
  catch (std::runtime_error const& e) { if (msg) *msg = e.what(); return true; }
  return false;
}

void test_damask_hardening() {
  using rvegen::phase_collection;
  // Injection through the phase name: newline must not create YAML lines.
  phase_collection<double> inj;
  inj.add("matrix", json{{"K", 3e9}, {"G", 1e9}});
  inj.add("a\nhomogenization: {evil: 1}", json{{"K", 3e9}, {"G", 1e9}});
  rvegen::damask_material_config<double> w;
  w.set_phases(&inj);
  std::ostringstream out;
  w.write(out);
  std::istringstream in{out.str()};
  std::string line;
  int top_homog = 0;
  while (std::getline(in, line))
    if (line.rfind("homogenization:", 0) == 0) ++top_homog;
  REQUIRE(top_homog == 1);

  // Impossible constants are rejected, never written as null.
  phase_collection<double> imp;
  imp.add("matrix", json{{"E", 1e9}, {"nu", 0.5}});
  REQUIRE(writes_throw(imp));
  phase_collection<double> neg;
  neg.add("matrix", json{{"K", -1.0}, {"G", 1.0}});
  REQUIRE(writes_throw(neg));
  phase_collection<double> lam;   // K = lambda + 2mu/3 < 0
  lam.add("matrix", json{{"lambda", -1e9}, {"mu", 1e9}});
  REQUIRE(writes_throw(lam));

  // Homogenization name is validated.
  rvegen::register_all_post_processes<>();
  bool hthrew = false;
  try {
    (void)rvegen::build_from_json(
        rvegen::post_process_registry_t<>::instance(),
        json{{"type", "damask_material_config"}, {"output_path", "x.yaml"},
             {"homogenization", "SX\nevil: 1"}});
  } catch (std::exception const&) { hthrew = true; }
  REQUIRE(hthrew);

  // Lattice is configurable and validated.
  phase_collection<double> hex;
  hex.add("matrix", json{{"K", 3e9}, {"G", 1e9}, {"lattice", "hP"}});
  rvegen::damask_material_config<double> wl;
  wl.set_phases(&hex);
  std::ostringstream lo;
  wl.write(lo);
  REQUIRE(lo.str().find("lattice: hP") != std::string::npos);
  REQUIRE(lo.str().find("\"C_13\"") != std::string::npos);
  phase_collection<double> badlat;
  badlat.add("matrix", json{{"K", 3e9}, {"G", 1e9}, {"lattice", "zz"}});
  REQUIRE(writes_throw(badlat));

  // No matrix phase -> clear error; matrix entry first otherwise.
  phase_collection<double> nomat;
  nomat.add("fibre", json{{"K", 3e9}, {"G", 1e9}});
  std::string msg;
  REQUIRE(writes_throw(nomat, &msg) && msg.find("matrix") != std::string::npos);
  std::string const y = out.str();
  auto first_entry = y.find("voxel id 0 (matrix)");
  REQUIRE(first_entry != std::string::npos);
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
  test_damask_hardening();
  test_by_index_unique_inclusions();
  test_by_index_overlap_and_order();
  test_all_inclusions();
  test_by_shape_type_and_precedence();
  test_by_index_ranges();
  test_assignment_errors();
  test_set_phases_hook_reaches_writers();
  test_damask_yaml();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "phases_test OK\n";
  return EXIT_SUCCESS;
}
