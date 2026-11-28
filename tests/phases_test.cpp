// JSON phases path (#7): phase_assignment forms, the generic
// post_process_base::set_phases hook, and phase-loading validation.

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
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

// ---- matrix phase ids, read back from the written files ------------------

struct parsed_grid {
  std::map<std::size_t, std::string> table;   // header "# phase N = name"
  std::string matrix_line;                    // "# matrix phase: ..." or ""
  std::string scheme_line;                    // "# id scheme: ..."
  std::vector<std::size_t> ids;
};

parsed_grid read_voxel_text(std::istream& in) {
  parsed_grid g;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line[0] == '#') {
      if (line.rfind("# phase ", 0) == 0) {
        auto eq = line.find(" = ");
        g.table[std::stoull(line.substr(8, eq - 8))] = line.substr(eq + 3);
      } else if (line.rfind("# matrix phase:", 0) == 0) {
        g.matrix_line = line;
      } else if (line.rfind("# id scheme:", 0) == 0) {
        g.scheme_line = line;
      }
      continue;
    }
    g.ids.push_back(std::stoull(line));
  }
  return g;
}

parsed_grid read_vtk_body(std::string const& path) {
  parsed_grid g;
  std::ifstream in{path};
  std::string line;
  bool body = false;
  while (std::getline(in, line)) {
    if (body) { if (!line.empty()) g.ids.push_back(std::stoull(line)); }
    else if (line.rfind("LOOKUP_TABLE", 0) == 0) body = true;
  }
  return g;
}

std::map<std::size_t, std::size_t> histogram(std::vector<std::size_t> const& v) {
  std::map<std::size_t, std::size_t> h;
  for (auto id : v) ++h[id];
  return h;
}

void test_matrix_phase_in_memory() {
  auto phases = make_phases();   // matrix = 1, fibre = 2, filler = 3
  auto s = sample_shapes();
  rvegen::phase_assignment<double>::from_json(
      json::parse(R"({"all_inclusions": "fibre", "matrix": "matrix"})"), phases)
      .apply(s);
  rvegen::voxel_writer<double> w{16, 16, 1};
  w.set_phases(&phases);

  // No matrix declared: behaviour as before, uncovered voxels are 0.
  auto const before = w.sample(s, {1.0, 1.0, 0.0});
  std::size_t zeros = 0;
  for (auto v : before) if (v == 0) ++zeros;
  REQUIRE(zeros > 0);

  phases.set_matrix("matrix");
  auto const after = w.sample(s, {1.0, 1.0, 0.0});
  std::size_t matrix_count = 0, fibre_count = 0, other = 0;
  for (auto v : after) {
    if (v == 1) ++matrix_count; else if (v == 2) ++fibre_count; else ++other;
  }
  REQUIRE(matrix_count == zeros);   // uncovered voxels take the matrix id
  REQUIRE(fibre_count > 0);
  REQUIRE(other == 0);              // no id 0 any more

  // A shape without a phase stays untagged (0) even with a matrix declared.
  s.push_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.05));
  s.back()->info().set_phase_name("");
  bool has_zero = false;
  for (auto v : w.sample(s, {1.0, 1.0, 0.0})) if (v == 0) has_zero = true;
  REQUIRE(has_zero);

  // Shape-index mode (no phases attached) is unchanged.
  rvegen::voxel_writer<double> idx{16, 16, 1};
  auto const g = idx.sample(sample_shapes(), {1.0, 1.0, 0.0});
  std::size_t idx_zero = 0;
  for (auto v : g) if (v == 0) ++idx_zero;
  REQUIRE(idx_zero == zeros);

  bool threw = false;
  try { phases.set_matrix("nope"); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
}

void test_matrix_phase_files_from_example(char const* cli, char const* example) {
  // Run the CLI on the shipped example (outputs redirected to private
  // names so this does not race the other CLI test) and read both grids
  // back from disk.
  std::ifstream cfg_in{example};
  auto cfg = json::parse(cfg_in);
  cfg["postprocess"][0]["output_path"] = "phases_files_test.vox";
  cfg["postprocess"][1]["output_path"] = "phases_files_test.vtk";
  { std::ofstream o{"phases_files_test.json"}; o << cfg.dump(); }
  const std::string cmd = std::string{"\""} + cli +
                          "\" phases_files_test.json > phases_files_test.log";
  REQUIRE(std::system(cmd.c_str()) == 0);

  std::ifstream vox{"phases_files_test.vox"};
  auto const v = read_voxel_text(vox);
  auto const k = read_vtk_body("phases_files_test.vtk");
  REQUIRE(v.ids.size() == 64 * 64);
  REQUIRE(v.ids == k.ids);   // voxel and VTK agree voxel by voxel

  REQUIRE(v.table.size() == 3);
  REQUIRE(v.table.at(1) == "matrix");
  REQUIRE(v.table.at(2) == "fibre");
  REQUIRE(v.table.at(3) == "filler");
  REQUIRE(v.matrix_line == "# matrix phase: matrix (id 1)");
  REQUIRE(v.scheme_line.find("matrix phase id") != std::string::npos);

  auto const h = histogram(v.ids);
  for (auto const& [id, n] : h) { (void)n; REQUIRE(v.table.count(id) == 1); }
  REQUIRE(h.count(0) == 0);          // no untagged voxel
  REQUIRE(h.count(1) == 1 && h.at(1) > 0);   // matrix voxels exist
  REQUIRE(h.count(2) == 1 && h.count(3) == 1);
  // Matrix count equals the uncovered count: voxels in shape-index
  // terms (id 0) are not available here, but fibre+filler+matrix must
  // add up to the whole grid with nothing else present.
  REQUIRE(h.at(1) + h.at(2) + h.at(3) == v.ids.size());

  // Sidecar name table matches the header table.
  std::ifstream side{"phases_files_test.vox.phases.json"};
  REQUIRE(static_cast<bool>(side));
  auto const sj = json::parse(side);
  REQUIRE(sj.at("matrix") == "matrix");
  REQUIRE(sj.at("phases").size() == 3);
  for (auto const& p : sj.at("phases"))
    REQUIRE(v.table.at(p.at("id").get<std::size_t>()) == p.at("name").get<std::string>());
  std::ifstream side_vtk{"phases_files_test.vtk.phases.json"};
  REQUIRE(static_cast<bool>(side_vtk));
  REQUIRE(json::parse(side_vtk) == sj);

  // The VTK file stays loadable: single title line within 256 chars.
  std::ifstream vtk{"phases_files_test.vtk"};
  std::string l1, title;
  std::getline(vtk, l1);
  std::getline(vtk, title);
  REQUIRE(l1 == "# vtk DataFile Version 3.0");
  REQUIRE(title.size() <= 256);
  REQUIRE(title.find(".phases.json") != std::string::npos);
}

void test_empty_phase_names_rejected() {
  bool threw = false;
  try {
    (void)rvegen::load_phases_from_json<double>(json::parse(R"([{"name": ""}])"));
  } catch (std::exception const& e) {
    threw = std::string{e.what()}.find("empty") != std::string::npos;
  }
  REQUIRE(threw);

  auto phases = make_phases();
  for (char const* text : {R"({"all_inclusions": ""})",
                           R"({"matrix": ""})",
                           R"({"by_shape_type": {"circle": ""}})",
                           R"({"by_index": {"0": ""}})"}) {
    threw = false;
    try {
      (void)rvegen::phase_assignment<double>::from_json(json::parse(text), phases);
    } catch (std::exception const& e) {
      threw = std::string{e.what()}.find("empty") != std::string::npos;
    }
    REQUIRE(threw);
  }
}

} // namespace

int main(int argc, char** argv) {
  test_phases_reject_material_and_unknown_keys();
  test_by_index_unique_inclusions();
  test_by_index_overlap_and_order();
  test_all_inclusions();
  test_by_shape_type_and_precedence();
  test_by_index_ranges();
  test_assignment_errors();
  test_set_phases_hook_reaches_writers();
  test_matrix_phase_in_memory();
  test_empty_phase_names_rejected();
  if (argc >= 3) test_matrix_phase_files_from_example(argv[1], argv[2]);
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "phases_test OK\n";
  return EXIT_SUCCESS;
}
