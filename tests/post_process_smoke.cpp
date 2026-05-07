// Smoke test: voxel_writer (registered as a post-process under the JSON key
// "voxel") produces the expected phase-id grid both when constructed
// directly and when built through the post_process registry from a JSON
// spec.

#include <array>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
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
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// One circle inclusion at (0.5, 0.5) r=0.25 in a [0,1]² domain. With a
// 4×4 grid this hits exactly the four interior voxels (centres at 0.375 /
// 0.625), and nothing else.
auto make_one_circle_rve() {
  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  auto c = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.25);
  c->make_bounding_box();
  shapes.push_back(std::move(c));
  return shapes;
}

void test_direct_voxelize() {
  rvegen::voxel_writer<double> pp{
      4, 4, 1, "post_process_smoke_direct.vox"};

  auto shapes = make_one_circle_rve();
  pp.run(shapes, {1.0, 1.0, 0.0});

  std::ifstream in{pp.output_path()};
  REQUIRE(in.is_open());

  std::string line;
  int header_lines = 0;
  std::vector<unsigned> values;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    if (line.front() == '#') {
      ++header_lines;
      continue;
    }
    values.push_back(static_cast<unsigned>(std::stoul(line)));
  }

  REQUIRE(header_lines == 4);
  REQUIRE(values.size() == 16);

  // The four interior voxels — (j, i) ∈ {1, 2} × {1, 2} — should be
  // labelled with the (1-based) circle index 1; everything else 0.
  for (std::size_t j = 0; j < 4; ++j) {
    for (std::size_t i = 0; i < 4; ++i) {
      const auto v = values[j * 4 + i];
      const bool interior = (i == 1 || i == 2) && (j == 1 || j == 2);
      REQUIRE(v == (interior ? 1u : 0u));
    }
  }

  std::remove(pp.output_path().c_str());
}

void test_registry_build_from_json() {
  rvegen::register_all_post_processes<>();

  auto& registry = rvegen::post_process_registry_t<>::instance();
  REQUIRE(registry.contains("voxel"));
  REQUIRE(registry.contains("gmsh_geo"));
  REQUIRE(registry.contains("svg"));
  REQUIRE(registry.contains("svg_3d"));
  REQUIRE(registry.contains("three_js"));
  REQUIRE(registry.contains("vtk_legacy"));

  const auto spec = nlohmann::json::parse(R"({
    "type": "voxel",
    "nx": 4, "ny": 4, "nz": 1,
    "output_path": "post_process_smoke_registry.vox"
  })");

  auto pp = rvegen::build_from_json(registry, spec);
  REQUIRE(pp != nullptr);

  auto shapes = make_one_circle_rve();
  pp->run(shapes, {1.0, 1.0, 0.0});

  std::ifstream in{"post_process_smoke_registry.vox"};
  REQUIRE(in.is_open());
  std::stringstream buf;
  buf << in.rdbuf();
  // Spot-check one of the four interior voxels is labelled "1".
  REQUIRE(buf.str().find("\n1\n") != std::string::npos);

  std::remove("post_process_smoke_registry.vox");
}

} // namespace

int main() {
  test_direct_voxelize();
  test_registry_build_from_json();

  if (failures != 0) {
    std::cerr << failures << " test(s) failed\n";
    return 1;
  }
  std::cout << "post_process_smoke: ok\n";
  return 0;
}
