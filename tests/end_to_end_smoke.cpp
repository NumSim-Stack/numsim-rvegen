// Phase 3 end-to-end smoke: complete RVE generation pipeline.
//
// Pipeline:
//   1. Construct an mt19937 engine with a known seed.
//   2. Build uniform distributions for circle position (x, y) and radius.
//   3. Wrap them in a circle_input.
//   4. Create only_inside_generator + number_of_inclusions termination.
//   5. Run compute() — get a vector of accepted circles.
//   6. Verify: count met, all inside box, no AABB overlap among accepted.
//   7. Feed the result through gmsh_geo_writer and voxel_writer; assert
//      writers produce sensible output.

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

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

void test_pipeline_circles_only_inside() {
  std::mt19937 engine{42}; // fixed seed → deterministic output

  // Position in [0.05, 0.95] so radius-0.05 circles always fit in [0,1] box.
  rvegen::uniform_real_distribution<double> pos_x{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> radius{0.03, 0.05, engine};

  auto input = std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius);
  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::move(input));

  rvegen::number_of_inclusions<double> termination{10};
  rvegen::only_inside_generator<double> generator;

  const std::array<double, 3> domain_box{1.0, 1.0, 0.0};
  auto shapes = generator.compute(inputs, termination, domain_box);

  // Termination should be met or max_attempts reached. With 10 small circles
  // in a unit box this should always succeed.
  REQUIRE(shapes.size() == 10);

  // Every accepted shape must be entirely inside the box.
  for (auto const& s : shapes) {
    auto const* bb = s->bounding_box();
    REQUIRE(bb != nullptr);
    REQUIRE(rvegen::aabb_inside(*bb, domain_box[0], domain_box[1], domain_box[2]));
  }

  // No two accepted shapes' AABBs may overlap (the generator's invariant).
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    for (std::size_t j = i + 1; j < shapes.size(); ++j) {
      REQUIRE(!rvegen::aabb_overlap(*shapes[i]->bounding_box(),
                                     *shapes[j]->bounding_box()));
    }
  }

  // Hand the result through both output writers.
  std::stringstream geo;
  rvegen::gmsh_geo_writer<double>{}.write(geo, shapes, domain_box);
  const auto geo_txt = geo.str();
  REQUIRE(geo_txt.find("Rectangle(1)") != std::string::npos); // domain
  REQUIRE(geo_txt.find("Disk(") != std::string::npos);        // at least one circle

  std::stringstream vox;
  rvegen::voxel_writer<double> writer{32, 32, 1};
  writer.write(vox, shapes, domain_box);
  const auto vox_txt = vox.str();
  REQUIRE(vox_txt.find("# rvegen voxel grid") != std::string::npos);

  // The voxel grid sample() should mark some voxels as inclusion (>0).
  const auto grid = writer.sample(shapes, domain_box);
  std::size_t inclusion_voxels = 0;
  for (auto v : grid) if (v > 0) ++inclusion_voxels;
  REQUIRE(inclusion_voxels > 0);
}

void test_termination_short_circuit() {
  // 0-target termination: generator should return immediately.
  std::mt19937 engine{7};
  rvegen::uniform_real_distribution<double> pos_x{0.0, 1.0, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.0, 1.0, engine};
  rvegen::constant_distribution<double> radius{0.05};

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));

  rvegen::number_of_inclusions<double> termination{0};
  rvegen::only_inside_generator<double> generator;

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0});
  REQUIRE(shapes.empty());
}

void test_constant_distribution_is_exact() {
  rvegen::constant_distribution<double> d{4.2};
  for (int i = 0; i < 10; ++i) {
    REQUIRE(d() == 4.2);
  }
}

void test_uniform_real_is_in_range() {
  std::mt19937 engine{123};
  rvegen::uniform_real_distribution<double> d{1.0, 2.0, engine};
  for (int i = 0; i < 100; ++i) {
    const auto v = d();
    REQUIRE(v >= 1.0 && v <= 2.0);
  }
}

} // namespace

int main() {
  test_constant_distribution_is_exact();
  test_uniform_real_is_in_range();
  test_termination_short_circuit();
  test_pipeline_circles_only_inside();

  if (failures > 0) {
    std::cerr << failures << " end-to-end smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "end_to_end_smoke: PASS\n";
  return EXIT_SUCCESS;
}
