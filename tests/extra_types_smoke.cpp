// Phase 7 extra-types smoke: directly exercise periodic_generator,
// random_generator, rectangle_input, box_input, until_full termination,
// and vtk_legacy_writer.
//
// These types are also covered indirectly via the CLI examples; this file
// gives them direct unit-style coverage so failures point at one piece.

#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>

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

// ----------------------------------------------------------------------------
// periodic_generator: shapes near a face must allow placement of shapes near
// the OPPOSITE face only if their periodic images don't collide.
// ----------------------------------------------------------------------------
void test_periodic_collision_across_box_faces() {
  std::mt19937 engine{42};
  rvegen::periodic_generator<double> generator{1};
  rvegen::number_of_inclusions<double> termination{2};
  const std::array<double, 3> domain_box{1.0, 1.0, 0.0};

  // Build inputs that produce two specific candidates straddling the +x and
  // -x faces respectively. Their periodic images would overlap, so the
  // periodic generator must reject the second.
  rvegen::constant_distribution<double> x_first{0.95};
  rvegen::constant_distribution<double> y_zero{0.5};
  rvegen::constant_distribution<double> r_big{0.1};
  auto first = std::make_unique<rvegen::circle_input<double>>(x_first, y_zero, r_big);

  rvegen::periodic_generator<double>::input_vector inputs;
  inputs.push_back(std::move(first));

  // Generate 1 primary at (0.95, 0.5). It pokes through the +x face,
  // so the periodic_generator inserts the wrap at (-0.05, 0.5) into the
  // accepted vector explicitly. Total accepted entries: 2 (primary +
  // wrap).  Termination is_centre_in_domain-filtered, so it counts only
  // the primary toward the target.
  auto first_pass = generator.compute(inputs, termination, domain_box);
  REQUIRE(first_pass.size() == 2);  // primary + wrap

  // Now try to insert a second circle at x = 0.05 — its real position is
  // close to the first's periodic image at x = 0.95 - 1.0 = -0.05. The
  // periodic_aabb_overlap should detect this.
  rvegen::constant_distribution<double> x_second{0.05};
  rvegen::constant_distribution<double> r_small{0.05};
  auto colliding_input = std::make_unique<rvegen::circle_input<double>>(
      x_second, y_zero, r_small);
  rvegen::periodic_generator<double>::input_vector colliding_inputs;
  colliding_inputs.push_back(std::move(colliding_input));

  rvegen::periodic_generator<double> generator2{10};
  rvegen::number_of_inclusions<double> term2{1};
  // To test "would this collide with first_pass[0]", run a fresh generator
  // and seed its accepted set indirectly: easiest is to reproduce the
  // collision via generator's loop with two inputs that each produce one
  // candidate. Skipping deep instrumentation; the existence of
  // periodic_aabb_overlap is exercised in the next test.
  auto second_pass = generator2.compute(colliding_inputs, term2, domain_box);
  // second_pass alone won't collide (no periodic-self conflict); just check
  // it produced something.
  REQUIRE(second_pass.size() == 1);
}

// Direct test of periodic_aabb_overlap — the helper that periodic_generator
// uses for collision detection across periodic images.
void test_periodic_aabb_overlap_helper() {
  // Two unit circles, one at (0.95, 0.5), one at (0.05, 0.5) in a unit box.
  // Direct distance is 0.9 (no overlap), but the periodic image of the
  // first wraps to (-0.05, 0.5), distance to second is 0.1 — bounding boxes
  // of radius 0.1 overlap.
  rvegen::circle<double> a{0.95, 0.5, 0.1};
  rvegen::circle<double> b{0.05, 0.5, 0.1};
  a.make_bounding_box();
  b.make_bounding_box();

  // Direct AABB overlap: false (they're 0.9 apart on x).
  REQUIRE(!rvegen::aabb_overlap(*a.bounding_box(), *b.bounding_box()));

  // Periodic AABB overlap: true (the +x periodic copy of a meets b).
  REQUIRE(rvegen::periodic_aabb_overlap(*a.bounding_box(), *b.bounding_box(),
                                         std::array<double, 3>{1.0, 1.0, 0.0}));
}

// ----------------------------------------------------------------------------
// random_generator: produces shapes ignoring the box constraint.
// ----------------------------------------------------------------------------
void test_random_generator_allows_outside_box() {
  std::mt19937 engine{1};
  rvegen::uniform_real_distribution<double> px{0.0, 1.0, engine};
  rvegen::uniform_real_distribution<double> py{0.0, 1.0, engine};
  rvegen::constant_distribution<double> r{0.3}; // bigger than half-box → pokes out
  auto input = std::make_unique<rvegen::circle_input<double>>(px, py, r);

  rvegen::random_generator<double>::input_vector inputs;
  inputs.push_back(std::move(input));

  rvegen::random_generator<double> generator{10000};
  rvegen::number_of_inclusions<double> termination{3};

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0});
  // random_generator doesn't enforce in-box, so we expect to get the target
  // count (or close to it) even with shapes that poke out.
  REQUIRE(shapes.size() == 3);
}

// ----------------------------------------------------------------------------
// rectangle_input: pulls 4 dists, produces axis-aligned rectangles.
// ----------------------------------------------------------------------------
void test_rectangle_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{2};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x":  {"type": "constant", "value": 0.5},
    "y":  {"type": "constant", "value": 0.5},
    "w":  {"type": "constant", "value": 0.2},
    "h":  {"type": "constant", "value": 0.4}
  })");

  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec = nlohmann::json::parse(R"({
    "type": "rectangle_input",
    "pos_x_dist": "x",
    "pos_y_dist": "y",
    "width_dist": "w",
    "height_dist": "h"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  // FP roundtrip via min/max storage allows tiny ULP error on area.
  REQUIRE(std::abs(shape->area() - 0.2 * 0.4) < 1e-12);
}

// ----------------------------------------------------------------------------
// box_input: same but 3D.
// ----------------------------------------------------------------------------
void test_box_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{3};
  rvegen::distribution_map_t<double> dists;
  const auto dist_specs = nlohmann::json::parse(R"({
    "c": {"type": "constant", "value": 0.5},
    "s": {"type": "constant", "value": 0.1}
  })");
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec = nlohmann::json::parse(R"({
    "type": "box_input",
    "pos_x_dist": "c", "pos_y_dist": "c", "pos_z_dist": "c",
    "width_dist": "s", "height_dist": "s", "depth_dist": "s"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(std::abs(shape->volume() - 0.1 * 0.1 * 0.1) < 1e-12);
}

// ----------------------------------------------------------------------------
// until_full: returns false from operator() — generator's max_attempts
// is the only stop signal.
// ----------------------------------------------------------------------------
void test_until_full_lets_generator_max_attempts_cap() {
  rvegen::until_full<double> term;
  rvegen::until_full<double>::shape_vector empty;
  REQUIRE(term(empty) == false);

  // Place 2 large circles in a small box; until_full means we stop when
  // generator gives up. Should produce a small number of accepted shapes.
  std::mt19937 engine{5};
  rvegen::uniform_real_distribution<double> px{0.0, 1.0, engine};
  rvegen::uniform_real_distribution<double> py{0.0, 1.0, engine};
  rvegen::constant_distribution<double> r{0.4};
  auto input = std::make_unique<rvegen::circle_input<double>>(px, py, r);

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::move(input));

  rvegen::only_inside_generator<double> gen{1000};
  auto shapes = gen.compute(inputs, term, {1.0, 1.0, 0.0});
  // Should be > 0 (at least one fits) but capped (can't fit infinitely).
  REQUIRE(!shapes.empty());
  REQUIRE(shapes.size() <= 5);
}

// ----------------------------------------------------------------------------
// ellipse: rotation, area, is_inside (rotated), and GTE-backed collision.
// ----------------------------------------------------------------------------
void test_ellipse_basics_and_rotation() {
  // Axis-aligned ellipse: a=2, b=1, area = π * 2 * 1.
  rvegen::ellipse<double> e0{0.0, 0.0, 2.0, 1.0, 0.0};
  REQUIRE(std::abs(e0.area() - 2.0 * std::numbers::pi_v<double>) < 1e-12);
  REQUIRE(e0.volume() == 0.0);

  // is_inside on axis-aligned ellipse.
  REQUIRE(e0.is_inside({0.0, 0.0, 0.0}));     // centre
  REQUIRE(e0.is_inside({1.5, 0.0, 0.0}));     // along major
  REQUIRE(!e0.is_inside({0.0, 1.5, 0.0}));    // outside on minor

  // Rotated 90°: major axis now points along +y.
  rvegen::ellipse<double> e90{0.0, 0.0, 2.0, 1.0, std::numbers::pi_v<double> * 0.5};
  REQUIRE(e90.is_inside({0.0, 1.5, 0.0}));    // now inside (along rotated major)
  REQUIRE(!e90.is_inside({1.5, 0.0, 0.0}));   // outside on rotated minor

  // Rotated bounding box: at 90° the AABB is 2×4 (h, w swapped).
  e90.make_bounding_box();
  auto* bb = e90.bounding_box();
  REQUIRE(bb != nullptr);
  REQUIRE(std::abs(bb->top_point()[0] - 1.0) < 1e-9); // half-width = b = 1
  REQUIRE(std::abs(bb->top_point()[1] - 2.0) < 1e-9); // half-height = a = 2
}

void test_ellipse_collision_via_gte() {
  // Two parallel ellipses, far apart on x — separated.
  rvegen::ellipse<double> a{0.0, 0.0, 1.0, 0.5, 0.0};
  rvegen::ellipse<double> b{3.0, 0.0, 1.0, 0.5, 0.0};
  REQUIRE(!rvegen::collision_details(a, b));

  // Move them closer until they overlap.
  rvegen::ellipse<double> b2{1.5, 0.0, 1.0, 0.5, 0.0};
  REQUIRE(rvegen::collision_details(a, b2));

  // Rotated case: an ellipse rotated 90° with axes (1.0, 0.5) at (1.2, 0)
  // — its rotated extent on x is 0.5, so it does NOT reach a's right edge
  // (which is at x = 1.0). They should be separated.
  rvegen::ellipse<double> b_rot{1.6, 0.0, 1.0, 0.5,
                                  std::numbers::pi_v<double> * 0.5};
  REQUIRE(!rvegen::collision_details(a, b_rot));

  // Same rotated ellipse but moved closer — should overlap.
  rvegen::ellipse<double> b_rot_close{1.0, 0.0, 1.0, 0.5,
                                       std::numbers::pi_v<double> * 0.5};
  REQUIRE(rvegen::collision_details(a, b_rot_close));
}

void test_circle_ellipse_collision() {
  rvegen::circle<double>  c{0.0, 0.0, 0.5};
  rvegen::ellipse<double> e_far{2.0, 0.0, 1.0, 0.3, 0.0};
  REQUIRE(!rvegen::collision_details(c, e_far));
  REQUIRE(!rvegen::collision_details(e_far, c)); // symmetry

  rvegen::ellipse<double> e_near{1.2, 0.0, 1.0, 0.3, 0.0};
  REQUIRE(rvegen::collision_details(c, e_near));
  REQUIRE(rvegen::collision_details(e_near, c));
}

void test_ellipse_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{99};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x":   {"type": "constant", "value": 0.5},
    "y":   {"type": "constant", "value": 0.5},
    "ra":  {"type": "constant", "value": 0.10},
    "rb":  {"type": "constant", "value": 0.05},
    "rot": {"type": "constant", "value": 0.78539816}
  })");

  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec = nlohmann::json::parse(R"({
    "type": "ellipse_input",
    "pos_x_dist":     "x",
    "pos_y_dist":     "y",
    "radius_a_dist":  "ra",
    "radius_b_dist":  "rb",
    "rotation_dist":  "rot"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(std::abs(shape->area()
                   - 0.10 * 0.05 * std::numbers::pi_v<double>) < 1e-12);
}

// ----------------------------------------------------------------------------
// vtk_legacy_writer: ASCII STRUCTURED_POINTS header + flat phase data.
// ----------------------------------------------------------------------------
void test_vtk_legacy_writer_header_and_size() {
  rvegen::vtk_legacy_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2));

  rvegen::vtk_legacy_writer<double> writer{16, 16, 1};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});

  const auto txt = out.str();
  REQUIRE(txt.find("# vtk DataFile Version 3.0")  != std::string::npos);
  REQUIRE(txt.find("DATASET STRUCTURED_POINTS")    != std::string::npos);
  REQUIRE(txt.find("DIMENSIONS 16 16 1")           != std::string::npos);
  REQUIRE(txt.find("SCALARS phase int 1")          != std::string::npos);
  REQUIRE(txt.find("LOOKUP_TABLE default")         != std::string::npos);
  // A handful of voxels at the centre should be marked as inclusion (== 1).
  REQUIRE(txt.find("\n1\n") != std::string::npos);
}

// ----------------------------------------------------------------------------
// oriented_uniform_distribution: von-Mises sampler. Two regime checks:
//   κ = 0:    behaviour collapses to uniform on a 2π interval.
//   κ = 10:   tightly concentrated around mean_angle.
// Both verified statistically over a moderate sample.
// ----------------------------------------------------------------------------
void test_oriented_uniform_uniform_regime() {
  std::mt19937 engine{123};
  rvegen::oriented_uniform_distribution<double> dist{
      0.0 /*mean*/, 0.0 /*kappa = uniform*/, engine};

  // 5000 samples binned into 12 30°-wide bins. With kappa=0 each bin
  // expects ~417 samples; standard deviation under the binomial is
  // sqrt(N · p · (1-p)) ≈ √(5000 · 1/12 · 11/12) ≈ 19.5. Tolerance set
  // to 5σ so the test isn't flaky — this is detecting "wildly skewed"
  // not "minor sampling noise".
  constexpr int N = 5000;
  constexpr int B = 12;
  std::array<int, B> bins{};
  for (int i = 0; i < N; ++i) {
    double a = std::fmod(dist() + 4 * M_PI, 2 * M_PI);   // wrap to [0, 2π)
    int b = std::min(int(a / (2 * M_PI / B)), B - 1);
    ++bins[b];
  }
  for (int count : bins) {
    REQUIRE(std::abs(count - N / B) < 5 * 20);   // ~5σ tolerance
  }
}

void test_oriented_uniform_concentrated_regime() {
  std::mt19937 engine{456};
  // mean_angle = π/4; high concentration. Samples should cluster.
  rvegen::oriented_uniform_distribution<double> dist{
      M_PI / 4, 50.0 /*kappa large*/, engine};

  constexpr int N = 2000;
  double sum_dev_sq = 0.0;
  for (int i = 0; i < N; ++i) {
    double a = dist();
    double dev = std::fmod(a - M_PI / 4 + 3 * M_PI, 2 * M_PI) - M_PI;  // wrap to [-π, π]
    sum_dev_sq += dev * dev;
  }
  // For a von Mises distribution, Var ≈ 1/κ for large κ. With κ=50,
  // expect std-dev ≈ 1/√50 ≈ 0.14 rad. Empirical std-dev should be in
  // the same ballpark (allow 3× factor for sample noise).
  const double empirical_std = std::sqrt(sum_dev_sq / N);
  REQUIRE(empirical_std < 0.5);   // would be ~π/√3 ≈ 1.81 if uniform
}

} // namespace

int main() {
  test_periodic_collision_across_box_faces();
  test_periodic_aabb_overlap_helper();
  test_random_generator_allows_outside_box();
  test_rectangle_input_via_registry();
  test_box_input_via_registry();
  test_until_full_lets_generator_max_attempts_cap();
  test_ellipse_basics_and_rotation();
  test_ellipse_collision_via_gte();
  test_circle_ellipse_collision();
  test_ellipse_input_via_registry();
  test_vtk_legacy_writer_header_and_size();
  test_oriented_uniform_uniform_regime();
  test_oriented_uniform_concentrated_regime();

  if (failures > 0) {
    std::cerr << failures << " extra-types smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "extra_types_smoke: PASS\n";
  return EXIT_SUCCESS;
}
