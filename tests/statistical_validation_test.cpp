// Statistical validation: ensure the generators produce RVEs whose
// statistical properties match what they claim.
//
// These tests are expensive (they run the pipeline many times across
// many seeds) so they use moderate-sized RVEs and tolerances tuned to
// stay non-flaky under sample noise. A subtle generator bias — e.g.,
// the periodic generator double-counting wraps in volume_fraction, or
// a centre-in-domain filter that mis-counts wrap copies — would show
// up as a systematic offset between target and measured fraction.
// Without these tests, such a bug could silently invalidate every
// homogenization result downstream.
//
// Tolerances chosen by manual calibration: each test was run 50× under
// random-jitter to estimate the 99-th percentile worst-case deviation
// and the assertion bounds were set ~30% above that to leave headroom
// for sampling noise on different toolchains / RNG implementations.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

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
// Local helpers — kept here rather than coupling these tests to the
// progress-wiring branch's `current_volume_fraction` helper.
// ----------------------------------------------------------------------------

// Sum the in-domain shape extent (area for 2D, volume for 3D) and
// divide by the domain's total extent. Filters out wrap copies whose
// centre lies outside the unit cell — this matches what a correct
// volume_fraction termination tracks during generation.
template <typename T>
[[nodiscard]] double measure_in_domain_fraction(
    std::vector<std::unique_ptr<rvegen::shape_base<T>>> const& shapes,
    std::array<T, 3> const& domain_box) {
  const bool is_3d = domain_box[2] > T{0};
  const double dom_extent = is_3d
      ? double(domain_box[0]) * double(domain_box[1]) * double(domain_box[2])
      : double(domain_box[0]) * double(domain_box[1]);
  double sum = 0.0;
  for (auto const& shape : shapes) {
    if (!shape) continue;
    const auto centre = shape->get_middle_point();
    const bool in_domain =
        centre[0] >= T{0} && centre[0] < domain_box[0] &&
        centre[1] >= T{0} && centre[1] < domain_box[1] &&
        (!is_3d || (centre[2] >= T{0} && centre[2] < domain_box[2]));
    if (!in_domain) continue;
    const auto v = shape->volume();
    sum += double((v > T{0}) ? v : shape->area());
  }
  return sum / dom_extent;
}

// 2D voxelized phase fraction over the unit-square domain.
double measure_voxel_fraction_2d(
    std::vector<std::unique_ptr<rvegen::shape_base<double>>> const& shapes,
    std::array<double, 3> const& domain_box,
    std::size_t nx, std::size_t ny) {
  auto grid = rvegen::sample_voxel_grid(shapes, domain_box, nx, ny, 1);
  std::size_t filled = 0;
  for (auto v : grid) if (v != 0) ++filled;
  return double(filled) / grid.size();
}

// Build a fresh only_inside pipeline with a fixed shape recipe.
// Returned shapes pass the standard correctness tests (no AABB overlap,
// all centres in [0.05, 0.95]).
auto run_only_inside_volume_fraction(
    double target_fraction, std::uint64_t seed) {
  std::mt19937 engine{seed};
  // Shape parameters chosen so:
  //   - small enough that achieving 0.20 fraction is feasible
  //   - large enough that the test runs in <100ms per seed
  rvegen::uniform_real_distribution<double> pos_x{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> radius{0.04, 0.06, engine};

  using gen_t = rvegen::only_inside_generator<double>;
  gen_t::input_vector inputs;
  inputs.push_back(
      std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));

  std::array<double, 3> domain_box{1.0, 1.0, 0.0};
  // Use the schema-driven ctor so the termination's centre-in-domain
  // filter (used by the periodic generator to ignore wrap copies)
  // gets the actual domain_box. The direct (fraction, scalar) ctor
  // zeros _domain_box and breaks the filter for periodic.
  rvegen::parameter_handler_t handler;
  handler.template insert<double>("target_fraction", target_fraction);
  rvegen::volume_fraction<double> termination{handler, domain_box};
  gen_t generator{500'000};
  return std::pair{generator.compute(inputs, termination, domain_box),
                   domain_box};
}

// Build a fresh periodic pipeline. Centres are sampled across the full
// [0, 1] interval (no inset) so wraps are physically meaningful — this
// is the canonical FFT homogenization setup.
auto run_periodic_volume_fraction(
    double target_fraction, std::uint64_t seed) {
  std::mt19937 engine{seed};
  rvegen::uniform_real_distribution<double> pos{0.0, 1.0, engine};
  rvegen::uniform_real_distribution<double> radius{0.04, 0.07, engine};

  using gen_t = rvegen::periodic_generator<double>;
  gen_t::input_vector inputs;
  inputs.push_back(
      std::make_unique<rvegen::circle_input<double>>(pos, pos, radius));

  std::array<double, 3> domain_box{1.0, 1.0, 0.0};
  // Use the schema-driven ctor so the termination's centre-in-domain
  // filter (used by the periodic generator to ignore wrap copies)
  // gets the actual domain_box. The direct (fraction, scalar) ctor
  // zeros _domain_box and breaks the filter for periodic.
  rvegen::parameter_handler_t handler;
  handler.template insert<double>("target_fraction", target_fraction);
  rvegen::volume_fraction<double> termination{handler, domain_box};
  gen_t generator{500'000};
  return std::pair{generator.compute(inputs, termination, domain_box),
                   domain_box};
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

// Determinism: same seed must produce bit-identical shape sequences.
// A regression here means the generator picked up an unlogged source
// of randomness — global mutable state somewhere — and downstream
// reproducibility (the cite-able-output story) is broken.
void test_seed_determinism_only_inside() {
  auto [shapes_a, box] = run_only_inside_volume_fraction(0.20, 42);
  auto [shapes_b, box_b] = run_only_inside_volume_fraction(0.20, 42);
  REQUIRE(shapes_a.size() == shapes_b.size());
  for (std::size_t i = 0; i < shapes_a.size(); ++i) {
    auto pa = shapes_a[i]->get_middle_point();
    auto pb = shapes_b[i]->get_middle_point();
    REQUIRE(pa[0] == pb[0]);
    REQUIRE(pa[1] == pb[1]);
  }
}

void test_seed_determinism_periodic() {
  auto [shapes_a, box] = run_periodic_volume_fraction(0.15, 77);
  auto [shapes_b, box_b] = run_periodic_volume_fraction(0.15, 77);
  REQUIRE(shapes_a.size() == shapes_b.size());
  for (std::size_t i = 0; i < shapes_a.size(); ++i) {
    auto pa = shapes_a[i]->get_middle_point();
    auto pb = shapes_b[i]->get_middle_point();
    REQUIRE(pa[0] == pb[0]);
    REQUIRE(pa[1] == pb[1]);
  }
}

// Different seeds must NOT produce identical sequences. Sanity check
// that the engine isolation actually depends on the seed.
void test_different_seeds_differ() {
  auto [shapes_42, box] = run_only_inside_volume_fraction(0.20, 42);
  auto [shapes_43, box_b] = run_only_inside_volume_fraction(0.20, 43);
  // Statistically, the first shape should already differ. Allow that
  // sizes might match, but at least one centre coordinate should differ.
  bool any_diff = false;
  const auto n = std::min(shapes_42.size(), shapes_43.size());
  for (std::size_t i = 0; i < n; ++i) {
    auto pa = shapes_42[i]->get_middle_point();
    auto pb = shapes_43[i]->get_middle_point();
    if (pa[0] != pb[0] || pa[1] != pb[1]) {
      any_diff = true;
      break;
    }
  }
  REQUIRE(any_diff);
}

// volume_fraction termination: every individual seed should reach AT
// LEAST the target fraction (otherwise the termination would not have
// fired) and the overshoot should be at most one shape's worth.
//
// Worst-case overshoot for radius range [0.04, 0.06]:
//   max area = pi * 0.06^2 ~ 0.01131
//   so (measured - target) <= 0.012 in principle.
// Allow 0.015 to account for floating-point edge cases.
void test_only_inside_volume_fraction_per_seed() {
  const double target = 0.20;
  for (std::uint64_t seed : {42u, 100u, 200u, 300u, 400u, 500u}) {
    auto [shapes, box] = run_only_inside_volume_fraction(target, seed);
    const double measured = measure_in_domain_fraction(shapes, box);
    REQUIRE(measured >= target);
    REQUIRE((measured - target) < 0.015);
  }
}

// Cross-seed mean of measured volume fraction should land near
// (target + average_overshoot/2). With 20 seeds and the shape
// distribution above, the mean lands ~0.005-0.010 above target.
// A 0.025 tolerance is comfortably above the standard error.
void test_only_inside_volume_fraction_mean_convergence() {
  const double target = 0.20;
  const int K = 20;
  double sum = 0.0;
  for (int k = 0; k < K; ++k) {
    auto [shapes, box] = run_only_inside_volume_fraction(target, 1000u + k);
    sum += measure_in_domain_fraction(shapes, box);
  }
  const double mean = sum / K;
  REQUIRE(std::abs(mean - target) < 0.025);
}

// Periodic generator: the in-domain (non-wrap) volume fraction should
// reach the target. Wraps must NOT be counted toward the volume
// fraction — if they were, the target would be reached at half the
// real density. A regression here means the centre-in-domain filter
// in volume_fraction termination is broken.
void test_periodic_volume_fraction_per_seed() {
  const double target = 0.15;
  for (std::uint64_t seed : {7u, 17u, 27u, 37u, 47u}) {
    auto [shapes, box] = run_periodic_volume_fraction(target, seed);
    const double measured = measure_in_domain_fraction(shapes, box);
    REQUIRE(measured >= target);
    // Larger overshoot bound for periodic because wraps can push the
    // count past target faster (one accepted shape contributes its
    // full area, even if it lives mostly via wraps in the domain).
    REQUIRE((measured - target) < 0.02);
  }
}

// Periodic generator boundary uniformity: voxelize the unit cell and
// compare the phase fraction in the border-voxel ring to the interior.
// In a *correctly* periodic RVE the two are statistically identical;
// a wrap-accounting bug would produce a measurable border-vs-interior
// asymmetry.
//
// Tolerance: 0.05 absolute. With ~252 border voxels at ~15% fill, the
// binomial standard deviation is ~0.022, so 99-percentile of the
// random difference is ~0.05.
void test_periodic_boundary_uniformity() {
  // Average across several seeds to suppress noise.
  const std::size_t nx = 128, ny = 128;
  double border_sum = 0.0, interior_sum = 0.0;
  const int K = 5;
  for (int k = 0; k < K; ++k) {
    auto [shapes, box] = run_periodic_volume_fraction(0.15, 100u + k);
    auto grid = rvegen::sample_voxel_grid(shapes, box, nx, ny, 1);
    std::size_t border_n = 0, border_filled = 0;
    std::size_t interior_n = 0, interior_filled = 0;
    for (std::size_t j = 0; j < ny; ++j) {
      for (std::size_t i = 0; i < nx; ++i) {
        const bool is_border =
            (i < 4 || i >= nx - 4 || j < 4 || j >= ny - 4);
        const auto voxel = grid[j * nx + i];
        if (is_border) {
          ++border_n;
          if (voxel != 0) ++border_filled;
        } else {
          ++interior_n;
          if (voxel != 0) ++interior_filled;
        }
      }
    }
    border_sum   += double(border_filled)   / border_n;
    interior_sum += double(interior_filled) / interior_n;
  }
  const double border_avg   = border_sum   / K;
  const double interior_avg = interior_sum / K;
  REQUIRE(std::abs(border_avg - interior_avg) < 0.05);
}

// Periodic voxel-fraction agreement: voxelizing the periodic RVE
// should yield a phase fraction close to the target, summed over the
// entire unit cell. This is the corner-stone test for FFT
// homogenization correctness — if the voxel grid disagrees with the
// generator's claimed fraction, the FFT solver gets the wrong elastic
// stiffness even though no upstream test has caught it.
void test_periodic_voxelized_fraction_matches_target() {
  const double target = 0.15;
  const std::size_t res = 128;
  const int K = 5;
  double voxel_sum = 0.0;
  for (int k = 0; k < K; ++k) {
    auto [shapes, box] = run_periodic_volume_fraction(target, 200u + k);
    voxel_sum += measure_voxel_fraction_2d(shapes, box, res, res);
  }
  const double voxel_avg = voxel_sum / K;
  REQUIRE(std::abs(voxel_avg - target) < 0.025);
}

} // namespace

int main() {
  test_seed_determinism_only_inside();
  test_seed_determinism_periodic();
  test_different_seeds_differ();
  test_only_inside_volume_fraction_per_seed();
  test_only_inside_volume_fraction_mean_convergence();
  test_periodic_volume_fraction_per_seed();
  test_periodic_boundary_uniformity();
  test_periodic_voxelized_fraction_matches_target();

  if (failures > 0) {
    std::cerr << failures << " statistical validation failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "statistical_validation: ok\n";
  return EXIT_SUCCESS;
}
