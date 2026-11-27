// Edge-case and statistics tests for the orientation samplers (#77).
//
// 1. von Mises (oriented_uniform_distribution): tiny κ in [1e-9, 1e-7]
//    used to cancel Best-Fisher's b to 0 and spin forever; it must now
//    return finite angles. Non-finite / negative κ throws.
// 2. ⟨cos 2(θ−μ)⟩ matches I2(κ)/I0(κ), the order parameter documented in
//    the header.
// 3. bingham_distribution: zero / non-finite mean axis and non-finite κ
//    throw; samples stay unit length, including for axes whose naive
//    squared norm would under- or overflow; a hopeless prolate κ hits
//    the iteration cap and throws quickly.

#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numbers>
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

template <typename F> bool throws_invalid_argument(F&& f) {
  try {
    f();
  } catch (std::invalid_argument const&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

constexpr double nan_v = std::numeric_limits<double>::quiet_NaN();
constexpr double inf_v = std::numeric_limits<double>::infinity();

void test_von_mises_tiny_kappa_returns_finite_angles() {
  constexpr double pi = std::numbers::pi;
  const double mean = 0.4;
  for (double kappa : {1e-9, 5e-9, 1e-8, 2e-8, 1e-7}) {
    std::mt19937 engine{7};
    rvegen::oriented_uniform_distribution<double> dist{mean, kappa, engine};
    for (int i = 0; i < 1000; ++i) {
      const double theta = dist();
      REQUIRE(std::isfinite(theta));
      REQUIRE(theta >= mean - pi - 1e-12 && theta <= mean + pi + 1e-12);
    }
  }
}

void test_von_mises_huge_kappa_is_point_mass() {
  // κ ≥ ~1e16 used to exhaust the rejection cap (cancellation in
  // c = κ(r − f)); above 1e12 the sampler now returns the mean exactly,
  // and just below it still samples with Var(θ) ≈ 1/κ.
  const double mean = 0.7;
  for (double kappa : {1e13, 1e16, 1e17, 1e300}) {
    std::mt19937 engine{9};
    rvegen::oriented_uniform_distribution<double> dist{mean, kappa, engine};
    for (int i = 0; i < 100; ++i) REQUIRE(dist() == mean);
  }
  std::mt19937 engine{9};
  const double kappa = 1e12;
  rvegen::oriented_uniform_distribution<double> dist{mean, kappa, engine};
  double sum_sq = 0.0;
  const int n = 5000;
  for (int i = 0; i < n; ++i) {
    const double d = dist() - mean;
    REQUIRE(std::isfinite(d));
    sum_sq += d * d;
  }
  const double kvar = kappa * sum_sq / n;
  REQUIRE(kvar > 0.9 && kvar < 1.1);
}

void test_von_mises_rejects_bad_kappa() {
  std::mt19937 engine{1};
  using dist_t = rvegen::oriented_uniform_distribution<double>;
  REQUIRE(throws_invalid_argument([&] { dist_t d{0.0, nan_v, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{0.0, inf_v, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{0.0, -1.0, engine}; }));
}

void test_von_mises_order_parameter_matches_i2_over_i0() {
  // cos 2(θ−μ) ∈ [−1, 1], so the standard error of the mean is ≤ 1/√N
  // ≈ 0.0045 at N = 50k; 0.02 is a > 4σ band. I1(2κ)/I0(2κ) (the old
  // documented formula) differs by ≥ 0.03 at these κ, so it would fail.
  const double mean = -1.1;
  const int n = 50000;
  for (double kappa : {1.0, 5.0, 20.0}) {
    std::mt19937 engine{2025};
    rvegen::oriented_uniform_distribution<double> dist{mean, kappa, engine};
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += std::cos(2.0 * (dist() - mean));
    const double measured = sum / n;
    const double expected =
        std::cyl_bessel_i(2.0, kappa) / std::cyl_bessel_i(0.0, kappa);
    if (std::abs(measured - expected) >= 0.02) {
      std::cerr << "kappa=" << kappa << " <cos2> measured=" << measured
                << " expected I2/I0=" << expected << "\n";
    }
    REQUIRE(std::abs(measured - expected) < 0.02);
  }
}

void test_bingham_rejects_bad_axis_and_kappa() {
  std::mt19937 engine{3};
  using dist_t = rvegen::bingham_distribution<double>;
  using arr = std::array<double, 3>;
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 0}, 5.0, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 0}, 0.0, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{nan_v, 0, 1}, 5.0, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, inf_v, 1}, 5.0, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 1}, nan_v, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 1}, inf_v, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 1}, -inf_v, engine}; }));

  // Same validation on the JSON / schema-driven path.
  rvegen::register_all_direction_distributions<>();
  const auto spec = nlohmann::json::parse(R"({
    "type": "bingham",
    "mean_axis_x": 0.0,
    "mean_axis_y": 0.0,
    "mean_axis_z": 0.0,
    "kappa":       5.0
  })");
  REQUIRE(throws_invalid_argument([&] {
    (void)rvegen::build_from_json(
        rvegen::direction_distribution_registry_t<>::instance(), spec,
        engine);
  }));
}

void test_bingham_samples_are_unit_length() {
  using arr = std::array<double, 3>;
  for (arr axis : {arr{0, 0, 1}, arr{1, 2, 3}, arr{1e-200, 0, 0},
                   arr{1e200, 1e200, 0}}) {
    for (double kappa : {0.0, 5.0, -5.0, 50.0}) {
      std::mt19937 engine{11};
      rvegen::bingham_distribution<double> dist{axis, kappa, engine};
      const auto& mu = dist.mean_axis();
      REQUIRE(std::abs(gte::Dot(mu, mu) - 1.0) < 1e-12);
      for (int i = 0; i < 500; ++i) {
        const auto v = dist.sample();
        REQUIRE(std::abs(gte::Length(v) - 1.0) < 1e-9);
      }
    }
  }
}

void test_bingham_iteration_cap_throws_quickly() {
  // An injected cap of 0 candidates must throw a runtime_error at once.
  std::mt19937 engine{5};
  rvegen::bingham_distribution<double> dist{std::array<double, 3>{0, 0, 1},
                                            1e5, engine};
  dist.set_max_iterations(0);
  const auto t0 = std::chrono::steady_clock::now();
  bool threw = false;
  try {
    (void)dist.sample();
  } catch (std::runtime_error const&) {
    threw = true;
  }
  const double secs = std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - t0)
                          .count();
  REQUIRE(threw);
  REQUIRE(secs < 0.5);
}

void test_bingham_kappa_range_and_default_cap() {
  using dist_t = rvegen::bingham_distribution<double>;
  using arr = std::array<double, 3>;
  std::mt19937 engine{6};
  // Outside the supported range: explicit construction-time rejection.
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 1}, 1e5 * 1.01, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 1}, 1e9, engine}; }));
  REQUIRE(throws_invalid_argument([&] { dist_t d{arr{0, 0, 1}, -1e11, engine}; }));
  // The default cap keeps the spurious-throw bound exp(-p·cap) ≤ e⁻³⁰,
  // using the documented acceptance lower bounds p.
  for (double kappa : {1e-3, 1.0, 50.0, 1e3, 1e5, -1.0, -50.0, -1e6, -1e10}) {
    dist_t d{arr{0, 0, 1}, kappa, engine};
    const double p = kappa > 0 ? 1.0 / (1.0 + 2.0 * kappa)
                               : 1.0 / (3.0 * (1.0 + std::sqrt(-kappa)));
    REQUIRE(p * static_cast<double>(d.max_iterations()) >= 30.0);
  }
  // Valid large prolate κ still samples (no random throws) — a few
  // draws at κ = 1e4 cost ~2e4 candidates each.
  dist_t d{arr{0, 0, 1}, 1e4, engine};
  for (int i = 0; i < 5; ++i) {
    const auto v = d.sample();
    REQUIRE(std::abs(v[2]) > 0.99);
  }
}

void test_bingham_prolate_high_kappa_is_fast_and_correct() {
  // Aligned-A2 case: κ = 1e5 used to cost ~2e5 candidates (~11 ms) per
  // draw. With the exponential envelope a cap of 64 candidates never
  // trips (old sampler would throw) and E[cos²θ] matches quadrature.
  using arr = std::array<double, 3>;
  for (double kappa : {1.0, 50.0, 1e3, 1e5}) {
    std::mt19937 engine{11};
    rvegen::bingham_distribution<double> d{arr{0, 0, 1}, kappa, engine};
    d.set_max_iterations(64);
    const int n = 20000;
    double m2 = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) {
      const auto v = d.sample();
      m2 += v[2] * v[2];
    }
    const double secs = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t0).count();
    REQUIRE(secs < 1.0);
    m2 /= n;
    // quadrature of E[x²] under exp(κ(x²-1)) on [0,1]; substitution
    // x = 1 - t resolves the boundary layer for large κ.
    double num = 0, den = 0;
    const int q = 2000000;
    const double tmax = std::min(1.0, 40.0 / kappa);
    for (int i = 0; i < q; ++i) {
      const double t = (i + 0.5) / q * tmax;
      const double w = std::exp(-kappa * t * (2 - t));
      num += w * (1 - t) * (1 - t);
      den += w;
    }
    const double expect = num / den;
    const double sd = std::sqrt(0.25 / n);  // loose bound on var(x²) <= 1/4
    if (std::abs(m2 - expect) > 5 * sd) {
      std::cerr << "kappa=" << kappa << " m2=" << m2 << " expect=" << expect
                << "\n";
    }
    REQUIRE(std::abs(m2 - expect) <= 5 * sd);
  }
}

void test_bingham_oblate_is_fast_and_correct() {
  // Oblate κ down to -1e10 used to cost ~1e5 candidates per draw; the
  // truncated-normal path is O(1) (cap 64 would trip on the old sampler) and
  // E[cos²θ] matches quadrature of exp(κu²) on [0, 1].
  using arr = std::array<double, 3>;
  for (double kappa : {-1.0, -1e3, -1e10}) {
    std::mt19937 engine{21};
    rvegen::bingham_distribution<double> d{arr{0, 0, 1}, kappa, engine};
    d.set_max_iterations(64);
    const int n = 20000;
    double m2 = 0;
    for (int i = 0; i < n; ++i) {
      const auto v = d.sample();
      REQUIRE(std::abs(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 1.0) < 1e-9);
      m2 += v[2] * v[2];
    }
    m2 /= n;
    const double a = -kappa;
    const double umax = std::min(1.0, 12.0 / std::sqrt(a));
    double num = 0, den = 0;
    const int q = 2000000;
    for (int i = 0; i < q; ++i) {
      const double u = (i + 0.5) / q * umax;
      const double w = std::exp(-a * u * u);
      num += w * u * u;
      den += w;
    }
    const double expect = num / den;
    const double tol = 6 * std::sqrt(expect / n);
    if (std::abs(m2 - expect) > tol) {
      std::cerr << "kappa=" << kappa << " m2=" << m2 << " expect=" << expect
                << "\n";
    }
    REQUIRE(std::abs(m2 - expect) <= tol);
  }
}

} // namespace

int main() {
  test_von_mises_tiny_kappa_returns_finite_angles();
  test_von_mises_huge_kappa_is_point_mass();
  test_von_mises_rejects_bad_kappa();
  test_von_mises_order_parameter_matches_i2_over_i0();
  test_bingham_rejects_bad_axis_and_kappa();
  test_bingham_samples_are_unit_length();
  test_bingham_iteration_cap_throws_quickly();
  test_bingham_kappa_range_and_default_cap();
  test_bingham_prolate_high_kappa_is_fast_and_correct();
  test_bingham_oblate_is_fast_and_correct();

  if (failures > 0) {
    std::cerr << failures << " orientation sampler test failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "orientation_sampler_test: PASS\n";
  return EXIT_SUCCESS;
}
