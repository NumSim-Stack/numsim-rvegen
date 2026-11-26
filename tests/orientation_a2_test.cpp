// A2 orientation-tensor input for the orientation samplers (#6).
//
// 1. Acceptance: N = 10,000 draws from von_mises_a2 (2D) and bingham_a2
//    (3D); the empirical A2 = ⟨p ⊗ p⟩ matches the input per component
//    within 3/√N, for several anisotropies.
// 2. Round trips: κ → A2 → κ and A2 → (axis, κ) → A2.
// 3. Non-axisymmetric 3D input: samples match the fitted tensor and
//    residual() equals the input-to-fit Frobenius distance.
// 4. Validation (trace, PSD, finiteness) and the JSON registry names
//    "von_mises_a2", "bingham_a2" and the "von_mises" alias.

#include <array>
#include <cmath>
#include <cstdio>
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

constexpr int n_samples = 10000;
const double tol = 3.0 / std::sqrt(double(n_samples));   // 0.03

using a2_3d = rvegen::orientation_tensor_3d<double>;

double max_abs_diff(a2_3d const& a, a2_3d const& b) {
  return std::max({std::abs(a.a11 - b.a11), std::abs(a.a22 - b.a22),
                   std::abs(a.a33 - b.a33), std::abs(a.a12 - b.a12),
                   std::abs(a.a13 - b.a13), std::abs(a.a23 - b.a23)});
}

a2_3d empirical_a2(rvegen::bingham_a2_distribution<double>& d) {
  a2_3d e{0, 0, 0, 0, 0, 0};
  for (int i = 0; i < n_samples; ++i) {
    const auto x = d();
    e.a11 += x[0] * x[0]; e.a22 += x[1] * x[1]; e.a33 += x[2] * x[2];
    e.a12 += x[0] * x[1]; e.a13 += x[0] * x[2]; e.a23 += x[1] * x[2];
  }
  const double n = n_samples;
  return {e.a11 / n, e.a22 / n, e.a33 / n, e.a12 / n, e.a13 / n, e.a23 / n};
}

// R·diag(l)·Rᵀ for the rotation taking ẑ to unit `n` (Rodrigues), used to
// build rotated axisymmetric inputs with a known axis.
a2_3d axisymmetric(std::array<double, 3> n, double lambda_axis) {
  const double iso = (1.0 - lambda_axis) / 2.0, ani = lambda_axis - iso;
  return {iso + ani * n[0] * n[0], iso + ani * n[1] * n[1],
          iso + ani * n[2] * n[2], ani * n[0] * n[1],
          ani * n[0] * n[2],       ani * n[1] * n[2]};
}

void test_2d_acceptance() {
  struct c { double a11, a22, a12; };
  for (c in : {c{0.5, 0.5, 0.0}, c{0.7, 0.3, 0.1}, c{0.9, 0.1, 0.05},
               c{0.2, 0.8, -0.3}, c{0.5, 0.5, 0.45}, c{1.0, 0.0, 0.0}}) {
    std::mt19937 engine{101};
    rvegen::von_mises_a2_distribution<double> d{in.a11, in.a22, in.a12,
                                                engine};
    double c2 = 0, s2 = 0, cs = 0;
    for (int i = 0; i < n_samples; ++i) {
      const double t = d();
      c2 += std::cos(t) * std::cos(t);
      s2 += std::sin(t) * std::sin(t);
      cs += std::cos(t) * std::sin(t);
    }
    c2 /= n_samples; s2 /= n_samples; cs /= n_samples;
    const double dev = std::max({std::abs(c2 - in.a11), std::abs(s2 - in.a22),
                                 std::abs(cs - in.a12)});
    std::printf("2D A2=(%.2f,%.2f,%.2f) kappa=%.4g mu=%.4f residual=%.1e "
                "max|dA2|=%.4f (tol %.3f)\n",
                in.a11, in.a22, in.a12, d.kappa(), d.mean_angle(),
                d.residual(), dev, tol);
    REQUIRE(dev < tol);
    REQUIRE(d.residual() < 1e-9);
  }
}

void test_3d_acceptance() {
  const double s3 = 1.0 / std::sqrt(3.0);
  struct c { const char* name; a2_3d a; };
  for (c in : {c{"isotropic", axisymmetric({0, 0, 1}, 1.0 / 3.0)},
               c{"prolate z 0.7", axisymmetric({0, 0, 1}, 0.7)},
               c{"prolate 111 0.9", axisymmetric({s3, s3, s3}, 0.9)},
               c{"oblate y 0.1", axisymmetric({0, 1, 0}, 0.1)},
               c{"oblate xz-diag 0.02",
                 axisymmetric({std::sqrt(0.5), 0, std::sqrt(0.5)}, 0.02)}}) {
    std::mt19937 engine{202};
    rvegen::bingham_a2_distribution<double> d{in.a, engine};
    const double dev = max_abs_diff(empirical_a2(d), in.a);
    std::printf("3D %-20s kappa=%9.4g residual=%.1e max|dA2|=%.4f (tol %.3f)\n",
                in.name, d.kappa(), d.residual(), dev, tol);
    REQUIRE(dev < tol);
    REQUIRE(d.residual() < 1e-9);
  }
}

void test_3d_non_axisymmetric_projection() {
  // Example from #6: three distinct eigenvalues.
  const a2_3d in{0.7, 0.2, 0.1, 0.05, 0.0, 0.0};
  std::mt19937 engine{303};
  rvegen::bingham_a2_distribution<double> d{in, engine};
  const auto fit = rvegen::watson_a2(d.mean_axis(), d.kappa());
  const auto emp = empirical_a2(d);
  const double dev_fit = max_abs_diff(emp, fit);
  const double r11 = in.a11 - fit.a11, r22 = in.a22 - fit.a22,
               r33 = in.a33 - fit.a33, r12 = in.a12 - fit.a12,
               r13 = in.a13 - fit.a13, r23 = in.a23 - fit.a23;
  const double frob = std::sqrt(r11 * r11 + r22 * r22 + r33 * r33 +
                                2 * (r12 * r12 + r13 * r13 + r23 * r23));
  std::printf("3D non-axisymmetric (0.7,0.2,0.1,a12=0.05): kappa=%.4g "
              "residual=%.4f max|emp-fit|=%.4f max|emp-input|=%.4f\n",
              d.kappa(), d.residual(), dev_fit, max_abs_diff(emp, in));
  REQUIRE(dev_fit < tol);
  REQUIRE(std::abs(d.residual() - frob) < 1e-12);
  REQUIRE(d.residual() > 0.05);   // genuinely non-axisymmetric
  REQUIRE(d.kappa() > 0);         // fibre-like → prolate fit
  // Prolate axis is the principal eigenvector, mostly along x.
  REQUIRE(std::abs(d.mean_axis()[0]) > 0.95);
}

void test_round_trips() {
  // 2D: κ → A2 → κ, several mean angles.
  for (double kappa : {0.0, 0.3, 2.0, 10.0, 150.0, 1e3}) {
    for (double mu : {0.0, 0.4, -1.2}) {
      const auto a = rvegen::von_mises_a2(mu, kappa);
      const auto fit = rvegen::fit_von_mises_a2(a[0], a[1], a[2]);
      REQUIRE(std::abs(fit.kappa - kappa) <= 1e-8 * std::max(1.0, kappa));
      if (kappa > 0) {
        REQUIRE(std::abs(std::remainder(fit.mean_angle - mu,
                                        std::numbers::pi)) < 1e-9);
      }
      const auto back = rvegen::von_mises_a2(fit.mean_angle, fit.kappa);
      REQUIRE(std::abs(back[0] - a[0]) < 1e-12);
      REQUIRE(std::abs(back[2] - a[2]) < 1e-12);
    }
  }
  // 3D: κ → A2 → (axis, κ) → A2.
  const double s3 = 1.0 / std::sqrt(3.0);
  for (double kappa : {-1e6, -200.0, -3.0, 0.0, 0.5, 4.0, 60.0, 1e4}) {
    for (std::array<double, 3> n : {std::array<double, 3>{0, 0, 1},
                                    std::array<double, 3>{s3, -s3, s3}}) {
      const auto a = rvegen::watson_a2(n, kappa);
      const auto fit = rvegen::fit_watson_a2(a);
      REQUIRE(std::abs(fit.kappa - kappa) <= 1e-6 * std::max(1.0, std::abs(kappa)));
      if (kappa != 0.0) {
        const double dot = fit.axis[0] * n[0] + fit.axis[1] * n[1] +
                           fit.axis[2] * n[2];
        REQUIRE(std::abs(std::abs(dot) - 1.0) < 1e-9);
      }
      REQUIRE(max_abs_diff(rvegen::watson_a2(fit.axis, fit.kappa), a) < 1e-10);
      REQUIRE(fit.residual < 1e-10);
    }
  }
  // Fully aligned 3D input: κ clamps to the sampler's max, tiny residual.
  const auto aligned = rvegen::fit_watson_a2(a2_3d{1, 0, 0, 0, 0, 0});
  REQUIRE(aligned.kappa == rvegen::bingham_distribution<double>::max_kappa);
  REQUIRE(aligned.residual < 1e-4);
}

void test_validation() {
  std::mt19937 engine{4};
  using vm = rvegen::von_mises_a2_distribution<double>;
  using bg = rvegen::bingham_a2_distribution<double>;
  const double nan_v = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(throws_invalid_argument([&] { vm d{0.7, 0.7, 0.0, engine}; }));  // trace
  REQUIRE(throws_invalid_argument([&] { vm d{0.5, 0.5, 0.6, engine}; }));  // not PSD
  REQUIRE(throws_invalid_argument([&] { vm d{nan_v, 0.5, 0.0, engine}; }));
  REQUIRE(throws_invalid_argument([&] { bg d{a2_3d{0.5, 0.5, 0.5, 0, 0, 0}, engine}; }));
  REQUIRE(throws_invalid_argument([&] { bg d{a2_3d{0.5, 0.5, 0.0, 0.6, 0, 0}, engine}; }));
  REQUIRE(throws_invalid_argument([&] { bg d{a2_3d{nan_v, 0.5, 0.5, 0, 0, 0}, engine}; }));
  // Slightly off-trace measured input is renormalised, not rejected.
  vm ok{0.703, 0.300, 0.1, engine};
  REQUIRE(std::isfinite(ok()));
}

void test_json_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_direction_distributions<>();
  std::mt19937 engine{5};

  auto vm = rvegen::build_from_json(
      rvegen::distribution_registry_t<>::instance(),
      nlohmann::json::parse(R"({"type": "von_mises_a2",
                                "a11": 0.8, "a22": 0.2, "a12": 0.0})"),
      engine);
  REQUIRE(vm != nullptr);
  REQUIRE(std::isfinite((*vm)()));

  auto alias = rvegen::build_from_json(
      rvegen::distribution_registry_t<>::instance(),
      nlohmann::json::parse(R"({"type": "von_mises",
                                "mean_angle": 0.3, "kappa": 4.0})"),
      engine);
  REQUIRE(alias != nullptr);
  REQUIRE(std::isfinite((*alias)()));

  auto bg = rvegen::build_from_json(
      rvegen::direction_distribution_registry_t<>::instance(),
      nlohmann::json::parse(R"({"type": "bingham_a2",
                                "a11": 0.7, "a22": 0.2, "a33": 0.1,
                                "a12": 0.05, "a13": 0.0, "a23": 0.0})"),
      engine);
  REQUIRE(bg != nullptr);
  const auto v = (*bg)();
  REQUIRE(std::abs(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] - 1.0) < 1e-9);
}

} // namespace

int main() {
  test_2d_acceptance();
  test_3d_acceptance();
  test_3d_non_axisymmetric_projection();
  test_round_trips();
  test_validation();
  test_json_registry();

  if (failures > 0) {
    std::cerr << failures << " orientation A2 test failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "orientation_a2_test: PASS\n";
  return EXIT_SUCCESS;
}
