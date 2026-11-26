#pragma once

// Advani-Tucker orientation tensor A2 = ⟨p ⊗ p⟩ ↔ sampler parameters
// (#6). Converts a measured second-order orientation tensor into the
// (mean, κ) of the two orientation samplers, and back:
//
//   * 2D, von Mises f(θ) ∝ exp(κ cos(θ − μ)) (oriented_uniform):
//       μ  = ½ atan2(2·a12, a11 − a22)
//       S2 = √((a11 − a22)² + (2·a12)²) = ⟨cos 2(θ − μ)⟩ = I2(κ)/I0(κ)
//     A 2D A2 with unit trace has exactly two degrees of freedom, so the
//     fit is exact (residual 0 up to rounding).
//
//   * 3D, axisymmetric Bingham / Watson f(x) ∝ exp(κ (μ·x)²):
//       A2_model = ((1 − m)/2)·I + ((3m − 1)/2)·μμᵀ, m = ⟨(μ·x)²⟩(κ)
//     A general A2 has three distinct eigenvalues; Watson only has two.
//     The fit projects onto the closest axisymmetric tensor that shares
//     A2's eigenframe: either the prolate fit (axis = eigenvector of the
//     largest eigenvalue, the other two averaged) or the oblate fit (axis
//     = eigenvector of the smallest eigenvalue), whichever is closer in
//     Frobenius norm. The axis is therefore the principal eigenvector for
//     fibre-like (prolate) tensors and the girdle normal for planar
//     (oblate) ones. The Frobenius distance between the (trace-
//     normalised) input and the fitted model is reported as `residual`.
//
// Both κ inversions are monotone 1-D root finds by bisection. κ is
// clamped to what the samplers support: von Mises S2 → 1 maps to the
// point-mass regime (κ > 1e12), Watson to bingham_distribution's
// [min_kappa, max_kappa]; a clamp shows up in `residual`.
//
// Input validation: finite components, trace within 1e-2 of 1 (the
// tensor is then renormalised to unit trace), positive semidefinite
// within 1e-9. Violations throw std::invalid_argument.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include <boost/math/quadrature/tanh_sinh.hpp>
#include <Mathematics/SymmetricEigensolver3x3.h>

#include "bingham_distribution.h"

namespace rvegen {

template <typename T = double>
struct von_mises_a2_fit {
  T mean_angle;   // μ in radians, in (−π/2, π/2]
  T kappa;        // von Mises concentration
  T residual;     // ‖A2_input − A2_model‖_F (trace-normalised input)
};

template <typename T = double>
struct watson_a2_fit {
  std::array<T, 3> axis;  // unit mean axis (sign arbitrary)
  T kappa;                // Watson / axisymmetric-Bingham concentration
  T residual;             // ‖A2_input − A2_model‖_F (trace-normalised input)
};

// Symmetric 3×3 A2 as its six unique components.
template <typename T = double>
struct orientation_tensor_3d {
  T a11, a22, a33, a12, a13, a23;
};

namespace detail {

inline constexpr double a2_trace_tolerance = 1e-2;
inline constexpr double a2_psd_tolerance = 1e-9;
// Above this κ oriented_uniform_distribution returns the mean exactly.
inline constexpr double von_mises_point_mass_kappa = 1e12;

// I1(κ)/I0(κ). std::cyl_bessel_i overflows near κ ≈ 700, so large κ uses
// the Hankel asymptotic series of both functions (the common
// e^κ/√(2πκ) prefactor cancels); 12 terms at κ ≥ 500 are exact to
// rounding.
template <typename T> T bessel_i1_over_i0(T k) {
  if (k <= T{0}) return T{0};
  if (k < T{500}) {
    return std::cyl_bessel_i(T{1}, k) / std::cyl_bessel_i(T{0}, k);
  }
  T s0{1}, s1{1}, t0{1}, t1{1};
  for (int j = 1; j <= 12; ++j) {
    const T odd = T(2 * j - 1) * T(2 * j - 1);
    t0 *= -(T{0} - odd) / (T(j) * T{8} * k);
    t1 *= -(T{4} - odd) / (T(j) * T{8} * k);
    s0 += t0;
    s1 += t1;
  }
  return s1 / s0;
}

// Monotone-increasing f on [lo, hi]: returns x with f(x) ≈ target.
template <typename T, typename F> T bisect(F&& f, T target, T lo, T hi) {
  for (int i = 0; i < 400; ++i) {
    const T mid = lo + (hi - lo) / T{2};
    if (mid <= lo || mid >= hi) break;
    if (f(mid) < target) lo = mid; else hi = mid;
  }
  return lo + (hi - lo) / T{2};
}

template <typename T> T frobenius_sym3(orientation_tensor_3d<T> const& a,
                                       orientation_tensor_3d<T> const& b) {
  const T d11 = a.a11 - b.a11, d22 = a.a22 - b.a22, d33 = a.a33 - b.a33;
  const T d12 = a.a12 - b.a12, d13 = a.a13 - b.a13, d23 = a.a23 - b.a23;
  return std::sqrt(d11 * d11 + d22 * d22 + d33 * d33 +
                   T{2} * (d12 * d12 + d13 * d13 + d23 * d23));
}

template <typename T> T check_trace(T tr) {
  if (!std::isfinite(tr) ||
      std::abs(tr - T{1}) > T(a2_trace_tolerance)) {
    throw std::invalid_argument{
        "orientation tensor A2: trace must be 1 (within 1e-2), got " +
        std::to_string(static_cast<double>(tr))};
  }
  return tr;
}

} // namespace detail

// ---------------------------------------------------------------- 2D ---

// S2(κ) = ⟨cos 2(θ − μ)⟩ = I2(κ)/I0(κ) for von Mises; 1 in the sampler's
// point-mass regime.
template <typename T> T von_mises_s2(T kappa) {
  if (!(kappa > T{0})) return T{0};
  if (kappa > T(detail::von_mises_point_mass_kappa)) return T{1};
  if (kappa < T{500}) {
    return std::cyl_bessel_i(T{2}, kappa) / std::cyl_bessel_i(T{0}, kappa);
  }
  // I2 = I0 − (2/κ) I1 (Bessel recurrence).
  return T{1} - T{2} * detail::bessel_i1_over_i0(kappa) / kappa;
}

// Inverse of von_mises_s2 on [0, 1]. S2 too close to 1 for any κ ≤ 1e12
// maps to 2e12 (point mass, S2 = 1).
template <typename T> T von_mises_kappa_from_s2(T s2) {
  if (!(s2 >= T{0} && s2 <= T{1})) {
    throw std::invalid_argument{
        "von_mises_kappa_from_s2: S2 must lie in [0, 1]"};
  }
  if (s2 == T{0}) return T{0};
  const T k_max = T(detail::von_mises_point_mass_kappa);
  if (s2 > von_mises_s2(k_max)) return T{2} * k_max;
  T hi{1};
  while (von_mises_s2(hi) < s2) hi *= T{2};
  return detail::bisect([](T k) { return von_mises_s2(k); }, s2, T{0}, hi);
}

// A2 of the von Mises density (a11, a22, a12).
template <typename T> std::array<T, 3> von_mises_a2(T mean_angle, T kappa) {
  const T s2 = von_mises_s2(kappa);
  const T c = std::cos(T{2} * mean_angle), s = std::sin(T{2} * mean_angle);
  return {T{0.5} * (T{1} + s2 * c), T{0.5} * (T{1} - s2 * c),
          T{0.5} * s2 * s};
}

template <typename T> von_mises_a2_fit<T> fit_von_mises_a2(T a11, T a22,
                                                           T a12) {
  if (!std::isfinite(a11) || !std::isfinite(a22) || !std::isfinite(a12)) {
    throw std::invalid_argument{
        "orientation tensor A2: components must be finite"};
  }
  const T tr = detail::check_trace(a11 + a22);
  a11 /= tr; a22 /= tr; a12 /= tr;
  const T d = a11 - a22, o = T{2} * a12;
  T s2 = std::hypot(d, o);
  // Eigenvalues are ½(1 ± S2): PSD ⇔ S2 ≤ 1.
  if (s2 > T{1} + T(detail::a2_psd_tolerance)) {
    throw std::invalid_argument{
        "orientation tensor A2: not positive semidefinite "
        "(a12² > a11·a22)"};
  }
  s2 = std::min(s2, T{1});
  von_mises_a2_fit<T> fit{};
  fit.mean_angle = s2 > T{0} ? T{0.5} * std::atan2(o, d) : T{0};
  fit.kappa = von_mises_kappa_from_s2(s2);
  const auto m = von_mises_a2(fit.mean_angle, fit.kappa);
  const T r11 = a11 - m[0], r22 = a22 - m[1], r12 = a12 - m[2];
  fit.residual = std::sqrt(r11 * r11 + r22 * r22 + T{2} * r12 * r12);
  return fit;
}

// ---------------------------------------------------------------- 3D ---

// m(κ) = ⟨(μ·x)²⟩ for the Watson density: ∫₀¹ t² w / ∫₀¹ w with
// w(t) = exp(κ(t² − 1)) for κ > 0 and exp(κt²) otherwise (both ≤ 1, so
// nothing overflows). tanh-sinh clusters its nodes at the endpoints,
// where the integrand concentrates for large |κ|.
template <typename T> T watson_second_moment(T kappa) {
  static const boost::math::quadrature::tanh_sinh<T> integrator;
  const T tol = std::numeric_limits<T>::epsilon() * T{1000};
  auto w = [kappa](T t) {
    return kappa > T{0} ? std::exp(kappa * (t - T{1}) * (t + T{1}))
                        : std::exp(kappa * t * t);
  };
  const T den = integrator.integrate(w, T{0}, T{1}, tol);
  const T num = integrator.integrate(
      [&w](T t) { return t * t * w(t); }, T{0}, T{1}, tol);
  return num / den;
}

// Inverse of watson_second_moment on (0, 1), clamped to the range
// bingham_distribution accepts.
template <typename T> T watson_kappa_from_second_moment(T m) {
  if (!(m >= T{0} && m <= T{1})) {
    throw std::invalid_argument{
        "watson_kappa_from_second_moment: m must lie in [0, 1]"};
  }
  constexpr T k_max = bingham_distribution<T>::max_kappa;
  constexpr T k_min = bingham_distribution<T>::min_kappa;
  auto g = [](T k) { return watson_second_moment(k); };
  if (m > T{1} / T{3}) {
    if (m >= g(k_max)) return k_max;
    T hi{1};
    while (g(hi) < m) hi *= T{2};
    return detail::bisect(g, m, T{0}, hi);
  }
  if (m < T{1} / T{3}) {
    if (m <= g(k_min)) return k_min;
    T lo{-1};
    while (g(lo) > m) lo *= T{2};
    return detail::bisect(g, m, lo, T{0});
  }
  return T{0};
}

// A2 of the Watson density with unit axis `axis`.
template <typename T>
orientation_tensor_3d<T> watson_a2(std::array<T, 3> const& axis, T kappa) {
  const T m = watson_second_moment(kappa);
  const T iso = (T{1} - m) / T{2}, ani = (T{3} * m - T{1}) / T{2};
  return {iso + ani * axis[0] * axis[0], iso + ani * axis[1] * axis[1],
          iso + ani * axis[2] * axis[2], ani * axis[0] * axis[1],
          ani * axis[0] * axis[2],       ani * axis[1] * axis[2]};
}

template <typename T>
watson_a2_fit<T> fit_watson_a2(orientation_tensor_3d<T> a) {
  for (T v : {a.a11, a.a22, a.a33, a.a12, a.a13, a.a23}) {
    if (!std::isfinite(v)) {
      throw std::invalid_argument{
          "orientation tensor A2: components must be finite"};
    }
  }
  const T tr = detail::check_trace(a.a11 + a.a22 + a.a33);
  a = {a.a11 / tr, a.a22 / tr, a.a33 / tr,
       a.a12 / tr, a.a13 / tr, a.a23 / tr};

  std::array<T, 3> eval{};
  std::array<std::array<T, 3>, 3> evec{};
  gte::SymmetricEigensolver3x3<T>{}(a.a11, a.a12, a.a13, a.a22, a.a23,
                                     a.a33, false, +1, eval, evec);
  if (eval[0] < -T(detail::a2_psd_tolerance)) {
    throw std::invalid_argument{
        "orientation tensor A2: not positive semidefinite"};
  }
  // Ascending eigenvalues. Prolate fit keeps λ₂ on its eigenvector and
  // averages λ₀, λ₁ (Frobenius cost |λ₁ − λ₀|/√2); oblate keeps λ₀ and
  // averages λ₁, λ₂ (cost |λ₂ − λ₁|/√2).
  const bool prolate = (eval[1] - eval[0]) <= (eval[2] - eval[1]);
  const int k = prolate ? 2 : 0;
  watson_a2_fit<T> fit{};
  fit.axis = evec[k];
  fit.kappa = watson_kappa_from_second_moment(
      std::clamp(eval[k], T{0}, T{1}));
  fit.residual = detail::frobenius_sym3(a, watson_a2(fit.axis, fit.kappa));
  return fit;
}

} // namespace rvegen
