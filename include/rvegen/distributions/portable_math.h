#pragma once

// Platform-independent elementary functions for the samplers (#130).
//
// Why this exists:
//   portable_random.h makes the uniform, normal and integer transforms
//   independent of the standard library. The orientation samplers (von
//   Mises, Bingham, uniform rotations) and the lognormal grain sizes also
//   need exp, sin, cos, acos and a cube root. The platform math library
//   (glibc, the MSVC runtime, Apple's libm) rounds these differently in the
//   last bit, and one different bit flips an accept test or shifts every
//   later draw. The functions below are defined exactly here instead.
//
// What they guarantee:
//   The same argument gives the same bits on every platform with IEEE-754
//   binary64 arithmetic in round-to-nearest: only +, -, *, /, std::sqrt
//   (correctly rounded) and exact operations (frexp, ldexp, floor, fmod)
//   are used, in a fixed order, and no product is fused into a following
//   addition (portable_detail::rounded, as in portable_random.h).
//   Outside the guarantee: -ffast-math, /fp:fast, x87 arithmetic.
//
// What they do not guarantee:
//   Correct rounding. The results are within a few units in the last place
//   of the exact value (tests/golden_random_test.cpp checks them against
//   the platform functions), which is far below what a sampler needs.
//   They are also slower than the platform functions; do not use them for
//   bulk geometry.
//
// Each function first rounds its argument (a product passed in cannot be
// fused into the function's first subtraction), so callers need no care
// there. Callers do need portable_detail::rounded() for their own
// "product then add" expressions.
//
// All functions compute in double. The templates accept float and return
// the double result rounded to float, which is as deterministic.

#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <type_traits>

#include "portable_random.h"

// Same as RVEGEN_PORTABLE_FP_BEGIN in portable_random.h (which is local to
// that header).
#if defined(__clang__)
#define RVEGEN_PORTABLE_MATH_FP _Pragma("clang fp contract(off)")
#else
#define RVEGEN_PORTABLE_MATH_FP
#endif

namespace rvegen {

namespace portable_detail {

// a * b, rounded before it can be fused into a following add.
[[nodiscard]] inline double mul(double a, double b) noexcept {
  return rounded(a * b);
}

// 1 / n! for n = 0 .. 22. Every n! up to 22! is exactly representable
// (22! = 2^19 * an odd number below 2^53), so each entry is one correctly
// rounded division of exact operands.
inline constexpr std::array<double, 23> inverse_factorial = [] {
  std::array<double, 23> c{};
  double f = 1.0;
  c[0] = 1.0;
  for (int n = 1; n <= 22; ++n) {
    f *= static_cast<double>(n);   // exact
    c[static_cast<std::size_t>(n)] = 1.0 / f;
  }
  return c;
}();

// Ratio of consecutive terms of the arcsine series,
// asin(x) = sum a_k, a_0 = x, a_k = a_(k-1) * x^2 * asin_ratio[k],
// asin_ratio[k] = (2k - 1)^2 / (2k (2k + 1)): small exact integers, one
// correctly rounded division each.
inline constexpr int asin_terms = 34;
inline constexpr std::array<double, asin_terms> asin_ratio = [] {
  std::array<double, asin_terms> c{};
  c[0] = 1.0;
  for (int k = 1; k < asin_terms; ++k) {
    const double odd = static_cast<double>(2 * k - 1);
    c[static_cast<std::size_t>(k)] =
        (odd * odd) / static_cast<double>(2 * k * (2 * k + 1));
  }
  return c;
}();

// pi/2 in three pieces; the first two have 33 significant bits, so n * p1
// and n * p2 are exact for |n| < 2^20 (the split of fdlibm's
// e_rem_pio2.c).
inline constexpr double pio2_1 = 0x1.921fb544p+0;     // 1.57079632673412561417
inline constexpr double pio2_2 = 0x1.0b4611a6p-34;    // 6.07710050630396597660e-11
inline constexpr double pio2_3 = 0x1.3198a2e037073p-69;   // 2.02226624879595063154e-21
inline constexpr double two_over_pi = 0x1.45f306dc9c883p-1;

// ln 2 = ln2_hi + ln2_lo; ln2_hi has 32 significant bits, so k * ln2_hi is
// exact for |k| < 2^21 (fdlibm's e_exp.c).
inline constexpr double ln2_hi = 0x1.62e42feep-1;         // 6.93147180369123816490e-01
inline constexpr double ln2_lo = 0x1.a39ef35793c76p-33;   // 1.90821492927058770002e-10
inline constexpr double inv_ln2 = 0x1.71547652b82fep+0;

// sin(r) and cos(r) for |r| <= pi/4 (a little beyond is fine): Taylor
// polynomials in Horner form, truncation error below 2^-60.
[[nodiscard]] inline double sin_kernel(double r) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  const double r2 = mul(r, r);
  double p = -inverse_factorial[19];
  for (int n = 17; n >= 3; n -= 2) {
    const double sign = ((n / 2) % 2 == 0) ? 1.0 : -1.0;
    p = mul(p, r2) + sign * inverse_factorial[static_cast<std::size_t>(n)];
  }
  // sin r = r + r^3 * p
  const double r3 = mul(r2, r);
  return r + mul(r3, p);
}

[[nodiscard]] inline double cos_kernel(double r) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  const double r2 = mul(r, r);
  double p = inverse_factorial[20];
  for (int n = 18; n >= 2; n -= 2) {
    const double sign = ((n / 2) % 2 == 0) ? 1.0 : -1.0;
    p = mul(p, r2) + sign * inverse_factorial[static_cast<std::size_t>(n)];
  }
  // cos r = 1 + r^2 * p
  return 1.0 + mul(r2, p);
}

// x = n * pi/2 + r with |r| <= pi/4 (about); returns n mod 4 in [0, 3].
[[nodiscard]] inline int reduce_pio2(double x, double& r) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  // Far arguments (beyond 2^19, about 83000 turns): fmod is exact, so the
  // result stays deterministic, but it folds with the double nearest to
  // 2 pi and the angle is then off by up to |x| * 4e-17 rad. The samplers
  // stay within a few turns.
  if (std::abs(x) > 0x1p19) x = std::fmod(x, 0x1.921fb54442d18p+2);
  const double n = std::floor(mul(x, two_over_pi) + 0.5);
  const double t1 = x - mul(n, pio2_1);    // n * pio2_1 is exact
  const double t2 = t1 - mul(n, pio2_2);   // n * pio2_2 is exact
  r = t2 - mul(n, pio2_3);
  // n mod 4, for negative n as well (n is an integer below 2^21).
  const int q = static_cast<int>(n - 4.0 * std::floor(n * 0.25));
  return q;
}

// asin(x) for |x| <= 0.5 (fixed-length series, truncation below 2^-60).
[[nodiscard]] inline double asin_small(double x) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  const double x2 = mul(x, x);
  double term = x;
  double sum = x;
  for (int k = 1; k < asin_terms; ++k) {
    term = mul(mul(term, x2), asin_ratio[static_cast<std::size_t>(k)]);
    sum = sum + term;
  }
  return sum;
}

} // namespace portable_detail

// e^x. NaN gives NaN; overflow gives +inf, underflow 0.
[[nodiscard]] inline double portable_exp(double x) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  using namespace portable_detail;
  x = rounded(x);
  if (!(x == x)) return x;
  if (x > 709.782712893384) return std::numeric_limits<double>::infinity();
  if (x < -745.2) return 0.0;
  const double k = std::floor(mul(x, inv_ln2) + 0.5);
  const double hi = x - mul(k, ln2_hi);   // k * ln2_hi is exact
  const double r = hi - mul(k, ln2_lo);   // |r| <= ln2 / 2 (about 0.3466)
  // Taylor polynomial of degree 14 in Horner form; the remainder is below
  // 0.3466^15 / 15! = 1e-19.
  double p = inverse_factorial[14];
  for (int n = 13; n >= 0; --n)
    p = mul(p, r) + inverse_factorial[static_cast<std::size_t>(n)];
  return std::ldexp(p, static_cast<int>(k));
}

// e^x - 1 without cancellation for small |x|.
[[nodiscard]] inline double portable_expm1(double x) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  using namespace portable_detail;
  x = rounded(x);
  if (!(std::abs(x) < 0.25)) return portable_exp(x) - 1.0;
  // x * (1 + x/2! + x^2/3! + ...), 15 terms (0.25^15 / 16! = 4e-23).
  double p = inverse_factorial[16];
  for (int n = 15; n >= 1; --n)
    p = mul(p, x) + inverse_factorial[static_cast<std::size_t>(n)];
  return mul(x, p);
}

// log(1 + x) for x > -1 without cancellation for small |x|.
[[nodiscard]] inline double portable_log1p(double x) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  using namespace portable_detail;
  x = rounded(x);
  if (!(std::abs(x) < 0.25)) return portable_log(1.0 + x);
  // log(1 + x) = 2 atanh(t), t = x / (2 + x), |t| < 1/7.
  const double t = x / (2.0 + x);
  const double t2 = mul(t, t);
  double p = 1.0 / 27.0;
  for (int n = 25; n >= 1; n -= 2) p = mul(p, t2) + 1.0 / static_cast<double>(n);
  return mul(2.0 * t, p);
}

// sin x and cos x for finite x.
[[nodiscard]] inline double portable_sin(double x) noexcept {
  using namespace portable_detail;
  x = rounded(x);
  if (!(std::abs(x) <= std::numeric_limits<double>::max()))
    return std::numeric_limits<double>::quiet_NaN();
  double r = 0.0;
  switch (reduce_pio2(x, r)) {
    case 0: return sin_kernel(r);
    case 1: return cos_kernel(r);
    case 2: return -sin_kernel(r);
    default: return -cos_kernel(r);
  }
}

[[nodiscard]] inline double portable_cos(double x) noexcept {
  using namespace portable_detail;
  x = rounded(x);
  if (!(std::abs(x) <= std::numeric_limits<double>::max()))
    return std::numeric_limits<double>::quiet_NaN();
  double r = 0.0;
  switch (reduce_pio2(x, r)) {
    case 0: return cos_kernel(r);
    case 1: return -sin_kernel(r);
    case 2: return -cos_kernel(r);
    default: return sin_kernel(r);
  }
}

// acos x on [-1, 1]; arguments outside by rounding are clamped, NaN
// gives NaN.
[[nodiscard]] inline double portable_acos(double x) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  using namespace portable_detail;
  x = rounded(x);
  if (!(x == x)) return x;
  constexpr double pi = std::numbers::pi;
  if (x >= 1.0) return 0.0;
  if (x <= -1.0) return pi;
  if (x > 0.5) {
    // acos x = 2 asin(sqrt((1 - x) / 2)); 1 - x is exact here.
    return 2.0 * asin_small(std::sqrt((1.0 - x) * 0.5));
  }
  if (x < -0.5) {
    return pi - 2.0 * asin_small(std::sqrt((1.0 + x) * 0.5));
  }
  return pi * 0.5 - asin_small(x);
}

// Cube root of a finite x (any sign).
[[nodiscard]] inline double portable_cbrt(double x) noexcept {
  RVEGEN_PORTABLE_MATH_FP
  using namespace portable_detail;
  x = rounded(x);
  if (x == 0.0 || !(std::abs(x) <= std::numeric_limits<double>::max()))
    return x;   // +-0, +-inf, NaN
  int e = 0;
  double m = std::frexp(std::abs(x), &e);   // exact, m in [0.5, 1)
  // Make the exponent a multiple of 3: m in [0.125, 1).
  int rem = e % 3;
  if (rem < 0) rem += 3;
  if (rem != 0) {
    m = std::ldexp(m, rem - 3);
    e += 3 - rem;
  }
  // Newton for y^3 = m from a linear start (within 12 %); six steps reach
  // the last bit, the count is fixed.
  double y = 0.42 + mul(0.6, m);
  for (int i = 0; i < 6; ++i) {
    const double y2 = mul(y, y);
    y = (mul(2.0, y) + m / y2) / 3.0;
  }
  const double root = std::ldexp(y, e / 3);
  return x < 0.0 ? -root : root;
}

// float (and other floating-point) arguments: computed in double.
#define RVEGEN_PORTABLE_MATH_TEMPLATE(name)                              \
  template <typename T>                                                  \
    requires(std::is_floating_point_v<T> && !std::is_same_v<T, double>)  \
  [[nodiscard]] inline T name(T x) noexcept {                            \
    return static_cast<T>(name(static_cast<double>(x)));                 \
  }
RVEGEN_PORTABLE_MATH_TEMPLATE(portable_exp)
RVEGEN_PORTABLE_MATH_TEMPLATE(portable_expm1)
RVEGEN_PORTABLE_MATH_TEMPLATE(portable_log1p)
RVEGEN_PORTABLE_MATH_TEMPLATE(portable_sin)
RVEGEN_PORTABLE_MATH_TEMPLATE(portable_cos)
RVEGEN_PORTABLE_MATH_TEMPLATE(portable_acos)
RVEGEN_PORTABLE_MATH_TEMPLATE(portable_cbrt)
#undef RVEGEN_PORTABLE_MATH_TEMPLATE
#undef RVEGEN_PORTABLE_MATH_FP

} // namespace rvegen
