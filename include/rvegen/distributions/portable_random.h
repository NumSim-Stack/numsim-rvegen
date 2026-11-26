#pragma once

// Toolchain-independent random transforms (#76).
//
// Why this exists:
//   The engines in <random> (std::mt19937, std::mt19937_64, ...) are
//   fully specified, so a fixed seed gives the same raw bit stream on
//   every standard library. The *distributions* are not:
//   std::uniform_real_distribution, std::normal_distribution and
//   std::uniform_int_distribution are implementation-defined, and
//   libstdc++, libc++ and MSVC turn the same bit stream into different
//   values. rvegen therefore samples through the transforms below, which
//   are defined exactly here and use the engine's raw output only.
//
// Reproducibility guarantee:
//   Same engine type + same seed + same sequence of calls gives
//   bit-identical results on any conforming toolchain, provided that
//   floating-point arithmetic is IEEE-754 binary32/binary64 with
//   round-to-nearest.
//   Only +, -, *, /, std::sqrt (correctly rounded under IEEE-754) and
//   exact scaling by powers of two are used; logarithms come from
//   `portable_log` below rather than the platform libm.
//   FMA contraction would change the last bits, so it is blocked here
//   regardless of -march / -mfma: Clang via `#pragma clang fp
//   contract(off)`, GCC via an empty asm barrier that forces every
//   product feeding an addition to be rounded first (see
//   `portable_detail::rounded`). MSVC does not contract under its
//   default /fp:precise.
//   Outside the guarantee: -ffast-math (or /fp:fast), x87 arithmetic
//   (32-bit x86 without SSE2), and anything that consumes these values
//   through libm transcendentals (e.g. the orientation samplers call
//   std::cos / std::exp, whose last bits differ between libms). Results
//   there stay statistically correct but may differ in the last bits.
//
// Transforms:
//   * random_bits<N>(engine): N bits built from successive engine words,
//     first word most significant. Each engine call contributes
//     min(W, remaining) bits taken from the *top* of the word, where W
//     is the engine's word width. For a 32-bit engine and N = 53 this
//     is (w1 << 21) | (w2 >> 11); for a 64-bit engine, w1 >> 11.
//   * canonical<T>(engine): random_bits<D>(engine) * 2^-D with
//     D = std::numeric_limits<T>::digits (53 for double), a uniform
//     value on the grid {0, 2^-D, ..., 1 - 2^-D}, i.e. in [0, 1).
//   * portable_uniform_real<T>{a, b}: a + (b - a) * canonical<T>(engine),
//     clamped to the largest value below b if rounding reaches b.
//   * portable_normal<T>{mean, stddev}: Marsaglia's polar method.
//     u = 2 * canonical - 1, v = 2 * canonical - 1 (u drawn first),
//     s = u*u + v*v, rejected unless 0 < s < 1;
//     f = sqrt(-2 * portable_log(s) / s). The call returns
//     mean + stddev * u * f and caches v * f, which the next call returns
//     as mean + stddev * (v * f) without touching the engine.
//     reset() drops the cached value.
//   * portable_uniform_int<I>{a, b}: inclusive [a, b] by rejection:
//     draw r = random_bits<64>(engine); with n = b - a + 1, reject
//     r >= 2^64 - (2^64 mod n), else return a + r mod n.
//     (n = 2^64 returns a + r directly.)

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <type_traits>

#if defined(__clang__)
#define RVEGEN_PORTABLE_FP_BEGIN _Pragma("clang fp contract(off)")
#else
#define RVEGEN_PORTABLE_FP_BEGIN
#endif

namespace rvegen {

namespace portable_detail {

template <typename Engine>
[[nodiscard]] consteval int engine_word_bits() {
  using result_type = typename Engine::result_type;
  static_assert(std::is_unsigned_v<result_type>,
                "portable_random: engine result_type must be unsigned");
  static_assert(Engine::min() == 0,
                "portable_random: engine must have min() == 0 "
                "(e.g. std::mt19937, std::mt19937_64)");
  constexpr std::uint64_t max = static_cast<std::uint64_t>(Engine::max());
  static_assert(max != 0 && ((max + 1) & max) == 0,
                "portable_random: engine max() must be 2^W - 1");
  return max == std::numeric_limits<std::uint64_t>::max()
             ? 64
             : std::countr_one(max);
}

// Returns `v` unchanged, but prevents GCC from fusing the product that
// produced it into a following add/sub (FMA contraction). Clang is
// handled by the pragma in RVEGEN_PORTABLE_FP_BEGIN.
template <typename T>
[[nodiscard]] inline T rounded(T v) noexcept {
#if defined(__GNUC__) && !defined(__clang__)
  asm("" : "+m"(v));
#endif
  return v;
}

} // namespace portable_detail

// Top-N bits from successive engine words; see the header comment.
template <int N, typename Engine>
[[nodiscard]] std::uint64_t random_bits(Engine& engine) {
  static_assert(N >= 1 && N <= 64, "random_bits: 1 <= N <= 64");
  constexpr int W = portable_detail::engine_word_bits<Engine>();
  std::uint64_t r = 0;
  int have = 0;
  while (have < N) {
    const int take = (N - have) < W ? (N - have) : W;
    const std::uint64_t word = static_cast<std::uint64_t>(engine());
    const std::uint64_t top = word >> (W - take);
    r = take == 64 ? top : ((r << take) | top);
    have += take;
  }
  return r;
}

// Uniform on [0, 1) with std::numeric_limits<T>::digits random bits.
template <typename T, typename Engine>
[[nodiscard]] T canonical(Engine& engine) {
  static_assert(std::is_floating_point_v<T>);
  constexpr int D = std::numeric_limits<T>::digits < 64
                        ? std::numeric_limits<T>::digits
                        : 64;
  // 2^-D is exact; the integer (< 2^D) converts to T exactly.
  return static_cast<T>(random_bits<D>(engine)) * std::ldexp(T{1}, -D);
}

// Natural logarithm for finite x > 0 using only exact scaling and
// +, -, *, / so the result does not depend on the platform libm.
// x = m * 2^k with m in [sqrt(1/2), sqrt(2)); log(m) = 2 atanh(t),
// t = (m - 1) / (m + 1), |t| <= 0.1716, summed as a fixed 14-term odd
// series in Horner form (truncation error < 2^-64 relative).
template <typename T>
[[nodiscard]] T portable_log(T x) {
  RVEGEN_PORTABLE_FP_BEGIN
  static_assert(std::is_floating_point_v<T>);
  int k = 0;
  T m = std::frexp(x, &k);  // exact: x = m * 2^k, m in [0.5, 1)
  if (m < std::numbers::sqrt2_v<T> / T{2}) {
    m = m * T{2};
    k = k - 1;
  }
  const T t = (m - T{1}) / (m + T{1});
  const T t2 = t * t;
  T p = T{1} / T{27};
  for (int n = 25; n >= 1; n -= 2) {
    const T tp = portable_detail::rounded(p * t2);
    p = tp + T{1} / T(n);
  }
  const T log_m = portable_detail::rounded(T{2} * t * p);
  const T k_ln2 = portable_detail::rounded(T(k) * std::numbers::ln2_v<T>);
  return k_ln2 + log_m;
}

// Drop-in replacement for std::uniform_real_distribution: [a, b).
// The caller validates a <= b (both finite).
template <typename T>
class portable_uniform_real {
public:
  using result_type = T;

  portable_uniform_real() = default;
  portable_uniform_real(T a, T b) noexcept : _a{a}, _b{b} {}

  template <typename Engine>
  [[nodiscard]] T operator()(Engine& engine) {
    RVEGEN_PORTABLE_FP_BEGIN
    const T u = canonical<T>(engine);
    if (!(_a < _b)) return _a;  // degenerate a == b (engine still advanced)
    const T span = _b - _a;
    const T scaled = portable_detail::rounded(span * u);
    const T x = _a + scaled;
    return x < _b ? x : std::nextafter(_b, _a);
  }

  [[nodiscard]] T a() const noexcept { return _a; }
  [[nodiscard]] T b() const noexcept { return _b; }
  void param(T a, T b) noexcept { _a = a; _b = b; }
  void reset() noexcept {}

private:
  T _a{0};
  T _b{1};
};

// Drop-in replacement for std::normal_distribution (Marsaglia polar).
// The caller validates stddev > 0 (finite).
template <typename T>
class portable_normal {
public:
  using result_type = T;

  portable_normal() = default;
  portable_normal(T mean, T stddev) noexcept : _mean{mean}, _stddev{stddev} {}

  template <typename Engine>
  [[nodiscard]] T operator()(Engine& engine) {
    RVEGEN_PORTABLE_FP_BEGIN
    if (_cached) {
      const T z = *_cached;
      _cached.reset();
      return scale(z);
    }
    T u{};
    T v{};
    T s{};
    do {
      const T cu = canonical<T>(engine);
      const T cv = canonical<T>(engine);
      u = portable_detail::rounded(T{2} * cu) - T{1};
      v = portable_detail::rounded(T{2} * cv) - T{1};
      const T uu = portable_detail::rounded(u * u);
      const T vv = portable_detail::rounded(v * v);
      s = uu + vv;
    } while (!(s < T{1}) || s == T{0});
    const T minus_two_log = T{-2} * portable_log(s);
    const T f = std::sqrt(minus_two_log / s);
    _cached = v * f;
    return scale(u * f);
  }

  [[nodiscard]] T mean() const noexcept { return _mean; }
  [[nodiscard]] T stddev() const noexcept { return _stddev; }
  void reset() noexcept { _cached.reset(); }

private:
  [[nodiscard]] T scale(T z) const noexcept {
    RVEGEN_PORTABLE_FP_BEGIN
    const T sz = portable_detail::rounded(_stddev * z);
    return _mean + sz;
  }

  T _mean{0};
  T _stddev{1};
  std::optional<T> _cached;
};

// Drop-in replacement for std::uniform_int_distribution: inclusive [a, b].
template <typename I>
class portable_uniform_int {
public:
  using result_type = I;
  static_assert(std::is_integral_v<I>);

  portable_uniform_int() = default;
  portable_uniform_int(I a, I b) noexcept : _a{a}, _b{b} {}

  template <typename Engine>
  [[nodiscard]] I operator()(Engine& engine) {
    using U = std::make_unsigned_t<I>;
    // n - 1 = b - a, computed in unsigned arithmetic (no overflow).
    const std::uint64_t n_minus_1 =
        static_cast<std::uint64_t>(static_cast<U>(static_cast<U>(_b) -
                                                  static_cast<U>(_a)));
    if (n_minus_1 == std::numeric_limits<std::uint64_t>::max()) {
      return static_cast<I>(static_cast<U>(_a) +
                            static_cast<U>(random_bits<64>(engine)));
    }
    constexpr std::uint64_t max64 = std::numeric_limits<std::uint64_t>::max();
    const std::uint64_t n = n_minus_1 + 1;
    const std::uint64_t rem = (max64 % n + 1) % n;  // 2^64 mod n
    // Accept r < 2^64 - rem (a whole number of copies of [0, n)).
    std::uint64_t r = 0;
    do {
      r = random_bits<64>(engine);
    } while (rem != 0 && r > max64 - rem);
    return static_cast<I>(static_cast<U>(_a) + static_cast<U>(r % n));
  }

  [[nodiscard]] I a() const noexcept { return _a; }
  [[nodiscard]] I b() const noexcept { return _b; }

private:
  I _a{0};
  I _b{std::numeric_limits<I>::max()};
};

} // namespace rvegen

#undef RVEGEN_PORTABLE_FP_BEGIN
