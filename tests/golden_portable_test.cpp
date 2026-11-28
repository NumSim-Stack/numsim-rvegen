// Golden test for the platform-independent random transforms and
// elementary functions (#130).
//
// This file depends on portable_random.h and portable_math.h only (no
// other rvegen header, no third-party library), so it builds wherever a
// C++20 compiler does. CI runs it against libstdc++, libc++ and the MSVC
// standard library; the hashes below must be the same on all of them.
//
// Each hash is FNV-1a over the bit patterns of a long stream of results
// (the arguments come from a seeded engine through exact operations), so
// one different bit anywhere in the stream changes it.
//
// If this test fails after an intentional change to a transform, rerun
// with `--dump` and replace the table, and say so in the change log:
// every seeded RVE changes with it.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numbers>
#include <random>
#include <string>

#include "rvegen/distributions/portable_math.h"
#include "rvegen/distributions/portable_random.h"

namespace {

int failures = 0;
bool dump = false;

#define REQUIRE(cond)                                                    \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::fprintf(stderr, "REQUIRE failed: %s at %s:%d\n", #cond,       \
                   __FILE__, __LINE__);                                  \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

struct fnv {
  std::uint64_t h = 1469598103934665603ull;
  void add(std::uint64_t bits) {
    for (int i = 0; i < 8; ++i) {
      h ^= (bits >> (8 * i)) & 0xffu;
      h *= 1099511628211ull;
    }
  }
  void add(double v) { add(std::bit_cast<std::uint64_t>(v)); }
};

void check(char const* name, std::uint64_t expected, std::uint64_t got) {
  if (dump) {
    std::printf("  {\"%s\", 0x%016llxull},\n", name,
                static_cast<unsigned long long>(got));
    return;
  }
  if (got != expected) {
    std::fprintf(stderr, "%s: expected 0x%016llx, got 0x%016llx\n", name,
                 static_cast<unsigned long long>(expected),
                 static_cast<unsigned long long>(got));
    ++failures;
  }
}

using rvegen::canonical;
using rvegen::portable_detail::rounded;

// Arguments: each generator draws in separate statements (the order of
// evaluation inside one expression differs between compilers) and uses
// exact operations or rounded products only.
using engine_t = std::mt19937_64;

// Uniform on [a, b).
struct uniform_arg {
  double a, b;
  double operator()(engine_t& e) const {
    const double c = canonical<double>(e);
    return a + rounded((b - a) * c);
  }
};
// +-(1 + c) * 2^k, k uniform in [lo, hi): every magnitude.
struct wide_arg {
  int lo, hi;
  bool both_signs;
  double operator()(engine_t& e) const {
    const double c0 = canonical<double>(e);
    const double c1 = canonical<double>(e);
    const double c2 = canonical<double>(e);
    const int k = lo + static_cast<int>(rounded(static_cast<double>(hi - lo) * c2));
    const double m = std::ldexp(1.0 + c1, k);
    return (both_signs && c0 < 0.5) ? -m : m;
  }
};
// 1 - (1 + c) * 2^-k, k in [1, 50]: towards 1 from below.
struct near_one_arg {
  double operator()(engine_t& e) const {
    const double c1 = canonical<double>(e);
    const double c2 = canonical<double>(e);
    const int k = 1 + static_cast<int>(rounded(50.0 * c2));
    return 1.0 - std::ldexp(1.0 + c1, -k - 1);
  }
};

constexpr std::size_t stream = 200000;

// Hash of f over `stream` arguments; also the largest error against the
// platform function `ref`, in units of the last place of the reference.
template <typename F, typename R, typename A>
std::uint64_t run(F f, R ref, A arg, double max_ulp, char const* name) {
  engine_t e{20260130};
  fnv h;
  double worst = 0.0;
  for (std::size_t i = 0; i < stream; ++i) {
    const double x = arg(e);
    const double got = f(x);
    h.add(got);
    const double want = ref(x);
    if (got != want) {
      const double ulp =
          std::nextafter(std::abs(want), std::numeric_limits<double>::infinity()) -
          std::abs(want);
      worst = std::max(worst, std::abs(got - want) / ulp);
    }
  }
  if (!(worst <= max_ulp)) {
    std::fprintf(stderr, "%s: %.1f ulp from the platform function (limit %.1f)\n",
                 name, worst, max_ulp);
    ++failures;
  }
  return h.h;
}

struct golden {
  char const* name;
  std::uint64_t hash;
};

// ---- expected hashes (identical on every platform) ---------------------
constexpr golden expected[] = {
  {"canonical_mt19937", 0xe147b75b02acee32ull},
  {"canonical_mt19937_64", 0x1ecd361dcda38ed5ull},
  {"uniform_real", 0x2d43120904cd5ab6ull},
  {"normal", 0x28a41a746fd6d636ull},
  {"uniform_int", 0x5dc101352a92a696ull},
  {"log", 0x6a1d0821e24f9d81ull},
  {"exp", 0xbdf49f9b00fb811cull},
  {"exp_small", 0xc1887f97a39cd1daull},
  {"expm1", 0x7b5eabfd93b565c5ull},
  {"log1p", 0xda3544092e07c9c8ull},
  {"log1p_negative", 0xc8ed9b1367a36ba6ull},
  {"sin", 0x3478b64525ac092bull},
  {"cos", 0x8805faea509d5e93ull},
  {"sin_far", 0x810ff9a5078d46cfull},
  {"cos_far", 0x9e3f4a8df3516ef1ull},
  {"acos", 0x91d70121217a9a2eull},
  {"acos_near_one", 0xb8fd427a5bb467b8ull},
  {"cbrt", 0xafc5df40020756fbull},
  {"edges", 0xc3dc7973ad7bec3cull},
};

std::uint64_t expected_hash(char const* name) {
  for (auto const& g : expected)
    if (std::strcmp(g.name, name) == 0) return g.hash;
  std::fprintf(stderr, "no golden named %s\n", name);
  ++failures;
  return 0;
}

void test_transforms() {
  {
    std::mt19937 e{42};
    fnv h;
    for (std::size_t i = 0; i < stream; ++i) h.add(canonical<double>(e));
    check("canonical_mt19937", expected_hash("canonical_mt19937"), h.h);
  }
  {
    std::mt19937_64 e{42};
    fnv h;
    for (std::size_t i = 0; i < stream; ++i) h.add(canonical<double>(e));
    check("canonical_mt19937_64", expected_hash("canonical_mt19937_64"), h.h);
  }
  {
    std::mt19937 e{42};
    rvegen::portable_uniform_real<double> u{-1.0, 3.0};
    fnv h;
    for (std::size_t i = 0; i < stream; ++i) h.add(u(e));
    check("uniform_real", expected_hash("uniform_real"), h.h);
  }
  {
    std::mt19937 e{42};
    rvegen::portable_normal<double> n{0.5, 2.0};
    fnv h;
    for (std::size_t i = 0; i < stream; ++i) h.add(n(e));
    check("normal", expected_hash("normal"), h.h);
  }
  {
    std::mt19937 e{42};
    rvegen::portable_uniform_int<std::uint64_t> u{3, 1000003};
    fnv h;
    for (std::size_t i = 0; i < stream; ++i) h.add(u(e));
    check("uniform_int", expected_hash("uniform_int"), h.h);
  }
}

void test_math() {
  using namespace rvegen;
  auto c = [&](char const* name, std::uint64_t got) {
    check(name, expected_hash(name), got);
  };
  // The limits are a few times the measured error (1 to 6 ulp against
  // glibc); the platform functions themselves differ by about 1 ulp.
  c("log", run([](double x) { return portable_log(x); },
               [](double x) { return std::log(x); },
               wide_arg{-1000, 1000, false}, 8.0, "log"));
  c("exp", run([](double x) { return portable_exp(x); },
               [](double x) { return std::exp(x); },
               uniform_arg{-700.0, 700.0}, 8.0, "exp"));
  c("exp_small", run([](double x) { return portable_exp(x); },
                     [](double x) { return std::exp(x); },
                     uniform_arg{-2.0, 2.0}, 8.0, "exp_small"));
  c("expm1", run([](double x) { return portable_expm1(x); },
                 [](double x) { return std::expm1(x); },
                 wide_arg{-50, 4, true}, 12.0, "expm1"));
  c("log1p", run([](double x) { return portable_log1p(x); },
                 [](double x) { return std::log1p(x); },
                 wide_arg{-50, 4, false}, 12.0, "log1p"));
  c("log1p_negative", run([](double x) { return portable_log1p(x); },
                          [](double x) { return std::log1p(x); },
                          uniform_arg{-0.999, 0.0}, 12.0, "log1p_negative"));
  c("sin", run([](double x) { return portable_sin(x); },
               [](double x) { return std::sin(x); },
               uniform_arg{-10.0, 10.0}, 8.0, "sin"));
  c("cos", run([](double x) { return portable_cos(x); },
               [](double x) { return std::cos(x); },
               uniform_arg{-10.0, 10.0}, 8.0, "cos"));
  c("sin_far", run([](double x) { return portable_sin(x); },
                   [](double x) { return std::sin(x); },
                   uniform_arg{-5e5, 5e5}, 16.0, "sin_far"));
  c("cos_far", run([](double x) { return portable_cos(x); },
                   [](double x) { return std::cos(x); },
                   uniform_arg{-5e5, 5e5}, 16.0, "cos_far"));
  c("acos", run([](double x) { return portable_acos(x); },
                [](double x) { return std::acos(x); },
                uniform_arg{-1.0, 1.0}, 16.0, "acos"));
  c("acos_near_one", run([](double x) { return portable_acos(x); },
                         [](double x) { return std::acos(x); },
                         near_one_arg{}, 16.0, "acos_near_one"));
  c("cbrt", run([](double x) { return portable_cbrt(x); },
                [](double x) { return std::cbrt(x); },
                wide_arg{-1000, 1000, true}, 12.0, "cbrt"));

  // Special values and the far-argument path of sin / cos.
  fnv h;
  const double inf = std::numeric_limits<double>::infinity();
  for (double x : {0.0, -0.0, 1.0, -1.0, 0.5, 1e-300, 5e-324, 1e7, -3e9,
                   1e15, 0x1p19, 0x1p19 + 1.0, 709.0, -745.0}) {
    h.add(portable_sin(x));
    h.add(portable_cos(x));
    h.add(portable_exp(x));
    h.add(portable_expm1(x));
    h.add(portable_cbrt(x));
    if (x >= -1.0 && x <= 1.0) h.add(portable_acos(x));
    if (x > -1.0) h.add(portable_log1p(x));
  }
  c("edges", h.h);

  REQUIRE(portable_exp(0.0) == 1.0);
  REQUIRE(portable_exp(710.0) == inf && portable_exp(-746.0) == 0.0);
  REQUIRE(portable_expm1(0.0) == 0.0 && portable_log1p(0.0) == 0.0);
  REQUIRE(portable_sin(0.0) == 0.0 && portable_cos(0.0) == 1.0);
  REQUIRE(portable_acos(1.0) == 0.0);
  REQUIRE(portable_acos(-1.0) == std::numbers::pi);
  REQUIRE(portable_acos(2.0) == 0.0 && portable_acos(-2.0) == std::numbers::pi);
  REQUIRE(portable_cbrt(27.0) == 3.0 && portable_cbrt(-8.0) == -2.0);
  REQUIRE(portable_cbrt(0.0) == 0.0 && portable_cbrt(inf) == inf);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (double v : {portable_exp(nan), portable_sin(nan), portable_cos(inf),
                   portable_acos(nan), portable_cbrt(nan),
                   portable_acos(inf), portable_acos(-inf),
                   portable_log1p(-2.0), portable_log1p(nan)})
    REQUIRE(v != v);
  REQUIRE(portable_log1p(-1.0) == -inf && portable_log1p(inf) == inf);
  REQUIRE(std::signbit(portable_sin(-0.0)) && !std::signbit(portable_sin(0.0)));
  // Near multiples of pi/2 the three-part reduction keeps the result
  // accurate in absolute terms.
  double worst = 0.0;
  for (int k = -2000; k <= 2000; ++k) {
    const double x = rounded(k * (std::numbers::pi / 2));
    worst = std::max({worst, std::abs(portable_sin(x) - std::sin(x)),
                      std::abs(portable_cos(x) - std::cos(x))});
  }
  REQUIRE(worst < 1e-18);
  // float arguments: computed in double, rounded once.
  REQUIRE(portable_exp(1.0f) == static_cast<float>(portable_exp(1.0)));
  REQUIRE(portable_cos(1.0f) == static_cast<float>(portable_cos(1.0)));
}

} // namespace

int main(int argc, char** argv) {
  dump = argc > 1 && std::strcmp(argv[1], "--dump") == 0;
  test_transforms();
  test_math();
  if (failures != 0) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  if (!dump) std::printf("golden_portable_test: all passed\n");
  return 0;
}
