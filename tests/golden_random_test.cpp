// Golden-value test for toolchain-independent sampling (#76).
//
// A fixed seed must give bit-identical draws on every standard library
// (libstdc++, libc++, MSVC). The values below were produced once and are
// compared with exact ==; they were cross-checked against an
// independent Python implementation of std::mt19937 + the transforms
// documented in rvegen/distributions/portable_random.h.
//
// If this test fails after an intentional change to a transform, rerun
// with `--dump` and replace the tables, and say so in the change log:
// every seeded RVE changes with it.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "rvegen/rvegen.h"

namespace {

int failures = 0;
bool dump = false;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

template <std::size_t N>
void check(char const* name, std::array<double, N> const& expected,
           std::vector<double> const& got) {
  if (dump) {
    std::printf("  // %s\n  {", name);
    for (std::size_t i = 0; i < got.size(); ++i) {
      std::printf("%s%a", i == 0 ? "" : ", ", got[i]);
    }
    std::printf("}\n");
    return;
  }
  REQUIRE(got.size() == N);
  for (std::size_t i = 0; i < N && i < got.size(); ++i) {
    if (!(got[i] == expected[i])) {
      std::fprintf(stderr, "%s[%zu]: expected %a, got %a\n", name, i,
                   expected[i], got[i]);
      ++failures;
    }
  }
}

template <typename F>
std::vector<double> take(std::size_t n, F&& f) {
  std::vector<double> v;
  for (std::size_t i = 0; i < n; ++i) v.push_back(static_cast<double>(f()));
  return v;
}

// ---- expected values --------------------------------------------------

constexpr std::array<double, 6> canonical_mt19937_42{
    0x1.7f87719b2fa8ep-2, 0x1.e6c406b85deb2p-1, 0x1.76c7e8d58f33ap-1,
    0x1.32835d7931965p-1, 0x1.3f86b3339110cp-3, 0x1.3f7a0e90ccbf8p-3};
constexpr std::array<double, 4> canonical_mt19937_64_42{
    0x1.82a3befaddcbcp-1, 0x1.472f1f73724ap-1, 0x1.81192cfe1cbcfp-1,
    0x1.171621fc50d68p-3};
constexpr std::array<double, 6> uniform_real_m1_3{
    0x1.fe1dc66cbea38p-2, 0x1.66c406b85deb2p+1, 0x1.ed8fd1ab1e674p+0,
    0x1.6506baf2632cap+0, -0x1.80f29998ddde8p-2, -0x1.810be2de6681p-2};
constexpr std::array<double, 7> normal_05_2{
    0x1.c9ab6227027acp-3, 0x1.7e514f96b40d3p+0, 0x1.c5e548e487596p+1,
    0x1.cb9dd5d1dccc8p+0, 0x1.03e65fdb97578p-5, 0x1.03a18745cd16p-5,
    0x1.04769a22367eap+1};
constexpr std::array<double, 8> uniform_int_0_9{
    0x1.2p+3, 0x0p+0, 0x1p+0,
    0x1p+2, 0x1.8p+1, 0x1p+1,
    0x1p+2, 0x1p+3};
constexpr std::array<double, 6> cluster_seeds_xy{
    0x1.678daf015bce4p-2, 0x1.dac6ee57c8f5ap-1, 0x1.b71654898d5b9p-2,
    0x1.ceba88058961p-1, 0x1.4fd69e215e73dp-2, 0x1.cf09eaf8b6253p-1};
constexpr std::array<double, 15> rve_circles_xyr{
    0x1.49d2d12e4145ap-3, 0x1.72a79cf54f737p-1, 0x1.736e2f325679ap-5,
    0x1.c3c8d5607e297p-1, 0x1.0fc495b06c14ep-1, 0x1.120b3f1ad1ee1p-4,
    0x1.00757d402387bp-1, 0x1.42d93c5c48b3p-3, 0x1.09da5d0e7bcc2p-5,
    0x1.4969a06e30a93p-1, 0x1.7c6958cbbab63p-1, 0x1.9986ed49d82f7p-5,
    0x1.9e778bc141e5ep-2, 0x1.38d481e450d2ap-3, 0x1.84308f03c5351p-5};

// Samplers that go through portable_math.h (#130): von Mises angles,
// Bingham directions, uniform rotations, lognormal grain sizes.
constexpr std::array<double, 6> von_mises_uniform{
    -0x1.f401b178fb325p-2, 0x1.90e2ce16db428p+1, 0x1.c1f610abedb6dp+0,
    0x1.d6fbc36d8470ap-1, -0x1.dc7e1140aa601p+0, -0x1.dc87ff4ce841dp+0};
constexpr std::array<double, 6> von_mises_k2_5{
    0x1.6d7dadeb913a5p-1, -0x1.07475d98c4e9p-1, -0x1.0a4edcb68629cp-1,
    0x1.532376363da2ap-4, -0x1.7aed7da494adp-6, -0x1.e396e51e8538p-8};
constexpr std::array<double, 6> von_mises_k500{
    0x1.51bfb97a8da6ep-2, 0x1.e8ef329f46521p-3, 0x1.e7e9b317ac84bp-3,
    0x1.2355446acf9bdp-2, 0x1.1b7b57edc0cefp-2, 0x1.1ca8168ca7cdfp-2};
constexpr std::array<double, 9> bingham_uniform{
    -0x1.6a97dd81fe83ep-3, 0x1.45a75f9d453c1p-1, -0x1.808c0fe0eb0c4p-1,
    0x1.9a4f69ec7f50ap-1, 0x1.5cfacbc7f27c4p-2, 0x1.f75c2a99655ap-2,
    -0x1.456d8d36d600bp-2, -0x1.4573645c63532p-2, -0x1.c95f15759b63cp-1};
constexpr std::array<double, 9> bingham_prolate_k5{
    -0x1.aa1efd531949cp-2, 0x1.3627323006618p-5, -0x1.d128db56b1c22p-1,
    0x1.72dfe74316007p-3, 0x1.205070c14aabep-1, 0x1.9cd27b1753d64p-1,
    -0x1.b81c5459c61e8p-3, -0x1.7faa69e23c68cp-1, -0x1.40af3c95f86e2p-1};
constexpr std::array<double, 9> bingham_oblate_km0_5{
    0x1.b833f03503871p-1, -0x1.7876b949b5246p-2, -0x1.6af49dac856a5p-2,
    -0x1.76bab5c61df43p-1, 0x1.5c633f86c461ep-1, 0x1.2a73ad5dce4ccp-5,
    -0x1.152b2a111160cp-4, 0x1.dea7e4d68c506p-1, -0x1.64d4f37625e76p-2};
constexpr std::array<double, 9> bingham_oblate_km40{
    0x1.e80c2aee01f7ap-1, -0x1.feea0928267b9p-3, -0x1.5d9b282e518fap-3,
    0x1.25c7f33a6e495p-1, -0x1.770abf4445212p-1, 0x1.7720b3276b3c4p-2,
    -0x1.6737ae843aa1cp-2, 0x1.a1bbe6ce52c4ep-1, -0x1.d6ac017ee947ap-2};
constexpr std::array<double, 8> uniform_quaternions{
    -0x1.ed972fbd8ca6cp-3, 0x1.81a8f0dbd4f7p-1, -0x1.3756d488ea176p-1,
    -0x1.1aff118792e5bp-4, 0x1.0d6edce848531p-1, 0x1.693168d2ae044p-2,
    0x1.49086ab0c589dp-1, 0x1.b93c2d2b67bb6p-2};
constexpr std::array<double, 6> lognormal_target_diameters{
    0x1.b480f4abb8d65p-2, 0x1.080d1fa135f16p-1, 0x1.6741ca555cff5p-1,
    0x1.144925f4fc276p-1, 0x1.a820bfd528d4p-2, 0x1.a82036f5d0693p-2};

// ---- tests ------------------------------------------------------------

void test_transforms() {
  {
    std::mt19937 e{42};
    check("canonical_mt19937_42", canonical_mt19937_42,
          take(6, [&] { return rvegen::canonical<double>(e); }));
  }
  {
    std::mt19937_64 e{42};
    check("canonical_mt19937_64_42", canonical_mt19937_64_42,
          take(4, [&] { return rvegen::canonical<double>(e); }));
  }
  {
    std::mt19937 e{42};
    rvegen::uniform_real_distribution<double> u{-1.0, 3.0, e};
    check("uniform_real_m1_3", uniform_real_m1_3, take(6, [&] { return u(); }));
  }
  {
    std::mt19937 e{42};
    rvegen::normal_distribution<double> n{0.5, 2.0, e};
    check("normal_05_2", normal_05_2, take(7, [&] { return n(); }));
  }
  {
    std::mt19937 e{42};
    rvegen::portable_uniform_int<std::size_t> u{0, 9};
    check("uniform_int_0_9", uniform_int_0_9, take(8, [&] { return u(e); }));
  }
  {
    std::mt19937 e{42};
    auto seeds = rvegen::cluster_seeds_2d<double>(1.0, 1.0, 1, 3, 0.1, e);
    std::vector<double> xy;
    for (auto const& s : seeds) {
      xy.push_back(s[0]);
      xy.push_back(s[1]);
    }
    check("cluster_seeds_xy", cluster_seeds_xy, xy);
  }
}

void test_samplers() {
  auto angles = [](double kappa) {
    std::mt19937 e{42};
    rvegen::oriented_uniform_distribution<double> d{0.3, kappa, e};
    return take(6, [&] { return d(); });
  };
  check("von_mises_uniform", von_mises_uniform, angles(0.0));
  check("von_mises_k2_5", von_mises_k2_5, angles(2.5));
  check("von_mises_k500", von_mises_k500, angles(500.0));

  auto directions = [](double kappa) {
    std::mt19937 e{42};
    rvegen::bingham_distribution<double> d{std::array<double, 3>{1.0, 2.0, 3.0},
                                           kappa, e};
    std::vector<double> v;
    for (int i = 0; i < 3; ++i) {
      const auto s = d();
      v.insert(v.end(), s.begin(), s.end());
    }
    return v;
  };
  check("bingham_uniform", bingham_uniform, directions(0.0));
  check("bingham_prolate_k5", bingham_prolate_k5, directions(5.0));
  check("bingham_oblate_km0_5", bingham_oblate_km0_5, directions(-0.5));
  check("bingham_oblate_km40", bingham_oblate_km40, directions(-40.0));

  {
    std::mt19937 e{42};
    rvegen::portable_uniform_real<double> u{0.0, 1.0};
    std::vector<double> q;
    for (int i = 0; i < 2; ++i) {
      const double u1 = u(e);
      const double u2 = u(e);
      const double u3 = u(e);
      const auto quat = rvegen::uniform_random_quaternion(u1, u2, u3);
      for (int k = 0; k < 4; ++k) q.push_back(quat[k]);
    }
    check("uniform_quaternions", uniform_quaternions, q);
  }
  {
    // The requested sizes only (they are fixed before the fit starts); no
    // iteration, no restart.
    std::mt19937 e{42};
    const auto r = rvegen::lognormal_grain_seeds_3d<double>(
        1.0, 1.0, 1.0, 0.45, 0.3, e, rvegen::voronoi_backend::in_house, true,
        0.01, 0, 0);
    REQUIRE(r.target_diameters.size() >= 6);
    std::vector<double> d(r.target_diameters.begin(),
                          r.target_diameters.begin() +
                              std::min<std::size_t>(6, r.target_diameters.size()));
    check("lognormal_target_diameters", lognormal_target_diameters, d);
  }
}

void test_portable_log_accuracy() {
  for (double x : {1e-300, 1e-10, 0.1, 0.5, 0.7071, 1.0, 1.5, 2.0, 10.0, 1e10}) {
    const double ref = std::log(x);
    const double got = rvegen::portable_log(x);
    REQUIRE(std::abs(got - ref) <= 4 * std::numeric_limits<double>::epsilon() *
                                        std::max(1.0, std::abs(ref)));
  }
  REQUIRE(rvegen::portable_log(1.0) == 0.0);
}

// Tiny end-to-end RVE: 5 circles, x, y ~ U(0.1, 0.9), r ~ N(0.05, 0.01).
void test_end_to_end_rve() {
  std::mt19937 engine{7};
  rvegen::uniform_real_distribution<double> pos_x{0.1, 0.9, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.1, 0.9, engine};
  rvegen::normal_distribution<double> radius{0.05, 0.01, engine};

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(
      std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));
  rvegen::number_of_inclusions<double> termination{5};
  rvegen::only_inside_generator<double> generator;
  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0});

  std::vector<double> xyr;
  for (auto const& s : shapes) {
    auto const& c = dynamic_cast<rvegen::circle<double> const&>(*s);
    xyr.push_back(c.center[0]);
    xyr.push_back(c.center[1]);
    xyr.push_back(c.radius);
  }
  check("rve_circles_xyr", rve_circles_xyr, xyr);
}

} // namespace

int main(int argc, char** argv) {
  dump = argc > 1 && std::strcmp(argv[1], "--dump") == 0;
  test_transforms();
  test_samplers();
  test_portable_log_accuracy();
  test_end_to_end_rve();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  if (!dump) std::cout << "golden_random_test: all passed\n";
  return 0;
}
