// Input validation and draw-order tests (#75, #76 part 1).
//
// 1. Every input's new_shape() assigns successive draws to the
//    documented fields in a fixed order (x, y, [z], sizes, rotation),
//    independent of the compiler's argument-evaluation order.
// 2. Size parameters are truncated to (0, inf): non-positive draws are
//    resampled (tube lengths included); size distributions with
//    P(x > 0) < 1e-3 are rejected at input construction; distributions
//    that can't report P(x > 0) hit the redraw cap as a backstop.
//    cluster_seeds_2d validates its arguments.
// 3. parameter_visitor_nlohmann rejects non-integer / negative /
//    out-of-range JSON values for integral targets.
// 4. uniform_real and normal distributions validate their parameters.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"
#include "rvegen/inputs/polyline_tube_input.h"

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

bool near(double a, double b, double tol = 1e-12) {
  return std::abs(a - b) <= tol;
}

// Returns 1, 2, 3, ... on successive calls. Passed for every distribution
// slot of an input, so the value a field receives is its draw index.
class counter_distribution final : public rvegen::distribution_base<double> {
public:
  double operator()() override { return ++_n; }
  [[nodiscard]] int calls() const noexcept { return static_cast<int>(_n); }

private:
  double _n{0};
};

// Cycles through a fixed list of values.
class sequence_distribution final : public rvegen::distribution_base<double> {
public:
  explicit sequence_distribution(std::vector<double> values)
      : _values{std::move(values)} {}
  double operator()() override {
    ++_calls;
    return _values[_i++ % _values.size()];
  }
  [[nodiscard]] std::size_t calls() const noexcept { return _calls; }

private:
  std::vector<double> _values;
  std::size_t _i{0};
  std::size_t _calls{0};
};

class fixed_direction final : public rvegen::direction_distribution_base<double> {
public:
  std::array<double, 3> operator()() override { return {1.0, 0.0, 0.0}; }
};

template <typename Shape>
Shape const& as(std::unique_ptr<rvegen::shape_base<double>> const& p) {
  auto const* s = dynamic_cast<Shape const*>(p.get());
  if (s == nullptr) throw std::logic_error{"unexpected shape type"};
  return *s;
}

template <typename E, typename F>
bool throws_containing(F&& f, std::string const& needle) {
  try {
    f();
  } catch (E const& e) {
    if (std::string{e.what()}.find(needle) != std::string::npos) return true;
    std::cerr << "  runtime_error without '" << needle << "': " << e.what()
              << "\n";
    return false;
  } catch (...) {
    std::cerr << "  wrong exception type\n";
    return false;
  }
  return false;
}

template <typename F>
bool throws_runtime_error_containing(F&& f, std::string const& needle) {
  return throws_containing<std::runtime_error>(std::forward<F>(f), needle);
}

template <typename F>
bool throws_invalid_argument_containing(F&& f, std::string const& needle) {
  return throws_containing<std::invalid_argument>(std::forward<F>(f), needle);
}

// ---- 1. Draw order -------------------------------------------------------

void test_draw_order() {
  {
    counter_distribution c;
    rvegen::circle_input<double> in{c, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::circle<double>>(s);
    REQUIRE(sh.center[0] == 1.0);
    REQUIRE(sh.center[1] == 2.0);
    REQUIRE(sh.radius == 3.0);
    REQUIRE(c.calls() == 3);
  }
  {
    counter_distribution c;
    rvegen::sphere_input<double> in{c, c, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::sphere<double>>(s);
    REQUIRE(sh.center[0] == 1.0);
    REQUIRE(sh.center[1] == 2.0);
    REQUIRE(sh.center[2] == 3.0);
    REQUIRE(sh.radius == 4.0);
    REQUIRE(c.calls() == 4);
  }
  {
    counter_distribution c;
    rvegen::rectangle_input<double> in{c, c, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::rectangle<double>>(s);
    auto const m = sh.get_middle_point();
    REQUIRE(near(m[0], 1.0));
    REQUIRE(near(m[1], 2.0));
    REQUIRE(near(sh.width(), 3.0));
    REQUIRE(near(sh.height(), 4.0));
    REQUIRE(c.calls() == 4);
  }
  {
    counter_distribution c;
    rvegen::box_input<double> in{c, c, c, c, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::box<double>>(s);
    auto const m = sh.get_middle_point();
    REQUIRE(near(m[0], 1.0));
    REQUIRE(near(m[1], 2.0));
    REQUIRE(near(m[2], 3.0));
    REQUIRE(near(sh.width(), 4.0));
    REQUIRE(near(sh.height(), 5.0));
    REQUIRE(near(sh.depth(), 6.0));
    REQUIRE(c.calls() == 6);
  }
  {
    counter_distribution c;
    rvegen::ellipse_input<double> in{c, c, c, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::ellipse<double>>(s);
    REQUIRE(sh.center[0] == 1.0);
    REQUIRE(sh.center[1] == 2.0);
    REQUIRE(sh.radius_a() == 3.0);
    REQUIRE(sh.radius_b() == 4.0);
    // rotation() is atan2-wrapped to (-pi, pi]; compare modulo 2*pi.
    REQUIRE(near(std::remainder(sh.rotation() - 5.0, 2 * std::numbers::pi), 0.0,
                 1e-9));
    REQUIRE(c.calls() == 5);
  }
  {
    counter_distribution c;
    rvegen::polyline_tube_input<double> in{c, c, c, c, c, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::polyline_tube<double>>(s);
    auto const& cl = sh.centerline();
    REQUIRE(cl.size() == 2);
    REQUIRE(cl[0][0] == 1.0 && cl[0][1] == 2.0 && cl[0][2] == 3.0);
    REQUIRE(cl[1][0] == 4.0 && cl[1][1] == 5.0 && cl[1][2] == 6.0);
    REQUIRE(sh.radius() == 7.0);
    REQUIRE(c.calls() == 7);
  }
  {
    // position 1,2,3; direction (4,5,6) normalised; length 7; radius 8.
    counter_distribution c;
    rvegen::polyline_tube_directional_input<double> in{c, c, c, c, c, c, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::polyline_tube<double>>(s);
    auto const& cl = sh.centerline();
    const double mag = std::sqrt(16.0 + 25.0 + 36.0);
    const std::array<double, 3> d{4.0 / mag, 5.0 / mag, 6.0 / mag};
    const std::array<double, 3> ctr{1.0, 2.0, 3.0};
    REQUIRE(cl.size() == 2);
    for (int k = 0; k < 3; ++k) {
      REQUIRE(near(cl[0][k], ctr[k] - 3.5 * d[k]));
      REQUIRE(near(cl[1][k], ctr[k] + 3.5 * d[k]));
    }
    REQUIRE(sh.radius() == 8.0);
    REQUIRE(c.calls() == 8);
  }
  {
    // position 1,2,3; direction fixed +x; length 4; radius 5.
    counter_distribution c;
    fixed_direction dir;
    rvegen::polyline_tube_oriented_input<double> in{c, c, c, dir, c, c};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::polyline_tube<double>>(s);
    auto const& cl = sh.centerline();
    REQUIRE(cl.size() == 2);
    REQUIRE(near(cl[0][0], -1.0) && near(cl[0][1], 2.0) && near(cl[0][2], 3.0));
    REQUIRE(near(cl[1][0], 3.0) && near(cl[1][1], 2.0) && near(cl[1][2], 3.0));
    REQUIRE(sh.radius() == 5.0);
    REQUIRE(c.calls() == 5);
  }
}

// ---- 2. Size truncation --------------------------------------------------

void test_negative_sizes_are_resampled() {
  rvegen::constant_distribution<double> pos{0.5};
  {
    sequence_distribution r{{-1.0, 0.5}};
    rvegen::circle_input<double> in{pos, pos, r};
    for (int i = 0; i < 20; ++i) {
      auto s = in.new_shape();
      REQUIRE(as<rvegen::circle<double>>(s).radius == 0.5);
    }
    REQUIRE(r.calls() == 40);
  }
  {
    sequence_distribution r{{0.0, -2.0, 0.25}};
    rvegen::sphere_input<double> in{pos, pos, pos, r};
    auto s = in.new_shape();
    REQUIRE(as<rvegen::sphere<double>>(s).radius == 0.25);
  }
  {
    sequence_distribution w{{-1.0, 0.5}};
    rvegen::rectangle_input<double> in{pos, pos, w, w};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::rectangle<double>>(s);
    REQUIRE(near(sh.width(), 0.5) && near(sh.height(), 0.5));
  }
  {
    sequence_distribution w{{-1.0, 0.5}};
    rvegen::box_input<double> in{pos, pos, pos, w, w, w};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::box<double>>(s);
    REQUIRE(near(sh.width(), 0.5) && near(sh.height(), 0.5) &&
            near(sh.depth(), 0.5));
  }
  {
    sequence_distribution a{{-1.0, std::numeric_limits<double>::quiet_NaN(), 0.5}};
    rvegen::ellipse_input<double> in{pos, pos, a, a, pos};
    auto s = in.new_shape();
    auto const& sh = as<rvegen::ellipse<double>>(s);
    REQUIRE(sh.radius_a() == 0.5 && sh.radius_b() == 0.5);
  }
  {
    sequence_distribution r{{-1.0, 0.5}};
    rvegen::polyline_tube_input<double> in{pos, pos, pos, pos, pos, pos, r};
    auto s = in.new_shape();
    REQUIRE(as<rvegen::polyline_tube<double>>(s).radius() == 0.5);
  }
  {
    sequence_distribution r{{-1.0, 0.5}};
    rvegen::constant_distribution<double> one{1.0};
    rvegen::polyline_tube_directional_input<double> in{pos, pos, pos, one, one,
                                                       one, one, r};
    auto s = in.new_shape();
    REQUIRE(as<rvegen::polyline_tube<double>>(s).radius() == 0.5);
  }
  {
    sequence_distribution r{{-1.0, 0.5}};
    rvegen::constant_distribution<double> one{1.0};
    fixed_direction dir;
    rvegen::polyline_tube_oriented_input<double> in{pos, pos, pos, dir, one, r};
    auto s = in.new_shape();
    REQUIRE(as<rvegen::polyline_tube<double>>(s).radius() == 0.5);
  }
  {
    // The #75 repro: r ~ N(0, 0.1) — half of the raw draws are negative.
    std::mt19937 engine{1234};
    rvegen::normal_distribution<double> r{0.0, 0.1, engine};
    rvegen::circle_input<double> in{pos, pos, r};
    for (int i = 0; i < 2000; ++i) {
      auto s = in.new_shape();
      REQUIRE(as<rvegen::circle<double>>(s).radius > 0.0);
    }
  }
}

void test_positive_probability() {
  std::mt19937 engine{1};
  using U = rvegen::uniform_real_distribution<double>;
  using N = rvegen::normal_distribution<double>;
  using C = rvegen::constant_distribution<double>;

  REQUIRE(*C{2.0}.positive_probability() == 1.0);
  REQUIRE(*C{0.0}.positive_probability() == 0.0);
  REQUIRE(*C{-1.0}.positive_probability() == 0.0);
  REQUIRE(*U(0.0, 1.0, engine).positive_probability() == 1.0);
  REQUIRE(*U(-1.0, 0.0, engine).positive_probability() == 0.0);
  REQUIRE(near(*U(-1.0, 3.0, engine).positive_probability(), 0.75));
  REQUIRE(*U(0.0, 0.0, engine).positive_probability() == 0.0);
  REQUIRE(*U(0.5, 0.5, engine).positive_probability() == 1.0);
  REQUIRE(near(*N(0.0, 1.0, engine).positive_probability(), 0.5));
  REQUIRE(near(*N(-3.0, 1.0, engine).positive_probability(), 0.0013498980316301,
               1e-12));
  // Distributions that can't tell report nullopt.
  sequence_distribution seq{{1.0}};
  REQUIRE(!seq.positive_probability().has_value());
}

// Size distributions with (almost) no mass on (0, inf) are rejected when
// the input is constructed, not mid-run.
void test_hopeless_size_distribution_rejected_at_construction() {
  std::mt19937 engine{1};
  rvegen::constant_distribution<double> pos{0.5};
  rvegen::constant_distribution<double> one{1.0};
  rvegen::constant_distribution<double> zero{0.0};
  rvegen::constant_distribution<double> neg{-1.0};
  rvegen::uniform_real_distribution<double> u_neg{-2.0, 0.0, engine};
  rvegen::uniform_real_distribution<double> u_point_neg{-1.0, -1.0, engine};
  rvegen::uniform_real_distribution<double> u_point_zero{0.0, 0.0, engine};
  rvegen::normal_distribution<double> n_far{-3.5, 1.0, engine};  // P(>0)=2.3e-4
  fixed_direction dir;
  using circ = rvegen::circle_input<double>;

  REQUIRE(throws_invalid_argument_containing([&] { circ{pos, pos, zero}; },
                                             "circle_input"));
  REQUIRE(throws_invalid_argument_containing([&] { circ{pos, pos, neg}; },
                                             "'radius'"));
  REQUIRE(throws_invalid_argument_containing([&] { circ{pos, pos, u_neg}; },
                                             "'radius'"));
  REQUIRE(throws_invalid_argument_containing([&] { circ{pos, pos, u_point_neg}; },
                                             "'radius'"));
  REQUIRE(throws_invalid_argument_containing([&] { circ{pos, pos, u_point_zero}; },
                                             "'radius'"));
  REQUIRE(throws_invalid_argument_containing([&] { circ{pos, pos, n_far}; },
                                             "'radius'"));
  REQUIRE(throws_invalid_argument_containing(
      [&] { rvegen::sphere_input<double>{pos, pos, pos, neg}; }, "sphere_input"));
  REQUIRE(throws_invalid_argument_containing(
      [&] { rvegen::rectangle_input<double>{pos, pos, one, neg}; }, "'height'"));
  REQUIRE(throws_invalid_argument_containing(
      [&] { rvegen::box_input<double>{pos, pos, pos, one, one, neg}; }, "'depth'"));
  REQUIRE(throws_invalid_argument_containing(
      [&] { rvegen::ellipse_input<double>{pos, pos, one, neg, pos}; }, "'radius_b'"));
  REQUIRE(throws_invalid_argument_containing(
      [&] {
        rvegen::polyline_tube_input<double>{pos, pos, pos, pos, pos, pos, neg};
      },
      "polyline_tube_input"));
  REQUIRE(throws_invalid_argument_containing(
      [&] {
        rvegen::polyline_tube_directional_input<double>{pos, pos, pos, one, one,
                                                        one, neg, one};
      },
      "'length'"));
  REQUIRE(throws_invalid_argument_containing(
      [&] {
        rvegen::polyline_tube_oriented_input<double>{pos, pos, pos, dir, zero, one};
      },
      "'length'"));
  REQUIRE(throws_invalid_argument_containing(
      [&] {
        rvegen::polyline_tube_oriented_input<double>{pos, pos, pos, dir, one, neg};
      },
      "'radius'"));

  // Mostly-negative but viable distributions are still accepted, and
  // don't fail mid-run: N(-3, 1) has P(>0) = 0.00135 (with the old
  // 1000-draw cap, ~26% of new_shape() calls threw).
  rvegen::normal_distribution<double> n_viable{-3.0, 1.0, engine};
  circ in{pos, pos, n_viable};
  for (int i = 0; i < 500; ++i) {
    auto s = in.new_shape();
    REQUIRE(as<rvegen::circle<double>>(s).radius > 0.0);
  }
  rvegen::uniform_real_distribution<double> u_viable{-1.0, 0.5, engine};
  circ in2{pos, pos, u_viable};
  for (int i = 0; i < 100; ++i) {
    auto s = in2.new_shape();
    REQUIRE(as<rvegen::circle<double>>(s).radius > 0.0);
  }
}

// A distribution that cannot report P(x > 0) bypasses the constructor
// check; the redraw cap is the backstop.
void test_redraw_cap_backstop() {
  rvegen::constant_distribution<double> pos{0.5};
  constexpr std::size_t cap = rvegen::shape_input_base<double>::max_size_draws;
  {
    sequence_distribution r{{-1.0}};
    rvegen::circle_input<double> in{pos, pos, r};
    REQUIRE(throws_runtime_error_containing([&] { (void)in.new_shape(); },
                                            "circle_input"));
    REQUIRE(throws_runtime_error_containing([&] { (void)in.new_shape(); },
                                            "'radius'"));
    REQUIRE(r.calls() == 2 * cap);
  }
  {
    rvegen::constant_distribution<double> one{1.0};
    sequence_distribution len{{0.0}};
    fixed_direction dir;
    rvegen::polyline_tube_oriented_input<double> in{pos, pos, pos, dir, len, one};
    REQUIRE(throws_runtime_error_containing([&] { (void)in.new_shape(); },
                                            "'length'"));
  }
}

// Tube lengths are truncated to (0, inf) like the radius.
void test_tube_length_resampled() {
  rvegen::constant_distribution<double> zero{0.0};
  rvegen::constant_distribution<double> one{1.0};
  {
    sequence_distribution len{{-1.0, 0.0, 2.0}};
    fixed_direction dir;
    rvegen::polyline_tube_oriented_input<double> in{zero, zero, zero, dir, len, one};
    auto s = in.new_shape();
    auto const& cl = as<rvegen::polyline_tube<double>>(s).centerline();
    REQUIRE(near(cl[0][0], -1.0) && near(cl[1][0], 1.0));
    REQUIRE(len.calls() == 3);
  }
  {
    sequence_distribution len{{-4.0, 2.0}};
    rvegen::polyline_tube_directional_input<double> in{zero, zero, zero, one, zero,
                                                       zero, len, one};
    auto s = in.new_shape();
    auto const& cl = as<rvegen::polyline_tube<double>>(s).centerline();
    REQUIRE(near(cl[0][0], -1.0) && near(cl[1][0], 1.0));
  }
}

void test_cluster_seeds_validation() {
  std::mt19937 e{1};
  constexpr double inf = std::numeric_limits<double>::infinity();
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  auto call = [&](double lx, double ly, double spread) {
    return [&e, lx, ly, spread] {
      (void)rvegen::cluster_seeds_2d<double>(lx, ly, 2, 3, spread, e);
    };
  };
  REQUIRE(throws_invalid_argument_containing(call(nan, 1.0, 0.1), "Lx and Ly"));
  REQUIRE(throws_invalid_argument_containing(call(1.0, inf, 0.1), "Lx and Ly"));
  REQUIRE(throws_invalid_argument_containing(call(-1.0, 1.0, 0.1), "Lx and Ly"));
  REQUIRE(throws_invalid_argument_containing(call(1.0, 1.0, 0.0), "cluster_spread"));
  REQUIRE(throws_invalid_argument_containing(call(1.0, 1.0, -0.1), "cluster_spread"));
  REQUIRE(throws_invalid_argument_containing(call(1.0, 1.0, nan), "cluster_spread"));
  REQUIRE(throws_invalid_argument_containing(call(1.0, 1.0, inf), "cluster_spread"));
  REQUIRE(throws_invalid_argument_containing(
      [&] { (void)rvegen::cluster_seeds_2d<double>(1.0, 1.0, 2, 3, 0.1, e, 0); },
      "max_attempts_per_seed"));
  REQUIRE(rvegen::cluster_seeds_2d<double>(1.0, 1.0, 2, 3, 0.1, e).size() == 6);
}

// ---- 3. JSON integral conversion ----------------------------------------

void test_json_integral_strictness() {
  auto read = [](char const* json, std::type_info const& tid) {
    const auto j = nlohmann::json::parse(json);
    rvegen::parameter_visitor_nlohmann v{j};
    return v.read("max_attempts", tid);
  };
  auto rejects = [&](char const* json, std::type_info const& tid) {
    return throws_runtime_error_containing([&] { (void)read(json, tid); },
                                           "'max_attempts'");
  };

  REQUIRE(std::any_cast<std::size_t>(read(R"({"max_attempts": 5})",
                                          typeid(std::size_t))) == 5);
  REQUIRE(rejects(R"({"max_attempts": -1})", typeid(std::size_t)));
  REQUIRE(rejects(R"({"max_attempts": 5.9})", typeid(std::size_t)));
  REQUIRE(rejects(R"({"max_attempts": "5"})", typeid(std::size_t)));
  REQUIRE(rejects(R"({"max_attempts": true})", typeid(std::size_t)));
  REQUIRE(rejects(R"({"max_attempts": 5.0})", typeid(std::size_t)));

  // Signed targets keep negatives, but still range-check.
  REQUIRE(std::any_cast<int>(read(R"({"max_attempts": -1})", typeid(int))) == -1);
  REQUIRE(rejects(R"({"max_attempts": 3000000000})", typeid(int)));
  REQUIRE(rejects(R"({"max_attempts": 4294967296})", typeid(std::uint32_t)));
  REQUIRE(rejects(R"({"max_attempts": 18446744073709551615})",
                  typeid(std::int64_t)));
  REQUIRE(std::any_cast<std::uint64_t>(
              read(R"({"max_attempts": 18446744073709551615})",
                   typeid(std::uint64_t))) ==
          std::numeric_limits<std::uint64_t>::max());

  // Vector<int> elements go through the same check.
  REQUIRE((std::any_cast<std::vector<int>>(
               read(R"({"max_attempts": [1, -2]})", typeid(std::vector<int>))) ==
           std::vector<int>{1, -2}));
  REQUIRE(rejects(R"({"max_attempts": [1, 2.5]})", typeid(std::vector<int>)));

  // Doubles are unaffected: integer JSON still reads as a double.
  REQUIRE(std::any_cast<double>(read(R"({"max_attempts": 5})", typeid(double))) ==
          5.0);
}

// ---- 5. `metadata` field (#72) -----------------------------------------

// Builds a circle_input through the registry with the given `metadata`
// value (or none when `metadata` is nullptr) and returns the produced
// shape's info blob.
nlohmann::json circle_info_with_metadata(char const* metadata) {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{72};
  rvegen::distribution_map_t<double> dists;
  for (char const* name : {"x", "y", "r"}) {
    nlohmann::json spec{{"type", "constant"}, {"value", 0.25}};
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  nlohmann::json input_spec{{"type", "circle_input"},
                            {"pos_x_dist", "x"},
                            {"pos_y_dist", "y"},
                            {"radius_dist", "r"},
                            {"phase_name", "fibre"}};
  if (metadata != nullptr) input_spec["metadata"] = metadata;
  auto input = rvegen::build_from_json(rvegen::input_registry_t<>::instance(),
                                       input_spec, dists);
  auto shape = input->new_shape();
  return shape->info().json();
}

void test_metadata_field() {
  const auto baseline = circle_info_with_metadata(nullptr);
  REQUIRE(baseline.value("phase_name", std::string{}) == "fibre");

  // Empty / whitespace-only is the same as absent.
  for (char const* blank : {"", " ", "  \t\n\r  "}) {
    bool ok = false;
    try {
      ok = circle_info_with_metadata(blank) == baseline;
    } catch (std::exception const& e) {
      std::cerr << "  metadata '" << blank << "' threw: " << e.what() << "\n";
    }
    REQUIRE(ok);
  }

  // A valid object still merges.
  const auto merged = circle_info_with_metadata(R"( {"source": "a.stl"} )");
  REQUIRE(merged.value("source", std::string{}) == "a.stl");
  REQUIRE(merged.value("phase_name", std::string{}) == "fibre");

  // Malformed or non-object JSON throws and names the key.
  for (char const* bad : {"{", "not json", "{\"a\": }", "[1, 2]", "5", "\"x\""}) {
    REQUIRE(throws_runtime_error_containing(
        [&] { (void)circle_info_with_metadata(bad); }, "'metadata'"));
  }
}

// ---- 4. Distribution parameters -----------------------------------------

void test_distribution_parameters() {
  std::mt19937 engine{1};
  constexpr double inf = std::numeric_limits<double>::infinity();
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  constexpr double big = std::numeric_limits<double>::max();
  using U = rvegen::uniform_real_distribution<double>;
  using N = rvegen::normal_distribution<double>;

  REQUIRE(throws_runtime_error_containing([&] { U{1.0, 0.0, engine}; },
                                          "uniform_real_distribution"));
  REQUIRE(throws_runtime_error_containing([&] { U{0.0, inf, engine}; }, "uniform"));
  REQUIRE(throws_runtime_error_containing([&] { U{nan, 1.0, engine}; }, "uniform"));
  REQUIRE(throws_runtime_error_containing([&] { U{-big, big, engine}; }, "uniform"));
  {
    U u{0.0, 1.0, engine};
    REQUIRE(throws_runtime_error_containing([&] { u.set_parameter(2.0, 1.0); },
                                            "uniform"));
    const double v = u();
    REQUIRE(v >= 0.0 && v < 1.0);
  }

  REQUIRE(throws_runtime_error_containing([&] { N{0.0, 0.0, engine}; },
                                          "normal_distribution"));
  REQUIRE(throws_runtime_error_containing([&] { N{0.0, -1.0, engine}; }, "normal"));
  REQUIRE(throws_runtime_error_containing([&] { N{nan, 1.0, engine}; }, "normal"));
  REQUIRE(throws_runtime_error_containing([&] { N{0.0, inf, engine}; }, "normal"));
  {
    N n{0.0, 1.0, engine};
    REQUIRE(std::isfinite(n()));
  }
}

} // namespace

int main() {
  test_draw_order();
  test_negative_sizes_are_resampled();
  test_positive_probability();
  test_hopeless_size_distribution_rejected_at_construction();
  test_redraw_cap_backstop();
  test_tube_length_resampled();
  test_cluster_seeds_validation();
  test_json_integral_strictness();
  test_distribution_parameters();
  test_metadata_field();

  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "input_validation_test: all passed\n";
  return 0;
}
