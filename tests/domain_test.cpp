// The domain type (#122): the one place where the dimension of the RVE is
// decided and its extents are validated, and the load-time check that the
// shapes of a config have the domain's dimension.

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"
#include "rvegen/pipeline.h"

namespace {

int failures = 0;
#define REQUIRE(cond)                                                    \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << "REQUIRE failed: " #cond " at " << __FILE__ << ":"    \
                << __LINE__ << "\n";                                     \
      ++failures;                                                        \
    }                                                                    \
  } while (false)

using json = nlohmann::json;
using rvegen::domain;

// The message of the std::invalid_argument f() throws; "" if it throws
// nothing or something else.
template <typename F>
std::string invalid(F&& f) {
  try { f(); } catch (std::invalid_argument const& e) { return e.what(); }
  catch (...) { return {}; }
  return {};
}
bool has(std::string const& s, char const* needle) {
  return s.find(needle) != std::string::npos;
}

void test_dimension_and_measure() {
  constexpr domain<double> unit{};
  static_assert(unit.is_2d() && unit.dimension() == 2 && unit.measure() == 1.0);
  constexpr domain<double> sheet{2.0, 3.0};
  static_assert(sheet.is_2d() && !sheet.is_3d() && sheet.measure() == 6.0);
  constexpr domain<double> brick{2.0, 3.0, 4.0};
  static_assert(brick.is_3d() && brick.dimension() == 3 && brick.measure() == 24.0);
  static_assert(brick[0] == 2.0 && brick[1] == 3.0 && brick[2] == 4.0);
  static_assert(sheet[2] == 0.0);

  // The older spelling of 2D, and the array conversions both ways.
  const std::array<double, 3> flat{1.5, 2.5, 0.0};
  const domain<double> d = flat;
  REQUIRE(d.is_2d() && d.measure() == 3.75);
  const std::array<double, 3>& back = d;
  REQUIRE(back == flat && d.extents() == flat);
  REQUIRE(d == domain<double>(1.5, 2.5) && !(d == domain<double>(1.5, 2.5, 1.0)));

  const domain<float> f{2.0f, 2.0f, 0.5f};
  REQUIRE(f.is_3d() && f.measure() == 2.0f);
}

void test_rejected_extents() {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  using box = std::array<double, 3>;
  REQUIRE(has(invalid([] { (void)domain<double>{box{0, 0, 0}}; }),
              "x and y extents must be > 0, got [0, 0, 0]"));
  REQUIRE(has(invalid([] { (void)domain<double>{0.0, 1.0}; }), "must be > 0"));
  REQUIRE(has(invalid([] { (void)domain<double>{1.0, 0.0, 1.0}; }), "must be > 0"));
  REQUIRE(has(invalid([] { (void)domain<double>{-1.0, 1.0}; }), "must be > 0"));
  REQUIRE(has(invalid([] { (void)domain<double>{1.0, -1.0, 1.0}; }), "must be > 0"));
  REQUIRE(has(invalid([] { (void)domain<double>{1.0, 1.0, -1.0}; }),
              "z extent must be >= 0"));
  for (double bad : {nan, inf, -inf}) {
    for (int axis = 0; axis < 3; ++axis) {
      box b{1, 1, 1};
      b[static_cast<std::size_t>(axis)] = bad;
      REQUIRE(has(invalid([&] { (void)domain<double>{b}; }), "must be finite"));
    }
  }
  // Through an interface that takes a domain: the conversion validates.
  rvegen::rve_generator_base<double>::shape_vector none;
  REQUIRE(has(invalid([&] { (void)rvegen::current_volume_fraction<double>(
                                none, box{1, 1, nan}); }),
              "must be finite"));
}

void test_from_json() {
  REQUIRE(domain<double>::from_json(json::parse("[2, 3]")).is_2d());
  REQUIRE(domain<double>::from_json(json::parse("[2, 3, 0]")).is_2d());
  REQUIRE(domain<double>::from_json(json::parse("[2, 3, 0.0]")) ==
          domain<double>(2.0, 3.0));
  const auto d3 = domain<double>::from_json(json::parse("[2, 3, 4]"));
  REQUIRE(d3.is_3d() && d3.measure() == 24.0);

  auto bad = [](char const* text) {
    return invalid([&] { (void)domain<double>::from_json(json::parse(text)); });
  };
  REQUIRE(has(bad("[1]"), "array of 2 or 3 numbers"));
  REQUIRE(has(bad("[1, 1, 1, 1]"), "array of 2 or 3 numbers"));
  REQUIRE(has(bad("{\"x\": 1}"), "array of 2 or 3 numbers"));
  REQUIRE(has(bad("\"1 1 1\""), "array of 2 or 3 numbers"));
  REQUIRE(has(bad("[1, \"a\"]"), "must be numbers"));
  REQUIRE(has(bad("[1, null, 1]"), "must be numbers"));
  REQUIRE(has(bad("[0, 0, 0]"), "must be > 0"));
  REQUIRE(has(bad("[0, 0]"), "must be > 0"));
  REQUIRE(has(bad("[1, -2]"), "must be > 0"));
  REQUIRE(has(bad("[1, 1, -0.5]"), "z extent must be >= 0"));
  const double inf = std::numeric_limits<double>::infinity();
  REQUIRE(has(invalid([&] { (void)domain<double>::from_json(json{inf, 1.0}); }),
              "must be finite"));
  REQUIRE(has(invalid([&] { (void)domain<double>::from_json(json{1.0, 1.0, -inf}); }),
              "must be finite"));
}

void test_shapes_and_inputs_declare_their_dimension() {
  REQUIRE(rvegen::circle<double>(0.5, 0.5, 0.1).dimension() == 2);
  REQUIRE(rvegen::sphere<double>(0.5, 0.5, 0.5, 0.1).dimension() == 3);
  // measure() is the size in the shape's own dimension - never "whichever
  // of area and volume is not zero".
  const rvegen::circle<double> c{0.5, 0.5, 0.1};
  REQUIRE(c.measure() == c.area() && c.volume() == 0.0);
  const rvegen::sphere<double> s{0.5, 0.5, 0.5, 0.1};
  REQUIRE(s.measure() == s.volume() && s.measure() > 0.0);
}

json base_config() {
  return json::parse(R"({
    "seed": 3,
    "distributions": {
      "x": {"type": "uniform_real", "a": 0.2, "b": 0.8},
      "r": {"type": "uniform_real", "a": 0.03, "b": 0.05}
    },
    "generator": {"type": "only_inside", "max_attempts": 10000},
    "termination": {"type": "number_of_inclusions", "target": 3}
  })");
}
const json circle_input = {{"type", "circle_input"}, {"pos_x_dist", "x"},
                           {"pos_y_dist", "x"}, {"radius_dist", "r"}};
const json sphere_input = {{"type", "sphere_input"}, {"pos_x_dist", "x"},
                           {"pos_y_dist", "x"}, {"pos_z_dist", "x"},
                           {"radius_dist", "r"}};

// A config that mixes dimensions fails when it is loaded - before any
// shape is placed and in --check as well - and the message names the input.
void test_mixed_dimensions_fail_at_load() {
  rvegen::register_all_types();
  auto message = [](json const& cfg, bool check_only) {
    rvegen::pipeline_options o;
    o.info = [](std::string const&) {};
    o.warn = [](std::string const&) {};
    return invalid([&] {
      if (check_only) rvegen::check_config(cfg, o);
      else (void)rvegen::generate(cfg, o);
    });
  };
  for (bool check_only : {true, false}) {
    auto c = base_config();
    c["domain_box"] = {1.0, 1.0, 1.0};
    c["shapes"] = json::array({sphere_input, circle_input});
    auto what = message(c, check_only);
    REQUIRE(has(what.c_str(), "shapes[1] ('circle_input'): produces 2D shapes, "
                              "but domain_box [1.0,1.0,1.0] is 3D"));

    c["domain_box"] = {1.0, 1.0};
    c["shapes"] = json::array({sphere_input});
    what = message(c, check_only);
    REQUIRE(has(what.c_str(), "shapes[0] ('sphere_input'): produces 3D shapes, "
                              "but domain_box [1.0,1.0,0.0] is 2D"));

    // Invalid boxes: rejected at load with the value in the message.
    c["shapes"] = json::array({circle_input});
    for (char const* box : {"[0, 0, 0]", "[1, -1]", "[1, 1, -1]", "[1]",
                            "\"unit\""}) {
      c["domain_box"] = json::parse(box);
      REQUIRE(has(message(c, check_only), "domain_box"));
    }

    // Matching dimensions: accepted in both spellings of 2D and in 3D.
    for (char const* box : {"[1, 1]", "[1, 1, 0]"}) {
      c["domain_box"] = json::parse(box);
      REQUIRE(message(c, check_only).empty());
    }
    c["domain_box"] = {1.0, 1.0, 1.0};
    c["shapes"] = json::array({sphere_input});
    REQUIRE(message(c, check_only).empty());
  }

  auto c = base_config();
  c["domain_box"] = {2.0, 1.0};
  c["shapes"] = json::array({circle_input});
  rvegen::pipeline_options o;
  o.info = [](std::string const&) {};
  const auto r = rvegen::generate(c, o);
  REQUIRE(r.domain_box.is_2d() && r.domain_box.measure() == 2.0);
  REQUIRE(r.shapes.size() == 3);
}

} // namespace

int main() {
  test_dimension_and_measure();
  test_rejected_extents();
  test_from_json();
  test_shapes_and_inputs_declare_their_dimension();
  test_mixed_dimensions_fail_at_load();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "domain_test: OK\n";
  return 0;
}
