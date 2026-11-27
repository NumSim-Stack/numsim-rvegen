// field_list optional fields (#1).
//
// 1. field_with_default: the schema marks it optional and carries
//    numsim-core's set_default; a missing key yields the default (both
//    through input_parameter_controller::accept() and through extract()
//    on a hand-built handler); a present key overrides it; range
//    annotations still reject out-of-range values.
// 2. field<..., false>: optional without default, extracted as
//    std::optional<T>.
// 3. A migrated type end to end: gmsh_geo_writer's optional mesh knobs.

#include <any>
#include <cstddef>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <numsim-core/input_parameter_controller.h>

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

template <typename T>
using typed_t = numsim_core::input_parameter<T, std::string,
                                             rvegen::parameter_handler_t>;

using fields = rvegen::field_list<
    rvegen::field<"req", double, true,
                  numsim_core::description_label<"required value">>,
    rvegen::field_with_default<"n", std::size_t, std::size_t{3},
                               rvegen::min_only<std::size_t{1}>,
                               numsim_core::description_label<"count (default 3)">>,
    rvegen::field_with_default<"label", std::string,
                               numsim_core::fixed_string{"abc"},
                               numsim_core::description_label<"label (default abc)">>,
    rvegen::field<"opt", int, false, rvegen::min_only<0>,
                  numsim_core::description_label<"optional, no default">>>;

// Run the JSON through the schema the way build_from_json does.
rvegen::parameter_handler_t accept(char const* json) {
  const auto j = nlohmann::json::parse(json);
  rvegen::parameter_visitor_nlohmann visitor{j};
  rvegen::parameter_handler_t handler;
  fields::schema().accept(visitor, handler);
  return handler;
}

template <typename T, template <class, class, class> class Check>
bool has_check(rvegen::parameter_controller_t const& schema,
               std::string const& key) {
  auto const& p = dynamic_cast<typed_t<T> const&>(schema.get(key));
  for (auto const& c : p.checks()) {
    if (dynamic_cast<Check<T, std::string, rvegen::parameter_handler_t> const*>(
            c.get()))
      return true;
  }
  return false;
}

template <typename F>
bool throws_invalid_argument(F&& f) {
  try {
    f();
  } catch (std::invalid_argument const&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

void test_schema_marks_optional_fields() {
  const auto s = fields::schema();
  REQUIRE((has_check<double, numsim_core::is_required>(s, "req")));
  REQUIRE(!(has_check<double, numsim_core::set_default>(s, "req")));
  REQUIRE(!(has_check<std::size_t, numsim_core::is_required>(s, "n")));
  REQUIRE((has_check<std::size_t, numsim_core::set_default>(s, "n")));
  REQUIRE(!(has_check<std::string, numsim_core::is_required>(s, "label")));
  REQUIRE((has_check<std::string, numsim_core::set_default>(s, "label")));
  REQUIRE(!(has_check<int, numsim_core::is_required>(s, "opt")));
  REQUIRE(!(has_check<int, numsim_core::set_default>(s, "opt")));
}

void test_missing_key_yields_default() {
  // Through the controller: set_default inserts the defaults.
  auto h = accept(R"({"req": 1.5})");
  REQUIRE(h.contains("n") && h.get<std::size_t>("n") == 3);
  REQUIRE(h.contains("label") && h.get<std::string>("label") == "abc");
  REQUIRE(!h.contains("opt"));
  const auto [req, n, label, opt] = fields::extract(h);
  REQUIRE(req == 1.5);
  REQUIRE(n == 3);
  REQUIRE(label == "abc");
  REQUIRE(!opt.has_value());

  // Hand-built handler that never went through accept(): extract()
  // falls back to the default on its own.
  rvegen::parameter_handler_t raw;
  raw.insert<double>("req", 2.0);
  const auto [req2, n2, label2, opt2] = fields::extract(raw);
  REQUIRE(req2 == 2.0);
  REQUIRE(n2 == 3);
  REQUIRE(label2 == "abc");
  REQUIRE(!opt2.has_value());
  REQUIRE(fields::value<"n">(raw) == 3);
  REQUIRE(fields::value<"req">(raw) == 2.0);
}

void test_present_key_overrides_default() {
  auto h = accept(R"({"req": 1.5, "n": 7, "label": "xyz", "opt": 4})");
  const auto [req, n, label, opt] = fields::extract(h);
  REQUIRE(req == 1.5);
  REQUIRE(n == 7);
  REQUIRE(label == "xyz");
  REQUIRE(opt.has_value() && *opt == 4);
  REQUIRE(fields::value<"label">(h) == "xyz");
}

void test_out_of_range_still_rejected() {
  REQUIRE(throws_invalid_argument([] { (void)accept(R"({"req": 1.0, "n": 0})"); }));
  REQUIRE(throws_invalid_argument([] { (void)accept(R"({"req": 1.0, "opt": -1})"); }));
  // Missing required key still fails.
  REQUIRE(throws_invalid_argument([] { (void)accept(R"({"n": 5})"); }));
}

void test_gmsh_geo_writer_defaults() {
  rvegen::register_all_post_processes<>();
  auto& reg = rvegen::post_process_registry_t<>::instance();
  {
    auto pp = rvegen::build_from_json(
        reg, nlohmann::json::parse(R"({"type": "gmsh_geo", "output_path": "a.geo"})"));
    auto const& w = dynamic_cast<rvegen::gmsh_geo_writer<double> const&>(*pp);
    REQUIRE(w.output_path() == "a.geo");
    REQUIRE(!w.periodic());
    REQUIRE(w.characteristic_length_min() == 0.0);
    REQUIRE(w.characteristic_length_max() == 0.0);
    REQUIRE(w.element_order() == 0);
    REQUIRE(w.algorithm() == 0);
    REQUIRE(!w.recombine_quads());
  }
  {
    auto pp = rvegen::build_from_json(reg, nlohmann::json::parse(R"({
      "type": "gmsh_geo", "output_path": "b.geo", "periodic": true,
      "char_length_min": 0.01, "char_length_max": 0.05,
      "element_order": 2, "algorithm": 6, "recombine_quads": true})"));
    auto const& w = dynamic_cast<rvegen::gmsh_geo_writer<double> const&>(*pp);
    REQUIRE(w.periodic());
    REQUIRE(w.characteristic_length_min() == 0.01);
    REQUIRE(w.characteristic_length_max() == 0.05);
    REQUIRE(w.element_order() == 2);
    REQUIRE(w.algorithm() == 6);
    REQUIRE(w.recombine_quads());
  }
  REQUIRE(throws_invalid_argument([&] {
    (void)rvegen::build_from_json(reg, nlohmann::json::parse(
        R"({"type": "gmsh_geo", "output_path": "c.geo", "element_order": 9})"));
  }));
}

// Default narrowing guard (compile-time): lossy conversions are flagged,
// exact ones are not.
static_assert(rvegen::detail::default_is_lossless<std::size_t>(3));
static_assert(rvegen::detail::default_is_lossless<double>(0.5));
static_assert(rvegen::detail::default_is_lossless<double>(1));
static_assert(rvegen::detail::default_is_lossless<bool>(false));
static_assert(!rvegen::detail::default_is_lossless<std::size_t>(-1));
static_assert(!rvegen::detail::default_is_lossless<int>(1.5));
static_assert(!rvegen::detail::default_is_lossless<float>(0.1));
static_assert(!rvegen::detail::default_is_lossless<int>(3000000000LL));

} // namespace

int main() {
  test_schema_marks_optional_fields();
  test_missing_key_yields_default();
  test_present_key_overrides_default();
  test_out_of_range_still_rejected();
  test_gmsh_geo_writer_defaults();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "field_list_defaults_test: all passed\n";
  return 0;
}
