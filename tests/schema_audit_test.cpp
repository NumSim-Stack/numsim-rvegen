// Audits every parameter schema across all rvegen registries to make sure
// new types don't slip in without GUI metadata. Asserts:
//
//   * Every field has a description_label policy attached — Tessera form
//     widgets surface this as a tooltip, and error messages reference it.
//   * Every INTEGRAL numeric field has at least one of: a range policy
//     (range<>, min_only<>, max_only<>) or a unit_label policy. Qt's
//     QSpinBox defaults to 0..99 if no range is given, which is wrong
//     for nearly every count/dimension we have.
//
// Floating-point fields are exempt from the second rule: distribution
// parameters (a/b/mean/value) are user-supplied bounds with no inherent
// constraint, and QDoubleSpinBox accepts a generous default range from
// the form-builder. String fields are exempt entirely (filenames,
// distribution-name references, free-form text).
//
// Implementation note: under the policy API, hint metadata lives on
// side-bases (range_hint_base, units_hint_base, description_hint_base)
// reachable via dynamic_cast. The audit downcasts each parameter to
// its concrete typed input_parameter<T, K, P> for known T (size_t,
// int, double, string), walks the resulting checks() list, and tests
// each check against the side-bases. Any T not in the known list is
// silently skipped — the audit is only as complete as its T-list, but
// extending it is one if-else.

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <typeindex>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "rvegen/rvegen.h"
#include "rvegen/shapes/box.h"
#include "rvegen/shapes/circle.h"
#include "rvegen/shapes/ellipse.h"
#include "rvegen/shapes/rectangle.h"
#include "rvegen/shapes/sphere.h"

namespace {

using base_t = numsim_core::input_parameter_base<
    std::string, rvegen::parameter_handler_t>;

template <typename T>
using typed_t = numsim_core::input_parameter<
    T, std::string, rvegen::parameter_handler_t>;

struct violation {
  std::string registry;
  std::string type;
  std::string field;
  std::string what;   // "missing description" | "integral field missing range/units"
};

std::vector<violation> all_violations;

bool is_integral_numeric(std::type_index t) {
  return t == typeid(int) || t == typeid(unsigned int)
      || t == typeid(std::size_t)
      || t == typeid(std::uint32_t) || t == typeid(std::uint64_t)
      || t == typeid(std::int32_t)  || t == typeid(std::int64_t);
}

// Walk the typed parameter's check list and report which side-bases
// are reachable. The caller picks T because input_parameter_base
// can't expose checks() polymorphically.
struct hint_summary {
  bool has_description = false;
  bool has_range = false;
  bool has_units = false;
};

template <typename T>
hint_summary inspect_typed(typed_t<T> const& p) {
  hint_summary s;
  for (auto const& check : p.checks()) {
    if (dynamic_cast<numsim_core::description_hint_base const*>(check.get()))
      s.has_description = true;
    if (dynamic_cast<numsim_core::range_hint_base const*>(check.get()))
      s.has_range = true;
    if (dynamic_cast<numsim_core::units_hint_base const*>(check.get()))
      s.has_units = true;
  }
  return s;
}

// Try the known T-list. Returns nullopt if this base parameter doesn't
// match any known type (audit silently skips it; extend the list if a
// new T appears in the schema).
std::optional<hint_summary> inspect(base_t const& base) {
  if (auto const* p = dynamic_cast<typed_t<int> const*>(&base))
    return inspect_typed(*p);
  if (auto const* p = dynamic_cast<typed_t<unsigned int> const*>(&base))
    return inspect_typed(*p);
  if (auto const* p = dynamic_cast<typed_t<std::size_t> const*>(&base))
    return inspect_typed(*p);
  if (auto const* p = dynamic_cast<typed_t<double> const*>(&base))
    return inspect_typed(*p);
  if (auto const* p = dynamic_cast<typed_t<float> const*>(&base))
    return inspect_typed(*p);
  if (auto const* p = dynamic_cast<typed_t<std::string> const*>(&base))
    return inspect_typed(*p);
  return std::nullopt;
}

template <typename Schema>
void audit_schema(Schema const& schema, std::string const& registry_name,
                  std::string const& type_name) {
  for (auto const& [field_key, param_ptr] : schema) {
    auto hints = inspect(*param_ptr);
    if (!hints) continue;  // unknown T; skip silently
    if (!hints->has_description) {
      all_violations.push_back({registry_name, type_name, field_key,
                                "missing description"});
    }
    if (is_integral_numeric(param_ptr->type_id())) {
      if (!hints->has_range && !hints->has_units) {
        all_violations.push_back({registry_name, type_name, field_key,
                                  "integral field missing range/units"});
      }
    }
  }
}

template <typename Registry>
void audit(Registry& reg, std::string const& registry_name) {
  for (auto const& type_key : reg.registered_types()) {
    audit_schema(reg.schema(type_key), registry_name, type_key);
  }
}

// Shapes are constructed via *_input registries rather than having their
// own top-level registry, so the registry-walking audit above never
// reaches them. Audit them directly via the static parameters() method
// every shape declares — Tessera will read these for the per-shape edit
// panel and the missing metadata used to be a silent gap.
template <typename Shape>
void audit_shape(std::string const& type_name) {
  audit_schema(Shape::parameters(), "shape", type_name);
}

} // namespace

int main() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_direction_distributions<>();
  rvegen::register_all_inputs<>();
  rvegen::register_all_generators<>();
  rvegen::register_all_terminations<>();
  rvegen::register_all_post_processes<>();

  audit(rvegen::distribution_registry_t<>::instance(),           "distribution");
  audit(rvegen::direction_distribution_registry_t<>::instance(), "direction_distribution");
  audit(rvegen::input_registry_t<>::instance(),                  "input");
  audit(rvegen::generator_registry_t<>::instance(),              "generator");
  audit(rvegen::termination_registry_t<>::instance(),            "termination");
  audit(rvegen::post_process_registry_t<>::instance(),           "post_process");

  audit_shape<rvegen::circle<>>(   "circle");
  audit_shape<rvegen::rectangle<>>("rectangle");
  audit_shape<rvegen::ellipse<>>(  "ellipse");
  audit_shape<rvegen::sphere<>>(   "sphere");
  audit_shape<rvegen::box<>>(      "box");

  if (!all_violations.empty()) {
    std::cerr << "schema_audit_test: " << all_violations.size()
              << " annotation gap(s):\n";
    for (auto const& v : all_violations) {
      std::cerr << "  - " << v.registry << "/" << v.type
                << " field '" << v.field << "': " << v.what << '\n';
    }
    return EXIT_FAILURE;
  }
  std::cout << "schema_audit_test: PASS\n";
  return EXIT_SUCCESS;
}
