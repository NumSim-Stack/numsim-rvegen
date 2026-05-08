// Audits every parameter schema across all rvegen registries to make sure
// new types don't slip in without GUI metadata. Asserts:
//
//   * Every field has a nonempty description() — Tessera form widgets
//     surface this as a tooltip, and CLI errors refer to it.
//   * Every INTEGRAL numeric field has at least one of .min(), .max(),
//     or .units() set — Qt's QSpinBox defaults to 0..99 if no range is
//     given, which is wrong for nearly every count/dimension we have.
//
// Floating-point fields are exempt from the second rule: distribution
// parameters (a/b/mean/value) are user-supplied bounds with no inherent
// constraint, and QDoubleSpinBox accepts a generous default range from
// the form-builder. String fields are exempt entirely (filenames,
// distribution-name references, free-form text).
//
// Failure mode: the test collects every offending (registry, type, field)
// across all registries before reporting, so one fix-then-build cycle
// surfaces the full set rather than one at a time.

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <typeindex>
#include <vector>

#include "rvegen/rvegen.h"

namespace {

struct violation {
  std::string registry;
  std::string type;
  std::string field;
  std::string what;   // "missing description" | "numeric field missing min/max/units"
};

std::vector<violation> all_violations;

bool is_integral_numeric(std::type_index t) {
  return t == typeid(int) || t == typeid(unsigned int)
      || t == typeid(std::size_t)
      || t == typeid(std::uint32_t) || t == typeid(std::uint64_t)
      || t == typeid(std::int32_t)  || t == typeid(std::int64_t);
}

template <typename Registry>
void audit(Registry& reg, std::string const& registry_name) {
  for (auto const& type_key : reg.registered_types()) {
    auto schema = reg.schema(type_key);
    for (auto const& [field_key, param_ptr] : schema) {
      if (param_ptr->description().empty()) {
        all_violations.push_back({registry_name, type_key, field_key,
                                  "missing description"});
      }
      if (is_integral_numeric(param_ptr->type_id())) {
        const bool has_hint =
            param_ptr->min().has_value() ||
            param_ptr->max().has_value() ||
            !param_ptr->units().empty();
        if (!has_hint) {
          all_violations.push_back({registry_name, type_key, field_key,
                                    "integral field missing min/max/units"});
        }
      }
    }
  }
}

} // namespace

int main() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  rvegen::register_all_generators<>();
  rvegen::register_all_terminations<>();
  rvegen::register_all_post_processes<>();

  audit(rvegen::distribution_registry_t<>::instance(), "distribution");
  audit(rvegen::input_registry_t<>::instance(),         "input");
  audit(rvegen::generator_registry_t<>::instance(),     "generator");
  audit(rvegen::termination_registry_t<>::instance(),   "termination");
  audit(rvegen::post_process_registry_t<>::instance(),  "post_process");

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
