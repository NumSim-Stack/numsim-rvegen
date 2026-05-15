#pragma once

#include <random>

#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "distribution_base.h"

namespace rvegen {

// "Distribution" that always returns the same value. Useful for fixed
// parameters where the JSON layer consistently passes a distribution object.
//
// Templated on Engine for compatibility with the (T, Engine) registry
// signature — the engine reference is held but never used.
template <typename T = double, typename Engine = std::mt19937>
class constant_distribution final : public distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;

  // Direct ctor without engine for non-registry use.
  explicit constant_distribution(value_type value) noexcept : _value{value} {}

  // Registry-friendly ctor — engine ignored but accepted for uniformity.
  constant_distribution(value_type value, engine_type& /*unused*/) noexcept
      : _value{value} {}

  using fields = field_list<
      field<"value", value_type, true,
            numsim_core::description_label<"the constant value the distribution always returns">>>;

  // Schema-driven ctor.
  constant_distribution(parameter_handler_t const& handler,
                        engine_type& /*unused*/)
      : _value{std::get<0>(fields::extract(handler))} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  [[nodiscard]] value_type operator()() noexcept override { return _value; }

  void set_value(value_type value) noexcept { _value = value; }
  [[nodiscard]] value_type value() const noexcept { return _value; }

private:
  value_type _value;
};

} // namespace rvegen
