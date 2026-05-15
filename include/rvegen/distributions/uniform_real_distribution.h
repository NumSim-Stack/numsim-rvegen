#pragma once

#include <random>

#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "distribution_base.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
class uniform_real_distribution final : public distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;

  // Direct C++ ctor — used by tests and library users not driving via JSON.
  uniform_real_distribution(value_type a, value_type b, engine_type& engine)
      : _dist(a, b), _engine{engine} {}

  using fields = field_list<
      field<"a", value_type, true,
            numsim_core::description_label<"lower bound of the uniform distribution (inclusive)">>,
      field<"b", value_type, true,
            numsim_core::description_label<"upper bound of the uniform distribution (inclusive)">>>;

  // Schema-driven ctor — what the registry calls. Reads `a`, `b` from the
  // handler via the field_list `extract` helper.
  uniform_real_distribution(parameter_handler_t const& handler,
                            engine_type& engine)
      : uniform_real_distribution(fields::extract(handler), engine) {}

  // Static schema — co-located with the type. Returned by registry::schema().
  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  [[nodiscard]] value_type operator()() override { return _dist(_engine); }

  void set_parameter(value_type a, value_type b) {
    _dist.param(typename std::uniform_real_distribution<value_type>::param_type{a, b});
  }

private:
  // Tuple-taking helper used by the schema-driven ctor. Lets us delegate
  // through `fields::extract(handler)` without re-extracting per element.
  uniform_real_distribution(std::tuple<value_type, value_type> ab,
                            engine_type& engine)
      : uniform_real_distribution(std::get<0>(ab), std::get<1>(ab), engine) {}

  std::uniform_real_distribution<value_type> _dist;
  engine_type& _engine;
};

} // namespace rvegen
