#pragma once

#include <random>

#include <numsim-core/input_parameter_controller.h>

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

  // Schema-driven ctor — what the registry calls. Reads `a`, `b` from the
  // handler that was previously populated by a parameter_visitor_*.
  uniform_real_distribution(parameter_handler_t const& handler,
                            engine_type& engine)
      : uniform_real_distribution(handler.template get<value_type>("a"),
                                  handler.template get<value_type>("b"),
                                  engine) {}

  // Static schema — co-located with the type. Returned by registry::schema().
  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<value_type>("a").template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"lower bound of the uniform distribution (inclusive)">>();
    s.template insert<value_type>("b").template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"upper bound of the uniform distribution (inclusive)">>();
    return s;
  }

  [[nodiscard]] value_type operator()() override { return _dist(_engine); }

  void set_parameter(value_type a, value_type b) {
    _dist.param(typename std::uniform_real_distribution<value_type>::param_type{a, b});
  }

private:
  std::uniform_real_distribution<value_type> _dist;
  engine_type& _engine;
};

} // namespace rvegen
