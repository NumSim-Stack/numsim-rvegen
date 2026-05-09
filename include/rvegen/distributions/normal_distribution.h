#pragma once

#include <random>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "distribution_base.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
class normal_distribution final : public distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;

  normal_distribution(value_type mean, value_type stddev, engine_type& engine)
      : _dist(mean, stddev), _engine{engine} {}

  // Schema-driven ctor — registry calls this.
  normal_distribution(parameter_handler_t const& handler, engine_type& engine)
      : normal_distribution(handler.template get<value_type>("mean"),
                            handler.template get<value_type>("stddev"),
                            engine) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<value_type>("mean").template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"mean of the normal distribution">>();
    s.template insert<value_type>("stddev").template add<numsim_core::is_required>()
        .template add<min_only<value_type{0}>>()
        .template add<numsim_core::description_label<"standard deviation of the normal distribution">>();
    return s;
  }

  [[nodiscard]] value_type operator()() override { return _dist(_engine); }

private:
  std::normal_distribution<value_type> _dist;
  engine_type& _engine;
};

} // namespace rvegen
