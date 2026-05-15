#pragma once

#include <random>

#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
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

  using fields = field_list<
      field<"mean", value_type, true,
            numsim_core::description_label<"mean of the normal distribution">>,
      field<"stddev", value_type, true,
            min_only<value_type{0}>,
            numsim_core::description_label<"standard deviation of the normal distribution">>>;

  // Schema-driven ctor — registry calls this.
  normal_distribution(parameter_handler_t const& handler, engine_type& engine)
      : normal_distribution(fields::extract(handler), engine) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  [[nodiscard]] value_type operator()() override { return _dist(_engine); }

private:
  // Tuple-taking helper used by the schema-driven ctor. Lets us delegate
  // through `fields::extract(handler)` without re-extracting per element.
  normal_distribution(std::tuple<value_type, value_type> ms,
                      engine_type& engine)
      : normal_distribution(std::get<0>(ms), std::get<1>(ms), engine) {}

  std::normal_distribution<value_type> _dist;
  engine_type& _engine;
};

} // namespace rvegen
