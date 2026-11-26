#pragma once

#include <cmath>
#include <numbers>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>

#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "distribution_base.h"
#include "portable_random.h"

namespace rvegen {

// Normal distribution N(mean, stddev^2).
//
// Validation: `mean` and `stddev` must be finite and `stddev > 0`;
// otherwise the ctor throws std::runtime_error.
//
// Reproducibility: draws use rvegen::portable_normal (Marsaglia polar
// method on 53-bit uniforms, with an in-house logarithm; see
// portable_random.h), not std::normal_distribution, so the same engine
// seed gives bit-identical values on libstdc++, libc++ and MSVC. Values
// come in pairs: every second call is served from a cached value and
// does not advance the engine.
template <typename T = double, typename Engine = std::mt19937>
class normal_distribution final : public distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;

  normal_distribution(value_type mean, value_type stddev, engine_type& engine)
      : _dist(checked_param(mean, stddev)), _engine{engine} {}

  using fields = field_list<
      field<"mean", value_type, true,
            numsim_core::description_label<"mean of the normal distribution">>,
      field<"stddev", value_type, true,
            min_only<value_type{0}>,
            numsim_core::description_label<"standard deviation of the normal distribution (finite, strictly positive)">>>;

  // Schema-driven ctor — registry calls this.
  normal_distribution(parameter_handler_t const& handler, engine_type& engine)
      : normal_distribution(fields::extract(handler), engine) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  [[nodiscard]] value_type operator()() override { return _dist(_engine); }

  // P(X > 0) = 0.5 * erfc(-mean / (stddev * sqrt(2))).
  [[nodiscard]] std::optional<value_type> positive_probability() const override {
    return value_type{0.5} *
           std::erfc(-_dist.mean() / (_dist.stddev() * std::numbers::sqrt2_v<value_type>));
  }

private:
  [[nodiscard]] static portable_normal<value_type> checked_param(
      value_type mean, value_type stddev) {
    if (!std::isfinite(mean) || !std::isfinite(stddev) ||
        !(stddev > value_type{0})) {
      throw std::runtime_error{
          "normal_distribution: invalid parameters mean=" +
          std::to_string(mean) + ", stddev=" + std::to_string(stddev) +
          " (need finite mean and finite stddev > 0)"};
    }
    return portable_normal<value_type>{mean, stddev};
  }

  // Tuple-taking helper used by the schema-driven ctor. Lets us delegate
  // through `fields::extract(handler)` without re-extracting per element.
  normal_distribution(std::tuple<value_type, value_type> ms,
                      engine_type& engine)
      : normal_distribution(std::get<0>(ms), std::get<1>(ms), engine) {}

  portable_normal<value_type> _dist;
  engine_type& _engine;
};

} // namespace rvegen
