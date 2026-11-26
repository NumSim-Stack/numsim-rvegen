#pragma once

#include <cmath>
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

// Uniform distribution on the half-open interval [a, b).
//
// Validation: `a` and `b` must be finite, `a <= b`, and `b - a` must not
// overflow; otherwise the ctor (and set_parameter) throws
// std::runtime_error.
//
// Reproducibility: draws use rvegen::portable_uniform_real (53-bit
// mantissa method, see portable_random.h), not
// std::uniform_real_distribution, so the same engine seed gives
// bit-identical values on libstdc++, libc++ and MSVC.
template <typename T = double, typename Engine = std::mt19937>
class uniform_real_distribution final : public distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;

  // Direct C++ ctor — used by tests and library users not driving via JSON.
  uniform_real_distribution(value_type a, value_type b, engine_type& engine)
      : _dist(checked_param(a, b)), _engine{engine} {}

  using fields = field_list<
      field<"a", value_type, true,
            numsim_core::description_label<"lower bound of the uniform distribution (inclusive, finite, <= b)">>,
      field<"b", value_type, true,
            numsim_core::description_label<"upper bound of the uniform distribution (exclusive: the range is [a, b))">>>;

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

  // Fraction of [a, b) above zero; for a == b, 1 if a > 0 else 0.
  [[nodiscard]] std::optional<value_type> positive_probability() const override {
    const value_type a = _dist.a();
    const value_type b = _dist.b();
    if (!(a < b)) return a > value_type{0} ? value_type{1} : value_type{0};
    if (a >= value_type{0}) return value_type{1};
    if (b <= value_type{0}) return value_type{0};
    return b / (b - a);
  }

  void set_parameter(value_type a, value_type b) {
    _dist = checked_param(a, b);
  }

private:
  [[nodiscard]] static portable_uniform_real<value_type> checked_param(
      value_type a, value_type b) {
    if (!std::isfinite(a) || !std::isfinite(b) || !(a <= b) ||
        !std::isfinite(b - a)) {
      throw std::runtime_error{
          "uniform_real_distribution: invalid bounds a=" + std::to_string(a) +
          ", b=" + std::to_string(b) +
          " (need finite a <= b with finite b - a)"};
    }
    return portable_uniform_real<value_type>{a, b};
  }

  // Tuple-taking helper used by the schema-driven ctor. Lets us delegate
  // through `fields::extract(handler)` without re-extracting per element.
  uniform_real_distribution(std::tuple<value_type, value_type> ab,
                            engine_type& engine)
      : uniform_real_distribution(std::get<0>(ab), std::get<1>(ab), engine) {}

  portable_uniform_real<value_type> _dist;
  engine_type& _engine;
};

} // namespace rvegen
