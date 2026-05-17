#pragma once

// Oriented (von-Mises) distribution for 2D rotation angles.
//
// Composites researchers measure fibre / inclusion orientations via μ-CT
// and feed them into RVE generators to reproduce industrially-realistic
// anisotropic packings. The distribution that physicists/engineers
// actually use is the von Mises distribution on a circle, parameterised
// by:
//   * mean_angle (μ): preferred orientation in radians
//   * kappa (κ): concentration; 0 = uniform, ∞ = perfectly aligned
//
// Relationship to the Advani-Tucker A2 tensor (the typical FOD input):
//   * mean_angle  = ½ atan2(2·A12, A11 - A22)
//   * order S2    = √((A11-A22)² + (2·A12)²) ∈ [0, 1]
//   * S2 → κ:    use S2 = I1(2κ)/I0(2κ); not closed-form, but
//     S2 ≈ tanh(κ) for small κ and S2 ≈ 1 - 1/(2κ) for large κ.
// Users with raw A2 should compute kappa once via a small helper and
// pass kappa here.
//
// 3D extension: Bingham distribution on the unit sphere, parameterised
// by 3×3 A2 directly. Deferred — see issue #6 for the design.

#include <cmath>
#include <functional>
#include <numbers>
#include <random>
#include <stdexcept>

#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "distribution_base.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
class oriented_uniform_distribution final : public distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;

  oriented_uniform_distribution(value_type mean_angle, value_type kappa,
                                 engine_type& engine)
      : _mean{mean_angle}, _kappa{kappa}, _engine{engine},
        _uniform{value_type{0}, value_type{1}} {
    if (_kappa < value_type{0}) {
      throw std::invalid_argument{
          "oriented_uniform_distribution: kappa must be ≥ 0"};
    }
  }

  using fields = field_list<
      field<"mean_angle", value_type, true,
            numsim_core::unit_label<"rad">,
            numsim_core::description_label<"preferred orientation angle (CCW from +x)">>,
      field<"kappa", value_type, true,
            min_only<value_type{0}>,
            numsim_core::description_label<"von Mises concentration; 0 = uniform, large = aligned (compute from Advani-Tucker A2 if needed)">>>;

  oriented_uniform_distribution(parameter_handler_t const& handler,
                                 engine_type& engine)
      : oriented_uniform_distribution(fields::extract(handler), engine) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  // Best-Fisher (1979) rejection sampler for the von Mises distribution.
  // Returns angle in [-π, π] + mean_angle (no wrap to [0, 2π) — downstream
  // shape ctors accept any real angle and normalise via atan2 on read).
  [[nodiscard]] value_type operator()() override {
    constexpr value_type pi = std::numbers::pi_v<value_type>;
    constexpr value_type two_pi = value_type{2} * pi;

    if (_kappa < value_type{1e-9}) {
      // Uniform on [-π, π].
      return _mean + (_uniform(_engine.get()) - value_type{0.5}) * two_pi;
    }

    const value_type a = value_type{1}
        + std::sqrt(value_type{1} + value_type{4} * _kappa * _kappa);
    const value_type b = (a - std::sqrt(value_type{2} * a))
        / (value_type{2} * _kappa);
    const value_type r = (value_type{1} + b * b) / (value_type{2} * b);

    while (true) {
      const value_type u1 = _uniform(_engine.get());
      const value_type z  = std::cos(pi * u1);
      const value_type f  = (value_type{1} + r * z) / (r + z);
      const value_type c  = _kappa * (r - f);
      const value_type u2 = _uniform(_engine.get());

      if (u2 < c * (value_type{2} - c)
          || u2 <= c * std::exp(value_type{1} - c)) {
        const value_type u3 = _uniform(_engine.get());
        const value_type sign = (u3 - value_type{0.5}) >= value_type{0}
            ? value_type{1} : value_type{-1};
        return _mean + sign * std::acos(f);
      }
    }
  }

  [[nodiscard]] value_type mean_angle() const noexcept { return _mean; }
  [[nodiscard]] value_type kappa() const noexcept { return _kappa; }

private:
  // Tuple-taking helper used by the schema-driven ctor. Lets us delegate
  // through `fields::extract(handler)` without re-extracting per element.
  oriented_uniform_distribution(std::tuple<value_type, value_type> mk,
                                engine_type& engine)
      : oriented_uniform_distribution(std::get<0>(mk), std::get<1>(mk),
                                       engine) {}

  value_type _mean;
  value_type _kappa;
  std::reference_wrapper<engine_type> _engine;
  std::uniform_real_distribution<value_type> _uniform;
};

} // namespace rvegen
