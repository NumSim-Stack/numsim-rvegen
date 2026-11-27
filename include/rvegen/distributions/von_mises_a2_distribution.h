#pragma once

// 2D von Mises orientation sampler parameterised by the Advani-Tucker
// orientation tensor A2 (a11, a22, a12) instead of (mean_angle, kappa)
// (#6). The tensor is fitted once at construction (orientation_tensor_fit.h)
// and sampling is delegated to oriented_uniform_distribution, so drawn
// angles θ reproduce ⟨cos²θ⟩ = a11, ⟨sin²θ⟩ = a22, ⟨cosθ sinθ⟩ = a12.
//
// Registered as "von_mises_a2". A separate registered name rather than
// optional fields on "oriented_uniform" / "von_mises": field_list has no
// optional-with-default fields yet (extract() throws on a missing key),
// and two fully-required schemas keep each config self-describing and
// schema-audited instead of needing a hand-written "exactly one of"
// rule.

#include <functional>
#include <random>
#include <tuple>

#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "distribution_base.h"
#include "orientation_tensor_fit.h"
#include "oriented_uniform_distribution.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
class von_mises_a2_distribution final : public distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;

  von_mises_a2_distribution(T a11, T a22, T a12, engine_type& engine)
      : von_mises_a2_distribution(fit_von_mises_a2<T>(a11, a22, a12),
                                  engine) {}

  using fields = field_list<
      field<"a11", T, true,
            numsim_core::description_label<"orientation tensor A2 component <cos²θ>; a11 + a22 = 1">>,
      field<"a22", T, true,
            numsim_core::description_label<"orientation tensor A2 component <sin²θ>; a11 + a22 = 1">>,
      field<"a12", T, true,
            numsim_core::description_label<"orientation tensor A2 component <cosθ·sinθ>; a12² ≤ a11·a22">>>;

  von_mises_a2_distribution(parameter_handler_t const& handler,
                            engine_type& engine)
      : von_mises_a2_distribution(fields::extract(handler), engine) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  [[nodiscard]] T operator()() override { return _dist(); }

  [[nodiscard]] T mean_angle() const noexcept { return _fit.mean_angle; }
  [[nodiscard]] T kappa() const noexcept { return _fit.kappa; }
  // ‖A2_input − A2_model‖_F; ~0 in 2D (the fit is exact).
  [[nodiscard]] T residual() const noexcept { return _fit.residual; }

private:
  von_mises_a2_distribution(std::tuple<T, T, T> a, engine_type& engine)
      : von_mises_a2_distribution(std::get<0>(a), std::get<1>(a),
                                  std::get<2>(a), engine) {}

  von_mises_a2_distribution(von_mises_a2_fit<T> fit, engine_type& engine)
      : _fit{fit}, _dist{fit.mean_angle, fit.kappa, engine} {}

  von_mises_a2_fit<T> _fit;
  oriented_uniform_distribution<T, Engine> _dist;
};

} // namespace rvegen
