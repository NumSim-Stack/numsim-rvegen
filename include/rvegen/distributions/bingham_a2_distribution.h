#pragma once

// 3D axisymmetric Bingham (Watson) direction sampler parameterised by
// the Advani-Tucker orientation tensor A2 (a11, a22, a33, a12, a13, a23)
// instead of (mean_axis, kappa) (#6). The tensor is fitted once at
// construction (orientation_tensor_fit.h) and sampling is delegated to
// bingham_distribution, so drawn unit vectors x reproduce ⟨x xᵀ⟩ = A2.
//
// Non-axisymmetric A2 (three distinct eigenvalues) cannot be matched
// exactly by the two-parameter Watson family: it is projected onto the
// closest axisymmetric tensor in A2's eigenframe (prolate or oblate,
// whichever is closer), and `residual()` reports the Frobenius distance
// between input and fit. Fully aligned tensors (e.g. diag(1, 0, 0)) are
// clamped to bingham_distribution's supported κ range; the clamp also
// shows up in `residual()`.
//
// Registered as "bingham_a2" (separate name rather than optional fields
// on "bingham"; see von_mises_a2_distribution.h for the rationale).

#include <array>
#include <functional>
#include <random>
#include <tuple>

#include <Mathematics/Vector3.h>
#include <numsim-core/input_parameter_controller.h>

#include "../schema/field_list.h"
#include "../types.h"
#include "bingham_distribution.h"
#include "direction_distribution_base.h"
#include "orientation_tensor_fit.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
class bingham_a2_distribution : public direction_distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;
  using vector_type = gte::Vector<3, T>;

  bingham_a2_distribution(orientation_tensor_3d<T> const& a2,
                          engine_type& engine)
      : bingham_a2_distribution(fit_watson_a2<T>(a2), engine) {}

  using fields = field_list<
      field<"a11", T, true,
            numsim_core::description_label<"orientation tensor A2 component <x²>; a11 + a22 + a33 = 1">>,
      field<"a22", T, true,
            numsim_core::description_label<"orientation tensor A2 component <y²>; a11 + a22 + a33 = 1">>,
      field<"a33", T, true,
            numsim_core::description_label<"orientation tensor A2 component <z²>; a11 + a22 + a33 = 1">>,
      field<"a12", T, true,
            numsim_core::description_label<"orientation tensor A2 component <x·y>">>,
      field<"a13", T, true,
            numsim_core::description_label<"orientation tensor A2 component <x·z>">>,
      field<"a23", T, true,
            numsim_core::description_label<"orientation tensor A2 component <y·z>">>>;

  bingham_a2_distribution(parameter_handler_t const& handler,
                          engine_type& engine)
      : bingham_a2_distribution(fields::extract(handler), engine) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  [[nodiscard]] std::array<T, 3> operator()() override { return _dist(); }
  [[nodiscard]] vector_type sample() { return _dist.sample(); }

  [[nodiscard]] std::array<T, 3> const& mean_axis() const noexcept {
    return _fit.axis;
  }
  [[nodiscard]] T kappa() const noexcept { return _fit.kappa; }
  // ‖A2_input − A2_model‖_F: 0 for axisymmetric input, > 0 when the
  // input had to be projected onto the Watson family.
  [[nodiscard]] T residual() const noexcept { return _fit.residual; }

private:
  bingham_a2_distribution(std::tuple<T, T, T, T, T, T> a,
                          engine_type& engine)
      : bingham_a2_distribution(
            orientation_tensor_3d<T>{std::get<0>(a), std::get<1>(a),
                                     std::get<2>(a), std::get<3>(a),
                                     std::get<4>(a), std::get<5>(a)},
            engine) {}

  bingham_a2_distribution(watson_a2_fit<T> fit, engine_type& engine)
      : _fit{fit}, _dist{fit.axis, fit.kappa, engine} {}

  watson_a2_fit<T> _fit;
  bingham_distribution<T, Engine> _dist;
};

} // namespace rvegen
