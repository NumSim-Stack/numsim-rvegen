#pragma once

// 3D axisymmetric Bingham (Watson) direction sampler parameterised by
// the Advani-Tucker orientation tensor A2 (a11, a22, a33, a12, a13, a23)
// instead of (mean_axis, kappa) (#6). The tensor is fitted once at
// construction (orientation_tensor_fit.h) and sampling is delegated to
// bingham_distribution, so drawn unit vectors x reproduce ⟨x xᵀ⟩ = A2.
//
// Only axisymmetric A2 (two equal eigenvalues) is reproduced exactly by
// the two-parameter Watson family. A tensor with three distinct
// eigenvalues would have to be projected onto the closest axisymmetric
// tensor in A2's eigenframe (prolate or oblate, whichever is closer), which
// silently loses the split between the two minor eigenvalues. Therefore
// construction throws std::invalid_argument when the Frobenius fit
// residual (trace-normalised input) exceeds `default_projection_tolerance`
// (1e-2) unless `allow_projection` is true (JSON key "allow_projection",
// default false), in which case the projection is used and `residual()`
// reports the distance. Fully aligned tensors (e.g. diag(1, 0, 0)) are
// clamped to bingham_distribution's supported κ range [−1e10, 1e10]; the
// clamp leaves a residual of ~1e-10.
//
// Registered as "bingham_a2" (separate name rather than optional fields
// on "bingham"; see von_mises_a2_distribution.h for the rationale).

#include <array>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>

#include <Mathematics/Vector3.h>
#include <numsim-core/input_parameter_controller.h>

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

  // Residual above which a non-axisymmetric A2 is rejected (Frobenius,
  // trace-normalised tensor) unless projection is explicitly allowed.
  static constexpr T default_projection_tolerance = T{1e-2};

  bingham_a2_distribution(orientation_tensor_3d<T> const& a2,
                          engine_type& engine, bool allow_projection = false)
      : bingham_a2_distribution(checked_fit(a2, allow_projection), engine) {}

  bingham_a2_distribution(parameter_handler_t const& handler,
                          engine_type& engine)
      : bingham_a2_distribution(
            orientation_tensor_3d<T>{handler.template get<T>("a11"),
                                     handler.template get<T>("a22"),
                                     handler.template get<T>("a33"),
                                     handler.template get<T>("a12"),
                                     handler.template get<T>("a13"),
                                     handler.template get<T>("a23")},
            engine,
            handler.contains("allow_projection")
                ? handler.template get<bool>("allow_projection")
                : false) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t p;
    p.template required<T>("a11")
        .template add<numsim_core::description_label<"orientation tensor A2 component <x²>; a11 + a22 + a33 must equal 1 within 1e-6. Only axisymmetric A2 (two equal eigenvalues) is reproduced exactly; see allow_projection">>();
    p.template required<T>("a22")
        .template add<numsim_core::description_label<"orientation tensor A2 component <y²>; a11 + a22 + a33 must equal 1 within 1e-6">>();
    p.template required<T>("a33")
        .template add<numsim_core::description_label<"orientation tensor A2 component <z²>; a11 + a22 + a33 must equal 1 within 1e-6">>();
    p.template required<T>("a12")
        .template add<numsim_core::description_label<"orientation tensor A2 component <x·y>">>();
    p.template required<T>("a13")
        .template add<numsim_core::description_label<"orientation tensor A2 component <x·z>">>();
    p.template required<T>("a23")
        .template add<numsim_core::description_label<"orientation tensor A2 component <y·z>">>();
    // Optional — default false.
    p.template insert<bool>("allow_projection")
        .template add<numsim_core::description_label<"default false: a non-axisymmetric A2 (three distinct eigenvalues; only two equal eigenvalues are reproduced exactly) is rejected when the fit residual exceeds 1e-2. true: project it onto the closest axisymmetric (Watson) tensor instead">>();
    return p;
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
  static watson_a2_fit<T> checked_fit(orientation_tensor_3d<T> const& a2,
                                      bool allow_projection) {
    const auto fit = fit_watson_a2<T>(a2);
    if (!allow_projection && !(fit.residual <= default_projection_tolerance)) {
      throw std::invalid_argument{
          "bingham_a2: the orientation tensor is not axisymmetric: the "
          "closest axisymmetric (Watson) tensor differs by a Frobenius "
          "residual of " +
          std::to_string(static_cast<double>(fit.residual)) +
          " (> " +
          std::to_string(static_cast<double>(default_projection_tolerance)) +
          "). Only axisymmetric A2 (two equal eigenvalues) is reproduced "
          "exactly; set \"allow_projection\": true (C++: pass "
          "allow_projection = true) to project onto the closest "
          "axisymmetric tensor instead"};
    }
    return fit;
  }

  bingham_a2_distribution(watson_a2_fit<T> fit, engine_type& engine)
      : _fit{fit}, _dist{fit.axis, fit.kappa, engine} {}

  watson_a2_fit<T> _fit;
  bingham_distribution<T, Engine> _dist;
};

} // namespace rvegen
