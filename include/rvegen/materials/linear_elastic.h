#pragma once

// Linear-elastic isotropic material parameterised by Young's modulus E
// and Poisson's ratio ν. The Voigt stiffness matrix follows the
// standard 3D isotropic form:
//
//   λ + 2μ   λ      λ      0   0   0
//   λ       λ+2μ   λ      0   0   0
//   λ       λ      λ+2μ   0   0   0
//   0       0      0       μ   0   0
//   0       0      0       0   μ   0
//   0       0      0       0   0   μ
//
// where λ = E·ν / ((1+ν)(1-2ν)) and μ = E / (2(1+ν)).
//
// 2D plane-stress / plane-strain reductions belong in a separate
// type — same E/ν but a different stiffness layout.

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "material_base.h"

namespace rvegen {

template <typename T = double>
class linear_elastic final : public material_base<T> {
public:
  using value_type = T;

  linear_elastic(value_type youngs_modulus, value_type poisson_ratio) noexcept
      : _E{youngs_modulus}, _nu{poisson_ratio} {}

  explicit linear_elastic(parameter_handler_t const& handler)
      : linear_elastic(handler.template get<value_type>("E"),
                       handler.template get<value_type>("nu")) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<value_type>("E")
        .template add<numsim_core::is_required>()
        .template add<min_only<value_type{0}>>()
        .template add<numsim_core::unit_label<"Pa">>()
        .template add<numsim_core::description_label<"Young's modulus (must be positive)">>();
    s.template insert<value_type>("nu")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::range<value_type{-1}, value_type{0.5}>>()
        .template add<numsim_core::description_label<"Poisson's ratio (-1 < nu < 0.5 for stable isotropic material)">>();
    return s;
  }

  [[nodiscard]] stiffness_tensor<value_type> stiffness() const override {
    stiffness_tensor<value_type> S{};
    const value_type lambda = _E * _nu / ((value_type{1} + _nu)
                                           * (value_type{1} - value_type{2} * _nu));
    const value_type mu = _E / (value_type{2} * (value_type{1} + _nu));
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        S.c[i][j] = lambda + (i == j ? value_type{2} * mu : value_type{0});
      }
    }
    for (int i = 3; i < 6; ++i) S.c[i][i] = mu;
    return S;
  }

  [[nodiscard]] std::string_view kind() const noexcept override {
    return "linear_elastic";
  }

  [[nodiscard]] value_type youngs_modulus() const noexcept { return _E; }
  [[nodiscard]] value_type poisson_ratio() const noexcept { return _nu; }

private:
  value_type _E;
  value_type _nu;
};

} // namespace rvegen
