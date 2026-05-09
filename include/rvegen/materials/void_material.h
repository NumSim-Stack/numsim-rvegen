#pragma once

// Void / pore — zero stiffness everywhere. Used in porosity studies
// (foams, cracks, voids in metallic / composite microstructures).
// FFT homogenization solvers handle this via a numerical floor on the
// stiffness; the consumer is responsible for that adjustment.

#include "../types.h"
#include "material_base.h"

namespace rvegen {

template <typename T = double>
class void_material final : public material_base<T> {
public:
  using value_type = T;

  void_material() = default;

  // Schema-driven ctor — schema is empty, so the handler is unused.
  // Accepted for registry uniformity.
  explicit void_material(parameter_handler_t const&) noexcept {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return parameter_controller_t{};
  }

  [[nodiscard]] stiffness_tensor<value_type> stiffness() const override {
    return stiffness_tensor<value_type>{};   // all zeros
  }

  [[nodiscard]] std::string_view kind() const noexcept override {
    return "void";
  }
};

} // namespace rvegen
