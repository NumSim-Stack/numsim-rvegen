#pragma once

#include <array>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "termination_base.h"

namespace rvegen {

// "Never stop by itself" termination. The generator's max_attempts is the
// only stopper, so this gives "pack as densely as possible until the
// algorithm gives up." Useful when you don't have a target inclusion count
// or volume fraction in mind.
template <typename T = double>
class until_full final : public termination_base<T> {
public:
  using shape_vector = typename termination_base<T>::shape_vector;

  until_full() = default;

  until_full(parameter_handler_t const& /*handler*/,
             std::array<T, 3> const& /*domain_box*/) noexcept {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return parameter_controller_t{};
  }

  [[nodiscard]] bool operator()(shape_vector const& /*accepted*/) const override {
    return false;
  }
};

} // namespace rvegen
