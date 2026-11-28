#pragma once

#include <array>
#include <cstddef>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "termination_base.h"

namespace rvegen {

// Stop after N inclusions (primaries) accepted.
//
// periodic_generator stores wrap copies alongside primaries so subsequent
// collision checks can use the standard non-periodic dispatch; a copy is
// the same physical inclusion as its primary and is not counted
// (shape_base::is_primary). The generators without copies
// (only_inside_generator, random_generator) call
// is_met(accepted, /*count_all=*/true): every accepted shape is a primary
// there, so the count is the vector's size, without a pass over it.
template <typename T = double>
class number_of_inclusions final : public termination_base<T> {
public:
  using shape_vector = typename termination_base<T>::shape_vector;

  explicit number_of_inclusions(std::size_t target) noexcept
      : _target{target} {}

  // Schema-driven ctor. Every termination is built with the domain box;
  // counting inclusions does not need it (shape_base::is_primary).
  number_of_inclusions(parameter_handler_t const& handler,
                       std::array<T, 3> const& /*domain_box*/)
      : _target{handler.template get<std::size_t>("target")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t p;
    p.template required<std::size_t>("target")
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"number of inclusions (primaries) to place before stopping">>();
    return p;
  }

  [[nodiscard]] bool operator()(shape_vector const& accepted) const override {
    return is_met(accepted, false);
  }

  [[nodiscard]] bool is_met(shape_vector const& accepted,
                            bool count_all) const override {
    if (count_all) return accepted.size() >= _target;
    std::size_t primary_count = 0;
    for (auto const& shape : accepted) {
      if (shape->is_primary()) ++primary_count;
    }
    return primary_count >= _target;
  }

  [[nodiscard]] std::size_t target() const noexcept { return _target; }
  void set_target(std::size_t n) noexcept { _target = n; }

  // Surface the configured count for the generator's progress callback —
  // Tessera's progress bar displays "X / N" when this is non-zero.
  [[nodiscard]] std::size_t target_count() const noexcept override {
    return _target;
  }

private:
  std::size_t _target;
};

} // namespace rvegen
