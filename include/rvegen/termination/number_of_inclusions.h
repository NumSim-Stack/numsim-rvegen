#pragma once

#include <array>
#include <cstddef>

#include <numsim-core/input_parameter_controller.h>

#include "../intersection/aabb_overlap.h"
#include "../types.h"
#include "termination_base.h"

namespace rvegen {

// Stop after N inclusions (primaries) accepted.
//
// "primary" = shape whose centre is in the domain box. periodic_generator
// stores wrap copies alongside primaries so subsequent collision checks
// can use the standard non-periodic dispatch; the wraps share storage but
// represent the same physical inclusion as their primary, so we don't
// count them. For non-periodic generators (only_inside, random) every
// accepted shape's centre is in the domain, so the filter is a no-op.
template <typename T = double>
class number_of_inclusions final : public termination_base<T> {
public:
  using shape_vector = typename termination_base<T>::shape_vector;

  explicit number_of_inclusions(std::size_t target) noexcept
      : _target{target}, _domain_box{T{0}, T{0}, T{0}} {}

  // Schema-driven ctor — captures domain_box for the centre-in-domain
  // primary filter.
  number_of_inclusions(parameter_handler_t const& handler,
                       std::array<T, 3> const& domain_box)
      : _target{handler.template get<std::size_t>("target")},
        _domain_box{domain_box} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::size_t>("target").template add<numsim_core::is_required>()
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"number of inclusions (primaries) to place before stopping">>();
    return s;
  }

  [[nodiscard]] bool operator()(shape_vector const& accepted) const override {
    bool const have_box =
        _domain_box[0] > T{0} && _domain_box[1] > T{0};
    if (!have_box) {
      // Direct ctor path or empty box: count everything (legacy behaviour).
      return accepted.size() >= _target;
    }
    std::size_t primary_count = 0;
    for (auto const& shape : accepted) {
      if (is_centre_in_domain(shape->get_middle_point(), _domain_box))
        ++primary_count;
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
  std::array<T, 3> _domain_box;
};

} // namespace rvegen
