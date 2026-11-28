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
                       domain<T> const& /*domain_box*/)
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
    return count(accepted, count_all) >= _target;
  }

  [[nodiscard]] bool is_met(shape_vector const& accepted, bool count_all,
                            termination_tally<T>& tally) const override {
    if (count_all) return accepted.size() >= _target;
    if (accepted.size() < tally.seen) tally = {};   // not the list it knew
    for (; tally.seen < accepted.size(); ++tally.seen) {
      if (accepted[tally.seen]->is_primary()) ++tally.count;
    }
    return tally.count >= _target;
  }

  [[nodiscard]] std::string kind() const override {
    return "number_of_inclusions";
  }
  [[nodiscard]] double target_value() const noexcept override {
    return static_cast<double>(_target);
  }
  [[nodiscard]] double achieved_value(shape_vector const& accepted,
                                      bool count_all) const override {
    return static_cast<double>(count(accepted, count_all));
  }

  [[nodiscard]] std::size_t target() const noexcept { return _target; }
  void set_target(std::size_t n) noexcept { _target = n; }

  // Surface the configured count for the generator's progress callback —
  // Tessera's progress bar displays "X / N" when this is non-zero.
  [[nodiscard]] std::size_t target_count() const noexcept override {
    return _target;
  }

private:
  // Inclusions in `accepted`: every shape with count_all, else the
  // primaries (periodic copies are not counted again).
  [[nodiscard]] static std::size_t count(shape_vector const& accepted,
                                         bool count_all) noexcept {
    if (count_all) return accepted.size();
    std::size_t primary_count = 0;
    for (auto const& shape : accepted) {
      if (shape->is_primary()) ++primary_count;
    }
    return primary_count;
  }

  std::size_t _target;
};

} // namespace rvegen
