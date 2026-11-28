#pragma once

#include <array>
#include <optional>

#include <numsim-core/input_parameter_controller.h>

#include "../shapes/periodic_wraps.h"
#include "../types.h"
#include "termination_base.h"

namespace rvegen {

// Stop when the total area (2D) or volume (3D) of the shapes that count as
// material of the RVE reaches the configured fraction of the domain.
//
// Which shapes count is decided in one place, is_counted_in_rve()
// (shapes/periodic_wraps.h): a periodic copy never does (its primary
// stands for the whole inclusion), a primary with copies always does, and
// a shape that crosses a face without being wrapped (random_generator)
// counts when its centre is inside the box. The `count_all` flag
// therefore only affects number_of_inclusions.
//
// Dimension is auto-detected: shape_base::area() is zero for 3D shapes,
// shape_base::volume() is zero for 2D shapes — sum whichever is non-zero
// per shape.
template <typename T = double>
class volume_fraction final : public termination_base<T> {
public:
  using value_type = T;
  using shape_vector = typename termination_base<T>::shape_vector;

  // Direct ctor — caller computes domain_size. Periodic copies are not
  // counted on this path either; the estimate for shapes that cross a face
  // without copies needs the box, i.e. the schema-driven ctor.
  volume_fraction(value_type target_fraction, value_type domain_size) noexcept
      : _target{target_fraction},
        _domain_size{domain_size} {}

  // Target and box, for C++ callers: the same counting as the schema-driven
  // ctor (shapes without copies count when their centre is in the box).
  volume_fraction(value_type target_fraction,
                  domain<value_type> const& domain_box)
      : _target{target_fraction},
        _domain_box{domain_box},
        _domain_size{domain_box.measure()} {}

  // Schema-driven ctor — the domain size is the box's area (2D) or
  // volume (3D).
  volume_fraction(parameter_handler_t const& handler,
                  domain<value_type> const& domain_box)
      : _target{handler.template get<value_type>("target_fraction")},
        _domain_box{domain_box},
        _domain_size{domain_box.measure()} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t p;
    p.template required<value_type>("target_fraction")
        .template add<numsim_core::range<value_type{0}, value_type{1}>>()
        .template add<numsim_core::unit_label<"fraction">>()
        .template add<numsim_core::description_label<"target volume (3D) or area (2D) fraction of inclusions">>();
    return p;
  }

  [[nodiscard]] bool operator()(shape_vector const& accepted) const override {
    return is_met(accepted, false);
  }

  [[nodiscard]] bool is_met(shape_vector const& accepted,
                            bool /*count_all*/) const override {
    return _domain_size > value_type{0} && fraction(accepted) >= _target;
  }

  [[nodiscard]] bool is_met(shape_vector const& accepted, bool /*count_all*/,
                            termination_tally<T>& tally) const override {
    if (!(_domain_size > value_type{0})) return false;
    if (accepted.size() < tally.seen) tally = {};   // not the list it knew
    for (; tally.seen < accepted.size(); ++tally.seen) {
      auto const& shape = accepted[tally.seen];
      if (counted(*shape)) tally.sum += shape->measure();
    }
    return tally.sum / _domain_size >= _target;
  }

  [[nodiscard]] std::string kind() const override { return "volume_fraction"; }
  [[nodiscard]] double target_value() const noexcept override {
    return static_cast<double>(_target);
  }
  [[nodiscard]] double achieved_value(shape_vector const& accepted,
                                      bool /*count_all*/) const override {
    return static_cast<double>(fraction(accepted));
  }

  // The fraction is_met() compares with the target (0 without a domain).
  [[nodiscard]] value_type fraction(shape_vector const& accepted) const {
    if (!(_domain_size > value_type{0})) return value_type{0};
    value_type sum{};
    for (auto const& shape : accepted) {
      if (counted(*shape)) sum += shape->measure();
    }
    return sum / _domain_size;
  }

  // Whether a shape adds to the fraction. (`count_all` is deliberately
  // ignored.) Without a box (direct construction) only the periodic copies
  // can be told apart.
  [[nodiscard]] bool counted(shape_base<T> const& shape) const {
    return _domain_box ? is_counted_in_rve(shape, *_domain_box)
                       : shape.is_primary();
  }

  [[nodiscard]] value_type target_fraction() const noexcept { return _target; }
  [[nodiscard]] value_type domain_size() const noexcept { return _domain_size; }

  // Surface the configured fraction for the generator's progress callback —
  // Tessera's progress bar reads (current_fraction / target_fraction) for
  // a percentage display when this is non-zero.
  [[nodiscard]] double target_volume_fraction() const noexcept override {
    return static_cast<double>(_target);
  }

private:
  value_type _target;
  std::optional<domain<value_type>> _domain_box;   // empty: direct ctor
  value_type _domain_size;
};

} // namespace rvegen
