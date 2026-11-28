#pragma once

#include <array>

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
        _domain_box{T{0}, T{0}, T{0}},
        _domain_size{domain_size} {}

  // Schema-driven ctor — derives domain_size from the box dimensions.
  // domain_box[2] == 0 means 2D RVE → use Lx*Ly; otherwise Lx*Ly*Lz.
  volume_fraction(parameter_handler_t const& handler,
                  std::array<value_type, 3> const& domain_box)
      : _target{handler.template get<value_type>("target_fraction")},
        _domain_box{domain_box},
        _domain_size{compute_domain_size(domain_box)} {}

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
    value_type sum{};
    bool const have_box =
        _domain_box[0] > value_type{0} && _domain_box[1] > value_type{0};
    for (auto const& shape : accepted) {
      // (`count_all` is deliberately ignored.) Without a box (direct
      // construction) only the periodic copies can be told apart.
      if (have_box ? !is_counted_in_rve(*shape, _domain_box)
                   : !shape->is_primary())
        continue;
      const auto v = shape->volume();
      sum += (v > value_type{0}) ? v : shape->area();
    }
    return _domain_size > value_type{0} && sum / _domain_size >= _target;
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
  static constexpr value_type compute_domain_size(
      std::array<value_type, 3> const& box) noexcept {
    auto size = box[0] * box[1];
    if (box[2] > value_type{0}) size *= box[2];
    return size;
  }

  value_type _target;
  std::array<value_type, 3> _domain_box;
  value_type _domain_size;
};

} // namespace rvegen
