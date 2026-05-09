#pragma once

#include <array>

#include <numsim-core/input_parameter_controller.h>

#include "../intersection/aabb_overlap.h"
#include "../types.h"
#include "termination_base.h"

namespace rvegen {

// Stop when the in-domain shapes' total area (2D) or volume (3D) reaches
// the configured fraction of the domain.
//
// "in-domain" means the shape's centre lies inside the domain box. This
// matters for periodic_generator, which inserts wrap copies of shapes
// near box faces — the wraps share the primary's geometry but live with
// their centres outside the domain. Counting wraps would double-count
// the same physical inclusion, so we filter by centre-in-domain.
// For non-periodic generators (only_inside, random) every accepted shape
// has its centre in the domain, so the filter is a no-op.
//
// Dimension is auto-detected: shape_base::area() is zero for 3D shapes,
// shape_base::volume() is zero for 2D shapes — sum whichever is non-zero
// per shape.
template <typename T = double>
class volume_fraction final : public termination_base<T> {
public:
  using value_type = T;
  using shape_vector = typename termination_base<T>::shape_vector;

  // Direct ctor — caller computes domain_size; centre-in-domain filter
  // is unavailable on this path. Use the schema-driven ctor for
  // periodic-correct accounting.
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
    parameter_controller_t s;
    s.template insert<value_type>("target_fraction")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::range<value_type{0}, value_type{1}>>()
        .template add<numsim_core::unit_label<"fraction">>()
        .template add<numsim_core::description_label<"target volume (3D) or area (2D) fraction of inclusions">>();
    return s;
  }

  [[nodiscard]] bool operator()(shape_vector const& accepted) const override {
    value_type sum{};
    bool const have_box =
        _domain_box[0] > value_type{0} && _domain_box[1] > value_type{0};
    for (auto const& shape : accepted) {
      // Filter wrap copies: only count shapes whose centre is in domain.
      if (have_box && !is_centre_in_domain(shape->get_middle_point(),
                                            _domain_box))
        continue;
      const auto v = shape->volume();
      sum += (v > value_type{0}) ? v : shape->area();
    }
    return _domain_size > value_type{0} && sum / _domain_size >= _target;
  }

  [[nodiscard]] value_type target_fraction() const noexcept { return _target; }
  [[nodiscard]] value_type domain_size() const noexcept { return _domain_size; }

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
