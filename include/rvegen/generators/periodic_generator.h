#pragma once

#include <cstddef>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "../intersection/aabb_overlap.h"
#include "../intersection/collision_dispatcher.h"
#include "../shapes/periodic_wraps.h"
#include "../types.h"
#include "rve_generator_base.h"

namespace rvegen {

// Place shapes with periodic boundary conditions: a shape near the +x face
// has a periodic image on the −x face, etc. This is the canonical generator
// for FFT homogenization, which assumes a periodic unit cell.
//
// Implementation strategy:
//   1. Sample a candidate anywhere in the box (no inside-only constraint).
//   2. Compute its periodic equivalence class — the primary plus a clone
//      per poking face/corner, each translated to the wrapped position.
//   3. Check the entire class for collision against the accepted vector
//      via the standard non-periodic two-stage dispatch. If any class
//      member collides with anything in accepted, reject the whole class.
//   4. On accept, push every class member into accepted. Subsequent
//      generation iterations then see the wraps as ordinary shapes.
//
// This avoids periodic-aware collision math at runtime — wraps are stored
// explicitly. Volume_fraction filters by is_centre_in_domain so wraps
// don't double-count. Output writers (voxel grids especially) get correct
// periodic behaviour at the box faces because the wraps are physical.
template <typename T = double>
class periodic_generator final : public rve_generator_base<T> {
public:
  using value_type = T;
  using shape_vector = typename rve_generator_base<T>::shape_vector;
  using input_vector = typename rve_generator_base<T>::input_vector;

  periodic_generator() = default;
  explicit periodic_generator(std::size_t max_attempts) noexcept
      : _max_attempts{max_attempts} {}

  explicit periodic_generator(parameter_handler_t const& handler)
      : _max_attempts{handler.template get<std::size_t>("max_attempts")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::size_t>("max_attempts")
        .template add<numsim_core::is_required>()
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"maximum placement attempts before giving up">>();
    return s;
  }

  [[nodiscard]] shape_vector
  compute(input_vector& inputs,
          termination_base<T> const& termination,
          std::array<value_type, 3> const& domain_box,
          progress_options const& opts = {}) override {
    shape_vector accepted;
    if (inputs.empty()) return accepted;

    std::size_t attempts = 0;
    std::size_t input_idx = 0;
    std::size_t last_emit = 0;

    while (!termination(accepted) && attempts < _max_attempts) {
      if (opts.cancel()) return accepted;
      ++attempts;
      auto& input = *inputs[input_idx];
      input_idx = (input_idx + 1) % inputs.size();

      auto primary = input.new_shape();
      primary->make_bounding_box();
      auto const* prim_bb = primary->bounding_box();
      if (prim_bb == nullptr) continue;

      // Self-feasibility: a shape too large to fit in the unit cell
      // has its own periodic image overlapping itself.
      if (self_periodic_collides(*prim_bb, domain_box)) continue;

      // Build the periodic equivalence class — primary + wraps.
      auto wraps = generate_periodic_wraps(*primary, domain_box);

      // Check every class member vs every accepted shape. The accepted
      // vector already contains all previously-inserted primaries AND
      // their wraps, so plain (non-periodic) two-stage dispatch suffices.
      bool collides = false;
      auto check_member = [&](shape_base<value_type> const& m) {
        auto const bound =
            collision_dispatcher<value_type>::instance().bind_lhs(m);
        auto const* mbb = m.bounding_box();
        for (auto const& other : accepted) {
          if (!aabb_overlap(*mbb, *other->bounding_box())) continue;
          if (bound(m, *other)) return true;
        }
        return false;
      };

      if (check_member(*primary)) continue;
      for (auto const& wrap : wraps) {
        if (check_member(*wrap)) { collides = true; break; }
      }
      if (collides) continue;

      // Accept: push primary first, then wraps. Centre-in-domain
      // accounting (volume_fraction termination) distinguishes them
      // by their middle_point, not by insertion role.
      accepted.push_back(std::move(primary));
      for (auto& w : wraps) accepted.push_back(std::move(w));

      if (accepted.size() - last_emit >= opts.emit_every) {
        opts.on_progress({accepted.size(),
                          termination.target_count(),
                          current_volume_fraction(accepted, domain_box)});
        last_emit = accepted.size();
      }
    }

    return accepted;
  }

  [[nodiscard]] std::size_t max_attempts() const noexcept { return _max_attempts; }
  void set_max_attempts(std::size_t n) noexcept { _max_attempts = n; }

private:
  // True if any non-zero periodic image of `bb` overlaps `bb` itself —
  // i.e. the shape is so large that its periodic copy collides with the
  // original. Such a candidate cannot be placed in this domain.
  static bool self_periodic_collides(bounding_box_base<value_type> const& bb,
                                      std::array<value_type, 3> const& domain_box) noexcept {
    const bool is_3d = domain_box[2] > value_type{0};
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          if (!is_3d && dz != 0) continue;
          if (dx == 0 && dy == 0 && dz == 0) continue;
          const std::array<value_type, 3> shift{
              static_cast<value_type>(dx) * domain_box[0],
              static_cast<value_type>(dy) * domain_box[1],
              static_cast<value_type>(dz) * domain_box[2]};
          if (aabb_overlap_shifted(bb, bb, shift)) return true;
        }
      }
    }
    return false;
  }

  std::size_t _max_attempts{100'000};
};

} // namespace rvegen
