#pragma once

#include <cstddef>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "../intersection/aabb_overlap.h"
#include "../intersection/collision_dispatcher.h"
#include "../types.h"
#include "rve_generator_base.h"

namespace rvegen {

// Place shapes anywhere — no inside-box constraint, no periodic wrap.
// Shapes can poke outside the RVE domain. Useful as a baseline / sanity
// generator and for non-RVE bulk-packing applications.
//
// Collision: direct AABB overlap against accepted shapes.
template <typename T = double>
class random_generator final : public rve_generator_base<T> {
public:
  using value_type = T;
  using shape_vector = typename rve_generator_base<T>::shape_vector;
  using input_vector = typename rve_generator_base<T>::input_vector;

  random_generator() = default;
  explicit random_generator(std::size_t max_attempts) noexcept
      : _max_attempts{max_attempts} {}

  explicit random_generator(parameter_handler_t const& handler)
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

      auto candidate = input.new_shape();
      candidate->make_bounding_box();
      auto const* cand_bb = candidate->bounding_box();
      if (cand_bb == nullptr) continue;

      // Two-stage check, with the dispatcher's outer lookup hoisted
      // (see only_inside_generator for the rationale).
      bool collides = false;
      auto const bound_dispatcher =
          collision_dispatcher<value_type>::instance().bind_lhs(*candidate);
      for (auto const& other : accepted) {
        auto const* other_bb = other->bounding_box();
        if (!aabb_overlap(*cand_bb, *other_bb)) continue;
        if (bound_dispatcher(*candidate, *other)) {
          collides = true;
          break;
        }
      }
      if (collides) continue;

      accepted.push_back(std::move(candidate));

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
  std::size_t _max_attempts{100'000};
};

} // namespace rvegen
