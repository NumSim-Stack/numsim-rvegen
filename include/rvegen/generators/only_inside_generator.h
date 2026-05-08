#pragma once

#include <cstddef>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "../intersection/aabb_overlap.h"
#include "../intersection/collision_dispatcher.h"
#include "../types.h"
#include "rve_generator_base.h"

namespace rvegen {

// Place shapes such that each fits entirely inside the RVE domain box and
// does not collide with any previously accepted shape.
//
// Algorithm: round-robin over inputs; for each step, sample a candidate from
// the next input, compute its bounding box, reject if (a) bounding box is
// not contained in [0, Lx] × [0, Ly] × [0, Lz] or (b) bounding box overlaps
// any accepted shape's bounding box. Stops when the termination predicate
// fires or after max_attempts iterations (whichever comes first).
//
// AABB overlap is a conservative collision check — true intersections are
// always rejected; some non-intersections are also rejected (false positives
// reduce achievable packing density). A precise pairwise dispatcher comes in
// a later phase.
template <typename T = double>
class only_inside_generator final : public rve_generator_base<T> {
public:
  using value_type = T;
  using shape_vector = typename rve_generator_base<T>::shape_vector;
  using input_vector = typename rve_generator_base<T>::input_vector;

  only_inside_generator() = default;
  explicit only_inside_generator(std::size_t max_attempts) noexcept
      : _max_attempts{max_attempts} {}

  // Schema-driven ctor.
  explicit only_inside_generator(parameter_handler_t const& handler)
      : _max_attempts{handler.template get<std::size_t>("max_attempts")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::size_t>("max_attempts")
        .template add<numsim_core::is_required>()
        .min(1.0)
        .description("maximum placement attempts before giving up");
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

      // (a) inside the domain box?
      if (!aabb_inside(*cand_bb,
                       domain_box[0], domain_box[1], domain_box[2])) {
        continue;
      }

      // (b) Two-stage collision check against accepted shapes:
      //     Stage 1: AABB overlap (cheap reject — most pairs don't overlap).
      //     Stage 2: collision_dispatcher (precise typed collision_details
      //              for registered pairs; AABB fallback otherwise).
      // The candidate's typeid is invariant across this inner loop, so
      // hoist the outer dispatcher lookup once via bind_lhs.
      // If the user hasn't called rvegen::register_all_collision_pairs(),
      // the bound dispatcher returns AABB results — same conservative
      // behaviour as before.
      bool collides = false;
      auto const bound_dispatcher =
          collision_dispatcher<value_type>::instance().bind_lhs(*candidate);
      for (auto const& other : accepted) {
        auto const* other_bb = other->bounding_box();
        if (!aabb_overlap(*cand_bb, *other_bb)) continue;       // stage 1
        if (bound_dispatcher(*candidate, *other)) {             // stage 2
          collides = true;
          break;
        }
      }
      if (collides) continue;

      accepted.push_back(std::move(candidate));

      if (accepted.size() - last_emit >= opts.emit_every) {
        opts.on_progress({accepted.size(), 0, 0.0});
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
