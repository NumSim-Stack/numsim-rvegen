#pragma once

#include <cstddef>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "../intersection/aabb_grid.h"
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
    parameter_controller_t p;
    p.template required<std::size_t>("max_attempts")
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"maximum placement attempts before giving up">>();
    return p;
  }

  [[nodiscard]] shape_vector
  compute(input_vector& inputs,
          termination_base<T> const& termination,
          domain<value_type> const& domain_box,
          progress_options const& opts = {}) override {
    this->clear_report();
    shape_vector accepted;
    if (inputs.empty()) {
      this->record_report(termination, accepted, true, 0, _max_attempts,
                          stop_reason::no_inputs);
      return accepted;
    }

    std::size_t attempts = 0;
    std::size_t input_idx = 0;
    std::size_t last_emit = 0;

    // Broad phase: bucket grid over the accepted AABBs (#83). The
    // termination predicate only depends on `accepted`, so it is
    // re-evaluated after each accept rather than on every attempt.
    // Every accepted shape is inside the box (hence a primary), so the
    // primaries filter is skipped (count_all): O(1) for number_of_inclusions.
    aabb_grid<value_type> grid;
    bool done = termination.is_met(accepted, true);

    while (!done && attempts < _max_attempts) {
      if (opts.cancel()) {
        this->record_report(termination, accepted, true, attempts,
                            _max_attempts, stop_reason::cancelled);
        return accepted;
      }
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
      //     Stage 1: grid broad phase -> AABB-overlapping neighbours only.
      //     Stage 2: collision_dispatcher (precise typed collision_details
      //              for registered pairs; AABB fallback otherwise).
      // The candidate's typeid is invariant across this inner loop, so
      // hoist the outer dispatcher lookup once via bind_lhs.
      // If the user hasn't called rvegen::register_all_collision_pairs(),
      // the bound dispatcher returns AABB results — same conservative
      // behaviour as before.
      auto const bound_dispatcher =
          collision_dispatcher<value_type>::instance().bind_lhs(*candidate);
      const bool collides =
          grid.any_overlap(*cand_bb, [&](std::size_t i) {
            return bound_dispatcher(*candidate, *accepted[i]);
          });
      if (collides) continue;

      grid.insert(*cand_bb);
      accepted.push_back(std::move(candidate));

      if (accepted.size() - last_emit >= opts.emit_every) {
        opts.on_progress({accepted.size(),
                          termination.target_count(),
                          current_volume_fraction(accepted, domain_box)});
        last_emit = accepted.size();
      }
      done = termination.is_met(accepted, true);
    }

    this->record_report(termination, accepted, true, attempts,
                        _max_attempts,
                        done ? stop_reason::reached : stop_reason::max_attempts);
    return accepted;
  }

  [[nodiscard]] std::size_t max_attempts() const noexcept { return _max_attempts; }
  void set_max_attempts(std::size_t n) noexcept { _max_attempts = n; }

private:
  std::size_t _max_attempts{100'000};
};

} // namespace rvegen
