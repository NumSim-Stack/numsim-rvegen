#pragma once

#include <cstddef>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "../intersection/aabb_grid.h"
#include "../intersection/aabb_overlap.h"
#include "../intersection/collision_dispatcher.h"
#include "../types.h"
#include "placement_gap.h"
#include "rve_generator_base.h"

namespace rvegen {

// Place shapes anywhere — no inside-box constraint, no periodic wrap.
// Shapes can poke outside the RVE domain. Useful as a baseline / sanity
// generator and for non-RVE bulk-packing applications.
//
// Collision: two-stage -- grid broad phase on AABBs, then the precise
// collision_dispatcher on the AABB-overlapping neighbours.
//
// Accounting: shapes may be centred outside the box and there are no
// periodic wrap copies, so every accepted shape is a primary. The
// termination predicate is therefore evaluated with count_all = true
// (see termination_base::is_met): number_of_inclusions counts every shape,
// while volume_fraction (and the progress volume fraction) estimates the
// material in the box: a shape that crosses a face is not wrapped here, so
// it counts when its centre is inside (is_counted_in_rve).
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
      : _max_attempts{handler.template get<std::size_t>("max_attempts")},
        _gap{handler} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t p;
    p.template required<std::size_t>("max_attempts")
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"maximum placement attempts before giving up">>();
    placement_gap<T>::declare(p);
    return p;
  }

  [[nodiscard]] shape_vector
  compute(input_vector& inputs,
          termination_base<T> const& termination,
          domain<value_type> const& domain_box,
          progress_options const& opts = {}) override {
    this->clear_report();
    this->report_gaps(static_cast<double>(_gap.min_gap),
                      static_cast<double>(_gap.min_gap_relative),
                      static_cast<double>(_gap.wall_gap));
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
    gap_neighbourhood<value_type> placed{_gap};
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

      // Two-stage check, with the dispatcher's outer lookup hoisted
      // (see only_inside_generator for the rationale). A shape may cross
      // a face here; with a wall gap it must then reach that far through.
      if (_gap.wall_gap > value_type{0} &&
          !boundary_clear(*candidate, domain_box, _gap.wall_gap, true)) {
        continue;
      }
      const value_type size = _gap.size_of(*candidate);
      if (placed.too_close(*candidate, size, accepted)) continue;

      placed.insert(*candidate, size);
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

  // Minimum distances between inclusions and to the faces (#111).
  [[nodiscard]] placement_gap<T> const& gap() const noexcept { return _gap; }
  void set_gap(placement_gap<T> const& gap) noexcept { _gap = gap; }

private:
  std::size_t _max_attempts{100'000};
  placement_gap<T> _gap{};
};

} // namespace rvegen
