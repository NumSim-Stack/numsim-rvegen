#pragma once

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "../inputs/shape_input_base.h"
#include "../shapes/shape_base.h"
#include "../termination/termination_base.h"

namespace rvegen {

// Snapshot of generator progress, passed to the user's callback every
// `emit_every` accepted shapes. Fields not exposed by the termination
// predicate (e.g. shapes_target) are zero-valued.
struct progress_info {
  std::size_t shapes_placed{0};
  std::size_t shapes_target{0};   // 0 if termination doesn't expose a target
  double      volume_fraction{0}; // 0 if not tracked
};

// Optional callbacks for long-running compute() calls. Default-constructed
// behaviour is no-op + never-cancel — i.e. identical to the pre-hooks API,
// so the CLI and existing tests build unchanged.
//
// `cancel`      — polled once per placement; returning true causes compute()
//                 to return the partial-shape vector immediately.
// `on_progress` — called every `emit_every` accepted shapes with a snapshot.
// `emit_every`  — minimum number of placements between successive callbacks.
struct progress_options {
  std::function<bool()>              cancel       = []{ return false; };
  std::function<void(progress_info)> on_progress  = [](progress_info){};
  std::size_t                        emit_every   = 100;
};

// Polymorphic placement strategy. Concrete implementations:
//   * only_inside_generator                  — shapes fully inside box
//   * random_generator                       — shapes anywhere (can poke out)
//   * periodic_generator                     — shapes wrap across box faces
//   * only_inside_samepack_heuristic         — packing heuristic + only-inside
//
// Each strategy is a separate class so the registry can pick by JSON name.
template <typename T>
class rve_generator_base {
public:
  using value_type = T;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;
  using input_vector = std::vector<std::unique_ptr<shape_input_base<T>>>;

  rve_generator_base() = default;
  rve_generator_base(rve_generator_base const&) = default;
  rve_generator_base(rve_generator_base&&) noexcept = default;
  rve_generator_base& operator=(rve_generator_base const&) = default;
  rve_generator_base& operator=(rve_generator_base&&) noexcept = default;
  virtual ~rve_generator_base() = default;

  // Run the strategy. Returns the accepted-shape vector. Inputs are owned by
  // the caller; the generator only borrows them. The termination predicate
  // is consulted between attempts.
  //
  // The optional `opts` argument provides progress reporting and cooperative
  // cancellation. Default-constructed `opts` matches the pre-hooks behaviour
  // (no callbacks, no cancellation) so existing call sites remain unchanged.
  [[nodiscard]] virtual shape_vector
  compute(input_vector& inputs,
          termination_base<T> const& termination,
          std::array<value_type, 3> const& domain_box,
          progress_options const& opts = {}) = 0;
};

// Snapshot the current in-domain volume / area fraction. Called from the
// concrete generators once per progress emission (so at most every
// `emit_every` placements — bounded O(N) cost). Filters wrap copies by
// the centre-in-domain test, matching how volume_fraction termination
// accounts: counting wrap copies would double-report a single physical
// inclusion. For random_generator (no domain constraint at all) the
// fraction is "in-domain extent / domain extent", which still moves
// monotonically and is meaningful as a progress indicator.
template <typename T>
[[nodiscard]] inline double current_volume_fraction(
    std::vector<std::unique_ptr<shape_base<T>>> const& accepted,
    std::array<T, 3> const& domain_box) noexcept {
  const bool is_3d = domain_box[2] > T{0};
  const double dom_extent = is_3d
      ? static_cast<double>(domain_box[0]) *
        static_cast<double>(domain_box[1]) *
        static_cast<double>(domain_box[2])
      : static_cast<double>(domain_box[0]) *
        static_cast<double>(domain_box[1]);
  if (dom_extent <= 0.0) return 0.0;

  double sum = 0.0;
  for (auto const& shape : accepted) {
    if (!shape) continue;
    const auto centre = shape->get_middle_point();
    const bool in_domain =
        centre[0] >= T{0} && centre[0] < domain_box[0] &&
        centre[1] >= T{0} && centre[1] < domain_box[1] &&
        (!is_3d || (centre[2] >= T{0} && centre[2] < domain_box[2]));
    if (!in_domain) continue;
    const auto v = shape->volume();
    sum += static_cast<double>((v > T{0}) ? v : shape->area());
  }
  return sum / dom_extent;
}

} // namespace rvegen
