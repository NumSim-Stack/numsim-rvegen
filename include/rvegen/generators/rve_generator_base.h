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

} // namespace rvegen
