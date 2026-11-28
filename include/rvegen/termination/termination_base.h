#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "../shapes/shape_base.h"

namespace rvegen {

// Polymorphic stop predicate for the generator's accept loop.
// operator() returns true to halt generation. Common implementations:
//   * number_of_inclusions: stop after N accepted shapes
//   * volume_fraction:      stop after sum-of-volumes >= target ratio of box
//   * max_iterations:       stop after N attempts (success or failure)
template <typename T>
class termination_base {
public:
  using value_type = T;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  termination_base() = default;
  termination_base(termination_base const&) = default;
  termination_base(termination_base&&) noexcept = default;
  termination_base& operator=(termination_base const&) = default;
  termination_base& operator=(termination_base&&) noexcept = default;
  virtual ~termination_base() = default;

  [[nodiscard]] virtual bool operator()(shape_vector const& accepted) const = 0;

  // Same predicate with explicit accounting. `count_all == false` is the
  // operator() behaviour: only primaries (centre in the domain box) count,
  // so periodic wrap copies are not double-counted. `count_all == true` is
  // for generators with no domain constraint (random_generator), where
  // every accepted shape is a distinct primary even when its centre lies
  // outside the box. Terminations that do not distinguish ignore it.
  [[nodiscard]] virtual bool is_met(shape_vector const& accepted,
                                    bool /*count_all*/) const {
    return (*this)(accepted);
  }

  // Optional progress hooks consumed by the generator base when filling
  // `progress_info` for the on_progress callback. Concrete terminations
  // override whichever applies; the defaults of 0 mean "unknown" and
  // surface in Tessera's progress bar as a count-only display rather
  // than a percentage.
  [[nodiscard]] virtual std::size_t target_count() const noexcept { return 0; }
  [[nodiscard]] virtual double target_volume_fraction() const noexcept { return 0.0; }
};

} // namespace rvegen
