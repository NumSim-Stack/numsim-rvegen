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

  // Optional progress hooks consumed by the generator base when filling
  // `progress_info` for the on_progress callback. Concrete terminations
  // override whichever applies; the defaults of 0 mean "unknown" and
  // surface in Tessera's progress bar as a count-only display rather
  // than a percentage.
  [[nodiscard]] virtual std::size_t target_count() const noexcept { return 0; }
  [[nodiscard]] virtual double target_volume_fraction() const noexcept { return 0.0; }
};

} // namespace rvegen
