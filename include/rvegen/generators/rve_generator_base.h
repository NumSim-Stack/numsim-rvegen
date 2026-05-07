#pragma once

#include <array>
#include <memory>
#include <vector>

#include "../inputs/shape_input_base.h"
#include "../shapes/shape_base.h"
#include "../termination/termination_base.h"

namespace rvegen {

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
  [[nodiscard]] virtual shape_vector
  compute(input_vector& inputs,
          termination_base<T> const& termination,
          std::array<value_type, 3> const& domain_box) = 0;
};

} // namespace rvegen
