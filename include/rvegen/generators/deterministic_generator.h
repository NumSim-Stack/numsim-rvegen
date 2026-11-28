#pragma once

// Base class for deterministic generators (weave, tow_weave, voronoi).
//
// These generators produce their shapes from parameters (and, for
// Voronoi, a seed) rather than by iterative Monte-Carlo placement, so
// the termination predicate and the shape inputs have nothing to drive.
// Per issue #73 (recommendation (a)) they sit under the ordinary
// `rve_generator_base` so one registry / one CLI path serves them, and
// the contract is:
//
//   * `inputs` are ignored (the config may omit the "shapes" section);
//   * `termination` is ignored: a deterministic generator runs exactly
//     once and its output size is fixed by its parameters;
//   * the pipeline's top-level `seed` reaches them through
//     `set_pipeline_seed()` (compute() has no engine to draw from);
//     only generators that sample (Voronoi) use it;
//   * `opts.cancel` is polled once before the build, `opts.on_progress`
//     is invoked once after it with shapes_placed == shapes_target.
//
// Derived classes implement `build(domain_box)` only.

#include <array>
#include <cstdint>
#include <vector>

#include "rve_generator_base.h"

namespace rvegen {

template <typename T>
class deterministic_generator : public rve_generator_base<T> {
public:
  using shape_vector = typename rve_generator_base<T>::shape_vector;
  using input_vector = typename rve_generator_base<T>::input_vector;

  // Seed of the surrounding pipeline (top-level "seed"). Called by the
  // pipeline after construction; generators that sample take their RNG
  // seed from it unless they have an explicit override of their own.
  virtual void set_pipeline_seed(std::uint64_t /*seed*/) {}

  [[nodiscard]] shape_vector
  compute(input_vector& /*inputs*/,
          termination_base<T> const& /*termination*/,
          std::array<T, 3> const& domain_box,
          progress_options const& opts = {}) final {
    if (opts.cancel()) return {};
    auto shapes = build(domain_box);
    progress_info info;
    info.shapes_placed = shapes.size();
    info.shapes_target = shapes.size();
    info.volume_fraction = current_volume_fraction<T>(shapes, domain_box);
    opts.on_progress(info);
    return shapes;
  }

protected:
  [[nodiscard]] virtual shape_vector
  build(std::array<T, 3> const& domain_box) const = 0;
};

} // namespace rvegen
