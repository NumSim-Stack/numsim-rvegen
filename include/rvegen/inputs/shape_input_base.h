#pragma once

#include <memory>

#include "../shapes/shape_base.h"

namespace rvegen {

// Polymorphic input descriptor. One per shape kind (circle_input, sphere_input,
// rectangle_input, ...). Owns or references the distributions for its
// parameters and produces fresh candidate shapes via new_shape().
//
// The generator drives the simulation by repeatedly calling new_shape() on
// each input until a termination condition fires.
template <typename T>
class shape_input_base {
public:
  using value_type = T;

  shape_input_base() = default;
  shape_input_base(shape_input_base const&) = default;
  shape_input_base(shape_input_base&&) noexcept = default;
  shape_input_base& operator=(shape_input_base const&) = default;
  shape_input_base& operator=(shape_input_base&&) noexcept = default;
  virtual ~shape_input_base() = default;

  // Sample a new candidate shape from this input's distributions. The shape
  // is returned without a bounding box computed; the caller is responsible
  // for calling make_bounding_box() before any geometric query.
  // [[nodiscard]] is mandatory: discarding the return value silently
  // destroys the freshly-allocated candidate.
  [[nodiscard]] virtual std::unique_ptr<shape_base<T>> new_shape() = 0;
};

} // namespace rvegen
