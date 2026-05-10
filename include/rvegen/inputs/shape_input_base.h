#pragma once

#include <memory>
#include <string>
#include <utility>

#include "../shapes/shape_base.h"

namespace rvegen {

// Polymorphic input descriptor. One per shape kind (circle_input, sphere_input,
// rectangle_input, ...). Owns or references the distributions for its
// parameters and produces fresh candidate shapes via new_shape().
//
// The generator drives the simulation by repeatedly calling new_shape() on
// each input until a termination condition fires.
//
// Phase tagging:
//   Each input may carry a `phase_name` that gets stamped onto every shape
//   it produces. Concrete inputs read this from the schema (optional
//   `phase_name` field, default empty) and call `tag(*shape)` from their
//   `new_shape()` implementations. The shape carries the tag through
//   clone / translate operations; per-phase output writers (gmsh Physical
//   groups, voxel grid integers, DAMASK material.config) consume it.
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

  [[nodiscard]] std::string const& phase_name() const noexcept {
    return _phase_name;
  }
  void set_phase_name(std::string name) noexcept {
    _phase_name = std::move(name);
  }

protected:
  // Concrete inputs call this from `new_shape()` to stamp the input's
  // phase name onto the shape before returning it.
  void tag(shape_base<T>& shape) const {
    shape.set_phase_name(_phase_name);
  }

  std::string _phase_name;
};

} // namespace rvegen
