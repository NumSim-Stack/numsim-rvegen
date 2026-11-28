#pragma once

// The axis-aligned extent of a shape, for code that only has a
// shape_base& (#135): the shape's bounding box when one has been computed,
// otherwise centre +- max_expansion() (never smaller than the shape).
//
// shape_extent is itself a bounding_box_base, so aabb_overlap(),
// aabb_inside() and aabb_grid take it directly.

#include <array>
#include <cstddef>
#include <memory>

#include "bounding_box_base.h"
#include "shape_base.h"

namespace rvegen {

template <typename T>
struct shape_extent final : bounding_box_base<T> {
  std::array<T, 3> top{}, bottom{};
  [[nodiscard]] std::array<T, 3> const& top_point() const override { return top; }
  [[nodiscard]] std::array<T, 3> const& bottom_point() const override { return bottom; }
  [[nodiscard]] std::unique_ptr<bounding_box_base<T>> clone() const override {
    return std::make_unique<shape_extent>(*this);
  }
};

template <typename T>
[[nodiscard]] inline shape_extent<T> extent_of(shape_base<T> const& s) {
  shape_extent<T> e;
  if (auto const* bb = s.bounding_box(); bb != nullptr) {
    e.top = bb->top_point();
    e.bottom = bb->bottom_point();
  } else {
    const auto mid = s.get_middle_point();
    const auto ext = s.max_expansion();
    for (std::size_t k = 0; k < 3; ++k) {
      e.top[k] = mid[k] + ext[k];
      e.bottom[k] = mid[k] - ext[k];
    }
  }
  return e;
}

} // namespace rvegen
