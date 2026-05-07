#pragma once

// First-class visitor-pattern dispatch for collision: shapes stored as
// std::variant<...>, dispatch via std::visit. Compile-time, exhaustively
// checked. Cost: every shape's storage size is `sizeof(largest variant
// alternative) + tag`, and std::visit hits a function-pointer-table cliff
// past ~10 alternatives in libstdc++.
//
// Use this when:
//   - The shape set is small and closed (typical for a single RVE).
//   - You want compile-time exhaustiveness: forgetting a pair fails to compile.
//
// Companion to shape_base/dispatch_table for the polymorphic case.

#include <variant>

#include "../intersection/aabb_overlap.h"
#include "box.h"
#include "circle.h"
#include "ellipse.h"
#include "rectangle.h"
#include "sphere.h"

namespace rvegen {

// 2D shape mix: circle + rectangle + ellipse.
template <typename T = double>
using shape_variant_2d = std::variant<circle<T>, rectangle<T>, ellipse<T>>;

// 3D shape mix: sphere + box.
template <typename T = double>
using shape_variant_3d = std::variant<sphere<T>, box<T>>;

// Generic visitor: calls collision_details(a, b) when an overload exists,
// else falls back to AABB overlap (conservative; same fallback as
// dispatch_table). Use with std::visit.
struct collide_visitor {
  template <typename A, typename B>
  bool operator()(A const& a, B const& b) const {
    if constexpr (requires { collision_details(a, b); })
      return collision_details(a, b);
    else
      return aabb_overlap(*a.bounding_box(), *b.bounding_box());
  }
};

// Free-function helper: collide(variant_a, variant_b).
template <typename Variant>
[[nodiscard]] bool collide(Variant const& a, Variant const& b) {
  return std::visit(collide_visitor{}, a, b);
}

} // namespace rvegen
