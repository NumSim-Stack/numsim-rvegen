#pragma once

// Runtime double-dispatch for shape collisions when the static type is lost
// (vector<unique_ptr<shape_base<T>>>). One singleton per T; users register
// the precise collision_details pairs they care about at startup; queries
// fall back to AABB overlap for unregistered pairs.
//
// Storage: nested unordered_map<type_index, unordered_map<type_index, fn>>.
// The outer key is invariant across an inner loop where the candidate
// (LHS) is fixed — bind_lhs(typeid(candidate)) hoists the outer lookup
// out of the loop, so the per-pair work is one inner-map find + one
// indirect call.
//
// Two-stage usage (the generators' hot loop):
//
//   auto bound = collision_dispatcher<T>::instance().bind_lhs(typeid(*cand));
//   for (auto const& other : accepted) {
//     if (!aabb_overlap(*cand_bb, *other->bounding_box())) continue;  // stage 1
//     if (bound(*cand, *other)) { ... }                               // stage 2
//   }
//
// See bench/dispatch_compare.cpp for the head-to-head measurements that
// motivate the design.

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include <boost/container/flat_map.hpp>

#include "../shapes/shape_base.h"
#include "aabb_overlap.h"
#include "collision_details.h"

namespace rvegen {

template <typename T = double>
class collision_dispatcher {
public:
  using fn_t          = bool (*)(shape_base<T> const&, shape_base<T> const&);
  // boost::container::flat_map: sorted-vector storage. At today's table
  // sizes (~3 entries per inner map, ~5 outer) this is within noise of
  // std::unordered_map. The win compounds as the shape catalogue grows
  // past ~10 types; keeping flat_map locks in predictable O(log N)
  // behaviour and avoids the hash-collision degradation that
  // unordered_map can suffer.
  using inner_table_t = boost::container::flat_map<std::type_index, fn_t>;
  using outer_table_t = boost::container::flat_map<std::type_index, inner_table_t>;

  collision_dispatcher(collision_dispatcher const&) = delete;
  collision_dispatcher& operator=(collision_dispatcher const&) = delete;

  static collision_dispatcher& instance() {
    static collision_dispatcher d;
    return d;
  }

  // Register both A→B and B→A so call sites don't need to canonicalise.
  template <typename A, typename B>
  void register_pair() {
    _outer[typeid(A)][typeid(B)] = &dispatch<A, B>;
    if constexpr (!std::is_same_v<A, B>)
      _outer[typeid(B)][typeid(A)] = &dispatch<B, A>;
  }

  // ---- Bound subobject ---------------------------------------------------
  // Holds a pointer to the inner map for one fixed LHS type. Lifetime is
  // tied to the dispatcher instance — the singleton lives forever, so
  // bound objects are safe to keep on the stack across an inner loop.
  class bound {
  public:
    constexpr bound() noexcept = default;
    constexpr explicit bound(inner_table_t const* inner) noexcept
        : _inner{inner} {}

    [[nodiscard]] bool operator()(shape_base<T> const& a,
                                   shape_base<T> const& b) const noexcept {
      if (!_inner) {
        return aabb_overlap(*a.bounding_box(), *b.bounding_box());
      }
      auto it = _inner->find(typeid(b));
      if (it == _inner->end()) {
        return aabb_overlap(*a.bounding_box(), *b.bounding_box());
      }
      return it->second(a, b);
    }

    [[nodiscard]] bool has_inner() const noexcept { return _inner != nullptr; }

  private:
    inner_table_t const* _inner{nullptr};
  };

  // Hoist the outer lookup. Cache the returned `bound` and reuse it
  // across all (lhs, rhs_i) calls in an inner loop.
  [[nodiscard]] bound bind_lhs(std::type_index lhs) const noexcept {
    auto it = _outer.find(lhs);
    return bound{it == _outer.end() ? nullptr : &it->second};
  }

  // Single-call API — does both lookups every time. Convenient for
  // sporadic callers; slower than bind_lhs in tight loops.
  [[nodiscard]] bool operator()(shape_base<T> const& a,
                                 shape_base<T> const& b) const noexcept {
    return bind_lhs(typeid(a))(a, b);
  }

  [[nodiscard]] bool contains(std::type_index const& a,
                               std::type_index const& b) const noexcept {
    auto it = _outer.find(a);
    if (it == _outer.end()) return false;
    return it->second.contains(b);
  }
  [[nodiscard]] std::size_t size() const noexcept {
    std::size_t n = 0;
    for (auto const& [_, inner] : _outer) n += inner.size();
    return n;
  }
  void clear() noexcept { _outer.clear(); }

private:
  collision_dispatcher() = default;

  template <typename A, typename B>
  static bool dispatch(shape_base<T> const& a, shape_base<T> const& b) {
    // Defensive guard against mis-registration: the registered key
    // (typeid(A), typeid(B)) is supposed to match the actual dynamic
    // types of `a` and `b`. If it doesn't, the static_cast below is UB.
    // In release builds this is a zero-cost contract; in debug builds
    // the assertion catches mis-registration immediately.
    assert(dynamic_cast<A const*>(&a) != nullptr &&
           "collision_dispatcher: lhs type mismatch with registered key");
    assert(dynamic_cast<B const*>(&b) != nullptr &&
           "collision_dispatcher: rhs type mismatch with registered key");
    return collision_details(static_cast<A const&>(a),
                             static_cast<B const&>(b));
  }

  outer_table_t _outer;
};

// Convenience: full two-stage check on a pair of shape_base*.
// For tight loops, prefer bind_lhs(typeid(*candidate)) above and call
// bound(*candidate, *other) per iteration.
template <typename T>
[[nodiscard]] bool collide_two_stage(shape_base<T> const& a,
                                      shape_base<T> const& b) noexcept {
  if (!aabb_overlap(*a.bounding_box(), *b.bounding_box())) return false;
  return collision_dispatcher<T>::instance()(a, b);
}

} // namespace rvegen
