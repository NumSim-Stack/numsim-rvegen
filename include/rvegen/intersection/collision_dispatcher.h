#pragma once

// Runtime double-dispatch for shape collisions when the static type is lost
// (vector<unique_ptr<shape_base<T>>>). Storage is a 2-D vector keyed by the
// numsim_core::static_indexing id that every concrete shape carries — O(1)
// lookup, no RTTI, no map.
//
// Two-stage usage (the generators' hot loop):
//
//   auto bound = collision_dispatcher<T>::instance().bind_lhs(*cand);
//   for (auto const& other : accepted) {
//     if (!aabb_overlap(*cand_bb, *other->bounding_box())) continue;  // stage 1
//     if (bound(*cand, *other)) { ... }                               // stage 2
//   }
//
// Falls back to AABB overlap on misses (unregistered pair, or a shape whose
// id exceeds the table — which only happens if the caller skips
// register_pair for that pair).
//
// See bench/dispatch_compare.cpp for the head-to-head measurements that
// motivate the design.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <numsim-core/static_indexing.h>

#include "../shapes/shape_base.h"
#include "aabb_overlap.h"
#include "collision_details.h"

namespace rvegen {

template <typename T = double>
class collision_dispatcher {
public:
  using fn_t      = bool (*)(shape_base<T> const&, shape_base<T> const&);
  using row_t     = std::vector<fn_t>;     // inner row, indexed by rhs id
  using table_t   = std::vector<row_t>;    // outer, indexed by lhs id

  collision_dispatcher(collision_dispatcher const&) = delete;
  collision_dispatcher& operator=(collision_dispatcher const&) = delete;

  static collision_dispatcher& instance() {
    static collision_dispatcher d;
    return d;
  }

  // Register both A→B and B→A so call sites don't need to canonicalise.
  template <typename A, typename B>
  void register_pair() {
    grow_to_fit(std::max(A::m_id, B::m_id));
    _table[A::m_id][B::m_id] = &dispatch<A, B>;
    if constexpr (!std::is_same_v<A, B>) {
      _table[B::m_id][A::m_id] = &dispatch<B, A>;
    }
  }

  // ---- Bound subobject ---------------------------------------------------
  // Holds a pointer to one row of the table for a fixed LHS id. Lifetime is
  // tied to the dispatcher singleton — the row pointer stays valid for the
  // process lifetime. Safe to keep on the stack across an inner loop.
  class bound {
  public:
    constexpr bound() noexcept = default;
    constexpr explicit bound(row_t const* row) noexcept : _row{row} {}

    [[nodiscard]] bool operator()(shape_base<T> const& a,
                                  shape_base<T> const& b) const noexcept {
      if (!_row) return aabb_fallback(a, b);
      const auto rhs = static_cast<std::size_t>(b.shape_id());
      if (rhs >= _row->size()) return aabb_fallback(a, b);
      auto fn = (*_row)[rhs];
      return fn ? fn(a, b) : aabb_fallback(a, b);
    }

    [[nodiscard]] bool has_inner() const noexcept { return _row != nullptr; }

  private:
    row_t const* _row{nullptr};
  };

  // Hoist the LHS lookup. Cache the returned `bound` and reuse it across
  // all (lhs, rhs_i) calls in an inner loop.
  [[nodiscard]] bound bind_lhs(shape_base<T> const& lhs) const noexcept {
    const auto idx = static_cast<std::size_t>(lhs.shape_id());
    if (idx >= _table.size()) return bound{};
    return bound{&_table[idx]};
  }

  // Single-call API — does both lookups every time. Convenient for
  // sporadic callers; slower than bind_lhs in tight loops.
  [[nodiscard]] bool operator()(shape_base<T> const& a,
                                shape_base<T> const& b) const noexcept {
    return bind_lhs(a)(a, b);
  }

  // Number of registered (lhs, rhs) pairs. Diagnostic use.
  [[nodiscard]] std::size_t size() const noexcept {
    std::size_t n = 0;
    for (auto const& row : _table) {
      for (auto fn : row) if (fn) ++n;
    }
    return n;
  }

  void clear() noexcept { _table.clear(); }

private:
  collision_dispatcher() = default;

  // Grow the table so both axes can hold an entry at index max_id. Called
  // from register_pair before storing the entry.
  void grow_to_fit(numsim_core::type_id max_id) {
    const std::size_t needed = static_cast<std::size_t>(max_id) + 1;
    if (_table.size() < needed) _table.resize(needed);
    for (auto& row : _table) {
      if (row.size() < needed) row.resize(needed, nullptr);
    }
  }

  template <typename A, typename B>
  static bool dispatch(shape_base<T> const& a, shape_base<T> const& b) {
    // Defensive guard against mis-registration: the registered slot at
    // (A::m_id, B::m_id) is supposed to match the actual dynamic types of
    // `a` and `b`. If it doesn't, the static_cast below is UB. In release
    // builds this is a zero-cost contract; in debug builds the assertion
    // catches mis-registration immediately.
    assert(dynamic_cast<A const*>(&a) != nullptr &&
           "collision_dispatcher: lhs type mismatch with registered slot");
    assert(dynamic_cast<B const*>(&b) != nullptr &&
           "collision_dispatcher: rhs type mismatch with registered slot");
    return collision_details(static_cast<A const&>(a),
                             static_cast<B const&>(b));
  }

  static bool aabb_fallback(shape_base<T> const& a,
                             shape_base<T> const& b) noexcept {
    auto const* abb = a.bounding_box();
    auto const* bbb = b.bounding_box();
    if (!abb || !bbb) return false;
    return aabb_overlap(*abb, *bbb);
  }

  table_t _table;
};

// Convenience: full two-stage check on a pair of shape_base const&.
// For tight loops, prefer bind_lhs(*candidate) above and call
// bound(*candidate, *other) per iteration.
template <typename T>
[[nodiscard]] inline bool
collide_two_stage(shape_base<T> const& a, shape_base<T> const& b) noexcept {
  if (!aabb_overlap(*a.bounding_box(), *b.bounding_box())) return false;
  return collision_dispatcher<T>::instance()(a, b);
}

} // namespace rvegen
