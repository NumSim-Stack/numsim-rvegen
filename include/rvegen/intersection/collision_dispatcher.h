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
// register_pair throws std::logic_error if two different shape types
// report the same static_indexing id (see shape_id_claims.h): the table
// would otherwise route one type's collisions to the other's narrow phase.
//
// See bench/dispatch_compare.cpp for the head-to-head measurements that
// motivate the design.

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <numsim-core/static_indexing.h>

#include "../shapes/shape_base.h"
#include "../shapes/shape_id_claims.h"
#include "aabb_overlap.h"
#include "collision_details.h"

namespace rvegen {

template <typename T = double>
class collision_dispatcher {
public:
  using fn_t      = bool (*)(shape_base<T> const&, shape_base<T> const&);
  using row_t     = std::vector<fn_t>;     // inner row, indexed by rhs id
  using table_t   = std::vector<row_t>;    // outer, indexed by lhs id
  // "closer than gap?" (#111): collision_details(a, b, gap), see
  // collision_gap.h. A second table, so the plain test above is untouched.
  using gap_fn_t    = bool (*)(shape_base<T> const&, shape_base<T> const&, T);
  using gap_row_t   = std::vector<gap_fn_t>;
  using gap_table_t = std::vector<gap_row_t>;

  collision_dispatcher(collision_dispatcher const&) = delete;
  collision_dispatcher& operator=(collision_dispatcher const&) = delete;

  static collision_dispatcher& instance() {
    static collision_dispatcher d;
    return d;
  }

  // Register both A→B and B→A so call sites don't need to canonicalise.
  // Throws std::logic_error if A or B shares its id with another
  // registered type; the table is left unchanged in that case.
  template <typename A, typename B>
  void register_pair() {
    _ids.template claim<A>("collision_dispatcher");
    _ids.template claim<B>("collision_dispatcher");
    grow_to_fit(std::max(A::m_id, B::m_id));
    _table[A::m_id][B::m_id] = &dispatch<A, B>;
    if constexpr (!std::is_same_v<A, B>) {
      _table[B::m_id][A::m_id] = &dispatch<B, A>;
    }
    // The gap-aware test is registered when collision_details(A, B, gap)
    // exists at this point (include collision_gap.h, as
    // register_collisions.h does, or declare your own overload before
    // registering). A pair without one falls back to the distance between
    // the bounding boxes when a gap is asked for: never too close, but it
    // may refuse a placement that would have been fine.
    if constexpr (has_gap_test<A, B>) {
      _gap_table[A::m_id][B::m_id] = &dispatch_gap<A, B>;
    }
    if constexpr (!std::is_same_v<A, B> && has_gap_test<B, A>) {
      _gap_table[B::m_id][A::m_id] = &dispatch_gap<B, A>;
    }
  }

  // ---- Bound subobject ---------------------------------------------------
  // Holds a pointer to one row of the table for a fixed LHS id. Lifetime is
  // tied to the dispatcher singleton — the row pointer stays valid for the
  // process lifetime. Safe to keep on the stack across an inner loop.
  class bound {
  public:
    constexpr bound() noexcept = default;
    constexpr explicit bound(row_t const* row,
                             gap_row_t const* gap_row = nullptr) noexcept
        : _row{row}, _gap_row{gap_row} {}

    [[nodiscard]] bool operator()(shape_base<T> const& a,
                                  shape_base<T> const& b) const noexcept {
      if (!_row) return aabb_fallback(a, b);
      const auto rhs = static_cast<std::size_t>(b.shape_id());
      if (rhs >= _row->size()) return aabb_fallback(a, b);
      auto fn = (*_row)[rhs];
      return fn ? fn(a, b) : aabb_fallback(a, b);
    }

    // True when a and b overlap or are closer than `gap`. gap <= 0 is the
    // plain test above.
    [[nodiscard]] bool operator()(shape_base<T> const& a,
                                  shape_base<T> const& b, T gap) const {
      if (!(gap > T{0})) return (*this)(a, b);
      if (_gap_row) {
        const auto rhs = static_cast<std::size_t>(b.shape_id());
        if (rhs < _gap_row->size()) {
          if (auto fn = (*_gap_row)[rhs]) return fn(a, b, gap);
        }
      }
      return aabb_fallback(a, b, gap);
    }

    [[nodiscard]] bool has_inner() const noexcept { return _row != nullptr; }

  private:
    row_t const* _row{nullptr};
    gap_row_t const* _gap_row{nullptr};
  };

  // Hoist the LHS lookup. Cache the returned `bound` and reuse it across
  // all (lhs, rhs_i) calls in an inner loop.
  [[nodiscard]] bound bind_lhs(shape_base<T> const& lhs) const noexcept {
    const auto idx = static_cast<std::size_t>(lhs.shape_id());
    if (idx >= _table.size()) return bound{};
    return bound{&_table[idx], &_gap_table[idx]};
  }

  // Single-call API — does both lookups every time. Convenient for
  // sporadic callers; slower than bind_lhs in tight loops.
  [[nodiscard]] bool operator()(shape_base<T> const& a,
                                shape_base<T> const& b) const noexcept {
    return bind_lhs(a)(a, b);
  }

  [[nodiscard]] bool operator()(shape_base<T> const& a,
                                shape_base<T> const& b, T gap) const {
    return bind_lhs(a)(a, b, gap);
  }

  // Number of registered pairs that have a gap-aware test.
  [[nodiscard]] std::size_t gap_size() const noexcept {
    std::size_t n = 0;
    for (auto const& row : _gap_table) {
      for (auto fn : row) if (fn) ++n;
    }
    return n;
  }

  // Number of registered (lhs, rhs) pairs. Diagnostic use.
  [[nodiscard]] std::size_t size() const noexcept {
    std::size_t n = 0;
    for (auto const& row : _table) {
      for (auto fn : row) if (fn) ++n;
    }
    return n;
  }

  void clear() noexcept {
    _table.clear();
    _gap_table.clear();
    _ids.clear();
  }

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
    if (_gap_table.size() < needed) _gap_table.resize(needed);
    for (auto& row : _gap_table) {
      if (row.size() < needed) row.resize(needed, nullptr);
    }
  }

  template <typename A, typename B>
  static constexpr bool has_gap_test =
      requires(A const& a, B const& b, T gap) {
        { collision_details(a, b, gap) } -> std::convertible_to<bool>;
      };

  template <typename A, typename B>
  static bool dispatch_gap(shape_base<T> const& a, shape_base<T> const& b,
                           T gap) {
    assert(dynamic_cast<A const*>(&a) != nullptr &&
           "collision_dispatcher: lhs type mismatch with registered slot");
    assert(dynamic_cast<B const*>(&b) != nullptr &&
           "collision_dispatcher: rhs type mismatch with registered slot");
    return collision_details(static_cast<A const&>(a),
                             static_cast<B const&>(b), gap);
  }

  // Distance between the bounding boxes <= gap.
  static bool aabb_fallback(shape_base<T> const& a, shape_base<T> const& b,
                            T gap) noexcept {
    auto const* abb = a.bounding_box();
    auto const* bbb = b.bounding_box();
    if (!abb || !bbb) return false;
    auto const alo = abb->bottom_point();
    auto const ahi = abb->top_point();
    auto const blo = bbb->bottom_point();
    auto const bhi = bbb->top_point();
    T d2{0};
    for (std::size_t k = 0; k < 3; ++k) {
      const T s = std::max({T{0}, alo[k] - bhi[k], blo[k] - ahi[k]});
      d2 += s * s;
    }
    return d2 <= gap * gap;
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
  gap_table_t _gap_table;
  shape_id_claims _ids;
};

// Convenience: full two-stage check on a pair of shape_base const&.
// For tight loops, prefer bind_lhs(*candidate) above and call
// bound(*candidate, *other) per iteration.
template <typename T>
[[nodiscard]] inline bool
collide_two_stage(shape_base<T> const& a, shape_base<T> const& b) noexcept {
  auto const* abb = a.bounding_box();
  auto const* bbb = b.bounding_box();
  if (!abb || !bbb) return false;
  if (!aabb_overlap(*abb, *bbb)) return false;
  return collision_dispatcher<T>::instance()(a, b);
}

} // namespace rvegen
