#pragma once

// Typed-pool storage: one homogeneous std::vector per shape type, held in a
// std::tuple. Eliminates runtime dispatch entirely — every collision check
// is a typed call resolved at compile time. The cost is a closed shape set
// (the template arguments) and changing the shape mix means changing types.
//
// Use this when:
//   - You want maximum performance (the math itself is the only cost).
//   - The shape set is fixed at compile time for this RVE run.
//   - You're willing to give up the heterogeneous shape_base interface.

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "../intersection/aabb_overlap.h"
#include "../intersection/collision_details.h"

namespace rvegen {

template <typename... Shapes>
class shape_pool {
public:
  using tuple_type = std::tuple<std::vector<Shapes>...>;

  // Add a shape by exact type. The compiler picks the right pool.
  template <typename S>
    requires (std::is_same_v<std::decay_t<S>, Shapes> || ...)
  void push(S&& shape) {
    std::get<std::vector<std::decay_t<S>>>(_pools).push_back(std::forward<S>(shape));
  }

  // Direct typed access to the pools — for tight inner loops that want to
  // know the concrete types (e.g. benchmarks, custom kernels).
  [[nodiscard]] tuple_type const& pools() const noexcept { return _pools; }
  [[nodiscard]] tuple_type& pools() noexcept { return _pools; }

  // Total accepted-shape count across all pools.
  [[nodiscard]] std::size_t size() const noexcept {
    return std::apply(
        [](auto const&... p) { return (p.size() + ...); }, _pools);
  }

  // True if the candidate (any shape type) collides with any accepted
  // shape across all pools. Fold expression over the tuple — every
  // (Candidate, Pool_i) check resolves at compile time:
  //   * collision_details(c, p) overload picked by ADL+overload resolution
  //   * if no overload exists for the pair, falls back to AABB overlap
  template <typename Candidate>
  [[nodiscard]] bool any_collides(Candidate const& c) const {
    return std::apply([&c](auto const&... pool) {
      return ((std::ranges::any_of(pool, [&c](auto const& other) {
        if constexpr (requires { collision_details(c, other); })
          return collision_details(c, other);
        else
          return aabb_overlap(*c.bounding_box(), *other.bounding_box());
      })) || ...);
    }, _pools);
  }

  // Iterate every shape across every pool, calling f(shape).
  // Concrete type preserved — f is invoked with the typed shape.
  template <typename F>
  void for_each(F&& f) const {
    std::apply([&f](auto const&... pool) {
      (..., std::ranges::for_each(pool, f));
    }, _pools);
  }

private:
  tuple_type _pools;
};

namespace detail {

// Iterate every unordered pair (a, b) with a ≠ b across all pools, calling
// f(a, b) with concrete types. Used by any_pair_collides and the bench.
template <typename Tuple, typename F, std::size_t I, std::size_t J>
void each_pair_at(Tuple const& t, F&& f) {
  if constexpr (I == J) {
    auto const& p = std::get<I>(t);
    for (std::size_t a = 0; a + 1 < p.size(); ++a)
      for (std::size_t b = a + 1; b < p.size(); ++b)
        f(p[a], p[b]);
  } else {
    auto const& pa = std::get<I>(t);
    auto const& pb = std::get<J>(t);
    for (auto const& a : pa)
      for (auto const& b : pb)
        f(a, b);
  }
}

template <typename Tuple, typename F, std::size_t I, std::size_t... Js>
void each_pair_one_first(Tuple const& t, F&& f, std::index_sequence<Js...>) {
  // Pairs (I, I), (I, I+1), (I, I+2), ...
  (each_pair_at<Tuple, F, I, I + Js>(t, f), ...);
}

template <typename Tuple, typename F, std::size_t... Is>
void each_pair_dispatch(Tuple const& t, F&& f, std::index_sequence<Is...>) {
  constexpr std::size_t N = std::tuple_size_v<Tuple>;
  (each_pair_one_first<Tuple, F, Is>(t, f, std::make_index_sequence<N - Is>{}),
   ...);
}

} // namespace detail

// Visit every unordered pair across all pools — concrete types preserved.
// O(N²/2) total invocations.
template <typename... Shapes, typename F>
void for_each_unordered_pair(shape_pool<Shapes...> const& pool, F&& f) {
  detail::each_pair_dispatch(
      pool.pools(), f,
      std::make_index_sequence<sizeof...(Shapes)>{});
}

// Apply collision_details (or AABB fallback) to every unordered pair.
// Returns the number of colliding pairs.
template <typename... Shapes>
[[nodiscard]] std::size_t
count_colliding_pairs(shape_pool<Shapes...> const& pool) {
  std::size_t hits = 0;
  for_each_unordered_pair(pool, [&hits](auto const& a, auto const& b) {
    if constexpr (requires { collision_details(a, b); }) {
      if (collision_details(a, b)) ++hits;
    } else {
      if (aabb_overlap(*a.bounding_box(), *b.bounding_box())) ++hits;
    }
  });
  return hits;
}

// Two-stage version: AABB pre-check inside the fold expression, then
// typed collision_details for pairs that pass. Equivalent to what the
// generator hot loop does, but on typed-pool storage with compile-time
// narrow-phase dispatch (no runtime indirection on the precise call).
template <typename... Shapes>
[[nodiscard]] std::size_t
count_two_stage_colliding_pairs(shape_pool<Shapes...> const& pool) {
  std::size_t hits = 0;
  for_each_unordered_pair(pool, [&hits](auto const& a, auto const& b) {
    if (!aabb_overlap(*a.bounding_box(), *b.bounding_box())) return; // stage 1
    if constexpr (requires { collision_details(a, b); }) {
      if (collision_details(a, b)) ++hits;                            // stage 2
    } else {
      ++hits;  // AABB already confirmed; no precise overload available
    }
  });
  return hits;
}

} // namespace rvegen
