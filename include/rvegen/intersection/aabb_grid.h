#pragma once

// Uniform bucket grid over accepted-shape AABBs: the broad phase of the
// placement loop (#83). Replaces the O(N) scan of every accepted shape per
// candidate with a lookup of the few cells the candidate's AABB covers.
//
// Semantics are those of the linear scan it replaces: for a candidate box,
// any_overlap() visits (in unspecified order) exactly the indices whose
// stored AABB overlaps the candidate's under the same inclusive test as
// aabb_overlap(), and short-circuits on the first predicate hit. Since
// "collides" is an OR over accepted shapes, the accept/reject decision --
// and hence the output for a fixed seed -- does not depend on visit order.
//
// Cell size adapts: it starts at the largest extent of the first inserted
// box, and when a later box would cover too many cells the grid rebuilds
// itself with a cell size matching that box (amortised O(1) per insert as
// the size only grows). Boxes with non-finite or absurd coordinates, and
// queries covering too many cells, fall back to a linear scan over the
// stored boxes, so correctness never depends on the cell size.
//
// Cells are keyed by a hash of the integer cell coordinates; a hash
// collision merges two cells, which only adds candidates (every candidate
// is re-checked against its stored AABB), never loses one.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "../shapes/bounding_box_base.h"

namespace rvegen {

template <typename T>
class aabb_grid {
public:
  using index_type = std::size_t;

  // Append a box; its index is the number of previous insertions, i.e. the
  // position of the shape in the generator's accepted vector.
  void insert(bounding_box_base<T> const& bb) {
    insert(bb.bottom_point(), bb.top_point());
  }

  void insert(std::array<T, 3> const& lo, std::array<T, 3> const& hi) {
    const index_type idx = _boxes.size();
    _boxes.push_back({lo, hi});
    if (_cell <= 0.0) init_cell_size(lo, hi);
    if (!add_to_cells(idx)) {
      // Too many cells at the current size. If the box is bigger than a
      // cell, grow the cells to fit it and rebuild; otherwise (non-finite
      // or out-of-range coordinates) keep it in the linear list.
      if (finite(lo, hi) && max_extent(lo, hi) > _cell) {
        _cell = std::max(_cell * 2.0, max_extent(lo, hi));
        rebuild();
      } else {
        _unindexed.push_back(idx);
      }
    }
  }

  // Calls pred(index) for every stored box overlapping `bb` (inclusive,
  // like aabb_overlap); returns true as soon as pred returns true.
  template <typename Pred>
  [[nodiscard]] bool any_overlap(bounding_box_base<T> const& bb,
                                 Pred&& pred) {
    return any_overlap(bb.bottom_point(), bb.top_point(),
                       static_cast<Pred&&>(pred));
  }

  template <typename Pred>
  [[nodiscard]] bool any_overlap(std::array<T, 3> const& lo,
                                 std::array<T, 3> const& hi, Pred&& pred) {
    if (_boxes.empty()) return false;
    cell_range r;
    if (_cell <= 0.0 || !cell_range_of(lo, hi, r)) {
      for (index_type i = 0; i < _boxes.size(); ++i) {
        if (overlaps(i, lo, hi) && pred(i)) return true;
      }
      return false;
    }
    if (++_stamp == 0) {   // stamp wrapped: reset
      std::fill(_seen.begin(), _seen.end(), 0u);
      _stamp = 1;
    }
    _seen.resize(_boxes.size(), 0u);
    auto visit = [&](index_type i) {
      if (_seen[i] == _stamp) return false;
      _seen[i] = _stamp;
      return overlaps(i, lo, hi) && pred(i);
    };
    for (index_type i : _unindexed) {
      if (visit(i)) return true;
    }
    for (std::int64_t x = r.lo[0]; x <= r.hi[0]; ++x) {
      for (std::int64_t y = r.lo[1]; y <= r.hi[1]; ++y) {
        for (std::int64_t z = r.lo[2]; z <= r.hi[2]; ++z) {
          auto it = _cells.find(key(x, y, z));
          if (it == _cells.end()) continue;
          for (index_type i : it->second) {
            if (visit(i)) return true;
          }
        }
      }
    }
    return false;
  }

  [[nodiscard]] std::size_t size() const noexcept { return _boxes.size(); }

private:
  struct box_t { std::array<T, 3> lo, hi; };
  struct cell_range { std::int64_t lo[3], hi[3]; };

  // Max cells one box may be registered in before the grid coarsens.
  static constexpr double max_cells_per_box = 125.0;
  // Coordinates beyond this (in cell units) are not indexed.
  static constexpr double max_cell_coord = 1e15;

  [[nodiscard]] static bool finite(std::array<T, 3> const& lo,
                                   std::array<T, 3> const& hi) noexcept {
    for (int k = 0; k < 3; ++k) {
      if (!std::isfinite(lo[k]) || !std::isfinite(hi[k])) return false;
    }
    return true;
  }

  [[nodiscard]] static double max_extent(std::array<T, 3> const& lo,
                                         std::array<T, 3> const& hi) noexcept {
    double m = 0.0;
    for (int k = 0; k < 3; ++k) {
      m = std::max(m, static_cast<double>(hi[k]) - static_cast<double>(lo[k]));
    }
    return m;
  }

  void init_cell_size(std::array<T, 3> const& lo, std::array<T, 3> const& hi) {
    const double e = finite(lo, hi) ? max_extent(lo, hi) : 0.0;
    _cell = e > 0.0 ? e : 1.0;
  }

  [[nodiscard]] bool overlaps(index_type i, std::array<T, 3> const& lo,
                              std::array<T, 3> const& hi) const noexcept {
    auto const& b = _boxes[i];
    return b.lo[0] <= hi[0] && b.hi[0] >= lo[0] &&
           b.lo[1] <= hi[1] && b.hi[1] >= lo[1] &&
           b.lo[2] <= hi[2] && b.hi[2] >= lo[2];
  }

  // Cell coordinates covered by [lo, hi]; false if non-finite, out of
  // range, or covering more than max_cells_per_box cells.
  [[nodiscard]] bool cell_range_of(std::array<T, 3> const& lo,
                                   std::array<T, 3> const& hi,
                                   cell_range& r) const noexcept {
    double count = 1.0;
    for (int k = 0; k < 3; ++k) {
      const double a = std::floor(static_cast<double>(lo[k]) / _cell);
      const double b = std::floor(static_cast<double>(hi[k]) / _cell);
      if (!(std::abs(a) < max_cell_coord) || !(std::abs(b) < max_cell_coord))
        return false;
      count *= (b - a + 1.0);
      if (!(count <= max_cells_per_box)) return false;
      r.lo[k] = static_cast<std::int64_t>(a);
      r.hi[k] = static_cast<std::int64_t>(b);
    }
    return true;
  }

  [[nodiscard]] static std::uint64_t key(std::int64_t x, std::int64_t y,
                                         std::int64_t z) noexcept {
    auto mix = [](std::uint64_t h, std::uint64_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
      return h;
    };
    std::uint64_t h = 0;
    h = mix(h, static_cast<std::uint64_t>(x));
    h = mix(h, static_cast<std::uint64_t>(y));
    h = mix(h, static_cast<std::uint64_t>(z));
    return h;
  }

  bool add_to_cells(index_type idx) {
    cell_range r;
    if (!cell_range_of(_boxes[idx].lo, _boxes[idx].hi, r)) return false;
    for (std::int64_t x = r.lo[0]; x <= r.hi[0]; ++x)
      for (std::int64_t y = r.lo[1]; y <= r.hi[1]; ++y)
        for (std::int64_t z = r.lo[2]; z <= r.hi[2]; ++z)
          _cells[key(x, y, z)].push_back(idx);
    return true;
  }

  void rebuild() {
    _cells.clear();
    _unindexed.clear();
    for (index_type i = 0; i < _boxes.size(); ++i) {
      if (!add_to_cells(i)) _unindexed.push_back(i);
    }
  }

  std::vector<box_t> _boxes;
  std::unordered_map<std::uint64_t, std::vector<index_type>> _cells;
  std::vector<index_type> _unindexed;
  std::vector<std::uint32_t> _seen;
  std::uint32_t _stamp{0};
  double _cell{0.0};
};

} // namespace rvegen
