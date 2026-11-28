#pragma once

// Bounding-volume tree over the triangles of a base mesh (#126).
//
// A mesh_inclusion pair test asks "which triangles reach into this box?".
// Scanning and transforming every triangle answers that in O(N) per call,
// about a millisecond for an 80 000-triangle mesh, and a placement run asks
// for every candidate against every neighbour. This tree answers in
// O(log N + k).
//
// The tree is built by GTE's AABBBVTreeOfTriangles (median split along the
// longest axis of the node box, so it is balanced). GTE ships line, ray and
// segment queries for it but no box query, and it keeps its own copy of the
// vertices; what is kept here is a compact copy of the node boxes and the
// triangle permutation, about a third of the size of the triangle list.
//
// It is built once per base mesh, in base coordinates, and shared by every
// placed copy like the triangle list itself: a copy differs only by a
// rotation, a uniform scale and a translation, which mesh_inclusion undoes
// on the query box before it asks.
//
// `collect` returns a superset: all triangles of every leaf whose box
// overlaps the query box. The order is the order of the tree walk. The
// median split uses std::nth_element, which may order equal keys differently
// from one standard library to the next, so callers must not let a result
// depend on the order or on which extra triangles a leaf brings along.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

#include <Mathematics/AABBBVTreeOfTriangles.h>
#include <Mathematics/Triangle.h>
#include <Mathematics/Vector3.h>

namespace rvegen::detail {

// Relative margin by which the tree queries widen a box or lower a bound,
// to cover the rounding of the transforms on the way: far above a few ulp
// (also for float), far below anything a caller could tell apart.
template <typename T>
inline constexpr T mesh_rounding_margin =
    std::max(static_cast<T>(1e-9), 64 * std::numeric_limits<T>::epsilon());

template <typename T>
class mesh_triangle_tree {
public:
  // Largest number of triangles in a leaf. One triangle per leaf needs
  // eight times the nodes and saves little: a caller applies its own exact
  // test to every candidate anyway.
  static constexpr std::size_t leaf_size = 8;

  explicit mesh_triangle_tree(std::vector<gte::Triangle3<T>> const& tris) {
    if (tris.empty()) return;
    if (tris.size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::length_error{
          "mesh_inclusion: more than 2^32 - 1 triangles in one mesh"};
    }
    std::vector<gte::Vector3<T>> vertices;
    std::vector<std::array<std::size_t, 3>> triangles;
    vertices.reserve(3 * tris.size());
    triangles.reserve(tris.size());
    for (auto const& t : tris) {
      const std::size_t first = vertices.size();
      vertices.push_back(t.v[0]);
      vertices.push_back(t.v[1]);
      vertices.push_back(t.v[2]);
      triangles.push_back({first, first + 1, first + 2});
    }
    std::size_t height = 0;
    while ((leaf_size << height) < tris.size() && height < 31) ++height;
    gte::AABBBVTreeOfTriangles<T> tree;
    tree.Create(vertices, triangles, height);

    constexpr std::size_t none = std::numeric_limits<std::size_t>::max();
    auto const& nodes = tree.GetNodes();
    _nodes.resize(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      auto const& src = nodes[i];
      auto& dst = _nodes[i];
      if (src.minIndex == none) continue;   // slot the build never reached
      for (int k = 0; k < 3; ++k) {
        dst.lo[k] = src.boundingVolume.box.min[k];
        dst.hi[k] = src.boundingVolume.box.max[k];
      }
      dst.first = static_cast<std::uint32_t>(src.minIndex);
      dst.count = static_cast<std::uint32_t>(src.maxIndex - src.minIndex + 1);
      dst.leaf = (src.leftChild == none);
    }
    auto const& partition = tree.GetPartition();
    _order.assign(partition.begin(), partition.end());
  }

  // Appends the indices of all triangles that may overlap the box [lo, hi]
  // (base coordinates, bounds included).
  void collect(std::array<T, 3> const& lo, std::array<T, 3> const& hi,
               std::vector<std::size_t>& out) const {
    if (_nodes.empty()) return;
    // The children of node i are 2i+1 and 2i+2, the height is at most 31:
    // a depth-first walk keeps at most one pending sibling per level.
    std::array<std::size_t, 40> stack{};
    std::size_t top = 0;
    stack[top++] = 0;
    while (top > 0) {
      const std::size_t i = stack[--top];
      auto const& node = _nodes[i];
      if (node.count == 0 ||
          node.lo[0] > hi[0] || node.hi[0] < lo[0] ||
          node.lo[1] > hi[1] || node.hi[1] < lo[1] ||
          node.lo[2] > hi[2] || node.hi[2] < lo[2]) {
        continue;
      }
      if (node.leaf) {
        for (std::uint32_t k = 0; k < node.count; ++k)
          out.push_back(_order[node.first + k]);
      } else {
        stack[top++] = 2 * i + 2;
        stack[top++] = 2 * i + 1;
      }
    }
  }

  // Largest value of a function g over all vertices of the mesh, by branch
  // and bound. `bound(lo, hi)` is an upper bound of g over a box, `value(i)`
  // the largest g over the three vertices of triangle i. A node is skipped
  // only when its bound is below the best value by more than `slack`, which
  // must cover the rounding of `bound`; the result is then the same number
  // a loop over all triangles would return.
  template <typename Bound, typename Value>
  [[nodiscard]] T maximise(Bound&& bound, Value&& value, T slack) const {
    T best = std::numeric_limits<T>::lowest();
    if (_nodes.empty()) return best;
    std::array<std::size_t, 40> stack{};
    std::size_t top = 0;
    stack[top++] = 0;
    while (top > 0) {
      const std::size_t i = stack[--top];
      auto const& node = _nodes[i];
      if (node.count == 0 || bound(node.lo, node.hi) + slack < best) continue;
      if (node.leaf) {
        for (std::uint32_t k = 0; k < node.count; ++k)
          best = std::max(best, value(std::size_t{_order[node.first + k]}));
      } else {
        // The more promising child last, so it is popped first.
        const std::size_t l = 2 * i + 1, r = 2 * i + 2;
        const bool left_first = bound(_nodes[l].lo, _nodes[l].hi) >=
                                bound(_nodes[r].lo, _nodes[r].hi);
        stack[top++] = left_first ? r : l;
        stack[top++] = left_first ? l : r;
      }
    }
    return best;
  }

  // Node access for a walk of two trees against each other. Node 0 is the
  // root, the children of an inner node i are 2i+1 and 2i+2.
  [[nodiscard]] bool empty() const noexcept { return _nodes.empty(); }
  [[nodiscard]] bool leaf(std::size_t i) const noexcept {
    return _nodes[i].leaf;
  }
  [[nodiscard]] std::array<T, 3> const& lo(std::size_t i) const noexcept {
    return _nodes[i].lo;
  }
  [[nodiscard]] std::array<T, 3> const& hi(std::size_t i) const noexcept {
    return _nodes[i].hi;
  }
  [[nodiscard]] std::span<std::uint32_t const>
  triangles(std::size_t i) const noexcept {
    return {_order.data() + _nodes[i].first, _nodes[i].count};
  }

  [[nodiscard]] std::size_t node_count() const noexcept {
    return _nodes.size();
  }
  // Bytes held by the tree, for the memory test.
  [[nodiscard]] std::size_t memory() const noexcept {
    return _nodes.capacity() * sizeof(node) +
           _order.capacity() * sizeof(std::uint32_t);
  }

private:
  struct node {
    std::array<T, 3> lo{};
    std::array<T, 3> hi{};
    std::uint32_t first{0};
    std::uint32_t count{0};   // 0: unused slot
    bool leaf{true};
  };
  std::vector<node> _nodes;
  std::vector<std::uint32_t> _order;   // triangle indices, grouped by leaf
};

} // namespace rvegen::detail
