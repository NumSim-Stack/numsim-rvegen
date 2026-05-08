#pragma once

// Runtime dispatch for shape→triangle_mesh conversion when the static type
// is lost (vector<unique_ptr<shape_base<T>>>). Mirrors collision_dispatcher's
// design: storage is a 1-D vector keyed by the numsim_core::static_indexing
// id every concrete shape carries — O(1) lookup, no RTTI.
//
// Differences from collision_dispatcher:
//   * 1-D table (single shape lookup, not a pair).
//   * Miss returns an EMPTY triangle_mesh and emits no exception. 2D shapes
//     have no 3D mesh and a renderer asking for one is reachable from real
//     user data (toggling "Show shapes" on a 2D RVE) — throwing across a
//     thread/renderer boundary would crash the GUI. The renderer skips
//     empty meshes silently.
//   * register_shape<Shape>() is idempotent; calling twice replaces the
//     entry rather than failing.
//   * clear() empties the table for test isolation.

#include <cassert>
#include <cstddef>
#include <vector>

#include <numsim-core/static_indexing.h>

#include "../shapes/shape_base.h"
#include "triangle_mesh.h"

namespace rvegen {

template <typename T = double>
class mesh_dispatcher {
public:
  using fn_t = triangle_mesh<T> (*)(shape_base<T> const&);

  mesh_dispatcher(mesh_dispatcher const&) = delete;
  mesh_dispatcher& operator=(mesh_dispatcher const&) = delete;

  static mesh_dispatcher& instance() {
    static mesh_dispatcher d;
    return d;
  }

  // Register a concrete shape's free `to_mesh(Shape const&)` overload.
  // Idempotent — calling twice with the same Shape replaces the entry.
  template <typename Shape>
  void register_shape() {
    grow_to_fit(Shape::m_id);
    _table[Shape::m_id] = &dispatch<Shape>;
  }

  // Convert a polymorphic shape to a triangle mesh. Returns an empty mesh
  // (verts.empty() && tris.empty()) when the shape's id is out of range or
  // unregistered — typical for 2D shapes and a sentinel the renderer can
  // recognise via `triangle_mesh::empty()`.
  [[nodiscard]] triangle_mesh<T>
  operator()(shape_base<T> const& s) const {
    const auto idx = static_cast<std::size_t>(s.shape_id());
    if (idx >= _table.size() || !_table[idx]) return {};
    return _table[idx](s);
  }

  // Number of registered shape conversions. Diagnostic use.
  [[nodiscard]] std::size_t size() const noexcept {
    std::size_t n = 0;
    for (auto fn : _table) if (fn) ++n;
    return n;
  }

  void clear() noexcept { _table.clear(); }

private:
  mesh_dispatcher() = default;

  void grow_to_fit(numsim_core::type_id max_id) {
    const std::size_t needed = static_cast<std::size_t>(max_id) + 1;
    if (_table.size() < needed) _table.resize(needed, nullptr);
  }

  template <typename Shape>
  static triangle_mesh<T> dispatch(shape_base<T> const& s) {
    // Defensive guard against mis-registration: the registered slot at
    // Shape::m_id is supposed to match the actual dynamic type of `s`.
    // If it doesn't, the static_cast below is UB. In release builds this
    // is zero-cost; in debug builds the assertion catches it immediately.
    assert(dynamic_cast<Shape const*>(&s) != nullptr &&
           "mesh_dispatcher: shape type mismatch with registered slot");
    return to_mesh(static_cast<Shape const&>(s));   // ADL on free function
  }

  std::vector<fn_t> _table;
};

} // namespace rvegen
