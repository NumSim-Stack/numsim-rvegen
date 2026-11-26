#pragma once

// Shared "render through mesh_dispatcher or warn" helper for the 3D writers
// (three_js_writer, svg_3d_writer) and the 2D svg_writer.
//
// Policy for shapes a writer cannot show: emit ONE warning on std::cerr per
// (writer, concrete shape type) per process and skip the shape. Never throw
// (a post-process must not abort a run over a cosmetic output) and never drop
// silently. `reset_unrendered_warnings()` re-arms the warnings (tests).

#include <cstddef>
#include <iostream>
#include <mutex>
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <utility>

#include "../shapes/shape_base.h"
#include "../registry/register_meshes.h"
#include "mesh_dispatcher.h"
#include "triangle_mesh.h"

namespace rvegen {

namespace detail {
struct unrendered_registry {
  std::mutex mtx;
  std::set<std::pair<std::string, std::type_index>> seen;
  static unrendered_registry& instance() {
    static unrendered_registry r;
    return r;
  }
};
} // namespace detail

inline void reset_unrendered_warnings() {
  auto& r = detail::unrendered_registry::instance();
  std::lock_guard<std::mutex> lk{r.mtx};
  r.seen.clear();
}

// Warn (once per writer + dynamic type) that `shape` is not rendered.
template <typename T>
inline void warn_unrendered_once(shape_base<T> const& shape,
                                 char const* writer_name) {
  auto& r = detail::unrendered_registry::instance();
  std::lock_guard<std::mutex> lk{r.mtx};
  if (!r.seen.emplace(writer_name, std::type_index{typeid(shape)}).second) return;
  std::cerr << "rvegen: " << writer_name << ": skipping unsupported shape type '"
            << typeid(shape).name()
            << "' (no mesh registered / not drawable); further shapes of this"
               " type are skipped silently\n";
}

// Mesh `shape` via the dispatcher (auto-registering built-ins if the
// dispatcher is empty). Returns an empty mesh and warns once if the
// shape cannot be meshed.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
mesh_or_warn(shape_base<T> const& shape, char const* writer_name) {
  auto& d = mesh_dispatcher<T>::instance();
  // Nobody else populates the singleton in library code; if the host app
  // never registered anything, install the built-in shapes.
  if (d.size() == 0) register_all_meshes<T>();
  auto m = d(shape);
  if (m.empty()) warn_unrendered_once(shape, writer_name);
  return m;
}

} // namespace rvegen
