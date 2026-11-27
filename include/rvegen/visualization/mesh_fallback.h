#pragma once

// Shared "render through mesh_dispatcher or warn" helper for the 3D writers
// (three_js_writer, svg_3d_writer) and the 2D svg_writer.
//
// Policy for shapes a writer cannot show: emit ONE warning on std::cerr per
// (writer, concrete shape type) per process and skip the shape. Never throw
// (a post-process must not abort a run over a cosmetic output) and never drop
// silently. `reset_unrendered_warnings()` re-arms the warnings (tests).

#include <cstddef>
#include <cstdlib>
#include <memory>
#if defined(__GNUC__) || defined(__clang__)
#include <cxxabi.h>
#endif
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
// Demangled type name where the ABI supports it, raw name otherwise.
inline std::string readable_type_name(std::type_info const& ti) {
#if defined(__GNUC__) || defined(__clang__)
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> p{
      abi::__cxa_demangle(ti.name(), nullptr, nullptr, &status), std::free};
  if (status == 0 && p) return p.get();
#endif
  return ti.name();
}

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
                                 char const* writer_name,
                                 char const* reason =
                                     "no mesh registered / not drawable") {
  auto& r = detail::unrendered_registry::instance();
  std::lock_guard<std::mutex> lk{r.mtx};
  if (!r.seen.emplace(writer_name, std::type_index{typeid(shape)}).second) return;
  std::cerr << "rvegen: " << writer_name << ": skipping unsupported shape type '"
            << detail::readable_type_name(typeid(shape))
            << "' (" << reason << "); further shapes of this"
               " type are skipped silently\n";
}

// Warn on stderr that a writer was handed a very large number of shapes
// (call once per writer run). `bytes_per_shape` is the measured output size.
inline void warn_large_output(char const* writer_name, std::size_t n_shapes,
                              std::size_t threshold,
                              std::size_t bytes_per_shape) {
  if (n_shapes <= threshold) return;
  std::cerr << "rvegen: " << writer_name << ": " << n_shapes
            << " shapes (above " << threshold << "); expect roughly "
            << (n_shapes * bytes_per_shape) / (1024 * 1024)
            << " MB of output (~" << bytes_per_shape
            << " bytes per shape, warning threshold ~50 MB); the file may be"
               " slow or impossible to open\n";
}

// Mesh `shape` via the dispatcher (auto-registering built-ins if the
// dispatcher is empty). Returns an empty mesh and warns once if the
// shape cannot be meshed.
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
mesh_or_warn(shape_base<T> const& shape, char const* writer_name) {
  auto& d = mesh_dispatcher<T>::instance();
  // Nobody else populates the singleton in library code; if the host app
  // has no conversion for this shape, install any missing built-ins
  // (host registrations, including overrides of built-ins, are kept). The
  // check-and-register runs under a lock so concurrent writers do not race
  // on the table; the lock also orders the registration before every
  // writer's subsequent read. (Hosts that register their own shapes must
  // still do so before starting writers.)
  {
    static std::mutex reg_mtx;
    static std::size_t seen_generation = static_cast<std::size_t>(-1);
    std::lock_guard<std::mutex> lk{reg_mtx};
    // Only touch the table when it changed since the last check, so the
    // steady state is read-only (no writes racing with concurrent meshing).
    if (!d.handles(shape) && d.generation() != seen_generation) {
      register_missing_meshes<T>();
      seen_generation = d.generation();
    }
  }
  auto m = d(shape);
  if (m.empty()) warn_unrendered_once(shape, writer_name);
  return m;
}

} // namespace rvegen
