#pragma once

// Registration-time guard for the static_indexing ids the dispatchers use
// as table keys. collision_dispatcher and mesh_dispatcher index flat
// tables by `Shape::m_id`; if two different shape types ever report the
// same id, the second registration silently overwrites the first and
// every later dispatch static_casts to the wrong type (seen with clang,
// where numsim-core's static_indexing handed every type id 0 and circles
// were dispatched as box x sphere). Claiming each id here at registration
// turns that into a loud std::logic_error instead.
//
// Only registration pays for this (one std::type_info compare per
// claim); the dispatch hot path is unchanged.

#include <cstddef>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

#include <numsim-core/static_indexing.h>

namespace rvegen {

class shape_id_claims {
public:
  // Record that `Shape` owns id `Shape::m_id`. Claiming the same id for
  // the same type again is a no-op (registration is idempotent); claiming
  // it for a different type throws.
  template <typename Shape>
  void claim(char const* owner) {
    claim(static_cast<std::size_t>(Shape::m_id), typeid(Shape), owner);
  }

  void claim(std::size_t id, std::type_info const& type, char const* owner) {
    if (_owner.size() <= id) _owner.resize(id + 1, nullptr);
    std::type_info const* held = _owner[id];
    if (held == nullptr) {
      _owner[id] = &type;
      return;
    }
    if (*held == type) return;
    throw std::logic_error(
        std::string(owner) + ": shape types '" + held->name() + "' and '" +
        type.name() + "' share static_indexing id " + std::to_string(id) +
        "; dispatch tables keyed by that id would mix them up. The ids come "
        "from numsim_core::static_indexing and must be unique per type "
        "(known clang bug in numsim-core where every id is 0).");
  }

  void clear() noexcept { _owner.clear(); }

private:
  std::vector<std::type_info const*> _owner;
};

} // namespace rvegen
