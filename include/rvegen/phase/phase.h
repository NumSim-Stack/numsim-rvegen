#pragma once

// Phase — a named region of the RVE that carries an opaque material
// configuration. rvegen does not interpret material parameters; the
// `material_config` JSON blob is forwarded verbatim to downstream
// FFT / FEM solvers (notably numsim-materials, which provides
// linear_elasticity, plasticity, damage, etc., keyed off the same
// JSON shape that goes through its `material_factory`).
//
// Why a separate type and not just a JSON blob:
//   * Output writers (voxel, gmsh, DAMASK) need a stable integer id
//     per phase to label cells / Physical groups / sections. Doing the
//     name → id mapping centrally in `phase_collection` avoids every
//     writer reinventing it (and disagreeing about ids).
//   * Shapes carry a phase NAME, not the full material_config. This
//     keeps shape_input small and lets multiple shapes share a phase
//     by name without copying material parameters.
//
// What this header lands today (foundation for #7):
//   * `phase<T>` struct — name + integer id + opaque material_config.
//   * `phase_collection<T>` — name → phase map with monotonic id
//     allocation.
//   * `load_phases_from_json` — builds a phase_collection from the
//     `"phases"` array in an rvegen JSON config.
//
// Out of scope here, ships in follow-up PRs against #7:
//   * `shape_input_base::phase_name()` field + JSON schema entry on
//     each input.
//   * voxel_writer / gmsh_geo_writer per-phase ids (needs (a)).
//   * DAMASK material.config writer.
//   * Hard validation against numsim-materials' factory (currently
//     opaque pass-through).

#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace rvegen {

template <typename T = double>
struct phase {
  using value_type = T;

  std::string name;
  std::size_t id{0};
  // Forwarded verbatim to downstream solvers — typically the JSON shape
  // expected by numsim-materials' material_factory (e.g. linear_elasticity
  // with K and G fields). rvegen treats this as opaque metadata.
  nlohmann::json material_config;
};

// Name → phase map with monotonic id allocation. Phase 0 is reserved
// for "void" / matrix gaps in voxel writers; user phases start at 1.
template <typename T = double>
class phase_collection {
public:
  using value_type = T;
  using phase_type = phase<value_type>;

  phase_collection() = default;

  // Adds a new phase. The id is auto-assigned and monotonic in
  // insertion order, starting at 1. Throws if the name is already
  // present (an early signal that the user double-defined a phase
  // in JSON).
  phase_type& add(std::string name, nlohmann::json material_config = {}) {
    if (_by_name.contains(name)) {
      throw std::runtime_error{"phase_collection: duplicate phase name '" +
                               name + "'"};
    }
    phase_type p{name, _next_id++, std::move(material_config)};
    auto [it, _] = _by_name.emplace(std::move(name), std::move(p));
    _ordered.push_back(&it->second);
    return it->second;
  }

  [[nodiscard]] bool contains(std::string const& name) const noexcept {
    return _by_name.contains(name);
  }

  [[nodiscard]] phase_type const& at(std::string const& name) const {
    auto it = _by_name.find(name);
    if (it == _by_name.end()) {
      throw std::runtime_error{"phase_collection: unknown phase '" + name +
                               "'"};
    }
    return it->second;
  }

  [[nodiscard]] std::size_t id_of(std::string const& name) const {
    return at(name).id;
  }

  [[nodiscard]] std::size_t size() const noexcept { return _ordered.size(); }
  [[nodiscard]] bool empty() const noexcept { return _ordered.empty(); }

  // Stable insertion-order iteration. Writers consuming this for
  // gmsh Physical groups / DAMASK sections want deterministic order,
  // so we expose a span over pointers into our owning map.
  [[nodiscard]] std::vector<phase_type const*> const& ordered() const noexcept {
    return _ordered_const_view();
  }

private:
  // Re-cast cache to const* on demand so callers can't mutate via the
  // returned vector but we still keep insertion order without
  // duplicating storage.
  std::vector<phase_type const*> const& _ordered_const_view() const noexcept {
    if (_ordered_const_cache.size() != _ordered.size()) {
      _ordered_const_cache.assign(_ordered.begin(), _ordered.end());
    }
    return _ordered_const_cache;
  }

  std::unordered_map<std::string, phase_type> _by_name;
  std::vector<phase_type*> _ordered;
  mutable std::vector<phase_type const*> _ordered_const_cache;
  std::size_t _next_id{1};
};

// Parses a JSON `"phases"` array into a phase_collection.
//
// Expected shape:
//   "phases": [
//     {"name": "matrix", "material": {"type": "linear_elasticity",
//                                       "K": 1.6e9, "G": 0.8e9, ...}},
//     {"name": "fibre",  "material": {"type": "linear_elasticity",
//                                       "K": 5.0e10, "G": 3.0e10, ...}}
//   ]
//
// The `"material"` sub-object is opaque — rvegen does not validate
// or instantiate it; downstream consumers (e.g. numsim-materials'
// factory) interpret it.
template <typename T = double>
inline phase_collection<T> load_phases_from_json(nlohmann::json const& phases_array) {
  phase_collection<T> phases;
  if (!phases_array.is_array()) {
    throw std::runtime_error{
        "load_phases_from_json: expected an array of phase objects"};
  }
  for (auto const& entry : phases_array) {
    if (!entry.contains("name") || !entry["name"].is_string()) {
      throw std::runtime_error{
          "load_phases_from_json: each phase must have a string 'name'"};
    }
    auto name = entry["name"].get<std::string>();
    auto material = entry.contains("material") ? entry["material"]
                                                : nlohmann::json::object();
    phases.add(std::move(name), std::move(material));
  }
  return phases;
}

} // namespace rvegen
