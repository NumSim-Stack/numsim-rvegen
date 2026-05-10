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
// Convention on phase ids:
//   User-defined phases get monotonic ids starting at 1. Id 0 is held
//   in reserve for "void" / matrix gaps so a future voxel writer can
//   use it as the default-fill marker without colliding with a real
//   phase. Today no writer consumes this convention; it is a
//   forward-looking reservation.

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

// Name → phase map with monotonic id allocation. `ordered()` exposes
// insertion-order iteration directly (no const-cache, so concurrent
// readers after the build phase are safe).
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
    auto& slot = _by_name
                     .try_emplace(name, phase_type{name, _next_id++,
                                                   std::move(material_config)})
                     .first->second;
    _ordered.push_back(&slot);
    return slot;
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

  // Stable insertion-order view. Pointers are valid for the lifetime
  // of the collection — `std::unordered_map` references are stable
  // across `insert`/`try_emplace`, and we never erase. Reading
  // `ordered()` from multiple threads after the build phase is
  // finished is safe (no mutation, no lazy cache rebuild).
  [[nodiscard]] std::vector<phase_type const*> const& ordered() const noexcept {
    return _ordered;
  }

private:
  std::unordered_map<std::string, phase_type> _by_name;
  std::vector<phase_type const*> _ordered;
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
// Validation:
//   * `phases_array` must be an array.
//   * Each entry must be an object with a string `"name"`.
//   * `"material"` is required to be an object if present (rvegen does
//     not interpret its contents, but accepting non-objects would let
//     downstream solver mismatches surface as obscure errors far from
//     the typo).
template <typename T = double>
inline phase_collection<T> load_phases_from_json(nlohmann::json const& phases_array) {
  phase_collection<T> phases;
  if (!phases_array.is_array()) {
    throw std::runtime_error{
        "load_phases_from_json: expected an array of phase objects"};
  }
  for (auto const& entry : phases_array) {
    if (!entry.is_object()) {
      throw std::runtime_error{
          "load_phases_from_json: each phase entry must be an object"};
    }
    if (!entry.contains("name") || !entry["name"].is_string()) {
      throw std::runtime_error{
          "load_phases_from_json: each phase must have a string 'name'"};
    }
    auto name = entry["name"].get<std::string>();
    nlohmann::json material = nlohmann::json::object();
    if (entry.contains("material")) {
      if (!entry["material"].is_object()) {
        throw std::runtime_error{
            "load_phases_from_json: 'material' for phase '" + name +
            "' must be an object"};
      }
      material = entry["material"];
    }
    phases.add(std::move(name), std::move(material));
  }
  return phases;
}

} // namespace rvegen
