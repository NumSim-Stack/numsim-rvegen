#pragma once

// Phase — a named geometric region of the RVE (e.g. "matrix", "fibre").
// A phase is a label only: rvegen generates geometry and does not carry
// material properties. Downstream solvers assign materials by phase name
// or id.
//
// Why a separate type:
//   * Output writers (voxel, gmsh, vtk) need a stable integer id per
//     phase to label cells / Physical groups. Doing the name → id mapping
//     centrally in `phase_collection` avoids every writer reinventing it
//     (and disagreeing about ids).
//   * Shapes carry a phase NAME, so multiple shapes share a phase by name.
//
// Convention on phase ids:
//   User-defined phases get monotonic ids starting at 1 (gmsh physical
//   tags must be positive, so a phase can never be id 0). Id 0 means
//   "untagged". If the config declares which phase is the matrix
//   (`phase_assignment.matrix`, recorded via `set_matrix`), the voxel and
//   VTK writers give cells that no shape covers THAT phase's id; without
//   a declared matrix phase, uncovered cells stay 0 (matrix / untagged).

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
  // `_ordered` holds pointers into `_by_name`; a copy would point into the
  // source. Moving keeps the (node-based) map nodes, so moves stay valid.
  phase_collection(phase_collection const&) = delete;
  phase_collection& operator=(phase_collection const&) = delete;
  phase_collection(phase_collection&&) noexcept = default;
  phase_collection& operator=(phase_collection&&) noexcept = default;

  // Adds a new phase. The id is auto-assigned and monotonic in
  // insertion order, starting at 1. Throws if the name is already
  // present (an early signal that the user double-defined a phase
  // in JSON).
  phase_type& add(std::string name) {
    if (_by_name.contains(name)) {
      throw std::runtime_error{"phase_collection: duplicate phase name '" +
                               name + "'"};
    }
    auto& slot = _by_name
                     .try_emplace(name, phase_type{name, _next_id++})
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

  // Declare which phase is the matrix (the region no shape covers).
  // Throws if the phase is unknown.
  void set_matrix(std::string const& name) {
    (void)at(name);
    _matrix = name;
  }
  [[nodiscard]] bool has_matrix() const noexcept { return !_matrix.empty(); }
  [[nodiscard]] std::string const& matrix_name() const noexcept {
    return _matrix;
  }
  // Id carried by uncovered cells: the matrix phase's id, or 0 when no
  // matrix phase is declared.
  [[nodiscard]] std::size_t matrix_id() const {
    return _matrix.empty() ? std::size_t{0} : id_of(_matrix);
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
  std::string _matrix;
};

// Parses a JSON `"phases"` array into a phase_collection.
//
// Expected shape:
//   "phases": [ {"name": "matrix"}, {"name": "fibre"} ]
//
// Validation:
//   * `phases_array` must be an array.
//   * Each entry must be an object with a non-empty string `"name"` and no other
//     keys. A `"material"` key is rejected explicitly: material
//     properties are out of scope for rvegen and belong to the
//     downstream solver, keyed by phase name or id.
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
    if (name.empty()) {
      throw std::runtime_error{
          "load_phases_from_json: phase 'name' must not be empty"};
    }
    for (auto const& [key, value] : entry.items()) {
      if (key == "name") continue;
      if (key == "material") {
        throw std::runtime_error{
            "load_phases_from_json: phase '" + name +
            "' has a 'material' entry; rvegen generates geometry only, so "
            "phases carry just a name. Assign materials in the downstream "
            "solver by phase name or id."};
      }
      throw std::runtime_error{"load_phases_from_json: phase '" + name +
                               "' has unknown key '" + key +
                               "' (only 'name' is allowed)"};
    }
    phases.add(std::move(name));
  }
  return phases;
}

} // namespace rvegen
