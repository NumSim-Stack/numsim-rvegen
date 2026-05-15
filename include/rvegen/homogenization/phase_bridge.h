#pragma once

// Bridge between `phase_collection` + generated shape vector and the
// analytical-homogenization routines in `mean_field.h`.
//
// The mean-field routines (`voigt_bound`, `mori_tanaka`, `hashin_*`,
// `self_consistent_moduli`) all take parallel `vector<T>` inputs
// indexed by phase. This header provides two primitives that turn a
// rvegen pipeline state (shapes + phase_collection) into those
// inputs:
//
//   * `compute_phase_volume_fractions(shapes, phases, box, nx, ny, nz)`
//     samples a voxel grid (same machinery as voxel_writer) and
//     counts voxels per phase id. Returns a `phase_volume_breakdown`
//     carrying `per_phase[i]` for the i-th phase in
//     `phases.ordered()` plus an `untagged_fraction` for voxels
//     outside every shape or inside an untagged shape.
//
//   * `extract_isotropic_moduli(phases)` walks `phases.ordered()`
//     and pulls (K, G) out of each phase's `material_config` blob.
//     The blob must look like
//       {"type": "linear_elasticity", "K": ..., "G": ...}
//     matching numsim-materials' linear-elasticity entry. Anisotropic
//     phases / non-elastic phases throw with a clear message.
//
// Composition: feeding `breakdown.per_phase` and the (K, G) vectors
// into any of the mean-field routines is one line. Method-dispatch
// wrappers (`homogenize_voigt` etc.) are deliberately NOT in this
// header — they belong in a follow-up that also tackles the
// "where do untagged voxels go" policy question (a matrix-phase
// convention, an explicit matrix-phase name argument, or
// renormalisation). For now the breakdown surfaces both numbers so
// the caller picks the policy.

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "../phase/phase.h"
#include "../postprocess/voxel_grid.h"
#include "../shapes/shape_base.h"

namespace rvegen::homogenization {

template <typename T = double>
struct phase_volume_breakdown {
  // Volume fractions for each phase in the collection, in
  // `phases.ordered()` order (= insertion order). Each entry is the
  // fraction of voxels assigned to that phase id by
  // `sample_voxel_grid(..., phases)`.
  std::vector<T> per_phase;

  // Fraction of voxels that ended up with phase id 0 — i.e. either
  // outside every shape (background) or inside a shape whose
  // `phase_name()` was empty (untagged). Callers typically attribute
  // this to the matrix phase, but the policy choice is theirs.
  T untagged_fraction{};
};

// Sample the RVE's voxel grid and return the per-phase + untagged
// volume-fraction breakdown.
//
// `nx`, `ny`, `nz` control sampling resolution. Higher values give
// tighter estimates at quadratic-or-cubic cost. The defaults
// (64×64×64 for 3D, 64×64×1 for 2D) match the order-of-magnitude
// scale used elsewhere in the codebase.
//
// Throws if a shape carries a `phase_name` not in `phases` (typo
// guard, same as `sample_voxel_grid`'s phase-aware overload).
template <typename T>
[[nodiscard]] phase_volume_breakdown<T> compute_phase_volume_fractions(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    phase_collection<T> const& phases,
    std::array<T, 3> const& domain_box,
    std::size_t nx = 64, std::size_t ny = 64, std::size_t nz = 64) {
  const auto grid = sample_voxel_grid(shapes, domain_box, nx, ny, nz, phases);
  const auto total = static_cast<T>(grid.size());

  phase_volume_breakdown<T> out;
  out.per_phase.assign(phases.ordered().size(), T{0});

  // Map phase id → index in `per_phase` (= position in `ordered()`).
  // Most phase ids fit in a small int range; use a flat vector
  // indexed by id with `npos` sentinel for unused ids.
  std::size_t max_id = 0;
  for (auto const* p : phases.ordered()) {
    if (p->id > max_id) max_id = p->id;
  }
  constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> id_to_index(max_id + 1, npos);
  for (std::size_t i = 0; i < phases.ordered().size(); ++i) {
    id_to_index[phases.ordered()[i]->id] = i;
  }

  std::size_t untagged_count = 0;
  for (auto v : grid) {
    if (v == 0) {
      ++untagged_count;
      continue;
    }
    if (v >= id_to_index.size() || id_to_index[v] == npos) {
      // Should be impossible because sample_voxel_grid only writes
      // ids it resolved from phases.id_of(). Defensive throw to make
      // a future invariant change loud.
      throw std::runtime_error{
          "compute_phase_volume_fractions: voxel carries phase id "
          + std::to_string(v) + " that doesn't map to any phase in "
          "the collection — invariant violation"};
    }
    ++out.per_phase[id_to_index[v]];
  }
  for (auto& f : out.per_phase) f /= total;
  out.untagged_fraction = static_cast<T>(untagged_count) / total;
  return out;
}

template <typename T = double>
struct isotropic_phase_moduli {
  // Bulk modulus per phase, in `phases.ordered()` order.
  std::vector<T> K;
  // Shear modulus per phase, in `phases.ordered()` order.
  std::vector<T> G;
};

// Walk the phase collection and extract (K, G) from each phase's
// `material_config` JSON. The expected blob shape matches
// numsim-materials' linear-elasticity entry:
//
//   {"type": "linear_elasticity", "K": <bulk>, "G": <shear>}
//
// Throws with a clear message if any phase is missing a config,
// has a non-isotropic type, or omits K/G. Numeric fields may be int
// or float in the JSON — both round-trip through `nlohmann::json::get<T>`.
template <typename T>
[[nodiscard]] isotropic_phase_moduli<T> extract_isotropic_moduli(
    phase_collection<T> const& phases) {
  isotropic_phase_moduli<T> out;
  out.K.reserve(phases.ordered().size());
  out.G.reserve(phases.ordered().size());
  for (auto const* p : phases.ordered()) {
    auto const& cfg = p->material_config;
    if (cfg.is_null() || !cfg.is_object()) {
      throw std::runtime_error{
          "extract_isotropic_moduli: phase '" + p->name +
          "' has no material_config object — homogenization needs "
          "a {type: linear_elasticity, K, G} blob per phase"};
    }
    if (!cfg.contains("type") || cfg["type"] != "linear_elasticity") {
      const std::string seen = cfg.contains("type")
          ? cfg["type"].dump() : std::string{"<missing>"};
      throw std::runtime_error{
          "extract_isotropic_moduli: phase '" + p->name +
          "' has material type " + seen + "; only "
          "'linear_elasticity' is supported by the analytical "
          "homogenization routines"};
    }
    if (!cfg.contains("K") || !cfg.contains("G")) {
      throw std::runtime_error{
          "extract_isotropic_moduli: phase '" + p->name +
          "' is missing 'K' or 'G' in material_config"};
    }
    out.K.push_back(cfg["K"].template get<T>());
    out.G.push_back(cfg["G"].template get<T>());
  }
  return out;
}

} // namespace rvegen::homogenization
