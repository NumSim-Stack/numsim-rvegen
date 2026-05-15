#pragma once

// One-call method-dispatch wrapper on top of the bridge + mean-field
// routines. Given an rvegen generation state — shape vector,
// `phase_collection`, domain box — and a `homogenization_method`
// enum, returns the effective (K, G).
//
// Internally:
//   1. `compute_phase_volume_fractions` samples a voxel grid and
//      returns the per-phase + untagged fractions.
//   2. Untagged voxels (background + untagged-shape voxels) are
//      attributed to a designated matrix phase identified by name
//      (default `"matrix"`). If the named phase doesn't exist in the
//      collection, we throw — better than silently dropping the
//      untagged fraction or renormalising without telling anyone.
//   3. `extract_isotropic_moduli` pulls (K, G) per phase.
//   4. Dispatches to the requested method:
//      - voigt / reuss / voigt_reuss_hill: build 6×6 isotropic
//        stiffness matrices per phase, then `voigt_bound` etc.
//      - mori_tanaka: matrix phase = the named one; everyone else
//        is an inclusion.
//      - hashin_shtrikman_lower / _upper: N-phase Berryman form.
//      - self_consistent: fixed-point iteration.
//
// All methods return `pair<K_eff, G_eff>` for API uniformity. Voigt
// / Reuss / VRH internally average 6×6 stiffness matrices and then
// pull (K, G) back out via the isotropic invariants of the result.

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../phase/phase.h"
#include "../shapes/shape_base.h"
#include "mean_field.h"
#include "phase_bridge.h"

namespace rvegen::homogenization {

enum class homogenization_method {
  voigt,
  reuss,
  voigt_reuss_hill,
  mori_tanaka,
  hashin_shtrikman_lower,
  hashin_shtrikman_upper,
  self_consistent
};

template <typename T = double>
struct homogenize_options {
  std::size_t nx{64};
  std::size_t ny{64};
  std::size_t nz{64};
  // The phase that absorbs background voxels and untagged-shape
  // voxels. Must exist in the collection. For Mori-Tanaka, this is
  // also the surrounding matrix; the other phases are inclusions.
  std::string matrix_phase_name{"matrix"};
};

namespace detail {

// Index of the matrix phase in `phases.ordered()`. Throws if the
// named phase isn't in the collection.
template <typename T>
[[nodiscard]] std::size_t matrix_phase_index(
    phase_collection<T> const& phases, std::string const& matrix_name) {
  auto const& ordered = phases.ordered();
  for (std::size_t i = 0; i < ordered.size(); ++i) {
    if (ordered[i]->name == matrix_name) return i;
  }
  throw std::runtime_error{
      "homogenize: matrix_phase_name '" + matrix_name +
      "' is not declared in the phase_collection — either add it or "
      "set opts.matrix_phase_name to a phase that exists."};
}

// Volume fractions with untagged voxels rolled into the matrix phase.
// Returned vector is parallel to `phases.ordered()` and sums to 1.
template <typename T>
[[nodiscard]] std::vector<T> volume_fractions_with_matrix_absorbing_untagged(
    phase_volume_breakdown<T> const& breakdown, std::size_t matrix_index) {
  std::vector<T> v = breakdown.per_phase;
  v[matrix_index] += breakdown.untagged_fraction;
  return v;
}

// Recover (K, G) from a 6×6 isotropic stiffness in Voigt notation:
//   C(0,0) = K + 4G/3,  C(0,1) = K − 2G/3,  C(3,3) = G
// (and the rest of the 6×6 follows from isotropy).
template <typename T>
[[nodiscard]] std::pair<T, T> recover_K_G(stiffness_matrix<T> const& C) {
  const T G = C(3, 3);
  // K = (C(0,0) + 2·C(0,1)) / 3 — the bulk modulus equals the trace
  // of the upper 3×3 stress-strain block divided by 9 (or any of
  // the equivalent forms; this one is numerically stablest because
  // it averages three off-diagonals).
  const T K = (C(0, 0) + T{2} * C(0, 1)) / T{3};
  return {K, G};
}

} // namespace detail

template <typename T = double>
[[nodiscard]] std::pair<T, T> homogenize(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    phase_collection<T> const& phases,
    std::array<T, 3> const& domain_box,
    homogenization_method method,
    homogenize_options<T> const& opts = {}) {
  if (phases.empty()) {
    throw std::runtime_error{
        "homogenize: phase_collection is empty"};
  }
  const auto matrix_idx = detail::matrix_phase_index<T>(
      phases, opts.matrix_phase_name);
  const auto breakdown = compute_phase_volume_fractions<T>(
      shapes, phases, domain_box, opts.nx, opts.ny, opts.nz);
  const auto volume_fractions =
      detail::volume_fractions_with_matrix_absorbing_untagged<T>(
          breakdown, matrix_idx);
  const auto moduli = extract_isotropic_moduli<T>(phases);

  switch (method) {
    case homogenization_method::voigt:
    case homogenization_method::reuss:
    case homogenization_method::voigt_reuss_hill: {
      // Build 6×6 isotropic stiffnesses per phase, then average.
      std::vector<stiffness_matrix<T>> Cs;
      Cs.reserve(moduli.K.size());
      for (std::size_t i = 0; i < moduli.K.size(); ++i) {
        Cs.push_back(isotropic_voigt_stiffness<T>(moduli.K[i], moduli.G[i]));
      }
      stiffness_matrix<T> C;
      if (method == homogenization_method::voigt) {
        C = voigt_bound<T>(Cs, volume_fractions);
      } else if (method == homogenization_method::reuss) {
        C = reuss_bound<T>(Cs, volume_fractions);
      } else {
        C = voigt_reuss_hill<T>(Cs, volume_fractions);
      }
      return detail::recover_K_G<T>(C);
    }
    case homogenization_method::mori_tanaka: {
      // The matrix phase is identified by `matrix_phase_name`. All
      // other phases are inclusions, in their `phases.ordered()` order.
      std::vector<T> K_inc, G_inc, v_inc;
      K_inc.reserve(moduli.K.size() - 1);
      G_inc.reserve(moduli.G.size() - 1);
      v_inc.reserve(volume_fractions.size() - 1);
      for (std::size_t i = 0; i < moduli.K.size(); ++i) {
        if (i == matrix_idx) continue;
        K_inc.push_back(moduli.K[i]);
        G_inc.push_back(moduli.G[i]);
        v_inc.push_back(volume_fractions[i]);
      }
      return mori_tanaka_moduli<T>(
          moduli.K[matrix_idx], moduli.G[matrix_idx],
          K_inc, G_inc, v_inc);
    }
    case homogenization_method::hashin_shtrikman_lower:
      return hashin_shtrikman_lower_n<T>(
          moduli.K, moduli.G, volume_fractions);
    case homogenization_method::hashin_shtrikman_upper:
      return hashin_shtrikman_upper_n<T>(
          moduli.K, moduli.G, volume_fractions);
    case homogenization_method::self_consistent:
      return self_consistent_moduli<T>(
          moduli.K, moduli.G, volume_fractions);
  }
  // Unreachable: switch is exhaustive over the enum.
  throw std::runtime_error{"homogenize: unknown method"};
}

} // namespace rvegen::homogenization
