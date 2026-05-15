#pragma once

// Mean-field analytical homogenization — Voigt and Reuss bounds.
//
// Given a multi-phase composite with phase stiffness matrices C_i (in
// 6×6 Voigt form) and volume fractions v_i (summing to 1), these are
// the textbook arithmetic and harmonic-mean bounds on the effective
// stiffness:
//
//   C_voigt = Σ v_i · C_i                          (upper bound, parallel rule of mixtures)
//   C_reuss = (Σ v_i · C_i^{-1})^{-1}              (lower bound, series rule of mixtures)
//
// They bracket any micromechanically realisable effective stiffness;
// the gap between them widens with phase contrast. For an isotropic
// quick estimate, a Voigt-Reuss-Hill average ((C_voigt + C_reuss)/2)
// is also exposed.
//
// Why this lives here, not in a constitutive lib:
//   * rvegen ships RVEs to FE/FFT solvers; downstream
//     numsim-materials models are full constitutive laws (linear
//     elasticity, plasticity, damage). The Voigt/Reuss bounds are an
//     averaging operation over a list of stiffnesses + volume
//     fractions, not a constitutive law themselves.
//   * Phase stiffness matrices come in via the phase model (#7) once
//     wired. For now this header takes raw 6×6 Voigt matrices so it
//     can stand alone.
//
// What this header lands today:
//   * Eigen-based 6×6 stiffness type alias.
//   * `voigt_bound` / `reuss_bound` / `voigt_reuss_hill` over a list
//     of phases.
//   * Free helpers `lame_to_voigt_stiffness` and
//     `isotropic_voigt_stiffness` to turn (λ, μ) or (K, G) into a
//     Voigt 6×6.
//   * `mori_tanaka` — Mori-Tanaka mean-field estimate for spherical
//     isotropic inclusions in an isotropic matrix (the textbook
//     closed-form case). Coincides with one of the Hashin-Shtrikman
//     bounds depending on whether the matrix is the soft or stiff
//     phase.
//
// Out of scope here, ships in follow-up PRs against #5:
//   * Hashin-Shtrikman bounds for general non-spherical inclusion
//     morphology.
//   * Mori-Tanaka for anisotropic phases / non-spherical inclusions
//     (needs a per-shape Eshelby tensor).
//   * Self-consistent (Hill) scheme — requires iterative solve.
//   * Wiring directly into the phase model so a "homogenize" post-
//     process consumes per-phase numsim-materials configs.
//   * 2D plane-stress / plane-strain reductions of the 3D bounds.

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace rvegen::homogenization {

template <typename T = double>
using stiffness_matrix = Eigen::Matrix<T, 6, 6>;

// Builds the Voigt-form 6×6 isotropic stiffness from Lamé parameters.
// Top-left 3×3 block is (λ + 2μ) on the diagonal, λ on the off-diagonals;
// bottom-right 3×3 block is μ on the diagonal (shear). Useful for tests
// and for callers who only have (E, ν) or (K, G) — convert to (λ, μ)
// first.
template <typename T = double>
[[nodiscard]] stiffness_matrix<T> lame_to_voigt_stiffness(T lambda, T mu) {
  stiffness_matrix<T> C = stiffness_matrix<T>::Zero();
  const T a = lambda + T{2} * mu;
  C(0, 0) = C(1, 1) = C(2, 2) = a;
  C(0, 1) = C(0, 2) = C(1, 0) = C(1, 2) = C(2, 0) = C(2, 1) = lambda;
  C(3, 3) = C(4, 4) = C(5, 5) = mu;
  return C;
}

// Convenience wrapper from bulk + shear moduli. λ = K - 2G/3 is the
// canonical bulk-to-Lamé conversion.
template <typename T = double>
[[nodiscard]] stiffness_matrix<T> isotropic_voigt_stiffness(T K, T G) {
  return lame_to_voigt_stiffness<T>(K - T{2} * G / T{3}, G);
}

// Voigt upper bound: C = Σ v_i · C_i. Volume fractions are not
// auto-normalised — caller is responsible for ensuring they sum to 1
// (the typical use case is `phase_volumes / total_volume` from the
// generator output, which is already normalised).
template <typename T = double>
[[nodiscard]] stiffness_matrix<T> voigt_bound(
    std::vector<stiffness_matrix<T>> const& stiffnesses,
    std::vector<T> const& volume_fractions) {
  if (stiffnesses.size() != volume_fractions.size()) {
    throw std::runtime_error{
        "voigt_bound: stiffnesses and volume_fractions size mismatch"};
  }
  if (stiffnesses.empty()) {
    throw std::runtime_error{"voigt_bound: at least one phase required"};
  }
  stiffness_matrix<T> C = stiffness_matrix<T>::Zero();
  for (std::size_t i = 0; i < stiffnesses.size(); ++i) {
    C.noalias() += volume_fractions[i] * stiffnesses[i];
  }
  return C;
}

// Reuss lower bound: C = (Σ v_i · C_i^{-1})^{-1}. Inverts each phase
// matrix via `FullPivLU` and uses its rank-revealing `isInvertible()`
// check, which is robust at engineering scales — comparing
// `determinant()` against `epsilon` would be meaningless for a 6×6
// stiffness with ~1e9 entries (det ~1e54). Throws if any phase is
// singular (e.g. a void phase with zero stiffness — for that case the
// user should drop the void from the average and renormalise the
// remaining phases, since the harmonic mean diverges).
template <typename T = double>
[[nodiscard]] stiffness_matrix<T> reuss_bound(
    std::vector<stiffness_matrix<T>> const& stiffnesses,
    std::vector<T> const& volume_fractions) {
  if (stiffnesses.size() != volume_fractions.size()) {
    throw std::runtime_error{
        "reuss_bound: stiffnesses and volume_fractions size mismatch"};
  }
  if (stiffnesses.empty()) {
    throw std::runtime_error{"reuss_bound: at least one phase required"};
  }
  stiffness_matrix<T> S = stiffness_matrix<T>::Zero();
  for (std::size_t i = 0; i < stiffnesses.size(); ++i) {
    Eigen::FullPivLU<stiffness_matrix<T>> lu{stiffnesses[i]};
    if (!lu.isInvertible()) {
      throw std::runtime_error{
          "reuss_bound: a phase stiffness is singular (e.g. void); "
          "drop it from the input and renormalise volume fractions"};
    }
    S.noalias() += volume_fractions[i] * lu.inverse();
  }
  return Eigen::FullPivLU<stiffness_matrix<T>>{S}.inverse();
}

// Voigt-Reuss-Hill average: (C_voigt + C_reuss) / 2. A common
// engineering rule-of-thumb estimate that sits between the bounds.
// Whether it lands closer to experiment than either bound alone is
// morphology-dependent and not a property of the average itself —
// for tightly-bracketed cases (low contrast) the bounds are close so
// it doesn't matter much; for high-contrast composites with one phase
// percolating the experiment can favour either bound.
template <typename T = double>
[[nodiscard]] stiffness_matrix<T> voigt_reuss_hill(
    std::vector<stiffness_matrix<T>> const& stiffnesses,
    std::vector<T> const& volume_fractions) {
  return (voigt_bound(stiffnesses, volume_fractions) +
          reuss_bound(stiffnesses, volume_fractions)) /
         T{2};
}

// Mori-Tanaka effective bulk and shear moduli for `n` isotropic
// spherical inclusion phases in an isotropic matrix. Returns the
// pair (K_eff, G_eff). For each inclusion i with bulk K_i, shear G_i
// and volume fraction v_i (where Σ v_i < 1 = inclusion total + matrix
// fills the rest):
//
//   K_eff = K_m + Σ v_i (K_i - K_m) / (1 + α_K · (K_i - K_m) / K_m')
//   G_eff = G_m + Σ v_i (G_i - G_m) / (1 + α_G · (G_i - G_m) / G_m')
//
// where the standard Eshelby-spherical alpha factors are
//   K_m'  = K_m + 4·G_m / 3                       (longitudinal modulus / 3)
//   G_m'  = G_m + G_m·(9·K_m + 8·G_m) / (6·(K_m + 2·G_m))
//   α_K   = 1 - v_total_inclusions     (= matrix volume fraction)
//   α_G   = 1 - v_total_inclusions
//
// This is the well-known dilute-strain-concentration form summed over
// multiple inclusion phases. Single-inclusion-type derivations are
// in Mori & Tanaka (1973) and Benveniste (1987). The multi-inclusion
// extension is straightforward: each inclusion's contribution is
// linear in its volume fraction and uses the *matrix* moduli (not
// the running average), so the sum is well-defined.
//
// Throws if Σ v_i > 1 (inclusions would over-fill the matrix), if
// any v_i < 0, or if K_m + 4G_m/3 (≈ longitudinal stiffness) is
// non-positive — the latter would indicate a non-physical matrix.
template <typename T = double>
[[nodiscard]] std::pair<T, T> mori_tanaka_moduli(
    T K_matrix, T G_matrix,
    std::vector<T> const& K_inclusions,
    std::vector<T> const& G_inclusions,
    std::vector<T> const& volume_fractions) {
  if (K_inclusions.size() != G_inclusions.size() ||
      K_inclusions.size() != volume_fractions.size()) {
    throw std::runtime_error{
        "mori_tanaka: K_inclusions / G_inclusions / volume_fractions "
        "size mismatch"};
  }
  if (K_matrix + T{4} * G_matrix / T{3} <= T{0}) {
    throw std::runtime_error{
        "mori_tanaka: matrix longitudinal modulus (K + 4G/3) must be positive"};
  }
  T v_inclusions_total{0};
  for (auto v : volume_fractions) {
    if (v < T{0}) {
      throw std::runtime_error{
          "mori_tanaka: negative volume fraction"};
    }
    v_inclusions_total += v;
  }
  // Allow a tiny float slop above 1 to absorb rounding from upstream
  // volume_fraction normalisation, but reject anything materially
  // over-filling the matrix.
  if (v_inclusions_total > T{1} + T{1e-9}) {
    throw std::runtime_error{
        "mori_tanaka: inclusion volume fractions sum to >1 — matrix "
        "would have negative fraction"};
  }
  // Eshelby-spherical alpha factors for the matrix.
  const T K_m_eff = K_matrix + T{4} * G_matrix / T{3};
  const T G_m_eff = G_matrix +
      G_matrix * (T{9} * K_matrix + T{8} * G_matrix) /
      (T{6} * (K_matrix + T{2} * G_matrix));
  const T alpha = T{1} - v_inclusions_total;   // matrix volume fraction

  T K_eff = K_matrix;
  T G_eff = G_matrix;
  for (std::size_t i = 0; i < K_inclusions.size(); ++i) {
    const T dK = K_inclusions[i] - K_matrix;
    const T dG = G_inclusions[i] - G_matrix;
    K_eff += volume_fractions[i] * dK / (T{1} + alpha * dK / K_m_eff);
    G_eff += volume_fractions[i] * dG / (T{1} + alpha * dG / G_m_eff);
  }
  return {K_eff, G_eff};
}

// Mori-Tanaka effective 6×6 isotropic stiffness — convenience wrapper
// that delegates to `mori_tanaka_moduli` and assembles the result via
// `isotropic_voigt_stiffness`. The Voigt-form output makes it
// drop-in-comparable with `voigt_bound` / `reuss_bound`.
template <typename T = double>
[[nodiscard]] stiffness_matrix<T> mori_tanaka(
    T K_matrix, T G_matrix,
    std::vector<T> const& K_inclusions,
    std::vector<T> const& G_inclusions,
    std::vector<T> const& volume_fractions) {
  const auto [K_eff, G_eff] = mori_tanaka_moduli<T>(
      K_matrix, G_matrix, K_inclusions, G_inclusions, volume_fractions);
  return isotropic_voigt_stiffness<T>(K_eff, G_eff);
}

} // namespace rvegen::homogenization
