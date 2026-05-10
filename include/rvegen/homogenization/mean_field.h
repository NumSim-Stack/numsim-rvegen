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
// What this header lands today (phase 1 of #5):
//   * Eigen-based 6×6 stiffness type alias.
//   * `voigt_bound` / `reuss_bound` / `voigt_reuss_hill` over a list
//     of phases.
//   * Free helper `lame_to_voigt_stiffness` to turn (λ, μ) into a
//     Voigt 6×6 — useful in tests and for callers who only have
//     isotropic moduli.
//
// Out of scope here, ships in follow-up PRs against #5:
//   * Mori-Tanaka, self-consistent, Hashin-Shtrikman bounds (each
//     needs a per-shape Eshelby tensor; non-trivial).
//   * Wiring directly into the phase model so a "homogenize" post-
//     process consumes per-phase numsim-materials configs.
//   * 2D plane-stress / plane-strain reductions of the 3D bounds.

#include <cstddef>
#include <stdexcept>
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

} // namespace rvegen::homogenization
