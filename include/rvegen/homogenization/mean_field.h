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
// Hashin-Shtrikman (HS) bounds for **two-phase isotropic mixtures**
// with spherical morphology are also exposed (`hashin_shtrikman_lower`
// / `_upper` / `_bounds`). For two phases, HS- == MT-with-soft-as-
// matrix and HS+ == MT-with-stiff-as-matrix; the explicit HS API
// frees callers from picking a matrix and makes the variational
// bracket on the effective moduli direct.
//
// N-phase HS bounds (`hashin_shtrikman_lower_n` / `_upper_n` /
// `_bounds_n`) use the Berryman / Walpole closed form:
//
//   K_HS = (Σ v_i / (K_i + 4G*/3))^{-1} - 4G*/3
//   G_HS = (Σ v_i / (G_i + ξ*))^{-1} - ξ*
//     ξ* = G* · (9K* + 8G*) / (6 (K* + 2G*))
//
// where (K*, G*) is the comparison medium — the soft phase's moduli
// for HS- and the stiff phase's for HS+. Defined only when the phases
// sort consistently in K and G (one phase has both the smallest K
// and the smallest G; another has both the largest K and the largest
// G). The 2-phase HS routines remain available; for `n == 2` the
// N-phase form is exactly equivalent and is cross-checked in tests.
//
// Self-consistent (Hill) scheme (`self_consistent_moduli`) — each
// phase is embedded in the (yet-unknown) effective medium itself,
// rather than in a designated matrix or in the soft/stiff comparison
// of HS. Solves the Berryman fixed-point pair:
//
//   Σ v_i · (K_i − K_eff) / (K_i + 4G_eff/3) = 0
//   Σ v_i · (G_i − G_eff) / (G_i + ξ_eff)    = 0
//     ξ_eff = G_eff·(9K_eff + 8G_eff) / (6(K_eff + 2G_eff))
//
// by fixed-point iteration, seeded with the Voigt-Reuss-Hill average.
// Throws if convergence isn't reached within `max_iter`. SC lands
// closer to experiment than the bounds for percolating networks
// (high inclusion fraction, similar phase morphology) — at the cost
// of needing an iterative solve and a tolerance argument.
//
// Out of scope here, ships in follow-up PRs against #5:
//   * Mori-Tanaka for anisotropic phases / non-spherical inclusions
//     (needs a per-shape Eshelby tensor).
//   * Wiring directly into the phase model so a "homogenize" post-
//     process consumes per-phase numsim-materials configs.
//   * 2D plane-stress / plane-strain reductions of the 3D bounds.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <string>
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

// Hashin-Shtrikman bounds for a two-phase isotropic mixture with
// spherical morphology — the tightest variational bounds without
// additional morphological information beyond the volume fractions.
//
// Convention: the caller passes phase 1 and phase 2 in either order;
// the routine internally identifies which is "soft" (lower K, lower
// G) and which is "stiff", and returns:
//   * HS- (lower bound): uses the soft phase as the comparison
//     medium. Equals Mori-Tanaka with the soft phase as matrix.
//   * HS+ (upper bound): uses the stiff phase as comparison medium.
//     Equals Mori-Tanaka with the stiff phase as matrix.
//
// Requires both phases to sort consistently in K and G — i.e. one
// phase has both K_1 ≤ K_2 and G_1 ≤ G_2 (the "two-phase well-ordered"
// case). When the moduli cross (one phase stiffer in K, the other
// stiffer in G), the bounds are not well-defined and the function
// throws. This is the documented limitation of HS for crossed
// orderings; resolving it needs the Walpole multi-comparison-medium
// generalisation.
template <typename T = double>
[[nodiscard]] std::pair<T, T> hashin_shtrikman_lower(
    T K_1, T G_1, T K_2, T G_2, T volume_fraction_2) {
  if (volume_fraction_2 < T{0} || volume_fraction_2 > T{1}) {
    throw std::runtime_error{
        "hashin_shtrikman: volume_fraction_2 must lie in [0, 1]"};
  }
  // Identify the soft phase by both K and G; the orderings must agree.
  const bool one_is_soft = (K_1 <= K_2) && (G_1 <= G_2);
  const bool two_is_soft = (K_2 <= K_1) && (G_2 <= G_1);
  if (!one_is_soft && !two_is_soft) {
    throw std::runtime_error{
        "hashin_shtrikman: phase moduli cross (one stiffer in K, the "
        "other stiffer in G); 2-phase bounds undefined — needs the "
        "multi-comparison-medium Walpole form"};
  }
  // Map to (Ks, Gs) = soft, (Kh, Gh) = hard, with v_h the hard-phase
  // volume fraction. HS- inserts the hard phase as inclusions in a
  // soft comparison medium.
  const T Ks = one_is_soft ? K_1 : K_2;
  const T Gs = one_is_soft ? G_1 : G_2;
  const T Kh = one_is_soft ? K_2 : K_1;
  const T Gh = one_is_soft ? G_2 : G_1;
  const T v_h = one_is_soft ? volume_fraction_2 : (T{1} - volume_fraction_2);
  const T v_s = T{1} - v_h;

  // Textbook 2-phase HS- (Hashin & Shtrikman 1963), in terms of the
  // soft phase's longitudinal modulus (Ks + 4Gs/3) and shear-mod-
  // resolvent (Gs + Gs·(9Ks + 8Gs)/(6(Ks + 2Gs))).
  const T K_L  = Ks + T{4} * Gs / T{3};
  const T G_L  = Gs + Gs * (T{9} * Ks + T{8} * Gs)
                       / (T{6} * (Ks + T{2} * Gs));
  const T K_eff = Ks + v_h / (T{1} / (Kh - Ks) + v_s / K_L);
  const T G_eff = Gs + v_h / (T{1} / (Gh - Gs) + v_s / G_L);
  return {K_eff, G_eff};
}

template <typename T = double>
[[nodiscard]] std::pair<T, T> hashin_shtrikman_upper(
    T K_1, T G_1, T K_2, T G_2, T volume_fraction_2) {
  if (volume_fraction_2 < T{0} || volume_fraction_2 > T{1}) {
    throw std::runtime_error{
        "hashin_shtrikman: volume_fraction_2 must lie in [0, 1]"};
  }
  const bool one_is_soft = (K_1 <= K_2) && (G_1 <= G_2);
  const bool two_is_soft = (K_2 <= K_1) && (G_2 <= G_1);
  if (!one_is_soft && !two_is_soft) {
    throw std::runtime_error{
        "hashin_shtrikman: phase moduli cross (one stiffer in K, the "
        "other stiffer in G); 2-phase bounds undefined — needs the "
        "multi-comparison-medium Walpole form"};
  }
  // HS+ uses the hard phase as the comparison medium — the formula
  // is symmetric to HS- with the roles flipped.
  const T Ks = one_is_soft ? K_1 : K_2;
  const T Gs = one_is_soft ? G_1 : G_2;
  const T Kh = one_is_soft ? K_2 : K_1;
  const T Gh = one_is_soft ? G_2 : G_1;
  const T v_h = one_is_soft ? volume_fraction_2 : (T{1} - volume_fraction_2);
  const T v_s = T{1} - v_h;

  const T K_L  = Kh + T{4} * Gh / T{3};
  const T G_L  = Gh + Gh * (T{9} * Kh + T{8} * Gh)
                       / (T{6} * (Kh + T{2} * Gh));
  const T K_eff = Kh + v_s / (T{1} / (Ks - Kh) + v_h / K_L);
  const T G_eff = Gh + v_s / (T{1} / (Gs - Gh) + v_h / G_L);
  return {K_eff, G_eff};
}

// Convenience wrapper returning both HS bounds at once. The returned
// pair is { lower={K,G}, upper={K,G} }.
template <typename T = double>
[[nodiscard]] std::pair<std::pair<T, T>, std::pair<T, T>>
hashin_shtrikman_bounds(T K_1, T G_1, T K_2, T G_2, T volume_fraction_2) {
  return {hashin_shtrikman_lower<T>(K_1, G_1, K_2, G_2, volume_fraction_2),
          hashin_shtrikman_upper<T>(K_1, G_1, K_2, G_2, volume_fraction_2)};
}

// N-phase Hashin-Shtrikman bounds via the Berryman / Walpole closed
// form. `K_phases`, `G_phases`, and `volume_fractions` must all be
// equal-sized; volume fractions must sum to 1 (with small float slop
// absorbed). The phases must sort consistently — for HS- the soft
// phase needs both the smallest K and the smallest G; for HS+ the
// stiff phase needs both the largest K and the largest G.
//
// Implementation note: the comparison medium (K_star, G_star) is
// derived per-bound, so the same input vector yields a well-defined
// lower bound iff `min_K_phase == min_G_phase` (one phase is the
// softest in both moduli) and similarly for the upper bound. We
// detect the cross-ordering case and throw with a clear message.
namespace detail {

template <typename T>
[[nodiscard]] std::pair<T, T> hashin_shtrikman_n_impl(
    std::vector<T> const& K_phases,
    std::vector<T> const& G_phases,
    std::vector<T> const& volume_fractions,
    T K_star, T G_star) {
  // Eshelby-spherical comparison medium resolvents.
  const T four_g_over_3 = T{4} * G_star / T{3};
  const T xi_star = G_star * (T{9} * K_star + T{8} * G_star) /
                    (T{6} * (K_star + T{2} * G_star));
  T inv_K_sum{0};
  T inv_G_sum{0};
  for (std::size_t i = 0; i < K_phases.size(); ++i) {
    inv_K_sum += volume_fractions[i] / (K_phases[i] + four_g_over_3);
    inv_G_sum += volume_fractions[i] / (G_phases[i] + xi_star);
  }
  return {T{1} / inv_K_sum - four_g_over_3,
          T{1} / inv_G_sum - xi_star};
}

template <typename T>
void hashin_shtrikman_n_validate(
    std::vector<T> const& K_phases,
    std::vector<T> const& G_phases,
    std::vector<T> const& volume_fractions) {
  if (K_phases.empty()) {
    throw std::runtime_error{
        "hashin_shtrikman_n: at least one phase required"};
  }
  if (K_phases.size() != G_phases.size() ||
      K_phases.size() != volume_fractions.size()) {
    throw std::runtime_error{
        "hashin_shtrikman_n: K_phases / G_phases / volume_fractions "
        "size mismatch"};
  }
  T sum{0};
  for (auto v : volume_fractions) {
    if (v < T{0}) {
      throw std::runtime_error{
          "hashin_shtrikman_n: negative volume fraction"};
    }
    sum += v;
  }
  if (sum < T{1} - T{1e-9} || sum > T{1} + T{1e-9}) {
    throw std::runtime_error{
        "hashin_shtrikman_n: volume fractions must sum to 1"};
  }
}

} // namespace detail

template <typename T = double>
[[nodiscard]] std::pair<T, T> hashin_shtrikman_lower_n(
    std::vector<T> const& K_phases,
    std::vector<T> const& G_phases,
    std::vector<T> const& volume_fractions) {
  detail::hashin_shtrikman_n_validate(K_phases, G_phases, volume_fractions);
  const auto min_K_it = std::min_element(K_phases.begin(), K_phases.end());
  const auto min_G_it = std::min_element(G_phases.begin(), G_phases.end());
  const auto min_K_idx = std::distance(K_phases.begin(), min_K_it);
  const auto min_G_idx = std::distance(G_phases.begin(), min_G_it);
  if (min_K_idx != min_G_idx) {
    throw std::runtime_error{
        "hashin_shtrikman_lower_n: phases don't sort consistently — "
        "the softest-in-K phase must also be softest in G; "
        "current input has the K minimum on a different phase than "
        "the G minimum"};
  }
  return detail::hashin_shtrikman_n_impl<T>(
      K_phases, G_phases, volume_fractions, *min_K_it, *min_G_it);
}

template <typename T = double>
[[nodiscard]] std::pair<T, T> hashin_shtrikman_upper_n(
    std::vector<T> const& K_phases,
    std::vector<T> const& G_phases,
    std::vector<T> const& volume_fractions) {
  detail::hashin_shtrikman_n_validate(K_phases, G_phases, volume_fractions);
  const auto max_K_it = std::max_element(K_phases.begin(), K_phases.end());
  const auto max_G_it = std::max_element(G_phases.begin(), G_phases.end());
  const auto max_K_idx = std::distance(K_phases.begin(), max_K_it);
  const auto max_G_idx = std::distance(G_phases.begin(), max_G_it);
  if (max_K_idx != max_G_idx) {
    throw std::runtime_error{
        "hashin_shtrikman_upper_n: phases don't sort consistently — "
        "the stiffest-in-K phase must also be stiffest in G; "
        "current input has the K maximum on a different phase than "
        "the G maximum"};
  }
  return detail::hashin_shtrikman_n_impl<T>(
      K_phases, G_phases, volume_fractions, *max_K_it, *max_G_it);
}

template <typename T = double>
[[nodiscard]] std::pair<std::pair<T, T>, std::pair<T, T>>
hashin_shtrikman_bounds_n(std::vector<T> const& K_phases,
                          std::vector<T> const& G_phases,
                          std::vector<T> const& volume_fractions) {
  return {hashin_shtrikman_lower_n<T>(K_phases, G_phases, volume_fractions),
          hashin_shtrikman_upper_n<T>(K_phases, G_phases, volume_fractions)};
}

// Self-consistent (Hill) effective moduli via fixed-point iteration
// on the Berryman pair (see top-of-file comment for the equations).
// Each phase is embedded in the effective medium itself, so K_eff
// and G_eff appear on both sides — solved by direct substitution.
//
// Seed: Voigt-Reuss-Hill average. With pure-real inputs the iteration
// is contractive for most practical contrasts. `max_iter` and `tol`
// guard against pathological inputs (extreme contrast in K vs G);
// non-convergence throws with the last residual so callers can debug.
template <typename T = double>
[[nodiscard]] std::pair<T, T> self_consistent_moduli(
    std::vector<T> const& K_phases,
    std::vector<T> const& G_phases,
    std::vector<T> const& volume_fractions,
    std::size_t max_iter = 200,
    T tol = T{1e-9}) {
  detail::hashin_shtrikman_n_validate(K_phases, G_phases, volume_fractions);

  // Seed with the Voigt-Reuss-Hill average (arithmetic + harmonic
  // means halved). Cheap to compute and a stable starting point.
  T K_v{0}, G_v{0};
  T inv_K_r{0}, inv_G_r{0};
  for (std::size_t i = 0; i < K_phases.size(); ++i) {
    K_v += volume_fractions[i] * K_phases[i];
    G_v += volume_fractions[i] * G_phases[i];
    inv_K_r += volume_fractions[i] / K_phases[i];
    inv_G_r += volume_fractions[i] / G_phases[i];
  }
  T K_eff = T{0.5} * (K_v + T{1} / inv_K_r);
  T G_eff = T{0.5} * (G_v + T{1} / inv_G_r);

  for (std::size_t it = 0; it < max_iter; ++it) {
    const T four_g_over_3 = T{4} * G_eff / T{3};
    const T xi_eff = G_eff * (T{9} * K_eff + T{8} * G_eff) /
                     (T{6} * (K_eff + T{2} * G_eff));

    // Solve Σ v_i (K_i - K_eff)/(K_i + 4G_eff/3) = 0 for K_eff:
    // Σ v_i K_i / (K_i + a) = K_eff · Σ v_i / (K_i + a)
    // where a = 4G_eff/3. Rearranging gives a direct substitution.
    T num_K{0}, den_K{0};
    T num_G{0}, den_G{0};
    for (std::size_t i = 0; i < K_phases.size(); ++i) {
      const T denom_K = K_phases[i] + four_g_over_3;
      const T denom_G = G_phases[i] + xi_eff;
      num_K += volume_fractions[i] * K_phases[i] / denom_K;
      den_K += volume_fractions[i] / denom_K;
      num_G += volume_fractions[i] * G_phases[i] / denom_G;
      den_G += volume_fractions[i] / denom_G;
    }
    const T K_new = num_K / den_K;
    const T G_new = num_G / den_G;
    const T dK = std::abs(K_new - K_eff);
    const T dG = std::abs(G_new - G_eff);
    K_eff = K_new;
    G_eff = G_new;
    // Mixed absolute+relative tolerance — both moduli are typically
    // O(1e9) or larger, so a fixed absolute tol of 1e-9 would never
    // trigger. Use a relative tolerance against the running estimate.
    if (dK <= tol * (T{1} + std::abs(K_eff)) &&
        dG <= tol * (T{1} + std::abs(G_eff))) {
      return {K_eff, G_eff};
    }
  }
  throw std::runtime_error{
      "self_consistent_moduli: did not converge within "
      + std::to_string(max_iter) +
      " iterations; consider increasing max_iter or loosening tol "
      "(or verify phase contrast isn't pathological)"};
}

} // namespace rvegen::homogenization
