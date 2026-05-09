# #111 — Mean-field analytical homogenization models

> Drafted issue + PR text. Branch: `feature/mean-field-homogenization`.
> **Status: design only — no implementation in this branch.**

## Issue body

Mori-Tanaka, self-consistent, Voigt-Reuss bounds, Hashin-Shtrikman
bounds — these are what engineers actually use FIRST, before
committing to a full FE-RVE simulation. Digimat MF (mean-field) is a
huge part of their commercial offering.

rvegen currently has zero analytical models — it generates RVEs and
hands them off; the engineer still needs another tool for the
analytical screening step. Adding mean-field models closes the
"microstructure → effective property" loop in rvegen itself.

### Proposed fix

A new module `include/rvegen/homogenization/`. Takes a shape vector +
per-phase material properties (stiffness tensors), returns effective
stiffness via the chosen model. Inclusion-shape-aware (Eshelby tensor
depends on aspect ratio).

```cpp
template <typename T>
struct mean_field_input {
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes;
    std::array<T, 3> domain_box;
    stiffness_tensor<T> matrix_stiffness;
    std::vector<stiffness_tensor<T>> phase_stiffness;  // per phase ID
    phase_assignment<T> const& assignment;             // from #113
};

template <typename T>
[[nodiscard]] stiffness_tensor<T>
voigt_reuss_bounds(mean_field_input<T> const& in);

template <typename T>
[[nodiscard]] stiffness_tensor<T>
hashin_shtrikman_bounds(mean_field_input<T> const& in);

template <typename T>
[[nodiscard]] stiffness_tensor<T>
mori_tanaka(mean_field_input<T> const& in);

template <typename T>
[[nodiscard]] stiffness_tensor<T>
self_consistent(mean_field_input<T> const& in,
                std::size_t max_iters = 50, T tol = 1e-6);
```

Each is a free function returning the homogenised stiffness. Plus a
post-process `mean_field_summary` that runs all four and writes a CSV
comparing the predictions.

### Math sketch

- **Voigt:** `C_eff = Σ φ_i * C_i` (upper bound)
- **Reuss:** `S_eff = Σ φ_i * S_i` (lower bound, via compliance)
- **Hashin-Shtrikman:** tighter bounds for two-phase isotropic
  composites; involves the Hashin-Shtrikman function.
- **Mori-Tanaka:** uses Eshelby's tensor for the inclusion shape.
  For aligned ellipsoidal inclusions: closed-form Eshelby tensor.
  For random orientations: orientational average.
- **Self-consistent:** iterative — assume effective medium ≡ matrix,
  compute Mori-Tanaka, update matrix, iterate.

### Acceptance

- [ ] All four models implemented for isotropic phases.
- [ ] Test: two-phase composite (E_matrix=3GPa, E_inclusion=200GPa,
      φ=0.30) — Voigt-Reuss bounds and Hashin-Shtrikman bounds match
      tabulated reference values within 1e-6.
- [ ] Test: dilute-limit Mori-Tanaka matches Eshelby's analytical
      result for a single ellipsoidal inclusion.
- [ ] Self-consistent converges within 20 iterations for typical
      composites.
- [ ] `mean_field_summary` post-process writes a CSV.

### Out of scope

- Anisotropic / orthotropic phases — extend later.
- Plastic / nonlinear constitutive models — these are FE territory,
  not mean-field.
- 3D-to-2D plane-stress / plane-strain reductions — handle via the
  domain_box[2] = 0 convention already established.

## PR body

Closes #111. **Currently draft — no implementation in this branch.**

Implementation work requires:
- ~80 LOC for `stiffness_tensor<T>` (4th-order tensor in Voigt
  notation, common operations).
- ~50 LOC each for Voigt and Reuss bounds.
- ~150 LOC for Hashin-Shtrikman bounds.
- ~250 LOC for Mori-Tanaka with Eshelby tensor (most of the math
  is the Eshelby tensor for ellipsoidal inclusions).
- ~150 LOC for self-consistent iteration.
- ~150 LOC for `mean_field_summary` post-process.
- ~300 LOC of tests against tabulated reference values from
  Mori-Tanaka 1973, Eshelby 1957 originals.

Estimated 2–3 days of focused work. Significant payoff: closes
the loop on "give me the effective stiffness" without requiring
a separate FFT/FE solver.
