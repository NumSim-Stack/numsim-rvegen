# #112 — Fibre orientation distribution (FOD) support

> Drafted issue + PR text. Branch: `feature/fibre-orientation-distribution`.
> **Status: design only — no implementation in this branch.**

## Issue body

Composites researchers measure fibre orientations via μ-CT and feed
the orientation tensor (Advani-Tucker A2) into generators. Today
rvegen's `ellipse_input` and future cylinder/capsule shapes accept
only uniform-random rotations — no way to import a measured
orientation distribution. Result: rvegen can't be used for "real"
composites where the orientation is anisotropic, which is most
industrial cases (injection-moulded short fibres in particular).

### Proposed fix

A new distribution type `oriented_uniform` that takes an Advani-Tucker
A2 tensor (3×3 symmetric) and samples rotations whose first/second
moments match it.

```json
"distributions": {
  "rotation_oriented": {
    "type": "oriented_uniform",
    "a11": 0.7, "a22": 0.2, "a33": 0.1,
    "a12": 0.05, "a13": 0.0, "a23": 0.0
  }
}
```

The distribution samples rotation angles (or unit vectors in 3D) that
statistically reproduce the input A2 over many draws. Implementation:
sample uniformly on the unit sphere, then accept/reject weighted by
the orientation distribution function (ODF) inferred from A2.

For 2D (ellipse rotation): A2 reduces to a single scalar — the
order parameter S = ⟨cos 2θ⟩. Sample θ from a circular distribution
(von Mises) with concentration matched to S.

For 3D (cylinder/capsule axis direction): A2 is a 3×3 symmetric tensor.
Sample axis directions from a Bingham distribution parameterised by
the eigenvectors and eigenvalues of A2.

### Acceptance

- [ ] `oriented_uniform` distribution registered.
- [ ] Test: sample N=10,000 rotations, compute the empirical A2, assert
      it matches the input within tolerance proportional to 1/sqrt(N).
- [ ] Compatible with existing `ellipse_input` (2D) and future
      cylinder/capsule inputs (3D).

### Optional follow-ups

- Pole-figure / ODF input (file formats: Neper's `.ori`, MTEX `.txt`)
  — converts to A2 internally.
- Higher-order tensors A4 for finer fidelity.

## PR body

Closes #112. **Currently draft — no implementation in this branch.**

Implementation work requires:
- New `oriented_uniform_distribution<T>` in `include/rvegen/distributions/`.
- A2 → von Mises (2D) or Bingham (3D) parameter mapping.
- Eigendecomposition of A2 (Eigen3 already a dependency).
- Tests asserting empirical-A2-from-sample matches input.

Estimated half-day to one day of focused work. Tracked here as the
canonical description for when it gets picked up.
