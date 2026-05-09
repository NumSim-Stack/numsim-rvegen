# #113 — Per-phase material property assignment

> Drafted issue + PR text. Branch: `feature/per-phase-materials`.
> **Status: design only — no implementation in this branch.**

## Issue body

The library produces phase IDs (0 = matrix, 1..N = inclusion indices)
but no mechanism to attach material properties per phase. Downstream
FFT/FE solvers need this association — currently the user has to
hand-edit material cards after generation, which defeats the
JSON-driven reproducibility story.

### Proposed fix

A new `materials` JSON section + a `phase_assignment` block:

```json
{
  "materials": {
    "matrix":      { "type": "linear_elastic", "E": 3.5e9,  "nu": 0.35 },
    "fibre":       { "type": "linear_elastic", "E": 240e9,  "nu": 0.25 },
    "void":        { "type": "void" }
  },
  "phase_assignment": {
    "matrix":      "matrix",
    "all_inclusions": "fibre"
  },

  ...

  "postprocess": [
    { "type": "voxel", "nx": 64, "ny": 64, "nz": 64,
      "output_path": "rve.vox",
      "emit_material_grid": true,
      "material_grid_path": "rve.mat" }
  ]
}
```

Voxel writer extension: when `emit_material_grid` is true, emit a
parallel grid where each cell carries the material ID resolved from
the phase ID via `phase_assignment`.

### Material types (initial set)

- `linear_elastic` — `E`, `nu` (or `K`, `G`); converts to stiffness
  tensor on demand.
- `orthotropic` — 9 stiffness components (or compliance form).
- `plastic_J2` — yield stress + hardening modulus.
- `void` — zero stiffness; used for porosity studies.

Each material is a registered type via the existing registry pattern
(new `material_registry_t`). Schema-driven ctor takes a
`parameter_handler_t` and the standard `parameters()` static returns
the schema with policy hints.

### Phase-assignment forms

- `"all_inclusions": "fibre"` — every shape gets the named material.
- `"by_shape_type": { "circle": "fibre1", "rectangle": "fibre2" }` —
  per-shape-type assignment.
- `"by_index": { "0": "matrix", "1-": "fibre" }` — explicit
  per-phase-id mapping with range syntax.

### Output extensions

- `voxel_writer` / `vtk_legacy_writer` gain optional
  `emit_material_grid` + `material_grid_path` fields. Phase grid
  unchanged; an additional file/dataset carries material IDs.
- `gmsh_geo_writer` emits `Physical Surface ("matrix") = { ... };`
  directives so gmsh's mesh carries material tags directly.
- New post-process `damask_material_config` — emits a `material.config`
  file in DAMASK's format for direct FFT-homogenization input.

### Acceptance

- [ ] `material_registry_t` + `material_base<T>` + at least
      `linear_elastic` and `void` types registered.
- [ ] `phase_assignment` resolution works for the three forms above.
- [ ] Voxel writer emits a parallel material grid when requested.
- [ ] gmsh writer emits `Physical Surface` per material.
- [ ] `damask_material_config` post-process works on the
      `circles_in_unit_box.json` example.

### Implementation sketch

```cpp
namespace rvegen {

template <typename T>
class material_base {
public:
  virtual ~material_base() = default;
  virtual stiffness_tensor<T> stiffness() const = 0;
  virtual std::string_view kind() const = 0;
};

template <typename T>
class linear_elastic final : public material_base<T> {
public:
  T E, nu;
  // ... ctors, parameters(), stiffness() implementation
};

template <typename T = double>
using material_registry_t = numsim_core::object_registry<
    material_base<T>, parameter_controller_t, parameter_handler_t const>;

}
```

`phase_assignment` is a small struct holding `unordered_map<size_t,
shared_ptr<material_base<T>>>`. The CLI builds it from the JSON
section after constructing materials. Each post-process that wants
material info gets a const reference.

### Out of scope

- Plastic / damage / nonlinear material models — deferred to
  follow-ups; the registry pattern accommodates them when needed.
- Per-orientation material variants (for FOD or polycrystal cases) —
  belongs with #112 / #114.
- Material parameter calibration tooling.

## PR body

Closes #113. **Currently draft — no implementation in this branch.**

Implementation work requires:
- ~6 new headers in `include/rvegen/materials/`.
- New registry slot (`material_registry_t`) wired through the CLI.
- Voxel + VTK writer schema extension (optional fields).
- New `damask_material_config` post-process.
- Tests covering material instantiation + phase assignment +
  voxel-grid emission.

Estimated 1–2 days of focused work. Tracked here as the canonical
description for when it gets picked up.
