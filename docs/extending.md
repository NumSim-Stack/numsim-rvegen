# Extending rvegen

How to add new types to rvegen — shapes, distributions, generators,
terminations, and post-processes. The library is registry-driven: every
new type is one new header plus one line of registration. Zero edits to
the CLI, the orchestrator, or any existing type.

This document is the canonical contributor reference. The
[README's brief catalogue](../README.md#adding-a-new-type) is a teaser;
this doc is the full recipe with worked examples.

---

## Anatomy of a registered type

Every registered type follows the same pattern:

1. **A C++ class** that inherits from a base (`shape_base<T>`,
   `distribution_base<T>`, etc.) — the runtime polymorphism layer.
2. **A schema-driven constructor** taking `parameter_handler_t const&`
   — pulls JSON values out by name.
3. **A static `parameters()` method** returning a
   `parameter_controller_t` — declares the JSON schema with units,
   ranges, and descriptions.
4. **A registration** — one line in the relevant
   `register_*.h` header, mapping a string name to the type.

The schema is what bridges JSON config and C++ construction. The
`schema_audit_test` enforces that every field carries a
`.description()`, and that integral fields carry at least one of
`.min()` / `.max()` / `.units()` so a downstream GUI form generator
(Tessera) gets enough metadata to build sensible widgets.

---

## Adding a new shape

Shapes are the geometric primitives the generator places into the RVE
domain. Each shape inherits from `shape_base<T>` (via `static_indexing`
for the dispatcher's id) and typically also from a GeometricTools
primitive for the geometric data.

### Required pieces

For a hypothetical `triangle<T>` 2D shape:

**1. Header `include/rvegen/shapes/triangle.h`**:

```cpp
#pragma once

#include <array>
#include <cstddef>

#include <Mathematics/Triangle.h>           // GTE primitive
#include <numsim-core/input_parameter_controller.h>
#include <numsim-core/static_indexing.h>

#include "../types.h"
#include "rectangle_bounding.h"
#include "shape_base.h"

namespace rvegen {

template <typename T = double>
class triangle
    : public numsim_core::static_indexing<triangle<T>, shape_base<T>>,
      public gte::Triangle<2, T> {
public:
  using value_type = T;
  using size_type = std::size_t;

  triangle() = default;

  // Direct C++ ctor — what gets called from make_from_tuple in the
  // schema-driven path, and what users would call directly.
  triangle(T ax, T ay, T bx, T by, T cx, T cy) {
    this->v[0] = {ax, ay};
    this->v[1] = {bx, by};
    this->v[2] = {cx, cy};
  }

  // Schema-driven ctor — pulls JSON fields by name.
  explicit triangle(parameter_handler_t const& handler)
      : triangle(handler.template get<T>("ax"),
                 handler.template get<T>("ay"),
                 handler.template get<T>("bx"),
                 handler.template get<T>("by"),
                 handler.template get<T>("cx"),
                 handler.template get<T>("cy")) {}

  // Schema. The audit test demands description on every field plus
  // min/max/units on integral fields. Floats get min/units when
  // physically meaningful (a length cannot be negative; a fraction
  // belongs in [0, 1]).
  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<T>("ax").template add<numsim_core::is_required>()
        .units("m").description("vertex A x-coordinate");
    s.template insert<T>("ay").template add<numsim_core::is_required>()
        .units("m").description("vertex A y-coordinate");
    s.template insert<T>("bx").template add<numsim_core::is_required>()
        .units("m").description("vertex B x-coordinate");
    s.template insert<T>("by").template add<numsim_core::is_required>()
        .units("m").description("vertex B y-coordinate");
    s.template insert<T>("cx").template add<numsim_core::is_required>()
        .units("m").description("vertex C x-coordinate");
    s.template insert<T>("cy").template add<numsim_core::is_required>()
        .units("m").description("vertex C y-coordinate");
    return s;
  }

  // ---- shape_base contract ---------------------------------------
  [[nodiscard]] T area() const override {
    // Standard 2D triangle area via cross product.
    const auto a = this->v[1] - this->v[0];
    const auto b = this->v[2] - this->v[0];
    return std::abs(a[0] * b[1] - a[1] * b[0]) * T{0.5};
  }
  [[nodiscard]] T volume() const override { return T{0}; }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    // Bounding-box half-extents from the three vertices.
    const auto xmin = std::min({this->v[0][0], this->v[1][0], this->v[2][0]});
    const auto xmax = std::max({this->v[0][0], this->v[1][0], this->v[2][0]});
    const auto ymin = std::min({this->v[0][1], this->v[1][1], this->v[2][1]});
    const auto ymax = std::max({this->v[0][1], this->v[1][1], this->v[2][1]});
    return {(xmax - xmin) * T{0.5}, (ymax - ymin) * T{0.5}, T{0}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    // Centroid.
    return {(this->v[0][0] + this->v[1][0] + this->v[2][0]) / T{3},
            (this->v[0][1] + this->v[1][1] + this->v[2][1]) / T{3},
            T{0}};
  }
  void set_middle_point(std::array<T, 3> middle_point) override {
    // Translate all three vertices by (target - current_centroid).
    const auto cur = get_middle_point();
    const T dx = middle_point[0] - cur[0];
    const T dy = middle_point[1] - cur[1];
    for (int i = 0; i < 3; ++i) {
      this->v[i][0] += dx;
      this->v[i][1] += dy;
    }
  }

  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    // Barycentric / sign-of-cross-product test. Implementation omitted;
    // see GTE's PointInTriangle predicate or roll your own.
    return /* point is in this triangle */ false;
  }

  void make_bounding_box() override {
    auto box = std::make_unique<rectangle_bounding<T>>();
    const auto xmin = std::min({this->v[0][0], this->v[1][0], this->v[2][0]});
    const auto xmax = std::max({this->v[0][0], this->v[1][0], this->v[2][0]});
    const auto ymin = std::min({this->v[0][1], this->v[1][1], this->v[2][1]});
    const auto ymax = std::max({this->v[0][1], this->v[1][1], this->v[2][1]});
    box->top_point()    = {xmax, ymax, T{0}};
    box->bottom_point() = {xmin, ymin, T{0}};
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<triangle<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return triangle::m_id;
  }
};

} // namespace rvegen
```

**Key invariants you must preserve:**

- The `static_indexing<self<T>, shape_base<T>>` interposing layer — gives
  the shape its dispatcher id. Without this, the shape can't be
  registered for collision dispatch.
- `make_bounding_box()` must populate `_bounding_box` with a fresh
  unique_ptr — the AABB-overlap stage of two-stage dispatch reads it.
- `is_inside(point)` must be correct — voxelization and the FFT
  pipeline depend on it.
- Construction order: `shape_base` is initialised *before* the GTE base.
  Don't read GTE-base fields (centre, axis, etc.) from inside
  `shape_base`'s constructor — they don't exist yet.

### Collision overloads

Add `collision_details(triangle, X)` for every other shape type in
`include/rvegen/intersection/collision_details.h`. The compiler tells you
which combinations are missing — try to build and follow the errors.

For a 2D shape, the overloads you need are pairs against `circle`,
`rectangle`, `ellipse`, and `triangle` itself.

Each overload typically delegates to a GTE intersection query:

```cpp
template <typename T>
[[nodiscard]] inline bool collision_details(
    triangle<T> const& a, circle<T> const& b) noexcept {
  // Reuse GTE's TIQuery.
  using Query = gte::TIQuery<T, gte::Triangle<2, T>, gte::Circle2<T>>;
  return Query{}(a, b).intersect;
}
```

### Registration

In `include/rvegen/registry/register_collisions.h`, add:

```cpp
d.template register_pair<triangle<T>, triangle<T>>();
d.template register_pair<triangle<T>, circle<T>>();
d.template register_pair<triangle<T>, rectangle<T>>();
d.template register_pair<triangle<T>, ellipse<T>>();
```

inside `register_all_collision_pairs<T>()`.

### Mesh override (optional, needed only for Tessera viewport)

In `include/rvegen/visualization/triangle_mesh.h` (or sibling
`triangle_mesh.h` next to the shape header), add:

```cpp
template <typename T>
[[nodiscard]] inline triangle_mesh<T>
to_mesh(triangle<T> const& t) {
  triangle_mesh<T> m;
  m.verts.push_back({t.v[0][0], t.v[0][1], T{0}});
  m.verts.push_back({t.v[1][0], t.v[1][1], T{0}});
  m.verts.push_back({t.v[2][0], t.v[2][1], T{0}});
  m.tris.push_back({0, 1, 2});
  return m;
}
```

Then register in `register_meshes.h`'s `register_all_meshes<T>()`:

```cpp
mesh_dispatcher<T>::instance().template register_shape<triangle<T>>();
```

If you skip the mesh override, the dispatcher returns an empty mesh
(graceful degradation; the rest of the library works fine). 2D-only
contexts that don't render in 3D can omit it entirely.

---

## Adding a new shape input

A `shape_input` is what the generator queries to *get* a candidate
shape. Conceptually it pairs distributions with the shape constructor:
"sample x from this distribution, sample y from that one, return a
new circle".

**Header `include/rvegen/inputs/triangle_input.h`**:

```cpp
template <typename T = double>
class triangle_input final : public shape_input_base<T> {
public:
  // Direct ctor — one distribution per shape parameter.
  triangle_input(distribution_ref<T> ax, distribution_ref<T> ay,
                 distribution_ref<T> bx, distribution_ref<T> by,
                 distribution_ref<T> cx, distribution_ref<T> cy)
      : _ax{ax}, _ay{ay}, _bx{bx}, _by{by}, _cx{cx}, _cy{cy} {}

  // Schema-driven ctor — looks up named distributions in the map.
  triangle_input(parameter_handler_t const& h,
                 distribution_map_t<T> const& dists)
      : triangle_input(
          dists.at(h.template get<std::string>("ax_dist")),
          dists.at(h.template get<std::string>("ay_dist")),
          dists.at(h.template get<std::string>("bx_dist")),
          dists.at(h.template get<std::string>("by_dist")),
          dists.at(h.template get<std::string>("cx_dist")),
          dists.at(h.template get<std::string>("cy_dist"))) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("ax_dist").template add<numsim_core::is_required>()
        .description("name of the distribution sampling vertex A x-coordinate");
    // ... and so on for ay_dist, bx_dist, ..., cy_dist
    return s;
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> new_shape() override {
    return std::make_unique<triangle<T>>(
        _ax.sample(), _ay.sample(), _bx.sample(), _by.sample(),
        _cx.sample(), _cy.sample());
  }

private:
  distribution_ref<T> _ax, _ay, _bx, _by, _cx, _cy;
};
```

Register in `include/rvegen/registry/register_inputs.h`'s
`register_all_inputs<T>()`:

```cpp
input_registry_t<T>::instance()
    .template register_type<triangle_input<T>>("triangle_input");
```

---

## Adding a new distribution

Distributions are the random-number sources fed to inputs. Each owns
a reference to an engine and exposes a `sample()` method.

**Header `include/rvegen/distributions/lognormal_distribution.h`**:

```cpp
template <typename T = double>
class lognormal_distribution final : public distribution_base<T> {
public:
  template <typename Engine>
  lognormal_distribution(T mu, T sigma, Engine& engine)
      : _dist{mu, sigma}, _engine_ref{engine} {}

  template <typename Engine>
  lognormal_distribution(parameter_handler_t const& h, Engine& engine)
      : lognormal_distribution(h.template get<T>("mu"),
                               h.template get<T>("sigma"),
                               engine) {}

  [[nodiscard]] T sample() override {
    return _dist(_engine_ref.get());
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<T>("mu").template add<numsim_core::is_required>()
        .description("mean of the underlying normal");
    s.template insert<T>("sigma").template add<numsim_core::is_required>()
        .min(0.0).description("std-dev of the underlying normal (must be positive)");
    return s;
  }

private:
  std::lognormal_distribution<T> _dist;
  std::reference_wrapper<std::mt19937> _engine_ref;
};
```

Register in `include/rvegen/registry/register_distributions.h`:

```cpp
distribution_registry_t<T>::instance()
    .template register_type<lognormal_distribution<T>>("lognormal");
```

---

## Adding a new generator, termination, or post-process

The pattern is the same — inherit from the base, add a schema-driven
ctor, expose `parameters()`, register in the relevant `register_*.h`.

- **Generators** inherit `rve_generator_base<T>`, override `compute()`.
  Honour the `progress_options` parameter (call `opts.cancel()` once per
  outer loop iteration, call `opts.on_progress()` after acceptance).
- **Terminations** inherit `termination_base<T>`, override
  `operator()(accepted)`. If your termination has a target the GUI
  could surface, override `target_count()` or
  `target_volume_fraction()`.
- **Post-processes** inherit `post_process_base<T>`, override
  `run(shapes, domain_box)`. Own your own output sink; the schema
  carries an `output_path` field.

---

## Testing your additions

The `schema_audit_test` will fail-loud if your `parameters()` is
missing required metadata. Run `ctest -R schema_audit_test` after
adding a new type to confirm.

For correctness, add to one of the existing smoke tests or write a
dedicated one. The pattern: register the type, build a small JSON
config that exercises it, run a one-shape pipeline, assert on
properties of the result.

---

## Common pitfalls

- **Forgetting `static_indexing` on a new shape.** Compiles fine, then
  collision dispatch silently fails because `m_id` doesn't exist.
- **Missing `make_bounding_box()` call before AABB queries.** The
  generator does this automatically; if you're poking at shapes
  directly in a test, remember to call it.
- **Schema field names that don't match ctor parameter extraction.**
  The schema declares `"radius"` but the ctor calls
  `h.template get<T>("r")` — the JSON visitor reads `"radius"`, the
  ctor doesn't find it, runtime error. The audit test doesn't catch
  this; only an integration smoke does.
- **Construction-order-of-bases bug.** `shape_base` runs before the
  GTE base. Reading GTE fields in shape_base's constructor is UB.
- **2D shape with z != 0 in the bounding box.** Voxelization wrongly
  assumes the shape extends into z, FFT pipeline gets surprised.
  Always set z to T{0} for 2D shapes.
