# Refactoring Plan — rvegen + numsim-core

## Premise

This plan assumes two things, both load-bearing:

1. **Reproducibility matters.** RVE generation feeds material simulation; same input must produce same output. This rules out lazy choices for RNG and dispatch.
2. **The codebase should get smaller, not larger.** The current direction has been to add layers; the right direction is to remove them. Several phases below are deletion phases.

The plan is sequenced so each phase ships an independent improvement, leaves the build green, and can be paused between phases. Phases 0–4 are conservative repair; phases 5–9 are structural; phase 10 is the modernization bridge if you want to commit to it.

The line counts and effort estimates are rough — they're meant to communicate scale, not to be hit exactly.

---

## Phase 0 — Inventory of completed work

The following items from the prior review (`~/.claude/plans/virtual-orbiting-dragonfly.md`) are already done in the working tree:

- **C1** `basic_base::operator=` typo (`rve_forward.h:187,200`)
- **C2** weak_ptr lifetime (eliminated by M1 consolidation)
- **C3 + M5** `circle`/`rectangle` default ctors removed; `rectangle::m_shape` initialized
- **C4** `circle_generator::operator()` returns a real shape
- **M1** three `insert_random_algo*` classes collapsed into one templated, ref-injected variant
- **M2** `algo.run()` re-enabled in `main()` and `test_setup()`
- **M6 + R1** `circle`/`rectangle` migrated to `numsim_core::static_indexing`
- Removed `circle_generator_` (weak_ptr variant) and `circle_generator_std_func`; deleted the redundant std_func block in `main()`

Everything below is what remains.

---

## Phase 1 — Stabilize & gate further work behind tests

**Goal:** never regress what's working. Until this phase ships, every later phase is guesswork.

**Steps:**
1. Confirm the post-Phase-0 build is green. (You're running this now.)
2. Add a `CTest` target with three smoke tests:
   - Construct one of each shape (`circle`, `rectangle`, `sphere`) and assert `index()` returns distinct values — guards M6/R1.
   - Run `insert_random_algo` for 100 inclusions, assert `m_shapes.size() == 6` (the `number_of_inclusions::operator()` threshold) — guards M1.
   - Round-trip the embedded JSON config through `rve_app::init()` without exceptions — guards M3 once it lands.
3. Capture a benchmark baseline: run `shape_benchmark(10000)` and `shape_benchmark(100000)`, save numbers to `bench/baseline.txt`. This is what later phases must not regress (or must regress with a documented reason).
4. Add a minimal CI config (GitHub Actions or GitLab CI) that does `cmake -S . -B build -DRVEGEN_WITH_VTK=OFF -DRVEGEN_WITH_VOXELIZATION=OFF && cmake --build build && ctest --test-dir build`. Without CI, the tests rot.

**Files touched:** `CMakeLists.txt`, new `tests/`, new `bench/`, new `.github/workflows/build.yml` or equivalent.

**Verification:** `ctest` passes locally; CI green on a push.

**Effort:** half a day. **Risk:** low. **Independent:** yes — depends on Phase 0 only.

---

## Phase 2 — CMake hygiene & dependency reduction

**Goal:** make a default dev build cheap. The current full-fat build pulls VTK, CGAL, and Boost — many minutes from clean. Most edits don't need any of them.

**Steps:**
1. Add CMake options:
   ```cmake
   option(RVEGEN_WITH_VTK            "Build VTK rendering"     OFF)
   option(RVEGEN_WITH_VOXELIZATION   "Build CGAL voxelization" OFF)
   option(RVEGEN_BUILD_BENCHMARKS    "Build benchmarks"        ON)
   ```
2. Wrap VTK includes/links with `if(RVEGEN_WITH_VTK)` and gate the rendering call sites with `#ifdef RVEGEN_WITH_VTK`. Same for CGAL/voxelization.
3. Replace `boost::timer::cpu_timer` with `std::chrono::steady_clock`. About six call sites in `main.cpp`. Delete `find_package(Boost ...)` once `boost::container::flat_map` is also gone (Phase 9).
4. Reduce default GeometricTools include surface — only the headers actually used by shape and intersection code.

**Files touched:** `CMakeLists.txt`, `main.cpp` (timer call sites), the rendering/voxelization functions get `#ifdef` guards.

**Verification:** the default build (`-DRVEGEN_WITH_VTK=OFF -DRVEGEN_WITH_VOXELIZATION=OFF`) compiles in seconds, not minutes. Phase 1 tests still pass.

**Effort:** half a day. **Risk:** low. **Independent:** yes.

---

## Phase 3 — Decompose `main.cpp`

**Goal:** turn the 1700-line scrapbook into a thin executable plus a library. This phase doesn't change behavior; it relocates code.

**Target layout:**
```
rvegen/
  include/rvegen/
    shapes.h                   (existing, trimmed)
    distributions.h            (existing, trimmed)
    rve_forward.h              (existing — until Phase 6)
    intersection.h             (overloads + warehouse, until Phase 9)
    voxel.h                    (existing)
    generators.h               (NEW: shape_generator_base + circle_generator)
    algorithms.h               (NEW: insert_random_algo, only_inside, number_of_inclusions)
    rve_app.h                  (NEW: rve_app, app_base, rve_traits)
    json_visitor.h             (NEW: input_parameter_visitor_nlohmann)
    vtk_scene.h                (NEW: SceneManager, axes, etc.) — guarded by RVEGEN_WITH_VTK
    voxelize.h                 (NEW: voxelizeMesh declaration) — guarded by RVEGEN_WITH_VOXELIZATION
  src/
    rve_app.cpp                (factory map, setup logic)
    json_visitor.cpp
    intersection_warehouse.cpp (the static-init lambdas) — until Phase 9
    vtk_render.cpp             (vtk_render_shapes, vtk_test, vtk_random_test)
    vtk_scene.cpp
    voxelize.cpp
    voxel_io.cpp               (saveVoxelsAsVTK — note: this is text I/O, NOT a VTK lib dep)
  apps/
    rvegen_cli.cpp             (the new main.cpp — ~150 lines)
  bench/
    shape_benchmark.cpp        (separate target)
  tests/
    smoke_tests.cpp            (from Phase 1)
  examples/
    only_inside_circles.json
```

**Steps (in this order — each compiles independently):**
1. Pull the JSON visitor into its own TU. Trivial — no dependencies on the rest.
2. Pull `voxelize.cpp` and `voxel_io.cpp`. Note: `saveVoxelsAsVTK` writes a `.vtk` text file but does *not* link against VTK. Don't put it under `RVEGEN_WITH_VTK`; put it under `RVEGEN_WITH_VOXELIZATION`.
3. Pull VTK rendering into `vtk_scene.cpp` and `vtk_render.cpp`. All under `RVEGEN_WITH_VTK`.
4. Pull `shape_benchmark`, `quick_benchmark`, and `test_setup` into `bench/` and `tests/` respectively. They become separate executables.
5. Move `rve_app`/`app_base`/`rve_traits` into `rve_app.h` + `rve_app.cpp`.
6. Move `insert_random_algo`/`only_inside`/`number_of_inclusions` into `algorithms.h`.
7. Move `circle_generator` and `shape_generator_base` into `generators.h`. Delete the now-unused `shape_generator_base` if no one inherits from it polymorphically (`circle_generator` is used directly as a template parameter, not via base ptr).
8. Whatever's left in `main.cpp` becomes `apps/rvegen_cli.cpp` — tightly scoped to argv parsing + JSON load + algo execution.

**Verification:** Phase 1 tests still pass; rendering/voxelization paths build only when their flags are on.

**Effort:** 2 days. **Risk:** medium — header dependencies are subtle, especially around `parameter_handler` and `data_info`. Move one file at a time, build between each. **Independent:** depends on Phase 1 (tests catch regressions) and Phase 2 (CMake flags).

---

## Phase 4 — Externalize configuration

**Goal:** stop hard-coding the JSON config inside the binary.

**Steps:**
1. `apps/rvegen_cli.cpp` reads `argv[1]` as a JSON file path. Use `nlohmann::json::parse(std::ifstream{path})`.
2. Move the embedded JSON string from `main.cpp` into `examples/only_inside_circles.json`.
3. Add a default: if no argv given, look for `./config.json`; if that's absent, print usage and exit non-zero.
4. The Phase 1 smoke test that round-trips JSON should now load `examples/only_inside_circles.json` from disk.

**Files touched:** `apps/rvegen_cli.cpp`, new `examples/`, `tests/smoke_tests.cpp`.

**Verification:** `./rvegen examples/only_inside_circles.json` produces the same output as the old embedded-JSON build.

**Effort:** 1 hour. **Risk:** low. **Independent:** depends on Phase 3.

---

## Phase 5 — Consolidate `numsim-core`

**Goal:** before rvegen migrates *to* something in numsim-core, make sure numsim-core has only one of each thing.

This is a **numsim-core PR**, not a rvegen one — but rvegen pins it.

**Audit findings (already established):**
- `factory_base_bones.h` + `factory_base_meat.h` (legacy)
- `registry_bones.h` + `registry_meat.h` (lower-level)
- `warehouse_bones.h` + `warehouse_meat.h` (container wrapper)
- `object_registry.h` (modern, schema-aware)

Three generations of factory/registry coexist. Pick **`object_registry`** as canonical (it has schema awareness, which the others lack), delete the other two.

**Steps:**
1. In `numsim-core`, grep for inbound users of each. Anything that references `factory_base_bones` or `registry_bones` outside numsim-core's own files needs migration first — but I suspect rvegen is the only consumer and it doesn't touch them.
2. Delete `factory_base_*.h`, `registry_*.h`, `warehouse_*.h` from `numsim-core/include/numsim-core/`.
3. Bump numsim-core's version (if it has one) and document in CHANGELOG.

**Verification:** numsim-core's own tests pass; rvegen still builds against it.

**Effort:** half a day. **Risk:** low *if* rvegen is the only consumer; otherwise audit consumers first. **Independent:** yes — but Phase 7 needs it.

---

## Phase 6 — Move `basic_base` into `numsim-core`

**Goal:** `basic_base` and `basic_base_imp` live in `namespace numsim_core` but are physically declared in rvegen's `rve_forward.h`. That's a layering smell — the namespace claim isn't backed by ownership.

**Steps:**
1. In numsim-core, create `include/numsim-core/basic_base.h` containing `basic_base_imp` and `basic_base` (lifted verbatim from `rvegen/rve_forward.h`).
2. In rvegen's `rve_forward.h`, replace those class definitions with `#include <numsim-core/basic_base.h>`.
3. What remains in `rvegen/rve_forward.h` is the rvegen-specific `numsim_rve::input_parameter` variant, `numsim_rve::data_info`, and `numsim_rve::query_map` alias — about 30 lines. Rename to `rve_types.h` while you're at it.
4. While moving, **simplify**: the `basic_base_imp` / `basic_base` split exists for the optional `Info=void` specialization. Consider whether the class even needs to be split. If `basic_base_imp` is only ever used through `basic_base`, collapse them. (Inspection: yes, `basic_base_imp` is only used as a base of `basic_base`. Collapse.)

**Verification:** rvegen builds; Phase 1 tests pass.

**Effort:** 1 day. **Risk:** low — it's a copy + include swap, plus the optional collapse. **Independent:** depends on Phase 5.

---

## Phase 7 — Replace `rve_app::m_factory` with `object_registry`

**Goal:** stop reinventing the registry. Resolves M3 (silent factory failures) as a side effect.

**Steps:**
1. Read `numsim-core/include/numsim-core/object_registry.h` end-to-end. Understand `register_type<Derived>(key)`, `create(key, args...)`, `schema(key)`, and how it surfaces unknown-key errors.
2. In `rve_app`, replace the parallel `m_factory` (`std::unordered_map<std::string, std::function<...>>`) and `m_input_parameter` (`std::unordered_map<std::string, parameter_handler>`) with three `object_registry`s: one for shapes, one for distributions, one for algorithms.
3. Convert each currently-hand-registered type (sphere, circle, rectangle, uniform/normal/constant distributions, only_inside) to `registry.register_type<Foo>("foo")`.
4. Replace the `setup()` loop's silent `continue` statements:
   ```cpp
   if(!data.contains("type") || !data["type"].is_string()) continue;
   ```
   with explicit error reporting:
   ```cpp
   if(!data.contains("type") || !data["type"].is_string()) {
     throw std::runtime_error(std::format("config entry '{}' missing 'type'", name));
   }
   ```
   Same for the unknown-factory case.
5. Update tests to assert that malformed JSON raises a clear error.

**Files touched:** `rve_app.cpp`, registration sites for each shape/distribution/algorithm, smoke tests.

**Verification:** existing JSON config still works; deliberately-broken JSON now reports the error instead of silently dropping the entry.

**Effort:** 1.5 days. **Risk:** medium — `object_registry` may not exactly match the calling convention; expect adapter code. **Independent:** depends on Phases 5 and 6.

---

## Phase 8 — Concurrency posture

**Goal:** decide and commit. Either rvegen is single-threaded (and says so loudly), or it supports parallel packing properly.

**Recommendation: option (3) below.** RVE generation is embarrassingly parallel and reproducibility-sensitive; the architecturally right answer is engine injection.

**Three options, pick one:**

**Option 1 — declare single-threaded.** Add to `README` and the public header:
```cpp
// rvegen is single-threaded. Concurrent calls into any rvegen function
// from multiple threads invoke undefined behavior.
```
Then move on. Honest, free, ships nothing functional.

**Option 2 — `thread_local` engine.** One-line change in `distributions.h`:
```cpp
static Engine& get() {
  static thread_local Engine instance{Device{}()};
  return instance;
}
```
Each thread gets its own MT19937 seeded from `std::random_device`. **Caveat:** seeds are now non-deterministic across runs — bad for reproducibility. Could be fixed by a process-level seed + per-thread offset, but that's effort.

**Option 3 — engine injection (recommended).** Distributions own their engine via a member; callers either pass an engine to `value(Engine&)` or construct distributions with an engine. API change:
```cpp
template <typename T, typename Engine> class uniform_distribution_wrapper {
public:
  uniform_distribution_wrapper(parameter_handler const& p, data_info& i, Engine& engine);
  T operator()() { return m_dist(m_engine); }
private:
  Engine& m_engine;
  std::uniform_real_distribution<T> m_dist;
};
```
Threading is now trivial: each thread constructs its own engine, passes it to its own distributions. Reproducibility is preserved because the engine is owned and seedable. Singleton `random_engine` class is **deleted**.

**Steps for Option 3:**
1. Refactor `distributions.h` to inject engine through ctor.
2. Update every call site in `rve_app::setup()` and tests.
3. Delete `random_engine` class entirely.
4. Add a `parallel_pack` example that runs `insert_random_algo` on N threads with N engines.

**Files touched:** `distributions.h`, `rve_app.cpp`, tests, possibly examples.

**Verification:** existing single-threaded runs produce *bit-identical* output to before (proves seeding still deterministic). New parallel example produces different but valid output.

**Effort:** 2 days. **Risk:** medium — touches every distribution call site. **Independent:** depends on Phase 7.

---

## Phase 9 — Concept-driven intersection dispatch

**Goal:** delete the `intersection_warehouse` singleton without paying for `std::visit`. Use C++20 concepts so that dispatch is fully compile-time, fully monomorphic, fully inlinable.

**Why not `std::visit`:** `std::visit` is fast only for small variants (typically ≤ 10–11 alternatives) where libstdc++/libc++ generate a switch-based jump table. Past that, both implementations fall back to a function-pointer table, which destroys inlining and is often slower than virtual calls. Even at 3–5 shape types we're fine today, but the design should scale to "as many shapes as you want" without re-architecting. Concepts give us that for free, with the bonus that overload resolution is exhaustively checked by the compiler.

**Design:**

1. **Define the `Shape` concept** (replaces `shape_base` inheritance):
   ```cpp
   template <typename S> concept Shape = requires(S const& s) {
     typename S::value_type;
     { S::dimension }      -> std::convertible_to<std::size_t>;
     { s.bounding_box() }  -> std::convertible_to<aligned_box<typename S::value_type, S::dimension>>;
   };
   ```

2. **Intersection becomes a constrained free-function overload set.** No warehouse, no lambda registration, no runtime lookup:
   ```cpp
   template <typename T> bool intersect(circle<T> const&, circle<T> const&);
   template <typename T> bool intersect(circle<T> const&, rectangle<T,2> const&);
   template <typename T> bool intersect(sphere<T> const&, sphere<T> const&);
   // ... etc, plus a concept-constrained fallback
   template <Shape A, Shape B> bool intersect(A const& a, B const& b)
     requires(!std::same_as<A,B>) { return intersect(b, a); }   // commutativity
   ```
   The compiler picks the right overload at the call site. Missing pairs fail to compile — that's the exhaustiveness check the warehouse never had.

3. **Heterogeneous storage problem.** The hot loop tests one trial shape against N existing shapes; today they live in a single `std::vector<unique_ptr<shape_base>>`. Two options:

   **Option A — homogeneous typed pools (recommended):**
   ```cpp
   template <Shape... Shapes> class shape_pool {
     std::tuple<std::vector<Shapes>...> m_pools;
   public:
     template <Shape S> void push(S&& s) {
       std::get<std::vector<S>>(m_pools).push_back(std::forward<S>(s));
     }
     template <Shape Candidate> bool any_intersect(Candidate const& c) const {
       return std::apply([&](auto const&... pool) {
         return ((std::ranges::any_of(pool, [&](auto const& other) {
           return intersect(c, other);
         })) || ...);
       }, m_pools);
     }
   };
   ```
   Fold expression unrolls at compile time. Each `intersect(c, other)` call is a direct, inlinable call to the right overload. Zero runtime dispatch. No heap allocation. This is roughly the structure CGAL uses internally for similar problems.

   **Option B — concept-constrained type-erased wrapper:**
   ```cpp
   class any_shape {
     // small-buffer-optimized erasure that stores any Shape S
     // dispatches intersect via a function pointer set at construction
   };
   ```
   Similar to `std::function`. Loses some inlining but keeps a single `std::vector<any_shape>`. Use this only if the algorithm genuinely cannot commit to a closed set of shape types at compile time.

   **Recommendation: Option A.** RVE generation is parameterized at compile time on the shape mix anyway; `shape_pool<circle<double>, sphere<double>>` is the natural specialization for a given simulation.

4. **The `insert_random_algo::run()` hot loop** becomes:
   ```cpp
   void run() {
     while (!m_termination(m_pool)) {
       auto candidate = m_generator();
       if (!m_pool.any_intersect(candidate)) {
         m_pool.push(std::move(candidate));
       }
     }
   }
   ```
   Note: no `unique_ptr`, no virtual call, no variant. Every step monomorphic.

5. **Delete:** `intersection_warehouse.h`, the local `singleton<T>` template, all `register_intersection<A,B>(...)` calls, the `shape_ptr` typedef, the `shape_base` class itself (replaced by the `Shape` concept), and the `shape_type_2D`/`shape_type_3D` variants (no longer needed).

6. **Run benchmarks** against the Phase 1 baseline. Expected: 3–10× faster on the inner loop because every call inlines, allocations are batched (vector growth), and there's no virtual dispatch or function-pointer indirection. The measured result might surprise you in either direction — capture it.

**Files touched:** delete `intersection_warehouse.h`, modify `algorithms.h` (the algo and pool), modify `intersection.h` (the constrained overload set), `shapes.h` (drop `shape_base` inheritance, satisfy `Shape` concept), delete static-init registration code.

**Verification:** Phase 1 tests pass with bit-identical packings (assuming Phase 8 fixed engine seeding). `shape_benchmark` shows perf improvement. Adding a new shape requires adding `intersect(NewShape, ExistingShape)` overloads — and the compiler tells you which ones are missing if you forget. `boost::container::flat_map` no longer used → drop the dependency.

**Effort:** 3 days. **Risk:** medium-high — this is the hot path *and* this is the change that retires `shape_base` inheritance. Add a regression test that pre-Phase-9 produces a packing with N shapes; post-Phase-9 must produce the same N (modulo RNG details). **Independent:** depends on Phase 3 (algorithms in their own TU) and Phase 8 (engines deterministic, so the regression test is meaningful).

---

## Phase 10 — Finish the concepts migration

**Goal:** Phase 9 introduces the `Shape` concept and retires `shape_base`. Phase 10 finishes the job: distributions, generators, and algorithms get the same treatment. The codebase exits this phase with concepts as the primary contract everywhere virtual dispatch used to be.

**Direction:**

1. **`Distribution` concept** (replaces `base_distribution<T>`):
   ```cpp
   template <typename D, typename T = std::invoke_result_t<D&>>
   concept Distribution = requires(D& d) {
     { d() } -> std::convertible_to<T>;
   };
   ```
   `uniform_distribution_wrapper`, `normal_distribution_wrapper`, `constant_distribution_wrapper` become value types satisfying `Distribution`. They no longer inherit from `base_distribution<T>`. Their engine is held via the injection done in Phase 8.

2. **Generators become constrained value types** (no `shape_generator_base`, no virtual `operator()`):
   ```cpp
   template <std::regular T, Distribution Px, Distribution Py, Distribution R>
   class circle_generator {
     Px& m_px; Py& m_py; R& m_r;
   public:
     circle_generator(Px& x, Py& y, R& r) : m_px(x), m_py(y), m_r(r) {}
     circle<T> operator()() { return circle<T>{ {m_px(), m_py()}, m_r() }; }
   };
   ```
   `circle_generator::operator()` returns `circle<T>` directly — no `unique_ptr`, no base class. This composes naturally with the `shape_pool` from Phase 9.

3. **Algorithms become free function templates** instead of classes inheriting from `basic_base`:
   ```cpp
   template <typename Generator, typename Termination, Shape... Shapes>
   shape_pool<Shapes...> insert_random(Generator& gen, Termination& term, shape_pool<Shapes...> pool = {}) {
     while (!term(pool)) {
       auto candidate = gen();
       if (!pool.any_intersect(candidate)) pool.push(std::move(candidate));
     }
     return pool;
   }
   ```
   No `insert_random_algo` class. No `basic_base` inheritance. The algorithm is just a function. Configuration (parameter handler, data_info) only needs to be passed to the *generator* and *distributions*, not the algorithm — which doesn't need configuration at all.

4. **Termination predicates become any callable** satisfying:
   ```cpp
   template <typename T, typename Pool> concept Termination = requires(T& t, Pool const& p) {
     { t(p) } -> std::convertible_to<bool>;
   };
   ```
   `number_of_inclusions` becomes a struct with `bool operator()(auto const& pool) const { return pool.size() > N; }` — no inheritance from `termination_function_base`, which gets deleted.

5. **`basic_base` retreats to its rightful scope.** Keep it only on types that are constructed from a `parameter_handler` via the registry (distribution wrappers, shape classes that take JSON-driven params). Algorithms, generators, free functions — none need it. After this pass, `basic_base` is purely "this type can be built from a config."

**Outcome:** roughly 30–40% line-count reduction across `main.cpp` (now `apps/rvegen_cli.cpp`), `algorithms.h`, `generators.h`, `distributions.h`. Every non-trivial dispatch is compile-time. Compile times go up modestly (more template instantiation), runtime goes down significantly (everything inlines).

**Files touched:** `distributions.h`, `generators.h`, `algorithms.h`, `shapes.h`, the registry registration sites in `rve_app.cpp`. Tests probably need adjustments since the types they construct change shape.

**Verification:** Phase 1 tests pass; bench shows further improvement. `grep -rn 'virtual'` in the project should return only destructors and the few VTK callbacks.

**Effort:** 4 days. **Risk:** medium (lower than Phase 9 because the disruptive change — retiring `shape_base` — is already done). **Independent:** depends on Phase 9 setting the `Shape` concept and Phase 8 injecting engines.

---

## Phase 11 — Hygiene pass

**Goal:** delete dead code, fix const-correctness, remove unused helpers. Do this last so you're not deleting comments that document patterns about to change.

**Items (from review):**
- **L1** Remove all commented-out code blocks: ImGui includes, alternative template signatures, alternate member declarations, the `algo.run()` comment in benchmarks, ImGui/GLFW/GLM wiring in CMakeLists.
- **L2** Add `const` to non-mutating getters across `basic_base`, `parameter_handler`, etc.
- **L3** Narrow singleton APIs: instead of returning mutable references, expose `register()`, `query()`, `sample()` methods.
- **L4** If algorithms keep an inheritance hierarchy, mark virtuals `override`; if they go the Phase 10 route, this is moot.
- **L5** Pull `parameter()` static methods out of distribution wrappers into a helper template.
- Remove `using weak_ptr = std::weak_ptr<T>` (orphaned after Phase 0).
- Remove `termination_function_base` if `number_of_inclusions` is the only descendant and it's never used polymorphically.
- Remove unused `shape_generator_base` if `circle_generator` is the only descendant after Phase 10.
- Replace `boost::timer::cpu_timer` references that survived Phase 2 (shouldn't be any).
- Remove `#include`s nobody uses anymore.

**Effort:** 1 day. **Risk:** low. **Independent:** do last.

---

## Sequencing summary

| Phase | Goal                        | Effort  | Risk     | Depends on |
|-------|-----------------------------|---------|----------|------------|
| 0     | Inventory (done)            | —       | —        | —          |
| 1     | Stabilize + tests + CI      | 0.5d    | low      | 0          |
| 2     | CMake flags + Boost out     | 0.5d    | low      | 1          |
| 3     | Decompose main.cpp          | 2d      | medium   | 1, 2       |
| 4     | argv config + examples      | 1h      | low      | 3          |
| 5     | numsim-core consolidation   | 0.5d    | low      | independent|
| 6     | Move basic_base             | 1d      | low      | 5          |
| 7     | object_registry migration   | 1.5d    | medium   | 5, 6       |
| 8     | Engine injection            | 2d      | medium   | 7          |
| 9     | Concept-driven dispatch     | 3d      | med-high | 3, 8       |
| 10    | Finish concepts migration   | 4d      | medium   | 9          |
| 11    | Hygiene pass                | 1d      | low      | last       |

**Total to Phase 9 (conservative end-state):** ~13 working days.
**Total through Phase 11 (full modernization):** ~18 working days (~3.5 weeks).

Note: Phase 10 is no longer "optional modernization" — it's the natural completion of Phase 9. Stopping at Phase 9 leaves an inconsistent codebase where shapes use concepts but distributions and algorithms still use inheritance. Stop at the end of Phase 8 if you want a non-concept resting point.

---

## Stop rules

The plan is built so you can stop at the end of any phase and ship a better-than-before codebase. Reasonable stop points if scope tightens:

- **After Phase 4:** the codebase is decomposed and configurable; bugs are fixed. Acceptable stopping point if the goal was just "stop bleeding."
- **After Phase 7:** the framework layer is consolidated and rvegen is using its parent project's facilities. Acceptable stopping point if you're prioritizing correctness over performance/threading.
- **After Phase 9:** all P0/P1 review items closed. The shape of the code is solid; only the modernization remains. Acceptable as a long-term resting point.
- **After Phase 11:** finished. Ship.

---

## What this plan deliberately does *not* do

- **No "rewrite from scratch."** The bones are good; rewriting would discard the genuinely-good parts (schema validation, static_indexing, the two-layer split).
- **No premature optimization.** Phase 9's perf work is tied to a benchmark baseline so it's measured, not guessed.
- **No new features.** Every phase is structural. New features (more shape types, packing algorithms, IO formats) wait until the structure is sound — adding them now would make every later phase harder.
- **No documentation phase.** Documentation should fall out of the structural work (header comments on the public API, README updated as CMake options stabilize). A standalone "write docs" phase tends to produce documentation that's already stale.

---

## Open questions to resolve before starting

1. **Is rvegen the only consumer of numsim-core?** If yes, Phase 5 (numsim-core consolidation) is straightforward. If no, it needs an audit of all consumers.
2. **Is single-threaded acceptable as a permanent stance?** If yes, skip Phase 8 Option 3 and ship Option 1 instead. If no, commit to Option 3.
3. **Is reproducibility a hard requirement?** If yes, Phase 8 Option 3 is the only acceptable choice. If no, Option 2 (`thread_local`) is fine.
4. **Phase 10 yes or no?** The conservative end-state is Phase 9. Phase 10 is a strategic bet on the codebase having a future and being worth modernizing.

Answers to (2)–(4) shape ~30% of the effort. Worth deciding before Phase 1.
