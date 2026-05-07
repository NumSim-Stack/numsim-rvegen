// Compare collision-dispatch strategies for heterogeneous shape vectors.
//
// Five contestants, all answering the same question — "do shape A and shape
// B intersect?" — but recovering the concrete type differently:
//
//   1. monomorphic         — homogeneous vector, direct typed call.
//                            Theoretical upper bound; impossible if shapes
//                            are heterogeneous, but useful as a yardstick.
//   2. aabb_only           — what the generator does today: AABB overlap via
//                            virtual bounding_box(). Conservative (false
//                            positives possible), no precise check.
//   3. dispatcher_table    — pairwise type_index → fn_ptr lookup, the
//                            original numsim-rvegen approach. Function
//                            pointer call inlines GTE math.
//   4. std_visit_variant   — vector<variant<...>> + std::visit. Compile-time
//                            but hits the libstdc++ function-table cliff
//                            past ~10 alternatives.
//   5. dynamic_cast_chain  — hand-rolled if/else inside a free function.
//                            Zero infrastructure but verbose to extend.
//
// Test load: heterogeneous 2D shapes (circle, rectangle, ellipse) in equal
// proportion. For each N, run all O(N²) pair checks and report
// time-per-pair. Uses the same RNG seed across runs for reproducibility.

#include <chrono>
#include <cstddef>
#include <memory>
#include <random>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include <benchmark/benchmark.h>

#include "rvegen/rvegen.h"

namespace {

using T = double;
using shape_ptr   = std::unique_ptr<rvegen::shape_base<T>>;
using shape_vec   = std::vector<shape_ptr>;
// Use the library type for the visitor variant — same shape mix as
// shape_pool below for fairness.
using shape_var   = rvegen::shape_variant_2d<T>;
using shape_vvec  = std::vector<shape_var>;
using circle_vec  = std::vector<rvegen::circle<T>>;
using pool_t      = rvegen::shape_pool<rvegen::circle<T>,
                                        rvegen::rectangle<T>,
                                        rvegen::ellipse<T>>;

// ---------------------------------------------------------------------------
// Shape generators — all use the same seed so the SAME shapes are produced
// across strategies. That way the per-pair work is identical and only the
// dispatch overhead differs.
// ---------------------------------------------------------------------------

shape_vec make_heterogeneous_polymorphic(std::size_t n) {
  std::mt19937 e{42};
  std::uniform_real_distribution<T> u{0.0, 1.0};
  shape_vec v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    switch (i % 3) {
      case 0:
        v.emplace_back(std::make_unique<rvegen::circle<T>>(u(e), u(e), 0.05));
        break;
      case 1:
        v.emplace_back(std::make_unique<rvegen::rectangle<T>>(u(e), u(e), 0.10, 0.08));
        break;
      case 2:
        v.emplace_back(std::make_unique<rvegen::ellipse<T>>(
            u(e), u(e), 0.06, 0.04, u(e) * 3.14159));
        break;
    }
    v.back()->make_bounding_box();
  }
  return v;
}

shape_vvec make_heterogeneous_variant(std::size_t n) {
  std::mt19937 e{42};
  std::uniform_real_distribution<T> u{0.0, 1.0};
  shape_vvec v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    switch (i % 3) {
      case 0: {
        rvegen::circle<T> c{u(e), u(e), 0.05};
        c.make_bounding_box();
        v.emplace_back(std::move(c));
        break;
      }
      case 1: {
        rvegen::rectangle<T> r{u(e), u(e), 0.10, 0.08};
        r.make_bounding_box();
        v.emplace_back(std::move(r));
        break;
      }
      case 2: {
        rvegen::ellipse<T> el{u(e), u(e), 0.06, 0.04, u(e) * 3.14159};
        el.make_bounding_box();
        v.emplace_back(std::move(el));
        break;
      }
    }
  }
  return v;
}

// Same shapes as the polymorphic and variant builders, but bucketed into
// the typed pool so the compiler sees concrete types end-to-end.
pool_t make_heterogeneous_pool(std::size_t n) {
  std::mt19937 e{42};
  std::uniform_real_distribution<T> u{0.0, 1.0};
  pool_t p;
  for (std::size_t i = 0; i < n; ++i) {
    switch (i % 3) {
      case 0: {
        rvegen::circle<T> c{u(e), u(e), 0.05};
        c.make_bounding_box();
        p.push(std::move(c));
        break;
      }
      case 1: {
        rvegen::rectangle<T> r{u(e), u(e), 0.10, 0.08};
        r.make_bounding_box();
        p.push(std::move(r));
        break;
      }
      case 2: {
        rvegen::ellipse<T> el{u(e), u(e), 0.06, 0.04, u(e) * 3.14159};
        el.make_bounding_box();
        p.push(std::move(el));
        break;
      }
    }
  }
  return p;
}

circle_vec make_homogeneous_circles(std::size_t n) {
  std::mt19937 e{42};
  std::uniform_real_distribution<T> u{0.0, 1.0};
  circle_vec v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    v.emplace_back(u(e), u(e), 0.05);
    v.back().make_bounding_box();
  }
  return v;
}

// ---------------------------------------------------------------------------
// Strategy 3: pairwise dispatcher table.
//
// The dispatcher used to live here as a local class; now it's a first-class
// library type (rvegen::collision_dispatcher) so production code (the
// generators) and the benchmark exercise the EXACT same implementation.
// ---------------------------------------------------------------------------
void ensure_collision_pairs_registered() {
  static bool once = []{
    rvegen::register_all_collision_pairs<T>();
    return true;
  }();
  (void)once;
}

// ---------------------------------------------------------------------------
// Strategy 5: dynamic_cast chain.
// ---------------------------------------------------------------------------
bool dispatch_dynamic_cast(rvegen::shape_base<T> const& a,
                            rvegen::shape_base<T> const& b) {
  if (auto* ca = dynamic_cast<rvegen::circle<T> const*>(&a)) {
    if (auto* cb = dynamic_cast<rvegen::circle<T> const*>(&b))
      return rvegen::collision_details(*ca, *cb);
    if (auto* rb = dynamic_cast<rvegen::rectangle<T> const*>(&b))
      return rvegen::collision_details(*ca, *rb);
    if (auto* eb = dynamic_cast<rvegen::ellipse<T> const*>(&b))
      return rvegen::collision_details(*ca, *eb);
  } else if (auto* ra = dynamic_cast<rvegen::rectangle<T> const*>(&a)) {
    if (auto* rb = dynamic_cast<rvegen::rectangle<T> const*>(&b))
      return rvegen::collision_details(*ra, *rb);
    if (auto* cb = dynamic_cast<rvegen::circle<T> const*>(&b))
      return rvegen::collision_details(*ra, *cb);
  } else if (auto* ea = dynamic_cast<rvegen::ellipse<T> const*>(&a)) {
    if (auto* eb = dynamic_cast<rvegen::ellipse<T> const*>(&b))
      return rvegen::collision_details(*ea, *eb);
    if (auto* cb = dynamic_cast<rvegen::circle<T> const*>(&b))
      return rvegen::collision_details(*ea, *cb);
  }
  return rvegen::aabb_overlap(*a.bounding_box(), *b.bounding_box());
}

// ---------------------------------------------------------------------------
// Strategy 4: std::visit on variant.
// (visitor lives in the library now: rvegen::collide_visitor)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Benchmarks.
// ---------------------------------------------------------------------------
template <typename Vec, typename Dispatch>
std::size_t pair_check_loop(Vec const& v, Dispatch&& d) {
  std::size_t hits = 0;
  for (std::size_t i = 0; i < v.size(); ++i)
    for (std::size_t j = i + 1; j < v.size(); ++j)
      if (d(v[i], v[j])) ++hits;
  return hits;
}

// Strategy 1: monomorphic — typed circle vector, direct call.
static void BM_monomorphic_circles(benchmark::State& s) {
  auto v = make_homogeneous_circles(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return rvegen::collision_details(a, b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_monomorphic_circles)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Strategy 2: AABB-only via virtual bounding_box.
static void BM_aabb_only(benchmark::State& s) {
  auto v = make_heterogeneous_polymorphic(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return rvegen::aabb_overlap(*a->bounding_box(), *b->bounding_box());
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_aabb_only)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Strategy 3: dispatcher table.
static void BM_dispatcher_table(benchmark::State& s) {
  ensure_collision_pairs_registered();
  auto v = make_heterogeneous_polymorphic(s.range(0));
  auto const& d = rvegen::collision_dispatcher<T>::instance();
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [&d](auto const& a, auto const& b) {
      return d(*a, *b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_dispatcher_table)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Strategy 3b: two-stage = AABB pre-check + dispatcher (the generator pattern).
// Uses the simple dispatcher() per pair — re-does the outer lookup each call.
static void BM_two_stage(benchmark::State& s) {
  ensure_collision_pairs_registered();
  auto v = make_heterogeneous_polymorphic(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return rvegen::collide_two_stage(*a, *b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_two_stage)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Strategy 3c: dispatcher with LHS hoisted out of the inner loop via
// bind_lhs. This is the actual generator hot-loop pattern.
static void BM_dispatcher_table_bound(benchmark::State& s) {
  ensure_collision_pairs_registered();
  auto v = make_heterogeneous_polymorphic(s.range(0));
  auto const& d = rvegen::collision_dispatcher<T>::instance();
  for (auto _ : s) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto const bound = d.bind_lhs(*v[i]);
      for (std::size_t j = i + 1; j < v.size(); ++j) {
        if (bound(*v[i], *v[j])) ++hits;
      }
    }
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_dispatcher_table_bound)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Strategy 3d: two-stage with bound LHS — what the generators actually do.
static void BM_two_stage_bound(benchmark::State& s) {
  ensure_collision_pairs_registered();
  auto v = make_heterogeneous_polymorphic(s.range(0));
  auto const& d = rvegen::collision_dispatcher<T>::instance();
  for (auto _ : s) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto const bound = d.bind_lhs(*v[i]);
      auto const* abb  = v[i]->bounding_box();
      for (std::size_t j = i + 1; j < v.size(); ++j) {
        if (!rvegen::aabb_overlap(*abb, *v[j]->bounding_box())) continue;
        if (bound(*v[i], *v[j])) ++hits;
      }
    }
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_two_stage_bound)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Two-stage with std::visit on a variant.
static void BM_two_stage_visit(benchmark::State& s) {
  auto v = make_heterogeneous_variant(s.range(0));
  for (auto _ : s) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      // bounding_box accessible via std::visit on the variant
      auto const* abb = std::visit([](auto const& x) { return x.bounding_box(); }, v[i]);
      for (std::size_t j = i + 1; j < v.size(); ++j) {
        auto const* bbb = std::visit([](auto const& x) { return x.bounding_box(); }, v[j]);
        if (!rvegen::aabb_overlap(*abb, *bbb)) continue;
        if (std::visit(rvegen::collide_visitor{}, v[i], v[j])) ++hits;
      }
    }
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_two_stage_visit)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Two-stage with dynamic_cast chain.
static void BM_two_stage_dynamic_cast(benchmark::State& s) {
  auto v = make_heterogeneous_polymorphic(s.range(0));
  for (auto _ : s) {
    std::size_t hits = 0;
    for (std::size_t i = 0; i < v.size(); ++i) {
      auto const* abb = v[i]->bounding_box();
      for (std::size_t j = i + 1; j < v.size(); ++j) {
        if (!rvegen::aabb_overlap(*abb, *v[j]->bounding_box())) continue;
        if (dispatch_dynamic_cast(*v[i], *v[j])) ++hits;
      }
    }
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_two_stage_dynamic_cast)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Two-stage on shape_pool — AABB + typed compile-time precise check.
static void BM_two_stage_shape_pool(benchmark::State& s) {
  auto p = make_heterogeneous_pool(s.range(0));
  for (auto _ : s) {
    auto hits = rvegen::count_two_stage_colliding_pairs(p);
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_two_stage_shape_pool)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// (Periodic-collision benches removed: periodic_generator now stores
// wrap copies of accepted shapes explicitly; collision check on the
// augmented vector uses standard non-periodic two-stage. See
// shapes/periodic_wraps.h.)

// Strategy 4: std::visit.
static void BM_std_visit(benchmark::State& s) {
  auto v = make_heterogeneous_variant(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return std::visit(rvegen::collide_visitor{}, a, b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_std_visit)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Strategy 5: dynamic_cast chain.
static void BM_dynamic_cast_chain(benchmark::State& s) {
  auto v = make_heterogeneous_polymorphic(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return dispatch_dynamic_cast(*a, *b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_dynamic_cast_chain)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// HOMOGENEOUS BENCHMARKS — circle-only.
//
// Same dispatch strategies, but every shape is a circle. The underlying math
// is now uniformly cheap (squared-distance compare), so the per-pair cost
// shows the DISPATCH OVERHEAD rather than the geometric query.
// ---------------------------------------------------------------------------

// Build a polymorphic (shape_base*) container of all circles.
shape_vec make_homogeneous_polymorphic_circles(std::size_t n) {
  std::mt19937 e{42};
  std::uniform_real_distribution<T> u{0.0, 1.0};
  shape_vec v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    v.emplace_back(std::make_unique<rvegen::circle<T>>(u(e), u(e), 0.05));
    v.back()->make_bounding_box();
  }
  return v;
}

// Variant of size 1 (circles only) to test std::visit's overhead on the
// trivial case.
using circle_var  = std::variant<rvegen::circle<T>>;
using circle_vvec = std::vector<circle_var>;

circle_vvec make_homogeneous_variant_circles(std::size_t n) {
  std::mt19937 e{42};
  std::uniform_real_distribution<T> u{0.0, 1.0};
  circle_vvec v;
  v.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    rvegen::circle<T> c{u(e), u(e), 0.05};
    c.make_bounding_box();
    v.emplace_back(std::move(c));
  }
  return v;
}

// AABB-only on homogeneous circles — virtual bounding_box() + AABB compare.
static void BM_homo_aabb_only(benchmark::State& s) {
  auto v = make_homogeneous_polymorphic_circles(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return rvegen::aabb_overlap(*a->bounding_box(), *b->bounding_box());
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_homo_aabb_only)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Dispatcher table on homogeneous circles.
static void BM_homo_dispatcher_table(benchmark::State& s) {
  ensure_collision_pairs_registered();
  auto v = make_homogeneous_polymorphic_circles(s.range(0));
  auto const& d = rvegen::collision_dispatcher<T>::instance();
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [&d](auto const& a, auto const& b) {
      return d(*a, *b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_homo_dispatcher_table)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Two-stage on homogeneous circles — most pairs reject at AABB.
static void BM_homo_two_stage(benchmark::State& s) {
  ensure_collision_pairs_registered();
  auto v = make_homogeneous_polymorphic_circles(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return rvegen::collide_two_stage(*a, *b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_homo_two_stage)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// std::visit on a 1-alternative variant — the easy case for the compiler.
static void BM_homo_std_visit(benchmark::State& s) {
  auto v = make_homogeneous_variant_circles(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return std::visit(rvegen::collide_visitor{}, a, b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_homo_std_visit)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// dynamic_cast chain on homogeneous circles — first cast hits.
static void BM_homo_dynamic_cast(benchmark::State& s) {
  auto v = make_homogeneous_polymorphic_circles(s.range(0));
  for (auto _ : s) {
    auto hits = pair_check_loop(v, [](auto const& a, auto const& b) {
      return dispatch_dynamic_cast(*a, *b);
    });
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_homo_dynamic_cast)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// Strategy 6: shape_pool typed pool.
//
// Storage = tuple<vector<circle>, vector<rectangle>, vector<ellipse>>.
// Each pair check is a typed call resolved at compile time — no dispatch,
// no virtual, no variant. Using rvegen::count_colliding_pairs which folds
// over all pool combinations.
// ---------------------------------------------------------------------------
static void BM_shape_pool(benchmark::State& s) {
  auto p = make_heterogeneous_pool(s.range(0));
  for (auto _ : s) {
    auto hits = rvegen::count_colliding_pairs(p);
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_shape_pool)->Arg(10)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

// Homogeneous shape_pool — single-pool tuple, one fold over circle×circle.
// Should compile down to the monomorphic case.
using circle_pool_t = rvegen::shape_pool<rvegen::circle<T>>;

circle_pool_t make_homogeneous_circle_pool(std::size_t n) {
  std::mt19937 e{42};
  std::uniform_real_distribution<T> u{0.0, 1.0};
  circle_pool_t p;
  for (std::size_t i = 0; i < n; ++i) {
    rvegen::circle<T> c{u(e), u(e), 0.05};
    c.make_bounding_box();
    p.push(std::move(c));
  }
  return p;
}

static void BM_homo_shape_pool(benchmark::State& s) {
  auto p = make_homogeneous_circle_pool(s.range(0));
  for (auto _ : s) {
    auto hits = rvegen::count_colliding_pairs(p);
    benchmark::DoNotOptimize(hits);
  }
  s.SetItemsProcessed(s.iterations()
                      * std::size_t{s.range(0)} * (s.range(0) - 1) / 2);
}
BENCHMARK(BM_homo_shape_pool)->Arg(100)->Arg(1000)->Unit(benchmark::kMicrosecond);

} // namespace

BENCHMARK_MAIN();
