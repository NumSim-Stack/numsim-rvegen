// Phase 8 concurrency smoke: prove the library is safe to use from multiple
// threads when each thread owns its own engine, distributions, and pipeline.
//
// The library has zero shared mutable state in the hot path — distributions
// hold a reference to a per-thread engine, and the registry singletons are
// read-only after the once-per-process register_all_*().
//
// What this test guarantees:
//   1. Same seed in different threads → identical shape vectors (proves the
//      pipeline is fully deterministic given the engine).
//   2. Different seeds in different threads → different shape vectors
//      (proves engines are independent; no cross-thread leakage).
//   3. N threads run to completion concurrently with no data races (run
//      under TSan if you want belt-and-braces; this test verifies behavioural
//      correctness).

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

#include "rvegen/rvegen.h"

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// Build and run a complete pipeline with the given seed. Returns the
// vector of (x, y, radius) tuples for the accepted circles — used for
// determinism comparisons across threads.
struct circle_record {
  double x, y, r;
  bool operator==(circle_record const&) const = default;
};

std::vector<circle_record> run_pipeline(std::uint32_t seed, std::size_t target) {
  std::mt19937 engine{seed};

  rvegen::uniform_real_distribution<double> px{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> py{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> r{0.03, 0.05, engine};

  auto input = std::make_unique<rvegen::circle_input<double>>(px, py, r);
  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::move(input));

  rvegen::number_of_inclusions<double> term{target};
  rvegen::only_inside_generator<double> gen{100'000};
  auto shapes = gen.compute(inputs, term, {1.0, 1.0, 0.0});

  std::vector<circle_record> records;
  records.reserve(shapes.size());
  for (auto const& s : shapes) {
    auto* c = static_cast<rvegen::circle<double> const*>(s.get());
    records.push_back({(*c)(0), (*c)(1), c->radius});
  }
  return records;
}

// ----------------------------------------------------------------------------
// Determinism: same seed in different threads → identical results.
// ----------------------------------------------------------------------------
void test_same_seed_same_result_across_threads() {
  constexpr std::size_t target = 8;
  std::vector<std::vector<circle_record>> results(4);
  std::vector<std::thread> workers;

  for (std::size_t i = 0; i < 4; ++i) {
    workers.emplace_back([i, &results, target] {
      results[i] = run_pipeline(/*seed=*/12345, target);
    });
  }
  for (auto& w : workers) w.join();

  for (std::size_t i = 1; i < 4; ++i) {
    REQUIRE(results[i] == results[0]);
  }
}

// ----------------------------------------------------------------------------
// Independence: different seeds in different threads → different results.
// ----------------------------------------------------------------------------
void test_different_seeds_diverge() {
  constexpr std::size_t target = 8;
  std::vector<std::vector<circle_record>> results(4);
  std::vector<std::thread> workers;

  for (std::size_t i = 0; i < 4; ++i) {
    workers.emplace_back([i, &results, target] {
      results[i] = run_pipeline(/*seed=*/1000 + static_cast<std::uint32_t>(i),
                                target);
    });
  }
  for (auto& w : workers) w.join();

  // No two should be identical.
  for (std::size_t i = 0; i < 4; ++i) {
    for (std::size_t j = i + 1; j < 4; ++j) {
      REQUIRE(results[i] != results[j]);
    }
  }
}

// ----------------------------------------------------------------------------
// Heavy parallel load: 8 threads, each generates a non-trivial RVE.
// Smoke for "no crashes, no deadlocks, all threads finish".
// ----------------------------------------------------------------------------
void test_parallel_load_completes() {
  constexpr std::size_t n_threads = 8;
  std::atomic<std::size_t> total_shapes{0};
  std::vector<std::thread> workers;

  for (std::size_t i = 0; i < n_threads; ++i) {
    workers.emplace_back([i, &total_shapes] {
      auto rec = run_pipeline(/*seed=*/static_cast<std::uint32_t>(i + 1),
                              /*target=*/15);
      total_shapes.fetch_add(rec.size(), std::memory_order_relaxed);
    });
  }
  for (auto& w : workers) w.join();

  // Each thread targets 15 shapes; with 8 threads we expect 120 total
  // (the generator should hit the target in plenty of attempts).
  REQUIRE(total_shapes.load() == n_threads * 15);
}

} // namespace

int main() {
  test_same_seed_same_result_across_threads();
  test_different_seeds_diverge();
  test_parallel_load_completes();

  if (failures > 0) {
    std::cerr << failures << " concurrency smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "concurrency_smoke: PASS\n";
  return EXIT_SUCCESS;
}
