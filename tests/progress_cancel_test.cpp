// Smoke test for the progress/cancel hooks on rve_generator_base::compute().
//
// 1. Callbacks fire at least once for a long-running placement, and
//    shapes_placed is monotonically non-decreasing.
// 2. Setting cancel() to return true halts the generator early — the
//    returned vector size is strictly less than the termination target.
// 3. Default-constructed progress_options matches the pre-hooks behaviour
//    (sanity check: the unchanged call sites in end_to_end_smoke and the
//    CLI still build by re-invoking compute() with no opts argument).

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
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

// circle_input borrows distributions by reference; build the distributions
// in the test scope (not in a helper function) so they outlive the inputs.

void test_progress_callback_fires_and_is_monotonic() {
  std::mt19937 engine{42};
  rvegen::uniform_real_distribution<double> pos_x{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> radius{0.01, 0.02, engine};

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));

  rvegen::number_of_inclusions<double> termination{500};
  rvegen::only_inside_generator<double> generator;

  std::vector<std::size_t> recorded;
  rvegen::progress_options opts;
  opts.emit_every = 50;
  opts.on_progress = [&recorded](rvegen::progress_info info) {
    recorded.push_back(info.shapes_placed);
  };

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0}, opts);
  REQUIRE(shapes.size() == 500);

  // At least one callback should have fired (500 / 50 ≈ 10 expected).
  REQUIRE(!recorded.empty());
  REQUIRE(recorded.size() >= 5);

  // shapes_placed is monotonically non-decreasing.
  for (std::size_t i = 1; i < recorded.size(); ++i) {
    REQUIRE(recorded[i] >= recorded[i - 1]);
  }

  // Last reported value is <= final accepted size.
  REQUIRE(recorded.back() <= shapes.size());
}

void test_cancel_halts_generator() {
  std::mt19937 engine{7};
  rvegen::uniform_real_distribution<double> pos_x{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> radius{0.01, 0.02, engine};

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));

  rvegen::number_of_inclusions<double> termination{10000}; // unreachable in a [0,1]² box
  rvegen::only_inside_generator<double> generator;
  generator.set_max_attempts(10'000'000);

  std::atomic<bool> cancel_flag{false};
  const std::size_t cancel_after_n_callbacks = 2;
  std::size_t callback_count = 0;

  rvegen::progress_options opts;
  opts.emit_every = 25;
  opts.cancel = [&cancel_flag] { return cancel_flag.load(); };
  opts.on_progress = [&](rvegen::progress_info) {
    if (++callback_count >= cancel_after_n_callbacks) {
      cancel_flag.store(true);
    }
  };

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0}, opts);

  // Cancel triggered well before the 10000-shape target → vector is partial.
  REQUIRE(shapes.size() < 10000);
  // Cancel should fire after at least a few placements were emitted.
  REQUIRE(callback_count >= cancel_after_n_callbacks);
}

void test_progress_carries_termination_target_count() {
  // number_of_inclusions termination should surface its target count
  // through progress_info::shapes_target every callback. Tessera reads
  // this to render "X / N" in the progress bar.
  std::mt19937 engine{11};
  rvegen::uniform_real_distribution<double> pos_x{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> radius{0.01, 0.02, engine};

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));

  constexpr std::size_t kTarget = 200;
  rvegen::number_of_inclusions<double> termination{kTarget};
  rvegen::only_inside_generator<double> generator;

  std::vector<rvegen::progress_info> snapshots;
  rvegen::progress_options opts;
  opts.emit_every = 50;
  opts.on_progress = [&snapshots](rvegen::progress_info info) {
    snapshots.push_back(info);
  };

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0}, opts);
  REQUIRE(shapes.size() == kTarget);
  REQUIRE(!snapshots.empty());

  for (auto const& snap : snapshots) {
    REQUIRE(snap.shapes_target == kTarget);
    // current_volume_fraction should be in (0, 1) for a 1x1 box with
    // 0.01-0.02 radius circles — non-zero (we placed shapes) and well
    // below 1.0 (200 small circles can't fill a unit box).
    REQUIRE(snap.volume_fraction > 0.0);
    REQUIRE(snap.volume_fraction < 1.0);
  }

  // Volume fraction should be monotonically non-decreasing as more
  // shapes accumulate (only_inside doesn't remove anything).
  for (std::size_t i = 1; i < snapshots.size(); ++i) {
    REQUIRE(snapshots[i].volume_fraction >= snapshots[i - 1].volume_fraction);
  }
}

void test_progress_target_unknown_for_until_full() {
  // until_full has no notion of "target" — both progress fields stay
  // at the default-zero, signalling "unbounded" to the consumer.
  std::mt19937 engine{13};
  rvegen::uniform_real_distribution<double> pos_x{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> radius{0.05, 0.07, engine};

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));

  rvegen::until_full<double> termination;
  rvegen::only_inside_generator<double> generator;
  generator.set_max_attempts(20'000);

  std::vector<rvegen::progress_info> snapshots;
  rvegen::progress_options opts;
  opts.emit_every = 5;
  opts.on_progress = [&snapshots](rvegen::progress_info info) {
    snapshots.push_back(info);
  };

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0}, opts);
  REQUIRE(!shapes.empty());

  // until_full leaves both progress targets at zero — the consumer
  // (Tessera's progress bar) should display count-only when it sees this.
  REQUIRE(termination.target_count() == 0);
  REQUIRE(termination.target_volume_fraction() == 0.0);
  for (auto const& snap : snapshots) {
    REQUIRE(snap.shapes_target == 0);
  }
}

void test_default_options_match_pre_hook_behaviour() {
  // Calling compute() with no opts argument compiles (default-constructed
  // progress_options is no-op + never-cancel) and produces the same result
  // as the historical signature.
  std::mt19937 engine{42};
  rvegen::uniform_real_distribution<double> pos_x{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> pos_y{0.05, 0.95, engine};
  rvegen::uniform_real_distribution<double> radius{0.03, 0.05, engine};

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::make_unique<rvegen::circle_input<double>>(pos_x, pos_y, radius));

  rvegen::number_of_inclusions<double> termination{10};
  rvegen::only_inside_generator<double> generator;

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0});
  REQUIRE(shapes.size() == 10);
}

} // namespace

int main() {
  test_default_options_match_pre_hook_behaviour();
  test_progress_callback_fires_and_is_monotonic();
  test_cancel_halts_generator();

  if (failures > 0) {
    std::cerr << failures << " progress/cancel test failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "progress_cancel_test: PASS\n";
  return EXIT_SUCCESS;
}
