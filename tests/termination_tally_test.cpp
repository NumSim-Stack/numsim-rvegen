// Running tally of the termination criteria (#153): the answer with a tally
// is the answer of the pass over all accepted shapes, the criteria look at
// every shape once, and the placement generators use the tally.

#include <array>
#include <cstddef>
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

using D = double;
using shape_vector = rvegen::termination_base<D>::shape_vector;
using tally_t = rvegen::termination_tally<D>;

std::size_t area_calls = 0;

// A circle that counts how often its area is asked for.
struct counting_circle final : rvegen::circle<D> {
  using rvegen::circle<D>::circle;
  [[nodiscard]] D area() const override {
    ++area_calls;
    return rvegen::circle<D>::area();
  }
};

// Circles in the unit square, about a third of them periodic copies and a
// few with the centre outside the box.
shape_vector random_shapes(std::size_t n, unsigned seed) {
  std::mt19937 engine{seed};
  auto uniform = [&](D a, D b) {
    return a + (b - a) * (static_cast<D>(engine() >> 8) / 16777216.0);
  };
  shape_vector out;
  for (std::size_t i = 0; i < n; ++i) {
    auto c = std::make_unique<counting_circle>(uniform(-0.1, 1.1),
                                               uniform(-0.1, 1.1),
                                               uniform(0.001, 0.02));
    if (engine() % 3 == 0) c->set_image_shift({1, 0, 0});
    out.push_back(std::move(c));
  }
  return out;
}

// Appending in uneven steps, the tally answers as the full pass does, and
// its sum is the sum of the full pass to the last bit.
void test_tally_equals_full_pass() {
  const rvegen::domain<D> box{std::array<D, 3>{1.0, 1.0, 0.0}};
  for (unsigned seed = 1; seed <= 5; ++seed) {
    auto pool = random_shapes(600, seed);
    const rvegen::volume_fraction<D> boxed{0.02 * seed, box};
    const rvegen::volume_fraction<D> unboxed{0.02 * seed, 1.0};
    const rvegen::number_of_inclusions<D> counted{40 * seed};
    for (bool count_all : {false, true}) {
      shape_vector accepted;
      tally_t t_boxed, t_unboxed, t_counted;
      std::size_t step = 1;
      bool seen_false = false, seen_true = false;
      for (std::size_t next = 0; next < pool.size();) {
        for (std::size_t k = 0; k < step && next < pool.size(); ++k, ++next)
          accepted.push_back(pool[next]->clone());
        step = step % 4 + 1;   // 1, 2, 3, 4 shapes between two questions
        const bool full = boxed.is_met(accepted, count_all);
        REQUIRE(boxed.is_met(accepted, count_all, t_boxed) == full);
        REQUIRE(t_boxed.seen == accepted.size());
        REQUIRE(t_boxed.sum / boxed.domain_size() == boxed.fraction(accepted));
        REQUIRE(unboxed.is_met(accepted, count_all, t_unboxed) ==
                unboxed.is_met(accepted, count_all));
        REQUIRE(t_unboxed.sum / unboxed.domain_size() ==
                unboxed.fraction(accepted));
        REQUIRE(counted.is_met(accepted, count_all, t_counted) ==
                counted.is_met(accepted, count_all));
        (full ? seen_true : seen_false) = true;
      }
      REQUIRE(seen_false && seen_true);
      // Asked again without new shapes: the same answer, nothing added.
      const auto before = t_boxed;
      REQUIRE(boxed.is_met(accepted, count_all, t_boxed) ==
              boxed.is_met(accepted, count_all));
      REQUIRE(t_boxed.sum == before.sum && t_boxed.seen == before.seen);
      // A shorter list is not the list the tally knew: it starts over.
      accepted.resize(accepted.size() / 2);
      REQUIRE(boxed.is_met(accepted, count_all, t_boxed) ==
              boxed.is_met(accepted, count_all));
      REQUIRE(t_boxed.sum / boxed.domain_size() == boxed.fraction(accepted));
      REQUIRE(counted.is_met(accepted, count_all, t_counted) ==
              counted.is_met(accepted, count_all));
    }
  }
}

// One look at every shape, however often the criterion is asked.
void test_tally_looks_at_each_shape_once() {
  const rvegen::domain<D> box{std::array<D, 3>{1.0, 1.0, 0.0}};
  const rvegen::volume_fraction<D> criterion{0.99, box};
  auto pool = random_shapes(2000, 9);
  shape_vector accepted;
  tally_t tally;
  area_calls = 0;
  std::size_t counted = 0;
  for (auto& s : pool) {
    counted += rvegen::is_counted_in_rve(*s, box) ? 1 : 0;
    accepted.push_back(std::move(s));
    REQUIRE(!criterion.is_met(accepted, false, tally));
  }
  REQUIRE(area_calls == counted);   // not 2000 * 2001 / 2
  REQUIRE(counted > 500 && counted < 2000);
}

// A criterion without a tally of its own is still asked the plain way.
struct plain_criterion final : rvegen::termination_base<D> {
  mutable std::size_t asked = 0;
  [[nodiscard]] bool operator()(shape_vector const& accepted) const override {
    ++asked;
    return accepted.size() >= 7;
  }
};

// volume_fraction that counts the passes over all shapes it is asked for.
struct spy_fraction final : rvegen::termination_base<D> {
  rvegen::volume_fraction<D> inner;
  mutable std::size_t full_passes = 0;
  mutable std::size_t with_tally = 0;
  spy_fraction(D target, rvegen::domain<D> const& box) : inner{target, box} {}
  [[nodiscard]] bool operator()(shape_vector const& accepted) const override {
    ++full_passes;
    return inner(accepted);
  }
  [[nodiscard]] bool is_met(shape_vector const& accepted,
                            bool count_all) const override {
    ++full_passes;
    return inner.is_met(accepted, count_all);
  }
  [[nodiscard]] bool is_met(shape_vector const& accepted, bool count_all,
                            tally_t& tally) const override {
    ++with_tally;
    return inner.is_met(accepted, count_all, tally);
  }
  [[nodiscard]] double achieved_value(shape_vector const& accepted,
                                      bool count_all) const override {
    return inner.achieved_value(accepted, count_all);
  }
  [[nodiscard]] double target_value() const noexcept override {
    return inner.target_value();
  }
};

// The three placement generators ask with the tally after every accepted
// shape and never for a pass over all shapes; the progress callback gets
// the fraction a pass over all shapes gives.
void test_generators_use_the_tally() {
  rvegen::register_all_collision_pairs<D>();
  using input_vector = rvegen::rve_generator_base<D>::input_vector;
  const rvegen::domain<D> box{std::array<D, 3>{1.0, 1.0, 0.0}};
  auto run = [&](rvegen::rve_generator_base<D>& gen) {
    std::mt19937 engine{5};
    rvegen::uniform_real_distribution<D> x{0.0, 1.0, engine};
    rvegen::uniform_real_distribution<D> rad{0.01, 0.02, engine};
    input_vector in;
    in.push_back(std::make_unique<rvegen::circle_input<D>>(x, x, rad));
    spy_fraction spy{0.3, box};
    rvegen::progress_options opts;
    opts.emit_every = 1;
    std::size_t reports = 0;
    double last = 0.0;
    opts.on_progress = [&](rvegen::progress_info i) {
      ++reports;
      REQUIRE(i.volume_fraction >= last);
      last = i.volume_fraction;
    };
    const auto shapes = gen.compute(in, spy, box, opts);
    REQUIRE(shapes.size() > 200);
    REQUIRE(spy.full_passes == 0);
    REQUIRE(spy.with_tally >= reports + 1);
    REQUIRE(reports > 200);
    REQUIRE(last == rvegen::current_volume_fraction<D>(shapes, box));
    REQUIRE(gen.last_report().has_value());
    REQUIRE(gen.last_report()->achieved >= 0.3);

    plain_criterion plain;
    std::mt19937 engine2{6};
    rvegen::uniform_real_distribution<D> x2{0.2, 0.8, engine2};
    rvegen::uniform_real_distribution<D> rad2{0.01, 0.02, engine2};
    input_vector in2;
    in2.push_back(std::make_unique<rvegen::circle_input<D>>(x2, x2, rad2));
    const auto seven = gen.compute(in2, plain, box);
    REQUIRE(plain.asked >= 7);
    std::size_t primaries = 0;
    for (auto const& s : seven) primaries += s->is_primary() ? 1 : 0;
    REQUIRE(seven.size() >= 7 && primaries <= 7);
  };
  rvegen::only_inside_generator<D> a{200000};
  rvegen::random_generator<D> b{200000};
  rvegen::periodic_generator<D> c{200000};
  run(a);
  run(b);
  run(c);
}

} // namespace

int main() {
  test_tally_equals_full_pass();
  test_tally_looks_at_each_shape_once();
  test_generators_use_the_tally();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "termination_tally_test: all checks passed\n";
  return EXIT_SUCCESS;
}
