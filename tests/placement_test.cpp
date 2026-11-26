// #83 placement loop:
//   1. random_generator termination counts every accepted shape (centres
//      outside the box included) -- fails on the old primaries-only count.
//   2. aabb_grid returns exactly the brute-force AABB-overlap set.
//   3. The grid-accelerated generators reproduce the reference linear-scan
//      placement for a fixed seed (random + only_inside, wide radius range
//      so the grid has to coarsen), and periodic output stays collision
//      free including wrap copies.

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include <nlohmann/json.hpp>

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

using namespace rvegen;
using D = double;
const std::array<D, 3> unit_box{1.0, 1.0, 0.0};

// Termination built through the registry so it carries the domain box (the
// direct ctor has no box and always counted everything).
std::unique_ptr<termination_base<D>> make_termination(char const* json) {
  register_all_terminations<D>();
  return build_from_json(termination_registry_t<D>::instance(),
                         nlohmann::json::parse(json), unit_box);
}

void test_random_counts_shapes_outside_the_box() {
  std::mt19937 engine{11};
  // Centres spread over [-2, 3]^2: only ~4% land in the unit box.
  uniform_real_distribution<D> px{-2.0, 3.0, engine};
  uniform_real_distribution<D> py{-2.0, 3.0, engine};
  uniform_real_distribution<D> pr{0.02, 0.03, engine};
  random_generator<D>::input_vector inputs;
  inputs.push_back(std::make_unique<circle_input<D>>(px, py, pr));

  register_all_collision_pairs<D>();
  random_generator<D> gen{1'000'000};

  auto n_term = make_termination(
      R"({"type":"number_of_inclusions","target":20})");
  auto shapes = gen.compute(inputs, *n_term, unit_box);
  // Old code counted only in-box centres, so it ran on to ~20 in-box
  // shapes (hundreds in total). Every accepted shape counts now.
  REQUIRE(shapes.size() == 20);

  // volume_fraction: stop at the first shape that reaches the target, with
  // all shapes (in or out of the box) contributing their area.
  std::mt19937 engine2{12};
  uniform_real_distribution<D> qx{-2.0, 3.0, engine2};
  uniform_real_distribution<D> qy{-2.0, 3.0, engine2};
  uniform_real_distribution<D> qr{0.02, 0.03, engine2};
  random_generator<D>::input_vector inputs2;
  inputs2.push_back(std::make_unique<circle_input<D>>(qx, qy, qr));
  auto v_term = make_termination(
      R"({"type":"volume_fraction","target_fraction":0.05})");
  auto shapes2 = gen.compute(inputs2, *v_term, unit_box);
  D sum = 0, sum_but_last = 0;
  for (std::size_t i = 0; i < shapes2.size(); ++i) {
    sum += shapes2[i]->area();
    if (i + 1 < shapes2.size()) sum_but_last += shapes2[i]->area();
  }
  REQUIRE(sum >= 0.05);
  REQUIRE(sum_but_last < 0.05);

  // only_inside is unchanged: all shapes are inside, count == target.
  std::mt19937 engine3{13};
  uniform_real_distribution<D> sx{0.1, 0.9, engine3};
  uniform_real_distribution<D> sy{0.1, 0.9, engine3};
  uniform_real_distribution<D> sr{0.02, 0.03, engine3};
  only_inside_generator<D>::input_vector inputs3;
  inputs3.push_back(std::make_unique<circle_input<D>>(sx, sy, sr));
  only_inside_generator<D> oi{100'000};
  auto shapes3 = oi.compute(inputs3, *n_term, unit_box);
  REQUIRE(shapes3.size() == 20);
}

void test_grid_matches_brute_force() {
  std::mt19937 rng{5};
  std::uniform_real_distribution<D> pos{-5.0, 5.0};
  std::uniform_real_distribution<D> ext{0.0, 1.0};
  aabb_grid<D> grid;
  std::vector<std::pair<std::array<D, 3>, std::array<D, 3>>> boxes;
  const D inf = std::numeric_limits<D>::infinity();

  auto add = [&](std::array<D, 3> lo, std::array<D, 3> hi) {
    boxes.push_back({lo, hi});
    grid.insert(lo, hi);
  };
  for (int i = 0; i < 400; ++i) {
    std::array<D, 3> lo{pos(rng), pos(rng), i % 2 ? pos(rng) : 0.0};
    std::array<D, 3> hi{lo[0] + ext(rng), lo[1] + ext(rng),
                        lo[2] + (i % 2 ? ext(rng) : 0.0)};
    if (i == 150) hi = {lo[0] + 40.0, lo[1] + 40.0, lo[2] + 40.0};  // huge
    if (i == 200) { lo = {-inf, -inf, -inf}; hi = {inf, inf, inf}; }
    if (i == 250) { lo = {1e300, 0, 0}; hi = {1e300, 1, 1}; }       // absurd
    if (i == 300) { lo = {2.0, 2.0, 2.0}; hi = lo; }                // point
    add(lo, hi);
  }
  for (int q = 0; q < 300; ++q) {
    std::array<D, 3> lo{pos(rng), pos(rng), q % 2 ? pos(rng) : 0.0};
    std::array<D, 3> hi{lo[0] + 3 * ext(rng), lo[1] + 3 * ext(rng),
                        lo[2] + (q % 2 ? 3 * ext(rng) : 0.0)};
    if (q == 7) hi = {lo[0] + 100, lo[1] + 100, lo[2] + 100};
    std::set<std::size_t> expect, got;
    for (std::size_t i = 0; i < boxes.size(); ++i) {
      auto const& [bl, bh] = boxes[i];
      if (bl[0] <= hi[0] && bh[0] >= lo[0] && bl[1] <= hi[1] &&
          bh[1] >= lo[1] && bl[2] <= hi[2] && bh[2] >= lo[2])
        expect.insert(i);
    }
    (void)grid.any_overlap(lo, hi, [&](std::size_t i) {
      REQUIRE(got.insert(i).second);   // each index visited once
      return false;                    // never stop: collect all
    });
    REQUIRE(got == expect);
  }
  // Short-circuit: stops at the first hit.
  int calls = 0;
  REQUIRE(grid.any_overlap(std::array<D, 3>{-10, -10, -10},
                           std::array<D, 3>{10, 10, 10},
                           [&](std::size_t) { ++calls; return true; }));
  REQUIRE(calls == 1);
}

// Reference: the pre-grid linear scan, same candidate stream.
template <typename Gen>
std::vector<std::array<D, 3>> reference_centres(
    typename Gen::input_vector& inputs, std::size_t target,
    std::size_t max_attempts, bool require_inside) {
  std::vector<std::unique_ptr<shape_base<D>>> accepted;
  std::size_t attempts = 0, idx = 0;
  while (accepted.size() < target && attempts < max_attempts) {
    ++attempts;
    auto& input = *inputs[idx];
    idx = (idx + 1) % inputs.size();
    auto cand = input.new_shape();
    cand->make_bounding_box();
    auto const* bb = cand->bounding_box();
    if (require_inside &&
        !aabb_inside(*bb, unit_box[0], unit_box[1], unit_box[2]))
      continue;
    bool collides = false;
    auto const bound =
        collision_dispatcher<D>::instance().bind_lhs(*cand);
    for (auto const& o : accepted) {
      if (aabb_overlap(*bb, *o->bounding_box()) && bound(*cand, *o)) {
        collides = true;
        break;
      }
    }
    if (!collides) accepted.push_back(std::move(cand));
  }
  std::vector<std::array<D, 3>> out;
  for (auto const& s : accepted) out.push_back(s->get_middle_point());
  return out;
}

template <typename Gen>
void compare_with_reference(bool inside, D lo_x, D hi_x) {
  constexpr std::size_t target = 600, attempts = 60'000;
  auto run = [&](bool use_reference) {
    std::mt19937 e{99};
    uniform_real_distribution<D> px{lo_x, hi_x, e};
    uniform_real_distribution<D> py{lo_x, hi_x, e};
    // Wide radius range: forces the grid to coarsen while filling.
    uniform_real_distribution<D> pr{0.002, 0.06, e};
    typename Gen::input_vector inputs;
    inputs.push_back(std::make_unique<circle_input<D>>(px, py, pr));
    if (use_reference)
      return reference_centres<Gen>(inputs, target, attempts, inside);
    Gen gen{attempts};
    termination_base<D>* t = nullptr;
    number_of_inclusions<D> term{target};
    t = &term;
    auto shapes = gen.compute(inputs, *t, unit_box);
    std::vector<std::array<D, 3>> out;
    for (auto const& s : shapes) out.push_back(s->get_middle_point());
    return out;
  };
  auto ref = run(true);
  auto got = run(false);
  REQUIRE(ref.size() > 100);
  REQUIRE(ref == got);
}

void test_generators_match_reference() {
  register_all_collision_pairs<D>();
  compare_with_reference<only_inside_generator<D>>(true, 0.0, 1.0);
  compare_with_reference<random_generator<D>>(false, -0.2, 1.2);
}

void test_periodic_wraps_still_checked() {
  std::mt19937 e{3};
  uniform_real_distribution<D> px{0.0, 1.0, e};
  uniform_real_distribution<D> py{0.0, 1.0, e};
  uniform_real_distribution<D> pr{0.03, 0.08, e};
  periodic_generator<D>::input_vector inputs;
  inputs.push_back(std::make_unique<circle_input<D>>(px, py, pr));
  register_all_collision_pairs<D>();
  periodic_generator<D> gen{30'000};
  auto term = make_termination(
      R"({"type":"number_of_inclusions","target":60})");
  auto shapes = gen.compute(inputs, *term, unit_box);
  REQUIRE(!shapes.empty());
  // No pair among primaries + wraps may collide precisely.
  auto& d = collision_dispatcher<D>::instance();
  for (std::size_t i = 0; i < shapes.size(); ++i)
    for (std::size_t k = i + 1; k < shapes.size(); ++k)
      REQUIRE(!d(*shapes[i], *shapes[k]));
  // Wraps exist (some circle crosses a face) and were accepted.
  std::size_t primaries = 0;
  for (auto const& s : shapes)
    if (is_centre_in_domain(s->get_middle_point(), unit_box)) ++primaries;
  REQUIRE(primaries == 60);
  REQUIRE(shapes.size() > primaries);
}

} // namespace

int main() {
  test_random_counts_shapes_outside_the_box();
  test_grid_matches_brute_force();
  test_generators_match_reference();
  test_periodic_wraps_still_checked();
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "placement_test OK\n";
  return 0;
}
