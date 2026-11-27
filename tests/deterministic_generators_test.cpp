// Registry coverage for the deterministic generators (weave, tow_weave,
// voronoi_2d, voronoi_3d) and the direction-aware tube input (#84, #73).

#include <array>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>

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

using json = nlohmann::json;
using gen_t = rvegen::rve_generator_base<double>;

auto build_gen(json const& spec) {
  return rvegen::build_from_json(rvegen::generator_registry_t<>::instance(),
                                 spec);
}

gen_t::shape_vector run_gen(json const& spec, std::array<double, 3> box) {
  auto g = build_gen(spec);
  gen_t::input_vector no_inputs;
  // Termination must be ignored by deterministic generators; pass one
  // that would otherwise stop immediately.
  rvegen::number_of_inclusions<double> never_used{1};
  return g->compute(no_inputs, never_used, box);
}

void test_registered() {
  auto& r = rvegen::generator_registry_t<>::instance();
  for (auto n : {"weave", "tow_weave", "voronoi_2d", "voronoi_3d"})
    REQUIRE(r.contains(n));
  REQUIRE(rvegen::input_registry_t<>::instance().contains(
      "polyline_tube_oriented_input"));
}

void test_weave() {
  json spec = {{"type", "weave"}, {"n_warp_yarns", 4}, {"n_weft_yarns", 2},
               {"yarn_radius", 0.05}, {"amplitude", 0.04},
               {"warp_phase_name", "warp"}, {"weft_phase_name", "weft"}};
  auto shapes = run_gen(spec, {1.0, 1.0, 0.4});
  REQUIRE(shapes.size() == 6);   // 4 warp + 2 weft
  REQUIRE(shapes.front()->phase_name() == "warp");
  REQUIRE(shapes[3]->phase_name() == "warp");
  REQUIRE(shapes[4]->phase_name() == "weft");
  REQUIRE(shapes.back()->phase_name() == "weft");

  // Invalid params surface as exceptions (schema range / ctor check).
  bool threw = false;
  try { run_gen({{"type", "weave"}, {"n_warp_yarns", 0}, {"n_weft_yarns", 2},
                 {"yarn_radius", 0.05}, {"amplitude", 0.0}}, {1, 1, 0.4}); }
  catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
}

void test_tow_weave() {
  json spec = {{"type", "tow_weave"}, {"n_warp_yarns", 2}, {"n_weft_yarns", 2},
               {"yarn_radius", 0.08}, {"amplitude", 0.05},
               {"n_fibres_per_tow", 7}};
  auto shapes = run_gen(spec, {1.0, 1.0, 0.4});
  REQUIRE(shapes.size() == 4 * 7);
}

void test_voronoi_2d() {
  json spec = {{"type", "voronoi_2d"}, {"n_seeds", 15}, {"seed", 5}};
  auto shapes = run_gen(spec, {2.0, 1.0, 0.0});
  REQUIRE(shapes.size() == 15);
  double area = 0;
  for (auto const& s : shapes) area += s->area();
  REQUIRE(std::abs(area - 2.0) < 1e-9);   // cells tile the domain

  // Same params -> same tessellation; different seed -> different.
  auto again = run_gen(spec, {2.0, 1.0, 0.0});
  REQUIRE(again[3]->area() == shapes[3]->area());
  spec["seed"] = 6;
  REQUIRE(run_gen(spec, {2.0, 1.0, 0.0})[3]->area() != shapes[3]->area());

  // Explicit seeds + phase tag.
  json two = {{"type", "voronoi_2d"}, {"seeds", {0.25, 0.5, 0.75, 0.5}},
              {"phase_name", "grain"}};
  auto halves = run_gen(two, {1.0, 1.0, 0.0});
  REQUIRE(halves.size() == 2);
  REQUIRE(std::abs(halves[0]->area() - 0.5) < 1e-12);
  REQUIRE(halves[0]->phase_name() == "grain");

  bool threw = false;   // 2D generator on a 3D box
  try { run_gen(spec, {1, 1, 1}); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
}

void test_voronoi_3d() {
  json spec = {{"type", "voronoi_3d"}, {"n_seeds", 8}, {"seed", 2}};
  auto shapes = run_gen(spec, {1.0, 1.0, 1.0});
  REQUIRE(shapes.size() == 8);
  double vol = 0;
  for (auto const& s : shapes) vol += s->volume();
  REQUIRE(std::abs(vol - 1.0) < 1e-6);

  bool threw = false;
  try { run_gen(spec, {1, 1, 0}); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
}

void test_oriented_input() {
  std::mt19937 eng{1};
  rvegen::distribution_map_t<double> dists;
  dists.emplace("c", std::make_shared<rvegen::constant_distribution<double>>(0.5));
  auto dir = rvegen::build_from_json(
      rvegen::direction_distribution_registry_t<>::instance(),
      json{{"type", "bingham"}, {"mean_axis_x", 1.0}, {"mean_axis_y", 0.0},
           {"mean_axis_z", 0.0}, {"kappa", 50.0}},
      eng);
  dists.emplace("d", std::make_shared<rvegen::direction_distribution_holder<double>>(
                         std::shared_ptr<rvegen::direction_distribution_base<double>>{
                             std::move(dir)}));
  json spec = {{"type", "polyline_tube_oriented_input"},
               {"position_x_dist", "c"}, {"position_y_dist", "c"},
               {"position_z_dist", "c"}, {"direction_dist", "d"},
               {"length_dist", "c"}, {"radius_dist", "c"},
               {"phase_name", "fibre"}};
  auto input = rvegen::build_from_json(rvegen::input_registry_t<>::instance(),
                                       spec, dists);
  auto shape = input->new_shape();
  REQUIRE(shape->phase_name() == "fibre");

  // A scalar distribution named as direction is rejected.
  spec["direction_dist"] = "c";
  bool threw = false;
  try { (void)rvegen::build_from_json(rvegen::input_registry_t<>::instance(),
                                      spec, dists); }
  catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
}

// The generator seed follows the pipeline (top-level) seed; an explicit
// generator "seed" key overrides it.
std::vector<double> areas_with_pipeline_seed(json const& spec,
                                             std::uint64_t pipeline_seed) {
  auto g = build_gen(spec);
  auto* det =
      dynamic_cast<rvegen::deterministic_generator<double>*>(g.get());
  REQUIRE(det != nullptr);
  if (det) det->set_pipeline_seed(pipeline_seed);
  gen_t::input_vector no_inputs;
  rvegen::number_of_inclusions<double> never_used{1};
  auto shapes = g->compute(no_inputs, never_used, {1.0, 1.0, 0.0});
  std::vector<double> a;
  for (auto const& s : shapes) a.push_back(s->area());
  return a;
}

void test_voronoi_pipeline_seed() {
  json spec = {{"type", "voronoi_2d"}, {"n_seeds", 12}};
  auto s1 = areas_with_pipeline_seed(spec, 1);
  auto s1b = areas_with_pipeline_seed(spec, 1);
  auto s2 = areas_with_pipeline_seed(spec, 2);
  REQUIRE(s1 == s1b);
  REQUIRE(s1 != s2);

  // Explicit generator seed wins over the pipeline seed.
  json fixed = spec;
  fixed["seed"] = 5;
  REQUIRE(areas_with_pipeline_seed(fixed, 1) ==
          areas_with_pipeline_seed(fixed, 2));
  // ... and keeps the sequence of a plain run with that seed.
  REQUIRE(areas_with_pipeline_seed(fixed, 7) ==
          areas_with_pipeline_seed(spec, 5));

  // Same for the 3D generator.
  json s3 = {{"type", "voronoi_3d"}, {"n_seeds", 6}};
  auto vol = [&](std::uint64_t seed) {
    auto g = build_gen(s3);
    dynamic_cast<rvegen::deterministic_generator<double>*>(g.get())
        ->set_pipeline_seed(seed);
    gen_t::input_vector none;
    rvegen::number_of_inclusions<double> t{1};
    auto shapes = g->compute(none, t, {1.0, 1.0, 1.0});
    std::vector<double> v;
    for (auto const& s : shapes) v.push_back(s->volume());
    return v;
  };
  REQUIRE(vol(1) != vol(2));
  REQUIRE(vol(3) == vol(3));
}

void test_unknown_distribution_name() {
  rvegen::distribution_map_t<double> dists;
  dists.emplace("c", std::make_shared<rvegen::constant_distribution<double>>(0.5));
  json spec = {{"type", "circle_input"}, {"pos_x_dist", "c"},
               {"pos_y_dist", "c"}, {"radius_dist", "nope"}};
  std::string msg;
  try { (void)rvegen::build_from_json(rvegen::input_registry_t<>::instance(),
                                      spec, dists); }
  catch (std::exception const& e) { msg = e.what(); }
  REQUIRE(msg.find("radius_dist") != std::string::npos);
  REQUIRE(msg.find("nope") != std::string::npos);
  REQUIRE(msg.find("unknown distribution") != std::string::npos);

  json tube = {{"type", "polyline_tube_input"}, {"start_x_dist", "c"},
               {"start_y_dist", "c"}, {"start_z_dist", "c"},
               {"end_x_dist", "c"}, {"end_y_dist", "c"}, {"end_z_dist", "c"},
               {"radius_dist", "missing_r"}};
  msg.clear();
  try { (void)rvegen::build_from_json(rvegen::input_registry_t<>::instance(),
                                      tube, dists); }
  catch (std::exception const& e) { msg = e.what(); }
  REQUIRE(msg.find("radius_dist") != std::string::npos);
  REQUIRE(msg.find("missing_r") != std::string::npos);
}

// Output of each deterministic generator goes through the voxel writer and
// produces real content.
std::size_t count_ids(rvegen::voxel_writer<double>& w,
                      gen_t::shape_vector const& shapes,
                      std::array<double, 3> box, std::size_t* unassigned) {
  auto const grid = w.sample(shapes, box);
  std::size_t inc = 0, zero = 0;
  for (auto v : grid) (v > 0 ? inc : zero)++;
  if (unassigned) *unassigned = zero;
  return inc;
}

void test_voxel_writer_round_trip() {
  {
    auto shapes = run_gen({{"type", "weave"}, {"n_warp_yarns", 4},
                           {"n_weft_yarns", 2}, {"yarn_radius", 0.05},
                           {"amplitude", 0.04}}, {1.0, 1.0, 0.4});
    rvegen::voxel_writer<double> w{24, 24, 12};
    REQUIRE(count_ids(w, shapes, {1.0, 1.0, 0.4}, nullptr) > 0);
    std::stringstream out;
    w.write(out, shapes, {1.0, 1.0, 0.4});
    REQUIRE(out.str().find("# rvegen voxel grid") != std::string::npos);
  }
  {
    auto shapes = run_gen({{"type", "tow_weave"}, {"n_warp_yarns", 2},
                           {"n_weft_yarns", 2}, {"yarn_radius", 0.08},
                           {"amplitude", 0.05}, {"n_fibres_per_tow", 7}},
                          {1.0, 1.0, 0.4});
    rvegen::voxel_writer<double> w{32, 32, 16};
    REQUIRE(count_ids(w, shapes, {1.0, 1.0, 0.4}, nullptr) > 0);
  }
  {
    auto shapes = run_gen({{"type", "voronoi_2d"}, {"n_seeds", 9}, {"seed", 3}},
                          {1.0, 1.0, 0.0});
    rvegen::voxel_writer<double> w{32, 32, 1};
    std::size_t unassigned = 99;
    REQUIRE(count_ids(w, shapes, {1.0, 1.0, 0.0}, &unassigned) == 32 * 32);
    REQUIRE(unassigned == 0);
  }
  {
    auto shapes = run_gen({{"type", "voronoi_3d"}, {"n_seeds", 6}, {"seed", 3}},
                          {1.0, 1.0, 1.0});
    rvegen::voxel_writer<double> w{12, 12, 12};
    std::size_t unassigned = 99;
    REQUIRE(count_ids(w, shapes, {1.0, 1.0, 1.0}, &unassigned) == 12 * 12 * 12);
    REQUIRE(unassigned == 0);
  }
}

} // namespace

int main() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_direction_distributions<>();
  rvegen::register_all_generators<>();
  rvegen::register_all_inputs<>();
  test_registered();
  test_weave();
  test_tow_weave();
  test_voronoi_2d();
  test_voronoi_3d();
  test_oriented_input();
  test_voronoi_pipeline_seed();
  test_unknown_distribution_name();
  test_voxel_writer_round_trip();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "deterministic_generators_test OK\n";
  return EXIT_SUCCESS;
}
