// Registry coverage for the deterministic generators (weave, tow_weave,
// voronoi_2d, voronoi_3d) and the direction-aware tube input (#84, #73).

#include <cmath>
#include <map>
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
               {"yarn_radius", 0.05}, {"amplitude", 0.06},
               {"warp_phase_name", "warp"}, {"weft_phase_name", "weft"}};
  auto shapes = run_gen(spec, {1.0, 1.0, 0.4});
  REQUIRE(shapes.size() == 6);   // 4 warp + 2 weft
  REQUIRE(shapes.front()->phase_name() == "warp");
  REQUIRE(shapes[3]->phase_name() == "warp");
  REQUIRE(shapes[4]->phase_name() == "weft");
  REQUIRE(shapes.back()->phase_name() == "weft");

  // "pattern" reaches the generator: twill 2/2 needs multiples of 4 yarns,
  // so 4 x 2 is accepted as plain and rejected as twill; 4 x 4 twill builds.
  json twill = {{"type", "weave"}, {"n_warp_yarns", 4}, {"n_weft_yarns", 4},
                {"yarn_radius", 0.05}, {"amplitude", 0.06},
                {"pattern", "twill_2_2"}};
  REQUIRE(run_gen(twill, {1.0, 1.0, 0.4}).size() == 8);
  for (auto const& bad : {json{{"n_weft_yarns", 2}}, json{{"pattern", "basket"}}}) {
    json spec2 = twill;
    spec2.update(bad);
    bool rejected = false;
    try { run_gen(spec2, {1.0, 1.0, 0.4}); }
    catch (std::exception const&) { rejected = true; }
    REQUIRE(rejected);
  }

  // Invalid params surface as exceptions (schema range / ctor check).
  bool threw = false;
  try { run_gen({{"type", "weave"}, {"n_warp_yarns", 0}, {"n_weft_yarns", 2},
                 {"yarn_radius", 0.05}, {"amplitude", 0.0}}, {1, 1, 0.4}); }
  catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
}

void test_tow_weave() {
  json spec = {{"type", "tow_weave"}, {"n_warp_yarns", 2}, {"n_weft_yarns", 2},
               {"yarn_radius", 0.08}, {"amplitude", 0.09},
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

// "periodic" and "backend" reach the tessellators: a periodic run owns every
// voxel of the box (wrap pieces included) with one value per grain.
void test_voronoi_periodic_and_backend_keys() {
  for (int dim : {2, 3}) {
    json spec = {{"type", dim == 2 ? "voronoi_2d" : "voronoi_3d"},
                 {"n_seeds", 8}, {"seed", 2}, {"periodic", true}};
    const std::array<double, 3> box{1.0, 1.0, dim == 2 ? 0.0 : 1.0};
    auto shapes = run_gen(spec, box);
    REQUIRE(shapes.size() > 8);   // 8 grains plus their wrap pieces
    const std::size_t n = 24;
    const auto grid =
        rvegen::sample_voxel_grid(shapes, box, n, n, dim == 2 ? 1 : n);
    std::vector<bool> seen(9, false);
    bool covered = true;
    for (auto v : grid) {
      covered = covered && v >= 1 && v <= 8;
      if (v <= 8) seen[v] = true;
    }
    REQUIRE(covered);
    for (std::size_t g = 1; g <= 8; ++g) REQUIRE(seen[g]);
  }
  json base = {{"type", "voronoi_3d"}, {"n_seeds", 6}, {"seed", 3}};
  for (char const* backend : {"auto", "in_house"}) {
    json spec = base;
    spec["backend"] = backend;
    REQUIRE(run_gen(spec, {1.0, 1.0, 1.0}).size() == 6);
  }
  json bad = base;
  bad["backend"] = "qhull";
  bool threw = false;
  try { run_gen(bad, {1.0, 1.0, 1.0}); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
  bad["backend"] = "voro_pp";
  threw = false;
  try { run_gen(bad, {1.0, 1.0, 1.0}); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw == !rvegen::voro_pp_available());
}

// Grain-size control from JSON (#8): "grain_diameter_mean" makes voronoi_3d
// generate seeds + Laguerre weights for a lognormal diameter distribution.
// Returns the equivalent diameter of every grain (one entry per grain_id).
std::vector<double> grain_diameters(gen_t::shape_vector const& shapes) {
  std::map<std::size_t, double> volume;
  for (auto const& s : shapes) {
    volume[s->info().template get<std::size_t>("grain_id")] = s->volume();
  }
  std::vector<double> d;
  for (auto const& [id, v] : volume) d.push_back(rvegen::equivalent_diameter(v));
  return d;
}

std::vector<double> check_lognormal_grains(
    json const& spec, double L, double mean, double sigma,
    std::size_t min_grains, double mean_tol, double sigma_tol) {
  const auto shapes = run_gen(spec, {L, L, L});
  const auto d = grain_diameters(shapes);
  REQUIRE(d.size() >= min_grains);
  // Grains tile the box.
  constexpr double pi = 3.14159265358979323846;
  double v_sum = 0.0, m = 0.0, m2 = 0.0;
  for (double x : d) {
    v_sum += pi / 6.0 * x * x * x;
    m += std::log(x);
  }
  REQUIRE(std::abs(v_sum - L * L * L) < 1e-6 * L * L * L);
  m /= static_cast<double>(d.size());
  for (double x : d) m2 += (std::log(x) - m) * (std::log(x) - m);
  const double s_ln = std::sqrt(m2 / static_cast<double>(d.size() - 1));
  // ln D ~ N(mu, sigma) with mu = ln(mean) - sigma^2 / 2.
  const double mu = std::log(mean) - sigma * sigma / 2.0;
  std::cout << "grains: N=" << d.size() << " mean lnD=" << m << " (target "
            << mu << ") std lnD=" << s_ln << " (target " << sigma << ")\n";
  REQUIRE(std::abs(m - mu) < mean_tol);
  REQUIRE(std::abs(s_ln - sigma) < sigma_tol);
  return d;
}

void test_voronoi_grain_size_keys() {
  json spec = {{"type", "voronoi_3d"}, {"grain_diameter_mean", 1.0},
               {"grain_diameter_sigma", 0.4}, {"periodic", true},
               {"seed", 1}, {"backend", "in_house"}};
  // A handful of grains with the built-in backend (its cost grows steeply
  // with the count): they must tile the box; 8 draws say nothing about a
  // distribution, so the bounds only reject a plain, unweighted result
  // being returned for a far-off request.
  const std::array<double, 3> small{1.9, 1.9, 1.9};
  const auto a = check_lognormal_grains(spec, 1.9, 1.0, 0.4, 5, 0.5, 0.35);
  if (rvegen::voro_pp_available()) {
    // Acceptance of #8: 100+ grains reproduce the requested distribution.
    json big = spec;
    big["backend"] = "voro_pp";
    (void)check_lognormal_grains(big, 5.5, 1.0, 0.4, 100, 0.08, 0.06);
  }

  // Same seed -> same grains; another seed -> different ones.
  REQUIRE(grain_diameters(run_gen(spec, small)) == a);
  json other = spec;
  other["seed"] = 2;
  REQUIRE(grain_diameters(run_gen(other, small)) != a);

  auto message = [](json const& sp, std::array<double, 3> box) {
    try { (void)run_gen(sp, box); }
    catch (std::exception const& e) { return std::string{e.what()}; }
    return std::string{};
  };
  // The grain count follows from the box: n_seeds / seeds contradict it.
  json both = spec;
  both["n_seeds"] = 10;
  REQUIRE(message(both, small).find("remove 'n_seeds'") != std::string::npos);
  both.erase("n_seeds");
  both["seeds"] = {0.5, 0.5, 0.5};
  REQUIRE(message(both, small).find("remove 'n_seeds'") != std::string::npos);
  // A box too small for two grains of the requested size.
  REQUIRE(message(spec, {1.0, 1.0, 1.0}).find("at least 2") != std::string::npos);
  // A tolerance that is not reached fails loudly instead of returning
  // grains of the wrong size.
  json strict = spec;
  strict["grain_size_tolerance"] = 1e-10;
  strict["grain_size_max_iterations"] = 3;
  REQUIRE(message(strict, small).find("did not reach the tolerance") !=
          std::string::npos);
  // Without the key the plain mode is unchanged (no weights).
  json plain = {{"type", "voronoi_3d"}, {"n_seeds", 6}, {"seed", 3}};
  REQUIRE(run_gen(plain, {1.0, 1.0, 1.0}).size() == 6);
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
                           {"amplitude", 0.06}}, {1.0, 1.0, 0.4});
    rvegen::voxel_writer<double> w{24, 24, 12};
    REQUIRE(count_ids(w, shapes, {1.0, 1.0, 0.4}, nullptr) > 0);
    std::stringstream out;
    w.write(out, shapes, {1.0, 1.0, 0.4});
    REQUIRE(out.str().find("# rvegen voxel grid") != std::string::npos);
  }
  {
    auto shapes = run_gen({{"type", "tow_weave"}, {"n_warp_yarns", 2},
                           {"n_weft_yarns", 2}, {"yarn_radius", 0.08},
                           {"amplitude", 0.09}, {"n_fibres_per_tow", 7}},
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

// A cancel callback that fires mid-build stops the deterministic generators
// (polled per yarn / cell) and yields no shapes.
void test_cancel_during_build() {
  auto run_cancel = [&](json const& spec, std::array<double, 3> box,
                        std::size_t after, std::size_t& polls) {
    auto g = build_gen(spec);
    gen_t::input_vector none;
    rvegen::number_of_inclusions<double> t{1};
    rvegen::progress_options o;
    polls = 0;
    o.cancel = [&] { return ++polls > after; };
    return g->compute(none, t, box, o);
  };
  std::size_t polls = 0;
  auto v2 = run_cancel({{"type", "voronoi_2d"}, {"n_seeds", 300}, {"seed", 1}},
                       {1.0, 1.0, 0.0}, 5, polls);
  REQUIRE(v2.empty() && polls == 6);   // stopped at the 6th poll, not after 300 cells
  // in_house: polled per cell (Voro++ only before its single call).
  auto v3 = run_cancel({{"type", "voronoi_3d"}, {"n_seeds", 40}, {"seed", 1},
                        {"backend", "in_house"}},
                       {1.0, 1.0, 1.0}, 3, polls);
  REQUIRE(v3.empty() && polls == 4);
  auto w = run_cancel({{"type", "weave"}, {"n_warp_yarns", 20}, {"n_weft_yarns", 20},
                       {"yarn_radius", 0.01}, {"amplitude", 0.012}},
                      {1.0, 1.0, 0.4}, 4, polls);
  REQUIRE(w.empty() && polls == 5);
  auto tw = run_cancel({{"type", "tow_weave"}, {"n_warp_yarns", 4}, {"n_weft_yarns", 4},
                        {"yarn_radius", 0.05}, {"amplitude", 0.06},
                        {"n_fibres_per_tow", 3}}, {1.0, 1.0, 0.4}, 10, polls);
  REQUIRE(tw.empty());
  // Not cancelled: full result.
  auto full = run_cancel({{"type", "voronoi_2d"}, {"n_seeds", 20}, {"seed", 1}},
                         {1.0, 1.0, 0.0}, 1000000, polls);
  REQUIRE(full.size() == 20);
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
  test_voronoi_periodic_and_backend_keys();
  test_voronoi_grain_size_keys();
  test_oriented_input();
  test_voronoi_pipeline_seed();
  test_unknown_distribution_name();
  test_voxel_writer_round_trip();
  test_cancel_during_build();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "deterministic_generators_test OK\n";
  return EXIT_SUCCESS;
}
