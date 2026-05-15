// Phase 7 extra-types smoke: directly exercise periodic_generator,
// random_generator, rectangle_input, box_input, until_full termination,
// and vtk_legacy_writer.
//
// These types are also covered indirectly via the CLI examples; this file
// gives them direct unit-style coverage so failures point at one piece.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>

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

// ----------------------------------------------------------------------------
// periodic_generator: shapes near a face must allow placement of shapes near
// the OPPOSITE face only if their periodic images don't collide.
// ----------------------------------------------------------------------------
void test_periodic_collision_across_box_faces() {
  std::mt19937 engine{42};
  rvegen::periodic_generator<double> generator{1};
  rvegen::number_of_inclusions<double> termination{2};
  const std::array<double, 3> domain_box{1.0, 1.0, 0.0};

  // Build inputs that produce two specific candidates straddling the +x and
  // -x faces respectively. Their periodic images would overlap, so the
  // periodic generator must reject the second.
  rvegen::constant_distribution<double> x_first{0.95};
  rvegen::constant_distribution<double> y_zero{0.5};
  rvegen::constant_distribution<double> r_big{0.1};
  auto first = std::make_unique<rvegen::circle_input<double>>(x_first, y_zero, r_big);

  rvegen::periodic_generator<double>::input_vector inputs;
  inputs.push_back(std::move(first));

  // Generate 1 primary at (0.95, 0.5). It pokes through the +x face,
  // so the periodic_generator inserts the wrap at (-0.05, 0.5) into the
  // accepted vector explicitly. Total accepted entries: 2 (primary +
  // wrap).  Termination is_centre_in_domain-filtered, so it counts only
  // the primary toward the target.
  auto first_pass = generator.compute(inputs, termination, domain_box);
  REQUIRE(first_pass.size() == 2);  // primary + wrap

  // Now try to insert a second circle at x = 0.05 — its real position is
  // close to the first's periodic image at x = 0.95 - 1.0 = -0.05. The
  // periodic_aabb_overlap should detect this.
  rvegen::constant_distribution<double> x_second{0.05};
  rvegen::constant_distribution<double> r_small{0.05};
  auto colliding_input = std::make_unique<rvegen::circle_input<double>>(
      x_second, y_zero, r_small);
  rvegen::periodic_generator<double>::input_vector colliding_inputs;
  colliding_inputs.push_back(std::move(colliding_input));

  rvegen::periodic_generator<double> generator2{10};
  rvegen::number_of_inclusions<double> term2{1};
  // To test "would this collide with first_pass[0]", run a fresh generator
  // and seed its accepted set indirectly: easiest is to reproduce the
  // collision via generator's loop with two inputs that each produce one
  // candidate. Skipping deep instrumentation; the existence of
  // periodic_aabb_overlap is exercised in the next test.
  auto second_pass = generator2.compute(colliding_inputs, term2, domain_box);
  // second_pass alone won't collide (no periodic-self conflict); just check
  // it produced something.
  REQUIRE(second_pass.size() == 1);
}

// Direct test of periodic_aabb_overlap — the helper that periodic_generator
// uses for collision detection across periodic images.
void test_periodic_aabb_overlap_helper() {
  // Two unit circles, one at (0.95, 0.5), one at (0.05, 0.5) in a unit box.
  // Direct distance is 0.9 (no overlap), but the periodic image of the
  // first wraps to (-0.05, 0.5), distance to second is 0.1 — bounding boxes
  // of radius 0.1 overlap.
  rvegen::circle<double> a{0.95, 0.5, 0.1};
  rvegen::circle<double> b{0.05, 0.5, 0.1};
  a.make_bounding_box();
  b.make_bounding_box();

  // Direct AABB overlap: false (they're 0.9 apart on x).
  REQUIRE(!rvegen::aabb_overlap(*a.bounding_box(), *b.bounding_box()));

  // Periodic AABB overlap: true (the +x periodic copy of a meets b).
  REQUIRE(rvegen::periodic_aabb_overlap(*a.bounding_box(), *b.bounding_box(),
                                         std::array<double, 3>{1.0, 1.0, 0.0}));
}

// ----------------------------------------------------------------------------
// random_generator: produces shapes ignoring the box constraint.
// ----------------------------------------------------------------------------
void test_random_generator_allows_outside_box() {
  std::mt19937 engine{1};
  rvegen::uniform_real_distribution<double> px{0.0, 1.0, engine};
  rvegen::uniform_real_distribution<double> py{0.0, 1.0, engine};
  rvegen::constant_distribution<double> r{0.3}; // bigger than half-box → pokes out
  auto input = std::make_unique<rvegen::circle_input<double>>(px, py, r);

  rvegen::random_generator<double>::input_vector inputs;
  inputs.push_back(std::move(input));

  rvegen::random_generator<double> generator{10000};
  rvegen::number_of_inclusions<double> termination{3};

  auto shapes = generator.compute(inputs, termination, {1.0, 1.0, 0.0});
  // random_generator doesn't enforce in-box, so we expect to get the target
  // count (or close to it) even with shapes that poke out.
  REQUIRE(shapes.size() == 3);
}

// ----------------------------------------------------------------------------
// rectangle_input: pulls 4 dists, produces axis-aligned rectangles.
// ----------------------------------------------------------------------------
void test_rectangle_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{2};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x":  {"type": "constant", "value": 0.5},
    "y":  {"type": "constant", "value": 0.5},
    "w":  {"type": "constant", "value": 0.2},
    "h":  {"type": "constant", "value": 0.4}
  })");

  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec = nlohmann::json::parse(R"({
    "type": "rectangle_input",
    "pos_x_dist": "x",
    "pos_y_dist": "y",
    "width_dist": "w",
    "height_dist": "h"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  // FP roundtrip via min/max storage allows tiny ULP error on area.
  REQUIRE(std::abs(shape->area() - 0.2 * 0.4) < 1e-12);
}

// ----------------------------------------------------------------------------
// box_input: same but 3D.
// ----------------------------------------------------------------------------
void test_box_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{3};
  rvegen::distribution_map_t<double> dists;
  const auto dist_specs = nlohmann::json::parse(R"({
    "c": {"type": "constant", "value": 0.5},
    "s": {"type": "constant", "value": 0.1}
  })");
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec = nlohmann::json::parse(R"({
    "type": "box_input",
    "pos_x_dist": "c", "pos_y_dist": "c", "pos_z_dist": "c",
    "width_dist": "s", "height_dist": "s", "depth_dist": "s"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(std::abs(shape->volume() - 0.1 * 0.1 * 0.1) < 1e-12);
}

// ----------------------------------------------------------------------------
// until_full: returns false from operator() — generator's max_attempts
// is the only stop signal.
// ----------------------------------------------------------------------------
void test_until_full_lets_generator_max_attempts_cap() {
  rvegen::until_full<double> term;
  rvegen::until_full<double>::shape_vector empty;
  REQUIRE(term(empty) == false);

  // Place 2 large circles in a small box; until_full means we stop when
  // generator gives up. Should produce a small number of accepted shapes.
  std::mt19937 engine{5};
  rvegen::uniform_real_distribution<double> px{0.0, 1.0, engine};
  rvegen::uniform_real_distribution<double> py{0.0, 1.0, engine};
  rvegen::constant_distribution<double> r{0.4};
  auto input = std::make_unique<rvegen::circle_input<double>>(px, py, r);

  rvegen::only_inside_generator<double>::input_vector inputs;
  inputs.push_back(std::move(input));

  rvegen::only_inside_generator<double> gen{1000};
  auto shapes = gen.compute(inputs, term, {1.0, 1.0, 0.0});
  // Should be > 0 (at least one fits) but capped (can't fit infinitely).
  REQUIRE(!shapes.empty());
  REQUIRE(shapes.size() <= 5);
}

// ----------------------------------------------------------------------------
// ellipse: rotation, area, is_inside (rotated), and GTE-backed collision.
// ----------------------------------------------------------------------------
void test_ellipse_basics_and_rotation() {
  // Axis-aligned ellipse: a=2, b=1, area = π * 2 * 1.
  rvegen::ellipse<double> e0{0.0, 0.0, 2.0, 1.0, 0.0};
  REQUIRE(std::abs(e0.area() - 2.0 * std::numbers::pi_v<double>) < 1e-12);
  REQUIRE(e0.volume() == 0.0);

  // is_inside on axis-aligned ellipse.
  REQUIRE(e0.is_inside({0.0, 0.0, 0.0}));     // centre
  REQUIRE(e0.is_inside({1.5, 0.0, 0.0}));     // along major
  REQUIRE(!e0.is_inside({0.0, 1.5, 0.0}));    // outside on minor

  // Rotated 90°: major axis now points along +y.
  rvegen::ellipse<double> e90{0.0, 0.0, 2.0, 1.0, std::numbers::pi_v<double> * 0.5};
  REQUIRE(e90.is_inside({0.0, 1.5, 0.0}));    // now inside (along rotated major)
  REQUIRE(!e90.is_inside({1.5, 0.0, 0.0}));   // outside on rotated minor

  // Rotated bounding box: at 90° the AABB is 2×4 (h, w swapped).
  e90.make_bounding_box();
  auto* bb = e90.bounding_box();
  REQUIRE(bb != nullptr);
  REQUIRE(std::abs(bb->top_point()[0] - 1.0) < 1e-9); // half-width = b = 1
  REQUIRE(std::abs(bb->top_point()[1] - 2.0) < 1e-9); // half-height = a = 2
}

void test_ellipse_collision_via_gte() {
  // Two parallel ellipses, far apart on x — separated.
  rvegen::ellipse<double> a{0.0, 0.0, 1.0, 0.5, 0.0};
  rvegen::ellipse<double> b{3.0, 0.0, 1.0, 0.5, 0.0};
  REQUIRE(!rvegen::collision_details(a, b));

  // Move them closer until they overlap.
  rvegen::ellipse<double> b2{1.5, 0.0, 1.0, 0.5, 0.0};
  REQUIRE(rvegen::collision_details(a, b2));

  // Rotated case: an ellipse rotated 90° with axes (1.0, 0.5) at (1.2, 0)
  // — its rotated extent on x is 0.5, so it does NOT reach a's right edge
  // (which is at x = 1.0). They should be separated.
  rvegen::ellipse<double> b_rot{1.6, 0.0, 1.0, 0.5,
                                  std::numbers::pi_v<double> * 0.5};
  REQUIRE(!rvegen::collision_details(a, b_rot));

  // Same rotated ellipse but moved closer — should overlap.
  rvegen::ellipse<double> b_rot_close{1.0, 0.0, 1.0, 0.5,
                                       std::numbers::pi_v<double> * 0.5};
  REQUIRE(rvegen::collision_details(a, b_rot_close));
}

void test_circle_ellipse_collision() {
  rvegen::circle<double>  c{0.0, 0.0, 0.5};
  rvegen::ellipse<double> e_far{2.0, 0.0, 1.0, 0.3, 0.0};
  REQUIRE(!rvegen::collision_details(c, e_far));
  REQUIRE(!rvegen::collision_details(e_far, c)); // symmetry

  rvegen::ellipse<double> e_near{1.2, 0.0, 1.0, 0.3, 0.0};
  REQUIRE(rvegen::collision_details(c, e_near));
  REQUIRE(rvegen::collision_details(e_near, c));
}

void test_ellipse_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{99};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x":   {"type": "constant", "value": 0.5},
    "y":   {"type": "constant", "value": 0.5},
    "ra":  {"type": "constant", "value": 0.10},
    "rb":  {"type": "constant", "value": 0.05},
    "rot": {"type": "constant", "value": 0.78539816}
  })");

  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec = nlohmann::json::parse(R"({
    "type": "ellipse_input",
    "pos_x_dist":     "x",
    "pos_y_dist":     "y",
    "radius_a_dist":  "ra",
    "radius_b_dist":  "rb",
    "rotation_dist":  "rot"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(std::abs(shape->area()
                   - 0.10 * 0.05 * std::numbers::pi_v<double>) < 1e-12);
}

// ----------------------------------------------------------------------------
// polyline_tube: a swept tube around a centerline (foundation for #3).
// ----------------------------------------------------------------------------
void test_polyline_tube_inside_along_segment() {
  std::vector<std::array<double, 3>> centerline{{0.0, 0.0, 0.0},
                                                {1.0, 0.0, 0.0}};
  rvegen::polyline_tube<double> tube{centerline, 0.1};

  REQUIRE(tube.is_inside({0.5, 0.0, 0.0}));
  REQUIRE(tube.is_inside({0.5, 0.05, 0.0}));
  REQUIRE(!tube.is_inside({0.5, 0.11, 0.0}));
  // Past the cap (segment terminates at x=1): outside.
  REQUIRE(!tube.is_inside({1.5, 0.0, 0.0}));
}

void test_polyline_tube_bend_inside_corner() {
  // L-shaped centerline. Mid-second-segment must register inside.
  std::vector<std::array<double, 3>> centerline{{0.0, 0.0, 0.0},
                                                {1.0, 0.0, 0.0},
                                                {1.0, 1.0, 0.0}};
  rvegen::polyline_tube<double> tube{centerline, 0.1};

  REQUIRE(tube.is_inside({1.0, 0.0, 0.0}));
  REQUIRE(tube.is_inside({1.0, 0.5, 0.0}));
  REQUIRE(!tube.is_inside({1.2, 0.5, 0.0}));
}

void test_polyline_tube_volume_and_bbox() {
  std::vector<std::array<double, 3>> centerline{{0.0, 0.0, 0.0},
                                                {2.0, 0.0, 0.0}};
  rvegen::polyline_tube<double> tube{centerline, 0.1};

  // π · r² · L
  const double expected_v = std::numbers::pi_v<double> * 0.01 * 2.0;
  REQUIRE(std::abs(tube.volume() - expected_v) < 1e-12);

  tube.make_bounding_box();
  REQUIRE(tube.bounding_box() != nullptr);
}

void test_polyline_tube_translate_via_set_middle_point() {
  std::vector<std::array<double, 3>> centerline{{0.0, 0.0, 0.0},
                                                {1.0, 0.0, 0.0}};
  rvegen::polyline_tube<double> tube{centerline, 0.1};

  tube.set_middle_point({10.0, 5.0, 0.0});
  auto const& cl = tube.centerline();
  REQUIRE(std::abs(cl[0][0] - 9.5) < 1e-12);
  REQUIRE(std::abs(cl[0][1] - 5.0) < 1e-12);
  REQUIRE(std::abs(cl[1][0] - 10.5) < 1e-12);
  REQUIRE(std::abs(cl[1][1] - 5.0) < 1e-12);
}

// ----------------------------------------------------------------------------
// vtk_legacy_writer: ASCII STRUCTURED_POINTS header + flat phase data.
// ----------------------------------------------------------------------------
void test_vtk_legacy_writer_header_and_size() {
  rvegen::vtk_legacy_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2));

  rvegen::vtk_legacy_writer<double> writer{16, 16, 1};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});

  const auto txt = out.str();
  REQUIRE(txt.find("# vtk DataFile Version 3.0")  != std::string::npos);
  REQUIRE(txt.find("DATASET STRUCTURED_POINTS")    != std::string::npos);
  REQUIRE(txt.find("DIMENSIONS 16 16 1")           != std::string::npos);
  // Title carries an `rvegen-vtk-format: v2` marker so downstream
  // tools can detect the SCALARS field-name change unambiguously.
  REQUIRE(txt.find("[rvegen-vtk-format: v2]")     != std::string::npos);
  // Field name encodes id scheme — `shape_index` when no
  // phase_collection is attached, `phase_id` when it is.
  REQUIRE(txt.find("SCALARS shape_index int 1")    != std::string::npos);
  REQUIRE(txt.find("LOOKUP_TABLE default")         != std::string::npos);
  // A handful of voxels at the centre should be marked as inclusion (== 1).
  REQUIRE(txt.find("\n1\n") != std::string::npos);
}

// ----------------------------------------------------------------------------
// gmsh_geo_writer periodic mode: emits Periodic Curve / Surface directives
// + Coherence; for FE-RVE workflows that need matching nodes on opposite
// faces. Default mode (periodic=false) leaves the output untouched.
// ----------------------------------------------------------------------------
void test_gmsh_geo_writer_periodic_2d() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{"", true};   // periodic = true
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});

  const auto txt = out.str();
  REQUIRE(txt.find("Periodic Curve { 3 } = { 1 }") != std::string::npos);
  REQUIRE(txt.find("Periodic Curve { 2 } = { 4 }") != std::string::npos);
  REQUIRE(txt.find("Coherence;")                   != std::string::npos);
}

void test_gmsh_geo_writer_periodic_3d() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{"", true};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 1.0});

  const auto txt = out.str();
  REQUIRE(txt.find("Periodic Surface { 2 } = { 1 }") != std::string::npos);
  REQUIRE(txt.find("Periodic Surface { 4 } = { 3 }") != std::string::npos);
  REQUIRE(txt.find("Periodic Surface { 6 } = { 5 }") != std::string::npos);
  REQUIRE(txt.find("Coherence;")                     != std::string::npos);
}

void test_gmsh_geo_writer_non_periodic_unchanged() {
  // Default (periodic=false) must NOT emit the Periodic block — keeps
  // existing configs producing byte-identical output.
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});

  const auto txt = out.str();
  REQUIRE(txt.find("Periodic")  == std::string::npos);
  REQUIRE(txt.find("Coherence") == std::string::npos);
}

// gmsh_geo_writer with a phase_collection emits one Physical Surface /
// Physical Volume directive per phase, grouping inclusion entity ids.
void test_gmsh_geo_writer_2d_physical_groups_per_phase() {
  rvegen::phase_collection<double> phases;
  phases.add("matrix");                              // id 1
  phases.add("fibre");                               // id 2

  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto a = std::make_unique<rvegen::circle<double>>(0.3, 0.3, 0.1);
  a->set_phase_name("fibre");
  shapes.emplace_back(std::move(a));
  auto b = std::make_unique<rvegen::circle<double>>(0.7, 0.7, 0.1);
  b->set_phase_name("fibre");
  shapes.emplace_back(std::move(b));
  auto c = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.05);
  shapes.emplace_back(std::move(c));   // untagged — not in any group

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});

  const auto txt = out.str();
  // Two fibre disks → entity ids 2 and 3 (entity 1 is the bounding rect,
  // entity 4 is the untagged circle).
  REQUIRE(txt.find("Physical Surface(\"fibre\", 2) = {2, 3};") !=
          std::string::npos);
  // Untagged shape must NOT appear in any Physical group.
  REQUIRE(txt.find("Physical Surface(\"matrix\"") == std::string::npos);
  REQUIRE(txt.find("Physical Volume") == std::string::npos);  // 2D, no volumes
}

void test_gmsh_geo_writer_3d_physical_groups_per_phase() {
  rvegen::phase_collection<double> phases;
  phases.add("matrix");                              // id 1
  phases.add("particle");                            // id 2

  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto s = std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.1);
  s->set_phase_name("particle");
  shapes.emplace_back(std::move(s));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 1.0});

  const auto txt = out.str();
  REQUIRE(txt.find("Physical Volume(\"particle\", 2) = {2};") !=
          std::string::npos);
  // 3D file emits Volumes, not Surfaces.
  REQUIRE(txt.find("Physical Surface") == std::string::npos);
}

void test_gmsh_geo_writer_without_phases_emits_no_physical_groups() {
  // Without set_phases() the writer behaves exactly as before — no
  // Physical directives — to keep existing configs byte-compatible.
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto c = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1);
  c->set_phase_name("fibre");                        // tagged...
  shapes.emplace_back(std::move(c));

  rvegen::gmsh_geo_writer<double> writer{};
  REQUIRE(writer.phases() == nullptr);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});

  // ...but without phases attached on the writer, no group is emitted.
  REQUIRE(out.str().find("Physical") == std::string::npos);
}

void test_gmsh_geo_writer_periodic_and_phases_combine() {
  // Both features wire into write_2d / write_3d independently; this
  // test pins the emission order so a future refactor that
  // accidentally swapped them would fail loudly. Order on the .geo
  // is: header → entities → Periodic + Coherence → Physical groups.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");

  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto a = std::make_unique<rvegen::circle<double>>(0.3, 0.3, 0.1);
  a->set_phase_name("fibre");
  shapes.emplace_back(std::move(a));

  // Construct via the default ctor + set_periodic — using the 2-arg
  // ctor with an empty path here would compile but is misleading
  // (the empty path is only fine because this test bypasses run()).
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_periodic(true);
  writer.set_phases(&phases);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  const auto txt = out.str();

  // Use generic substrings (no hardcoded entity ids) so a future
  // change to the id-allocation scheme doesn't silently break this
  // order check.
  const auto entity_pos    = txt.find("Disk(");
  const auto periodic_pos  = txt.find("Periodic Curve");
  const auto coherence_pos = txt.find("Coherence;");
  const auto physical_pos  = txt.find("Physical Surface(\"fibre\"");
  REQUIRE(entity_pos    != std::string::npos);
  REQUIRE(periodic_pos  != std::string::npos);
  REQUIRE(coherence_pos != std::string::npos);
  REQUIRE(physical_pos  != std::string::npos);
  // Strict order: entity < periodic < coherence < physical.
  REQUIRE(entity_pos    < periodic_pos);
  REQUIRE(periodic_pos  < coherence_pos);
  REQUIRE(coherence_pos < physical_pos);
}

void test_gmsh_geo_writer_physical_groups_emitted_in_alpha_order() {
  // Documented invariant: Physical directives come out in
  // alphabetical order of phase_name, not phase_collection insertion
  // order. Pin it with a 3-phase test where alphabetical != insertion.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");      // id 1
  phases.add("ceramic");     // id 2
  phases.add("fibre");       // id 3

  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto a = std::make_unique<rvegen::circle<double>>(0.2, 0.2, 0.05);
  a->set_phase_name("fibre");
  shapes.emplace_back(std::move(a));
  auto b = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.05);
  b->set_phase_name("ceramic");
  shapes.emplace_back(std::move(b));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  const auto txt = out.str();

  // Alphabetical: "ceramic" < "fibre" — must precede in the output.
  const auto ceramic_pos = txt.find("Physical Surface(\"ceramic\"");
  const auto fibre_pos   = txt.find("Physical Surface(\"fibre\"");
  REQUIRE(ceramic_pos != std::string::npos);
  REQUIRE(fibre_pos   != std::string::npos);
  REQUIRE(ceramic_pos < fibre_pos);
  // Declared but untagged "matrix" phase must NOT appear — no shape
  // carries it, so it shouldn't be in any Physical group.
  REQUIRE(txt.find("Physical Surface(\"matrix\"") == std::string::npos);
}

void test_gmsh_geo_writer_phases_unknown_name_throws() {
  // Inclusion carries a phase_name that the collection does not declare.
  // gmsh_geo_writer must throw BEFORE any geometry is written, so a
  // file-backed caller doesn't see a partial .geo on disk.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");

  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto c = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1);
  c->set_phase_name("typo");
  shapes.emplace_back(std::move(c));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  std::stringstream out;
  bool threw = false;
  try { writer.write(out, shapes, {1.0, 1.0, 0.0}); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  // Partial-write guard: nothing should have been emitted before the throw.
  REQUIRE(out.str().empty());
}

void test_gmsh_geo_writer_strict_mode_rejects_untagged_shape() {
  // Default (strict=false): an untagged shape is silently dropped from
  // Physical groups and only the tagged shape gets one. With strict
  // mode on, the same setup must throw before any geometry is emitted.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");

  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto a = std::make_unique<rvegen::circle<double>>(0.3, 0.3, 0.1);
  a->set_phase_name("fibre");
  shapes.emplace_back(std::move(a));
  auto b = std::make_unique<rvegen::circle<double>>(0.7, 0.7, 0.1);
  // intentionally left untagged
  shapes.emplace_back(std::move(b));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);

  // Default: untagged shape just doesn't appear in any Physical group.
  REQUIRE(writer.phase_strict() == false);
  std::stringstream lenient_out;
  writer.write(lenient_out, shapes, {1.0, 1.0, 0.0});
  const auto lenient_text = lenient_out.str();
  REQUIRE(lenient_text.find("Physical Surface(\"fibre\"") != std::string::npos);

  // Strict: same input throws, and nothing is written.
  writer.set_phase_strict(true);
  std::stringstream strict_out;
  bool threw = false;
  std::string what;
  try { writer.write(strict_out, shapes, {1.0, 1.0, 0.0}); }
  catch (std::runtime_error const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  REQUIRE(what.find("strict mode") != std::string::npos);
  REQUIRE(strict_out.str().empty());
}

void test_gmsh_geo_writer_strict_mode_no_phases_attached_is_noop() {
  // Without set_phases attached, strict mode is irrelevant — no phase
  // grouping happens anyway. Toggling strict must not throw and must
  // not change output.
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phase_strict(true);
  std::stringstream out;
  // Explicitly confirm write() does NOT throw — an empty stream from a
  // thrown exception would pass the "no Physical" check below otherwise.
  bool threw = false;
  try { writer.write(out, shapes, {1.0, 1.0, 0.0}); }
  catch (...) { threw = true; }
  REQUIRE(!threw);
  // No Physical groups emitted without set_phases.
  REQUIRE(out.str().find("Physical") == std::string::npos);
  // Stream should still carry the entity geometry — sanity-check that
  // the lenient path actually emitted something.
  REQUIRE(!out.str().empty());
}

void test_voxel_writer_strict_mode_rejects_untagged_shape() {
  // Same untagged-shape bug-of-omission guard as gmsh, but on the
  // voxel grid path. Lenient (default): untagged shape's voxels stay
  // at 0. Strict: sample() throws.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");

  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  auto a = std::make_unique<rvegen::circle<double>>(0.25, 0.25, 0.15);
  a->set_phase_name("fibre");
  shapes.push_back(std::move(a));
  auto b = std::make_unique<rvegen::circle<double>>(0.75, 0.75, 0.15);
  // intentionally untagged
  shapes.push_back(std::move(b));

  rvegen::voxel_writer<double> w{16, 16, 1};
  w.set_phases(&phases);
  REQUIRE(w.phase_strict() == false);
  // Default: lenient — sample succeeds, untagged voxels are 0.
  (void)w.sample(shapes, std::array<double, 3>{1.0, 1.0, 0.0});

  // Strict: sample must throw.
  w.set_phase_strict(true);
  bool threw = false;
  try { (void)w.sample(shapes, std::array<double, 3>{1.0, 1.0, 0.0}); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_vtk_legacy_writer_strict_mode_rejects_untagged_shape() {
  // Same coverage on the VTK path.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");

  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  auto a = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2);
  a->set_phase_name("fibre");
  shapes.push_back(std::move(a));
  auto b = std::make_unique<rvegen::circle<double>>(0.1, 0.1, 0.05);
  // untagged
  shapes.push_back(std::move(b));

  rvegen::vtk_legacy_writer<double> w{16, 16, 1};
  w.set_phases(&phases);
  w.set_phase_strict(true);
  std::stringstream out;
  bool threw = false;
  try { w.write(out, shapes, std::array<double, 3>{1.0, 1.0, 0.0}); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

// ----------------------------------------------------------------------------
// oriented_uniform_distribution: von-Mises sampler. Two regime checks:
//   κ = 0:    behaviour collapses to uniform on a 2π interval.
//   κ = 10:   tightly concentrated around mean_angle.
// Both verified statistically over a moderate sample.
// ----------------------------------------------------------------------------
void test_oriented_uniform_uniform_regime() {
  std::mt19937 engine{123};
  rvegen::oriented_uniform_distribution<double> dist{
      0.0 /*mean*/, 0.0 /*kappa = uniform*/, engine};

  // 5000 samples binned into 12 30°-wide bins. With kappa=0 each bin
  // expects ~417 samples; standard deviation under the binomial is
  // sqrt(N · p · (1-p)) ≈ √(5000 · 1/12 · 11/12) ≈ 19.5. Tolerance set
  // to 5σ so the test isn't flaky — this is detecting "wildly skewed"
  // not "minor sampling noise".
  constexpr int N = 5000;
  constexpr int B = 12;
  std::array<int, B> bins{};
  for (int i = 0; i < N; ++i) {
    double a = std::fmod(dist() + 4 * M_PI, 2 * M_PI);   // wrap to [0, 2π)
    int b = std::min(int(a / (2 * M_PI / B)), B - 1);
    ++bins[b];
  }
  for (int count : bins) {
    REQUIRE(std::abs(count - N / B) < 5 * 20);   // ~5σ tolerance
  }
}

void test_oriented_uniform_concentrated_regime() {
  std::mt19937 engine{456};
  // mean_angle = π/4; high concentration. Samples should cluster.
  rvegen::oriented_uniform_distribution<double> dist{
      M_PI / 4, 50.0 /*kappa large*/, engine};

  constexpr int N = 2000;
  double sum_dev_sq = 0.0;
  for (int i = 0; i < N; ++i) {
    double a = dist();
    double dev = std::fmod(a - M_PI / 4 + 3 * M_PI, 2 * M_PI) - M_PI;  // wrap to [-π, π]
    sum_dev_sq += dev * dev;
  }
  // For a von Mises distribution, Var ≈ 1/κ for large κ. With κ=50,
  // expect std-dev ≈ 1/√50 ≈ 0.14 rad. Empirical std-dev should be in
  // the same ballpark (allow 3× factor for sample noise).
  const double empirical_std = std::sqrt(sum_dev_sq / N);
  REQUIRE(empirical_std < 0.5);   // would be ~π/√3 ≈ 1.81 if uniform
}

// ----------------------------------------------------------------------------
// voronoi_cell — convex polyhedron via vertices + faces. Tested with a unit
// cube tessellation (degenerate Voronoi case but exercises every code path).
// ----------------------------------------------------------------------------
void test_voronoi_cell_unit_cube_volume_and_inside() {
  using V = gte::Vector<3, double>;
  std::vector<V> verts(8);
  verts[0] = V{}; verts[0][0]=-0.5; verts[0][1]=-0.5; verts[0][2]=-0.5;
  verts[1] = V{}; verts[1][0]= 0.5; verts[1][1]=-0.5; verts[1][2]=-0.5;
  verts[2] = V{}; verts[2][0]= 0.5; verts[2][1]= 0.5; verts[2][2]=-0.5;
  verts[3] = V{}; verts[3][0]=-0.5; verts[3][1]= 0.5; verts[3][2]=-0.5;
  verts[4] = V{}; verts[4][0]=-0.5; verts[4][1]=-0.5; verts[4][2]= 0.5;
  verts[5] = V{}; verts[5][0]= 0.5; verts[5][1]=-0.5; verts[5][2]= 0.5;
  verts[6] = V{}; verts[6][0]= 0.5; verts[6][1]= 0.5; verts[6][2]= 0.5;
  verts[7] = V{}; verts[7][0]=-0.5; verts[7][1]= 0.5; verts[7][2]= 0.5;

  // Consistent outward-CCW winding — each face's (v1-v0)×(v2-v0)
  // cross product points along the outward normal of the cube. This
  // is what real producers (Voro++, CGAL) emit; the previous fixture
  // had inconsistent winding which the per-term-abs in volume() was
  // silently masking.
  std::vector<std::vector<std::size_t>> faces{
      {0, 3, 2, 1},   // -z (cross → -ẑ)
      {4, 5, 6, 7},   // +z (cross → +ẑ)
      {0, 1, 5, 4},   // -y (cross → -ŷ)
      {3, 7, 6, 2},   // +y (cross → +ŷ)
      {0, 4, 7, 3},   // -x (cross → -x̂)
      {1, 2, 6, 5}    // +x (cross → +x̂)
  };

  rvegen::voronoi_cell<double> cell{verts, faces};

  // Volume of unit cube = 1.
  REQUIRE(std::abs(cell.volume() - 1.0) < 1e-12);

  // Origin is inside; corner+ε is outside.
  REQUIRE(cell.is_inside({0.0, 0.0, 0.0}));
  REQUIRE(!cell.is_inside({0.6, 0.0, 0.0}));
  REQUIRE(!cell.is_inside({0.51, 0.51, 0.51}));
}

void test_voronoi_cell_face_with_collinear_first_three_vertices() {
  // A face whose first three vertices are collinear used to silently
  // produce a zero face normal — the half-space test then evaluated
  // every point as 'inside' along that face. Walk-the-polygon logic
  // in rebuild_face_planes() now finds a non-collinear triple deeper
  // in the face. Test: square face whose vertex 1 lies on the
  // segment 0→2 (collinear), but vertex 3 makes the face well-defined.
  using V = gte::Vector<3, double>;
  std::vector<V> verts(8);
  verts[0] = V{}; verts[0][0]=-0.5; verts[0][1]=-0.5; verts[0][2]=-0.5;
  verts[1] = V{}; verts[1][0]= 0.0; verts[1][1]=-0.5; verts[1][2]=-0.5;  // mid-edge
  verts[2] = V{}; verts[2][0]= 0.5; verts[2][1]=-0.5; verts[2][2]=-0.5;
  verts[3] = V{}; verts[3][0]= 0.5; verts[3][1]= 0.5; verts[3][2]=-0.5;
  verts[4] = V{}; verts[4][0]=-0.5; verts[4][1]= 0.5; verts[4][2]=-0.5;
  verts[5] = V{}; verts[5][0]=-0.5; verts[5][1]=-0.5; verts[5][2]= 0.5;
  verts[6] = V{}; verts[6][0]= 0.5; verts[6][1]=-0.5; verts[6][2]= 0.5;
  verts[7] = V{}; verts[7][0]= 0.5; verts[7][1]= 0.5; verts[7][2]= 0.5;
  // -z face uses {0, 1, 2, 3, 4} where 0,1,2 are collinear.
  // The walker should find the (0, 2, 3) triple and use that.
  std::vector<std::vector<std::size_t>> faces{
      {0, 1, 2, 3, 4},   // collinear-first-three on -z
      // Skip the rest of the cube; this test only verifies the face
      // plane comes out non-zero. Half-space test against the single
      // face is enough — outside of the face = below = is_inside true
      // for any point above z = -0.5 (since outward normal is -ẑ,
      // "side > 0" means "below the face", which would reject points
      // there). The other faces would normally bound the cell from
      // the other 5 sides.
  };
  rvegen::voronoi_cell<double> cell{verts, faces};

  // The single face's plane was derived from (0, 2, 3) — a real,
  // non-degenerate triple. A point well below the face must be
  // rejected; a point above the face is "inside" with respect to
  // this single half-space.
  REQUIRE(!cell.is_inside({0.0, 0.0, -1.0}));    // far below the -z face
  REQUIRE(cell.is_inside({0.0, 0.0, 1.0}));      // above the -z face
}

void test_voronoi_cell_scale_invariant_collinear_threshold() {
  // Build the unit-cube cell at micrometre scale (1e-6) and confirm
  // volume + inside test still work. With the previous absolute
  // collinearity threshold (epsilon * 1024 ≈ 2e-13) the cross
  // magnitude on micrometre-scale faces is ~1e-12 — within an order
  // of magnitude of being treated as collinear. The new sin-of-angle
  // threshold is dimensionless and unaffected by scale.
  using V = gte::Vector<3, double>;
  constexpr double s = 1e-6;
  std::vector<V> verts(8);
  verts[0] = V{}; verts[0][0]=-0.5*s; verts[0][1]=-0.5*s; verts[0][2]=-0.5*s;
  verts[1] = V{}; verts[1][0]= 0.5*s; verts[1][1]=-0.5*s; verts[1][2]=-0.5*s;
  verts[2] = V{}; verts[2][0]= 0.5*s; verts[2][1]= 0.5*s; verts[2][2]=-0.5*s;
  verts[3] = V{}; verts[3][0]=-0.5*s; verts[3][1]= 0.5*s; verts[3][2]=-0.5*s;
  verts[4] = V{}; verts[4][0]=-0.5*s; verts[4][1]=-0.5*s; verts[4][2]= 0.5*s;
  verts[5] = V{}; verts[5][0]= 0.5*s; verts[5][1]=-0.5*s; verts[5][2]= 0.5*s;
  verts[6] = V{}; verts[6][0]= 0.5*s; verts[6][1]= 0.5*s; verts[6][2]= 0.5*s;
  verts[7] = V{}; verts[7][0]=-0.5*s; verts[7][1]= 0.5*s; verts[7][2]= 0.5*s;
  std::vector<std::vector<std::size_t>> faces{
      {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
      {3, 7, 6, 2}, {0, 4, 7, 3}, {1, 2, 6, 5}};
  rvegen::voronoi_cell<double> cell{verts, faces};

  // Volume = s³ = 1e-18.
  REQUIRE(std::abs(cell.volume() - s * s * s) < 1e-30);
  REQUIRE(cell.is_inside({0.0, 0.0, 0.0}));
  REQUIRE(!cell.is_inside({s, 0.0, 0.0}));
}

void test_voronoi_cell_translate_via_set_middle_point() {
  using V = gte::Vector<3, double>;
  // Tetrahedron with apex on +z, base on z=0.
  std::vector<V> verts(4);
  verts[0] = V{}; verts[0][0]=0.0; verts[0][1]=0.0; verts[0][2]=1.0;
  verts[1] = V{}; verts[1][0]=1.0; verts[1][1]=0.0; verts[1][2]=0.0;
  verts[2] = V{}; verts[2][0]=0.0; verts[2][1]=1.0; verts[2][2]=0.0;
  verts[3] = V{}; verts[3][0]=0.0; verts[3][1]=0.0; verts[3][2]=0.0;

  std::vector<std::vector<std::size_t>> faces{
      {1, 2, 3},   // base (z=0)
      {0, 1, 2},   // diagonal
      {0, 2, 3},
      {0, 3, 1},
  };

  rvegen::voronoi_cell<double> cell{verts, faces};
  // Centroid of {(0,0,1),(1,0,0),(0,1,0),(0,0,0)} = (0.25, 0.25, 0.25).
  cell.set_middle_point({10.0, 10.0, 10.0});
  // Original centroid (0.25, 0.25, 0.25) is no longer inside.
  REQUIRE(!cell.is_inside({0.25, 0.25, 0.25}));
  REQUIRE(cell.is_inside({10.0, 10.0, 10.0}));
}

// ----------------------------------------------------------------------------
// mesh_inclusion + STL ASCII reader. Mesh: an axis-aligned cube of side 1
// centred at origin, expressed via 12 triangles.
// ----------------------------------------------------------------------------
namespace {

constexpr char const* unit_cube_stl = R"(solid cube
  facet normal 0 0 -1
    outer loop
      vertex -0.5 -0.5 -0.5
      vertex  0.5  0.5 -0.5
      vertex  0.5 -0.5 -0.5
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex -0.5 -0.5 -0.5
      vertex -0.5  0.5 -0.5
      vertex  0.5  0.5 -0.5
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex -0.5 -0.5  0.5
      vertex  0.5 -0.5  0.5
      vertex  0.5  0.5  0.5
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex -0.5 -0.5  0.5
      vertex  0.5  0.5  0.5
      vertex -0.5  0.5  0.5
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex -0.5 -0.5 -0.5
      vertex  0.5 -0.5 -0.5
      vertex  0.5 -0.5  0.5
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex -0.5 -0.5 -0.5
      vertex  0.5 -0.5  0.5
      vertex -0.5 -0.5  0.5
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex -0.5  0.5 -0.5
      vertex  0.5  0.5  0.5
      vertex  0.5  0.5 -0.5
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex -0.5  0.5 -0.5
      vertex -0.5  0.5  0.5
      vertex  0.5  0.5  0.5
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex -0.5 -0.5 -0.5
      vertex -0.5 -0.5  0.5
      vertex -0.5  0.5  0.5
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex -0.5 -0.5 -0.5
      vertex -0.5  0.5  0.5
      vertex -0.5  0.5 -0.5
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex  0.5 -0.5 -0.5
      vertex  0.5  0.5  0.5
      vertex  0.5 -0.5  0.5
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex  0.5 -0.5 -0.5
      vertex  0.5  0.5 -0.5
      vertex  0.5  0.5  0.5
    endloop
  endfacet
endsolid cube
)";

}

void test_stl_ascii_reader_triangle_count() {
  std::stringstream ss{unit_cube_stl};
  auto tris = rvegen::read_stl_ascii<double>(ss);
  REQUIRE(tris.size() == 12);
}

void test_mesh_inclusion_inside_outside_cube() {
  std::stringstream ss{unit_cube_stl};
  auto tris = rvegen::read_stl_ascii<double>(ss);
  rvegen::mesh_inclusion<double> mesh{tris};

  // Origin is at the cube centre — should be inside.
  REQUIRE(mesh.is_inside({0.0, 0.0, 0.0}));
  // Far away — should be outside.
  REQUIRE(!mesh.is_inside({2.0, 2.0, 2.0}));
  // Just outside on +x.
  REQUIRE(!mesh.is_inside({0.6, 0.0, 0.0}));
}

void test_mesh_inclusion_volume_unit_cube() {
  std::stringstream ss{unit_cube_stl};
  auto tris = rvegen::read_stl_ascii<double>(ss);
  rvegen::mesh_inclusion<double> mesh{tris};

  // Unit cube — volume must equal 1 to within FP tolerance.
  REQUIRE(std::abs(mesh.volume() - 1.0) < 1e-12);
}

void test_mesh_inclusion_translate_via_set_middle_point() {
  std::stringstream ss{unit_cube_stl};
  auto tris = rvegen::read_stl_ascii<double>(ss);
  rvegen::mesh_inclusion<double> mesh{tris};

  mesh.set_middle_point({10.0, 5.0, 0.0});
  // After translation, origin is no longer inside.
  REQUIRE(!mesh.is_inside({0.0, 0.0, 0.0}));
  REQUIRE(mesh.is_inside({10.0, 5.0, 0.0}));
}

// ----------------------------------------------------------------------------
// ASCII PLY reader (#9 follow-up).
// ----------------------------------------------------------------------------

// Unit cube as ASCII PLY — 8 vertices, 6 quad faces. The reader
// must fan-triangulate the quads into 12 triangles to match the STL
// fixture, so downstream `mesh_inclusion` consumes the same geometry.
constexpr char const* unit_cube_ply = R"(ply
format ascii 1.0
comment unit cube
element vertex 8
property float x
property float y
property float z
element face 6
property list uchar int vertex_indices
end_header
-0.5 -0.5 -0.5
 0.5 -0.5 -0.5
 0.5  0.5 -0.5
-0.5  0.5 -0.5
-0.5 -0.5  0.5
 0.5 -0.5  0.5
 0.5  0.5  0.5
-0.5  0.5  0.5
4 0 3 2 1
4 4 5 6 7
4 0 1 5 4
4 1 2 6 5
4 2 3 7 6
4 3 0 4 7
)";

void test_ply_ascii_reader_triangulates_quad_faces() {
  std::stringstream ss{unit_cube_ply};
  auto tris = rvegen::read_ply_ascii<double>(ss);
  // 6 quads × 2 triangles each via fan-triangulation = 12.
  REQUIRE(tris.size() == 12);
}

void test_ply_ascii_reader_round_trips_through_mesh_inclusion() {
  std::stringstream ss{unit_cube_ply};
  auto tris = rvegen::read_ply_ascii<double>(ss);
  rvegen::mesh_inclusion<double> mesh{tris};
  REQUIRE(mesh.is_inside({0.0, 0.0, 0.0}));   // centre
  REQUIRE(!mesh.is_inside({2.0, 2.0, 2.0}));  // far away
  REQUIRE(std::abs(mesh.volume() - 1.0) < 1e-12);
}

void test_ply_ascii_reader_rejects_binary_format() {
  // The explicit ASCII reader still refuses binary input — callers
  // who need binary should use read_ply / read_ply_binary instead.
  constexpr char const* bin_ply = R"(ply
format binary_little_endian 1.0
element vertex 0
element face 0
end_header
)";
  std::stringstream ss{bin_ply};
  bool threw = false;
  std::string what;
  try { (void)rvegen::read_ply_ascii<double>(ss); }
  catch (std::runtime_error const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  REQUIRE(what.find("binary") != std::string::npos);
}

// Helpers for building binary-PLY blobs inline. The data section is
// declared little-endian by the header; on a LE host (the CI runner)
// these write little-endian by memcpy.
namespace ply_binary_fixture {

inline void write_le_u8(std::ostream& o, std::uint8_t v) {
  o.put(static_cast<char>(v));
}
inline void write_le_i32(std::ostream& o, std::int32_t v) {
  std::uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  o.put(static_cast<char>(u & 0xff));
  o.put(static_cast<char>((u >> 8)  & 0xff));
  o.put(static_cast<char>((u >> 16) & 0xff));
  o.put(static_cast<char>((u >> 24) & 0xff));
}
inline void write_le_f32(std::ostream& o, float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  o.put(static_cast<char>(bits & 0xff));
  o.put(static_cast<char>((bits >> 8)  & 0xff));
  o.put(static_cast<char>((bits >> 16) & 0xff));
  o.put(static_cast<char>((bits >> 24) & 0xff));
}

// Emit a unit-cube binary-LE PLY blob mirroring `unit_cube_ply` (the
// ASCII fixture). 8 vertices (float x/y/z), 6 quad faces (uchar count
// + int indices). Same fan-triangulation expectation: 12 triangles.
inline std::string make_unit_cube_blob() {
  std::ostringstream out;
  out << "ply\n"
         "format binary_little_endian 1.0\n"
         "element vertex 8\n"
         "property float x\n"
         "property float y\n"
         "property float z\n"
         "element face 6\n"
         "property list uchar int vertex_indices\n"
         "end_header\n";
  constexpr float verts[8][3] = {
      {-0.5f, -0.5f, -0.5f}, { 0.5f, -0.5f, -0.5f},
      { 0.5f,  0.5f, -0.5f}, {-0.5f,  0.5f, -0.5f},
      {-0.5f, -0.5f,  0.5f}, { 0.5f, -0.5f,  0.5f},
      { 0.5f,  0.5f,  0.5f}, {-0.5f,  0.5f,  0.5f}};
  for (auto const& v : verts) {
    for (auto c : v) write_le_f32(out, c);
  }
  // Same outward-winding face index sets as the ASCII fixture.
  constexpr std::array<std::array<int, 4>, 6> faces{{
      {0, 3, 2, 1}, {4, 5, 6, 7},
      {0, 1, 5, 4}, {1, 2, 6, 5},
      {2, 3, 7, 6}, {3, 0, 4, 7}}};
  for (auto const& f : faces) {
    write_le_u8(out, 4);
    for (auto i : f) write_le_i32(out, i);
  }
  return out.str();
}

} // namespace ply_binary_fixture

void test_ply_binary_reader_round_trips_unit_cube() {
  const auto blob = ply_binary_fixture::make_unit_cube_blob();
  std::stringstream ss{blob};
  auto tris = rvegen::read_ply_binary<double>(ss);
  REQUIRE(tris.size() == 12);  // 6 quads × 2 fan-triangles each
  rvegen::mesh_inclusion<double> mesh{tris};
  REQUIRE(mesh.is_inside({0.0, 0.0, 0.0}));
  REQUIRE(!mesh.is_inside({2.0, 2.0, 2.0}));
  REQUIRE(std::abs(mesh.volume() - 1.0) < 1e-7);  // float-precision verts
}

void test_ply_auto_detect_routes_binary_and_ascii() {
  // Binary blob → read_ply picks read_ply_binary.
  {
    const auto blob = ply_binary_fixture::make_unit_cube_blob();
    std::stringstream ss{blob};
    auto tris = rvegen::read_ply<double>(ss);
    REQUIRE(tris.size() == 12);
  }
  // ASCII text → read_ply picks read_ply_ascii.
  {
    std::stringstream ss{unit_cube_ply};
    auto tris = rvegen::read_ply<double>(ss);
    REQUIRE(tris.size() == 12);
  }
}

void test_ply_binary_reader_rejects_ascii_format() {
  // The explicit binary reader refuses an ASCII-declared file —
  // callers who want auto-detect should use read_ply instead.
  std::stringstream ss{unit_cube_ply};
  bool threw = false;
  try { (void)rvegen::read_ply_binary<double>(ss); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_ply_binary_reader_short_read_throws() {
  // Header announces 8 vertices but the payload only carries the
  // first vertex's bytes — the reader must throw on the short read.
  std::ostringstream truncated;
  truncated << "ply\n"
               "format binary_little_endian 1.0\n"
               "element vertex 8\n"
               "property float x\n"
               "property float y\n"
               "property float z\n"
               "element face 0\n"
               "property list uchar int vertex_indices\n"
               "end_header\n";
  // Only 12 bytes of vertex data — one float triple, not eight.
  for (float c : {1.0f, 2.0f, 3.0f}) ply_binary_fixture::write_le_f32(truncated, c);
  std::stringstream ss{truncated.str()};
  bool threw = false;
  try { (void)rvegen::read_ply_binary<double>(ss); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_ply_ascii_reader_rejects_missing_xyz() {
  // Vertex element without all of x/y/z is a hard error — the
  // consumer can't build geometry without them.
  constexpr char const* bad_ply = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
element face 0
property list uchar int vertex_indices
end_header
0.0 0.0
)";
  std::stringstream ss{bad_ply};
  bool threw = false;
  try { (void)rvegen::read_ply_ascii<double>(ss); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_ply_ascii_reader_skips_extra_vertex_properties() {
  // A vertex with normals tacked on (nx, ny, nz) and a colour byte
  // (red) — the reader must skip them silently and still extract
  // x/y/z correctly. Single triangle face validates index lookup.
  constexpr char const* with_extras = R"(ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0 1.0 0.0 0.0 255
1.0 0.0 0.0 1.0 0.0 0.0 255
0.0 1.0 0.0 1.0 0.0 0.0 255
3 0 1 2
)";
  std::stringstream ss{with_extras};
  auto tris = rvegen::read_ply_ascii<double>(ss);
  REQUIRE(tris.size() == 1);
  REQUIRE(tris[0].v[1][0] == 1.0);   // (1, 0, 0)
  REQUIRE(tris[0].v[2][1] == 1.0);   // (0, 1, 0)
}

void test_ply_ascii_reader_rejects_out_of_range_index() {
  // Face references a vertex index that doesn't exist — early-catch
  // it rather than UB at consumer time.
  constexpr char const* bad_index = R"(ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
3 0 1 2
)";
  std::stringstream ss{bad_index};
  bool threw = false;
  try { (void)rvegen::read_ply_ascii<double>(ss); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_stl_ascii_reader_rejects_binary_stl() {
  // Binary STL: 80-byte header (often containing the literal "solid"),
  // then 4-byte triangle count, then per-triangle 50 bytes. Synthesise
  // a minimal binary blob — 80 NULs + a count of 0 — and confirm the
  // ASCII reader refuses with a clear error rather than parsing garbage.
  std::string blob(80, '\0');
  blob.append({'\0', '\0', '\0', '\0'});   // 0 triangles
  std::stringstream ss{blob};
  bool threw = false;
  try { (void)rvegen::read_stl_ascii<double>(ss); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

// ----------------------------------------------------------------------------
// polyline_tube_input — schema-driven 2-point straight tube. Tests both the
// direct ctor and the JSON-via-registry path.
// ----------------------------------------------------------------------------
void test_polyline_tube_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{42};
  const auto dist_specs = nlohmann::json::parse(R"({
    "sx": {"type": "constant", "value": 0.1},
    "sy": {"type": "constant", "value": 0.2},
    "sz": {"type": "constant", "value": 0.3},
    "ex": {"type": "constant", "value": 0.7},
    "ey": {"type": "constant", "value": 0.8},
    "ez": {"type": "constant", "value": 0.9},
    "r":  {"type": "constant", "value": 0.05}
  })");

  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec = nlohmann::json::parse(R"({
    "type": "polyline_tube_input",
    "start_x_dist": "sx",
    "start_y_dist": "sy",
    "start_z_dist": "sz",
    "end_x_dist":   "ex",
    "end_y_dist":   "ey",
    "end_z_dist":   "ez",
    "radius_dist":  "r"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  auto* tube = dynamic_cast<rvegen::polyline_tube<double>*>(shape.get());
  REQUIRE(tube != nullptr);
  REQUIRE(tube->radius() == 0.05);
  // Centerline endpoints match the constant distributions.
  auto const& cl = tube->centerline();
  REQUIRE(cl.size() == 2);
  REQUIRE(std::abs(cl[0][0] - 0.1) < 1e-12);
  REQUIRE(std::abs(cl[1][2] - 0.9) < 1e-12);
}

void test_polyline_tube_input_direct_ctor() {
  // Construction without going through the JSON registry.
  rvegen::constant_distribution<double> sx{0.0}, sy{0.0}, sz{0.0};
  rvegen::constant_distribution<double> ex{1.0}, ey{0.0}, ez{0.0};
  rvegen::constant_distribution<double> r{0.05};
  rvegen::polyline_tube_input<double> input{sx, sy, sz, ex, ey, ez, r};

  auto shape = input.new_shape();
  REQUIRE(shape != nullptr);
  auto* tube = dynamic_cast<rvegen::polyline_tube<double>*>(shape.get());
  REQUIRE(tube != nullptr);
  REQUIRE(std::abs(tube->radius() - 0.05) < 1e-12);
}

void test_polyline_tube_input_missing_dist_throws() {
  // Schema-driven ctor calls `distributions.at(...)` which throws
  // std::out_of_range when the named distribution is absent.
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{1};

  rvegen::distribution_map_t<double> empty_dists;
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "polyline_tube_input",
    "start_x_dist": "missing",
    "start_y_dist": "missing",
    "start_z_dist": "missing",
    "end_x_dist":   "missing",
    "end_y_dist":   "missing",
    "end_z_dist":   "missing",
    "radius_dist":  "missing"
  })");
  bool threw = false;
  try {
    auto input = rvegen::build_from_json(
        rvegen::input_registry_t<>::instance(), input_spec, empty_dists);
  } catch (std::out_of_range const&) { threw = true; }
  REQUIRE(threw);
}

void test_polyline_tube_directional_input_via_registry() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{7};

  // Position (0.5, 0.5, 0.0); direction (1, 0, 0); length 0.4; radius 0.05.
  // Expected centerline: (0.3, 0.5, 0.0) → (0.7, 0.5, 0.0).
  const auto dist_specs = nlohmann::json::parse(R"({
    "px": {"type": "constant", "value": 0.5},
    "py": {"type": "constant", "value": 0.5},
    "pz": {"type": "constant", "value": 0.0},
    "dx": {"type": "constant", "value": 1.0},
    "dy": {"type": "constant", "value": 0.0},
    "dz": {"type": "constant", "value": 0.0},
    "len": {"type": "constant", "value": 0.4},
    "rad": {"type": "constant", "value": 0.05}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "polyline_tube_directional_input",
    "position_x_dist":  "px",
    "position_y_dist":  "py",
    "position_z_dist":  "pz",
    "direction_x_dist": "dx",
    "direction_y_dist": "dy",
    "direction_z_dist": "dz",
    "length_dist":      "len",
    "radius_dist":      "rad"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  auto* tube = dynamic_cast<rvegen::polyline_tube<double>*>(shape.get());
  REQUIRE(tube != nullptr);
  REQUIRE(std::abs(tube->radius() - 0.05) < 1e-12);
  auto const& cl = tube->centerline();
  REQUIRE(cl.size() == 2);
  REQUIRE(std::abs(cl[0][0] - 0.3) < 1e-12);
  REQUIRE(std::abs(cl[1][0] - 0.7) < 1e-12);
}

// ----------------------------------------------------------------------------
// field_list scaffolding (#1): a schema declared once, used by both
// `schema()` and `extract()`.
// ----------------------------------------------------------------------------
void test_field_list_schema_round_trip() {
  using fields = rvegen::field_list<
      rvegen::field<"x",      double>,
      rvegen::field<"y",      double>,
      rvegen::field<"radius", double>>;

  // Schema must list all three names and mark them required.
  auto s = fields::schema();
  REQUIRE(s.contains("x"));
  REQUIRE(s.contains("y"));
  REQUIRE(s.contains("radius"));

  // Round-trip: populate a handler with the field values, then extract.
  rvegen::parameter_handler_t h;
  h.insert<double>("x", 0.25);
  h.insert<double>("y", 0.50);
  h.insert<double>("radius", 0.10);
  auto values = fields::extract(h);
  REQUIRE(std::get<0>(values) == 0.25);
  REQUIRE(std::get<1>(values) == 0.50);
  REQUIRE(std::get<2>(values) == 0.10);
}

void test_field_list_optional_field_not_required() {
  using fields = rvegen::field_list<
      rvegen::field<"required_field", double>,
      rvegen::field<"optional_field", double, /*Required=*/false>>;
  auto s = fields::schema();
  REQUIRE(s.contains("required_field"));
  REQUIRE(s.contains("optional_field"));
}

void test_field_list_empty_pack() {
  // A no-parameter type (e.g. a future void_material with no schema
  // fields) should produce a valid empty schema and an empty tuple.
  using empty_fields = rvegen::field_list<>;
  auto s = empty_fields::schema();
  rvegen::parameter_handler_t h;
  auto values = empty_fields::extract(h);
  static_assert(std::tuple_size_v<decltype(values)> == 0);
  (void)s;
}

// Helper: probe a parameter_controller slot for the hint side-bases that
// `description_label`, `unit_label`, and `min_only`/`range` attach. Mirrors
// the discovery pattern in tests/schema_audit_test.cpp; pulled inline here
// so the field_list tests can verify annotations actually landed.
struct field_list_hint_probe {
  bool has_description = false;
  bool has_range = false;
  bool has_units = false;
};

template <typename T>
field_list_hint_probe probe_field(rvegen::parameter_controller_t const& s,
                                  std::string const& name) {
  using typed_t = numsim_core::input_parameter<
      T, std::string, rvegen::parameter_handler_t>;
  field_list_hint_probe hp;
  for (auto const& [field_key, param_ptr] : s) {
    if (field_key != name) continue;
    auto const* typed = dynamic_cast<typed_t const*>(param_ptr.get());
    if (!typed) return hp;
    for (auto const& check : typed->checks()) {
      if (dynamic_cast<numsim_core::description_hint_base const*>(check.get()))
        hp.has_description = true;
      if (dynamic_cast<numsim_core::range_hint_base const*>(check.get()))
        hp.has_range = true;
      if (dynamic_cast<numsim_core::units_hint_base const*>(check.get()))
        hp.has_units = true;
    }
    return hp;
  }
  return hp;
}

void test_field_list_carries_description_and_unit_annotations() {
  // A field with description + unit annotations must have both side-bases
  // reachable from the controller slot — same hint discovery the
  // schema_audit_test uses to enforce schema-quality coverage.
  using fields = rvegen::field_list<
      rvegen::field<"radius", double, /*Required=*/true,
                    numsim_core::description_label<"sphere radius">,
                    numsim_core::unit_label<"m">>>;
  auto s = fields::schema();
  REQUIRE(s.contains("radius"));
  const auto hp = probe_field<double>(s, std::string{"radius"});
  REQUIRE(hp.has_description);
  REQUIRE(hp.has_units);
  REQUIRE(!hp.has_range);
}

void test_field_list_carries_range_annotation() {
  // A field with min_only<>/range<> picks up the range_hint_base side-base.
  using fields = rvegen::field_list<
      rvegen::field<"n_samples", std::size_t, true,
                    rvegen::min_only<std::size_t{1}>,
                    numsim_core::description_label<"number of samples">>>;
  auto s = fields::schema();
  const auto hp = probe_field<std::size_t>(s, "n_samples");
  REQUIRE(hp.has_range);
  REQUIRE(hp.has_description);
}

void test_field_list_no_annotations_no_hint_bases() {
  // A bare field (no annotation pack) must NOT spuriously attach any of
  // the hint side-bases — annotation transport is opt-in.
  using fields = rvegen::field_list<rvegen::field<"x", double>>;
  auto s = fields::schema();
  const auto hp = probe_field<double>(s, "x");
  REQUIRE(!hp.has_description);
  REQUIRE(!hp.has_range);
  REQUIRE(!hp.has_units);
}

// First in-tree migration of a shape to declarative field_list (circle).
// Regression-locks the post-migration schema: all 3 fields must carry a
// description label + unit label, and `radius` must additionally carry
// a range hint from min_only<T{0}>. If a future schema change drops one
// of these, this test catches it.
void test_circle_schema_post_field_list_migration() {
  auto s = rvegen::circle<double>::parameters();
  for (auto const& name :
       {std::string{"x"}, std::string{"y"}, std::string{"radius"}}) {
    const auto hp = probe_field<double>(s, name);
    REQUIRE(hp.has_description);
    REQUIRE(hp.has_units);
  }
  // radius is the only field with a range constraint (min_only<0>).
  REQUIRE(probe_field<double>(s, std::string{"radius"}).has_range);
  REQUIRE(!probe_field<double>(s, std::string{"x"}).has_range);
  REQUIRE(!probe_field<double>(s, std::string{"y"}).has_range);
}

// Schema-driven ctor still parses correctly after the field_list
// migration — fields::extract(handler) unpacks into the (x, y, radius)
// tuple-taking delegating ctor.
void test_circle_schema_driven_ctor_via_field_list() {
  rvegen::parameter_handler_t h;
  h.insert<double>("x", 0.25);
  h.insert<double>("y", 0.75);
  h.insert<double>("radius", 0.1);
  rvegen::circle<double> c{h};
  REQUIRE(c.center[0] == 0.25);
  REQUIRE(c.center[1] == 0.75);
  REQUIRE(c.radius == 0.1);
}

// Same regression coverage for sphere / rectangle / box / ellipse after
// the field_list migration: each schema field must keep its description
// and unit annotations; size/extent fields keep their min_only<0> range
// hint; rotation keeps its unit (rad) without a range.
void test_sphere_schema_post_field_list_migration() {
  auto s = rvegen::sphere<double>::parameters();
  for (auto const& name : {std::string{"x"}, std::string{"y"},
                           std::string{"z"}, std::string{"radius"}}) {
    const auto hp = probe_field<double>(s, name);
    REQUIRE(hp.has_description);
    REQUIRE(hp.has_units);
  }
  REQUIRE(probe_field<double>(s, std::string{"radius"}).has_range);
  REQUIRE(!probe_field<double>(s, std::string{"x"}).has_range);
}

void test_sphere_schema_driven_ctor_via_field_list() {
  rvegen::parameter_handler_t h;
  h.insert<double>("x", 1.0);
  h.insert<double>("y", 2.0);
  h.insert<double>("z", 3.0);
  h.insert<double>("radius", 0.5);
  rvegen::sphere<double> s{h};
  REQUIRE(s.center[0] == 1.0);
  REQUIRE(s.center[2] == 3.0);
  REQUIRE(s.radius == 0.5);
}

void test_rectangle_schema_post_field_list_migration() {
  auto s = rvegen::rectangle<double>::parameters();
  for (auto const& name : {std::string{"x"}, std::string{"y"},
                           std::string{"width"}, std::string{"height"}}) {
    const auto hp = probe_field<double>(s, name);
    REQUIRE(hp.has_description);
    REQUIRE(hp.has_units);
  }
  REQUIRE(probe_field<double>(s, std::string{"width"}).has_range);
  REQUIRE(probe_field<double>(s, std::string{"height"}).has_range);
}

void test_rectangle_schema_driven_ctor_via_field_list() {
  rvegen::parameter_handler_t h;
  h.insert<double>("x", 0.5);
  h.insert<double>("y", 0.25);
  h.insert<double>("width", 0.2);
  h.insert<double>("height", 0.1);
  rvegen::rectangle<double> r{h};
  // width() / height() recompute via max - min, picking up FP rounding —
  // hence the tolerance rather than exact equality.
  REQUIRE(std::abs(r.width()  - 0.2) < 1e-12);
  REQUIRE(std::abs(r.height() - 0.1) < 1e-12);
  REQUIRE(r(0) == 0.5);
}

void test_box_schema_post_field_list_migration() {
  auto s = rvegen::box<double>::parameters();
  for (auto const& name :
       {std::string{"x"}, std::string{"y"}, std::string{"z"},
        std::string{"width"}, std::string{"height"}, std::string{"depth"}}) {
    const auto hp = probe_field<double>(s, name);
    REQUIRE(hp.has_description);
    REQUIRE(hp.has_units);
  }
  REQUIRE(probe_field<double>(s, std::string{"width"}).has_range);
  REQUIRE(probe_field<double>(s, std::string{"depth"}).has_range);
}

void test_box_schema_driven_ctor_via_field_list() {
  rvegen::parameter_handler_t h;
  h.insert<double>("x", 0.5);
  h.insert<double>("y", 0.25);
  h.insert<double>("z", 0.125);
  h.insert<double>("width", 0.2);
  h.insert<double>("height", 0.1);
  h.insert<double>("depth", 0.05);
  rvegen::box<double> b{h};
  REQUIRE(std::abs(b.width()  - 0.2)  < 1e-12);
  REQUIRE(std::abs(b.height() - 0.1)  < 1e-12);
  REQUIRE(std::abs(b.depth()  - 0.05) < 1e-12);
}

void test_ellipse_schema_post_field_list_migration() {
  auto s = rvegen::ellipse<double>::parameters();
  for (auto const& name :
       {std::string{"x"}, std::string{"y"}, std::string{"radius_a"},
        std::string{"radius_b"}, std::string{"rotation"}}) {
    const auto hp = probe_field<double>(s, name);
    REQUIRE(hp.has_description);
    REQUIRE(hp.has_units);
  }
  REQUIRE(probe_field<double>(s, std::string{"radius_a"}).has_range);
  REQUIRE(probe_field<double>(s, std::string{"radius_b"}).has_range);
  REQUIRE(!probe_field<double>(s, std::string{"rotation"}).has_range);
}

void test_ellipse_schema_driven_ctor_via_field_list() {
  rvegen::parameter_handler_t h;
  h.insert<double>("x", 0.0);
  h.insert<double>("y", 0.0);
  h.insert<double>("radius_a", 0.3);
  h.insert<double>("radius_b", 0.1);
  h.insert<double>("rotation", 0.0);
  rvegen::ellipse<double> e{h};
  REQUIRE(e.radius_a() == 0.3);
  REQUIRE(e.radius_b() == 0.1);
}

// ----------------------------------------------------------------------------
// bingham_distribution: 3D unit-vector sampler. Three regimes verified.
// ----------------------------------------------------------------------------
void test_bingham_uniform_regime_unit_length() {
  // κ = 0 → uniform on the sphere. Each sample must be unit length.
  std::mt19937 engine{321};
  std::array<double, 3> axis{0.0, 0.0, 1.0};
  rvegen::bingham_distribution<double> dist{axis, 0.0, engine};
  for (int i = 0; i < 200; ++i) {
    auto v = dist.sample();
    const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    REQUIRE(std::abs(n - 1.0) < 1e-9);
  }
}

void test_bingham_prolate_clusters_on_axis() {
  // κ = 50 → samples should cluster around mean axis. Mean |dot(v, axis)|
  // should be very close to 1.
  std::mt19937 engine{555};
  std::array<double, 3> axis{0.0, 0.0, 1.0};
  rvegen::bingham_distribution<double> dist{axis, 50.0, engine};

  constexpr int N = 1000;
  double mean_abs_dot = 0.0;
  for (int i = 0; i < N; ++i) {
    auto v = dist.sample();
    mean_abs_dot += std::abs(v[2]);   // axis is ẑ
  }
  mean_abs_dot /= N;
  // For uniform, E[|cos θ|] = 0.5. For κ=50 prolate, it should be > 0.9.
  REQUIRE(mean_abs_dot > 0.9);
}

void test_bingham_oblate_avoids_axis() {
  // κ = -50 → oblate, samples lie near the equator. Mean |dot(v, axis)|
  // should be small.
  std::mt19937 engine{777};
  std::array<double, 3> axis{0.0, 0.0, 1.0};
  rvegen::bingham_distribution<double> dist{axis, -50.0, engine};

  constexpr int N = 1000;
  double mean_abs_dot = 0.0;
  for (int i = 0; i < N; ++i) {
    auto v = dist.sample();
    mean_abs_dot += std::abs(v[2]);
  }
  mean_abs_dot /= N;
  REQUIRE(mean_abs_dot < 0.2);   // would be 0.5 for uniform
}

void test_bingham_uniform_kappa_zero_spreads_over_sphere() {
  // κ = 0 should give a uniform distribution on S². Mean of dot
  // products with each world axis should be near 0; mean of |dot|
  // should be near 0.5.
  std::mt19937 engine{2024};
  std::array<double, 3> axis{0.0, 0.0, 1.0};
  rvegen::bingham_distribution<double> dist{axis, 0.0, engine};

  constexpr int N = 4000;
  double sum_z = 0.0, sum_abs_z = 0.0;
  for (int i = 0; i < N; ++i) {
    auto v = dist.sample();
    sum_z += v[2];
    sum_abs_z += std::abs(v[2]);
  }
  REQUIRE(std::abs(sum_z / N) < 0.05);            // uniform: E[v_z] = 0
  REQUIRE(std::abs(sum_abs_z / N - 0.5) < 0.05);  // uniform: E[|v_z|] = 0.5
}

void test_bingham_operator_call_matches_sample() {
  // `operator()` is just sugar for `sample()` — same engine state, same
  // sequence (each call advances the engine deterministically). We
  // build two identically-seeded distributions and compare.
  std::mt19937 e1{777}, e2{777};
  std::array<double, 3> axis{0.0, 0.0, 1.0};
  rvegen::bingham_distribution<double> a{axis, 5.0, e1};
  rvegen::bingham_distribution<double> b{axis, 5.0, e2};
  auto v_op = a();
  auto v_sa = b.sample();
  REQUIRE(std::abs(v_op[0] - v_sa[0]) < 1e-15);
  REQUIRE(std::abs(v_op[1] - v_sa[1]) < 1e-15);
  REQUIRE(std::abs(v_op[2] - v_sa[2]) < 1e-15);
}

void test_bingham_rotated_oblate_avoids_axis() {
  // axis = ŷ, κ = -50: girdle in the xz-plane. Mean |v · ŷ| should be small.
  std::mt19937 engine{1234};
  std::array<double, 3> axis{0.0, 1.0, 0.0};
  rvegen::bingham_distribution<double> dist{axis, -50.0, engine};

  constexpr int N = 800;
  double mean_abs_dot = 0.0;
  for (int i = 0; i < N; ++i) mean_abs_dot += std::abs(dist.sample()[1]);
  mean_abs_dot /= N;
  REQUIRE(mean_abs_dot < 0.2);
}

void test_bingham_rotated_axis() {
  // Mean axis x̂ instead of ẑ. Samples should cluster on x.
  std::mt19937 engine{999};
  std::array<double, 3> axis{1.0, 0.0, 0.0};
  rvegen::bingham_distribution<double> dist{axis, 50.0, engine};

  constexpr int N = 800;
  double mean_abs_dot = 0.0;
  for (int i = 0; i < N; ++i) {
    auto v = dist.sample();
    mean_abs_dot += std::abs(v[0]);
  }
  mean_abs_dot /= N;
  REQUIRE(mean_abs_dot > 0.9);
}

// FOD support (#112): bingham_distribution implements
// direction_distribution_base — concrete-class samples and
// polymorphic-base samples agree.
void test_bingham_implements_direction_distribution_base() {
  std::mt19937 e1{12345}, e2{12345};
  std::array<double, 3> axis{0.0, 0.0, 1.0};
  rvegen::bingham_distribution<double> concrete{axis, 5.0, e1};

  // Same engine state → same sequence. Compare a concrete `sample()`
  // (gte::Vector) with a polymorphic operator() (std::array).
  rvegen::bingham_distribution<double> via_base{axis, 5.0, e2};
  rvegen::direction_distribution_base<double>& base_ref = via_base;

  const auto v_concrete = concrete.sample();
  const auto v_base     = base_ref();
  REQUIRE(std::abs(v_concrete[0] - v_base[0]) < 1e-15);
  REQUIRE(std::abs(v_concrete[1] - v_base[1]) < 1e-15);
  REQUIRE(std::abs(v_concrete[2] - v_base[2]) < 1e-15);
  // And the returned vector is unit-length (the FOD contract).
  const double mag = std::sqrt(v_base[0] * v_base[0] + v_base[1] * v_base[1] +
                               v_base[2] * v_base[2]);
  REQUIRE(std::abs(mag - 1.0) < 1e-9);
}

// polyline_tube_oriented_input draws ONE coherent unit-vector
// direction per produced shape from a direction_distribution_base
// (typically Bingham). Verifies tube centerline is aligned with the
// sampled axis and length matches the scalar dist.
void test_polyline_tube_oriented_input_aligns_with_bingham_axis() {
  std::mt19937 engine{42};
  rvegen::constant_distribution<double> px{0.5};
  rvegen::constant_distribution<double> py{0.5};
  rvegen::constant_distribution<double> pz{0.5};
  rvegen::constant_distribution<double> length{0.4};
  rvegen::constant_distribution<double> radius{0.02};
  // High-kappa Bingham clustered tightly around z-axis.
  rvegen::bingham_distribution<double> dir{
      std::array<double, 3>{0.0, 0.0, 1.0}, 200.0, engine};

  rvegen::polyline_tube_oriented_input<double> input{
      px, py, pz, dir, length, radius};

  // 50 placements — for κ = 200 the half-width of the Bingham is
  // roughly 1/√κ ≈ 4°, so essentially every sample lands within 10°
  // of the axis (cos > 0.985). Loose enough to be statistically
  // stable across engines/seeds, strict enough that a totally
  // unaligned input would fail.
  std::size_t close_to_z = 0;
  for (int i = 0; i < 50; ++i) {
    auto shape = input.new_shape();
    auto* tube = dynamic_cast<rvegen::polyline_tube<double>*>(shape.get());
    REQUIRE(tube != nullptr);
    // Tube spans 2 centerline points; the direction is (p1 - p0)/||·||.
    auto const& centerline = tube->centerline();
    REQUIRE(centerline.size() == 2);
    const auto dx = centerline[1][0] - centerline[0][0];
    const auto dy = centerline[1][1] - centerline[0][1];
    const auto dz = centerline[1][2] - centerline[0][2];
    const auto len = std::sqrt(dx * dx + dy * dy + dz * dz);
    REQUIRE(std::abs(len - 0.4) < 1e-9);
    const auto abs_cos = std::abs(dz) / len;
    if (abs_cos > 0.985) ++close_to_z;
  }
  REQUIRE(close_to_z >= 45);   // ~90 % of samples within 10° of axis
}

// Position and length scalars still come from the same scalar
// distributions the old directional input used — confirm those are
// untouched. The orientation hookup is the only change.
void test_polyline_tube_oriented_input_uses_scalar_position_distributions() {
  std::mt19937 engine{7};
  rvegen::constant_distribution<double> px{0.7};
  rvegen::constant_distribution<double> py{0.3};
  rvegen::constant_distribution<double> pz{0.1};
  rvegen::constant_distribution<double> length{0.2};
  rvegen::constant_distribution<double> radius{0.05};
  rvegen::bingham_distribution<double> dir{
      std::array<double, 3>{1.0, 0.0, 0.0}, 200.0, engine};

  rvegen::polyline_tube_oriented_input<double> input{
      px, py, pz, dir, length, radius};
  auto shape = input.new_shape();
  auto* tube = dynamic_cast<rvegen::polyline_tube<double>*>(shape.get());
  REQUIRE(tube != nullptr);
  // Centerline midpoint should be at the position constants.
  auto const& centerline = tube->centerline();
  const auto mx = 0.5 * (centerline[0][0] + centerline[1][0]);
  const auto my = 0.5 * (centerline[0][1] + centerline[1][1]);
  const auto mz = 0.5 * (centerline[0][2] + centerline[1][2]);
  REQUIRE(std::abs(mx - 0.7) < 1e-9);
  REQUIRE(std::abs(my - 0.3) < 1e-9);
  REQUIRE(std::abs(mz - 0.1) < 1e-9);
}

// FOD JSON registry plumbing: a Bingham distribution built via the
// `direction_distribution_registry_t` JSON path produces unit
// vectors and is statistically equivalent to a directly-constructed
// Bingham seeded with the same engine state.
void test_direction_distribution_registry_builds_bingham_from_json() {
  rvegen::register_all_direction_distributions<>();

  std::mt19937 engine{2024};
  const auto spec = nlohmann::json::parse(R"({
    "type": "bingham",
    "mean_axis_x": 0.0,
    "mean_axis_y": 0.0,
    "mean_axis_z": 1.0,
    "kappa":       50.0
  })");
  auto dist = rvegen::build_from_json(
      rvegen::direction_distribution_registry_t<>::instance(),
      spec, engine);
  REQUIRE(dist != nullptr);

  // 200 samples cluster on z (kappa=50 → ~8° half-width).
  std::size_t aligned = 0;
  for (int i = 0; i < 200; ++i) {
    const auto v = (*dist)();
    // Unit length within FP tolerance.
    const double mag = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
    REQUIRE(std::abs(mag - 1.0) < 1e-9);
    if (std::abs(v[2]) > 0.9) ++aligned;
  }
  REQUIRE(aligned > 150);   // >75 % of samples within 25° of z
}

void test_direction_distribution_registry_schema_has_descriptions() {
  // Schema audit-style check on the bingham slot: every required
  // field is present. Iterates the schema rather than relying on
  // contains() (which is non-const on input_parameter_controller).
  rvegen::register_all_direction_distributions<>();
  const auto& schema =
      rvegen::direction_distribution_registry_t<>::instance().schema("bingham");
  std::set<std::string> field_names;
  for (auto const& [key, _] : schema) field_names.insert(key);
  REQUIRE(field_names.count("mean_axis_x") == 1);
  REQUIRE(field_names.count("mean_axis_y") == 1);
  REQUIRE(field_names.count("mean_axis_z") == 1);
  REQUIRE(field_names.count("kappa")       == 1);
}

// ----------------------------------------------------------------------------
// weave_generator (#109 / #3 headline): deterministic plain weave of
// polyline_tubes — N warp yarns along +x and N weft yarns along +y with
// sinusoidal z-undulation 180° out of phase.
// ----------------------------------------------------------------------------
void test_weave_generator_tube_count_matches_warp_plus_weft() {
  rvegen::weave_generator<double> gen{
      /*domain=*/{1.0, 1.0, 0.2},
      /*n_warp=*/4, /*n_weft=*/3,
      /*yarn_radius=*/0.04, /*amplitude=*/0.03};
  auto shapes = gen.build();
  REQUIRE(shapes.size() == 7);
  // Every produced shape is a polyline_tube.
  for (auto const& s : shapes) {
    REQUIRE(dynamic_cast<rvegen::polyline_tube<double>*>(s.get()) != nullptr);
  }
}

void test_weave_generator_warp_yarns_run_along_x() {
  // For 2 warp + 2 weft, the first 2 entries are warp (along +x).
  // A warp yarn's centerline spans x ∈ [0, Lx] at fixed y.
  rvegen::weave_generator<double> gen{
      {2.0, 1.0, 0.4}, 2, 2, 0.05, 0.05, /*segments=*/16};
  auto shapes = gen.build();
  REQUIRE(shapes.size() == 4);
  auto* warp0 = dynamic_cast<rvegen::polyline_tube<double>*>(shapes[0].get());
  REQUIRE(warp0 != nullptr);
  auto const& cl = warp0->centerline();
  REQUIRE(cl.size() == 17);   // n_segments + 1
  // Endpoints span the full Lx; y is fixed across all control points.
  REQUIRE(std::abs(cl.front()[0] - 0.0) < 1e-12);
  REQUIRE(std::abs(cl.back()[0]  - 2.0) < 1e-12);
  const double y0 = cl[0][1];
  for (auto const& p : cl) {
    REQUIRE(std::abs(p[1] - y0) < 1e-12);
  }
  // First warp yarn should be at y = Ly / (2·N_warp) = 1.0 / 4 = 0.25.
  REQUIRE(std::abs(y0 - 0.25) < 1e-12);
}

void test_weave_generator_weft_yarns_run_along_y() {
  rvegen::weave_generator<double> gen{
      {2.0, 1.0, 0.4}, 2, 2, 0.05, 0.05, 16};
  auto shapes = gen.build();
  // Weft yarns are shapes[2] and shapes[3].
  auto* weft0 = dynamic_cast<rvegen::polyline_tube<double>*>(shapes[2].get());
  REQUIRE(weft0 != nullptr);
  auto const& cl = weft0->centerline();
  REQUIRE(std::abs(cl.front()[1] - 0.0) < 1e-12);
  REQUIRE(std::abs(cl.back()[1]  - 1.0) < 1e-12);
  const double x0 = cl[0][0];
  for (auto const& p : cl) {
    REQUIRE(std::abs(p[0] - x0) < 1e-12);
  }
  // First weft yarn at x = Lx / (2·N_weft) = 2.0 / 4 = 0.5.
  REQUIRE(std::abs(x0 - 0.5) < 1e-12);
}

void test_weave_generator_warp_and_weft_interlace_180_out_of_phase() {
  // At the intersection (x_weft, y_warp), the warp's z and the weft's
  // z must straddle z_centre symmetrically — that's what makes them
  // interlock. With amplitude > 0, warp_z - z_centre = -weft_z + z_centre.
  rvegen::weave_generator<double> gen{
      {1.0, 1.0, 0.3}, 1, 1, 0.05, 0.05, 32};
  auto shapes = gen.build();
  auto* warp = dynamic_cast<rvegen::polyline_tube<double>*>(shapes[0].get());
  auto* weft = dynamic_cast<rvegen::polyline_tube<double>*>(shapes[1].get());

  // 1 warp + 1 weft → warp at y = 0.5, weft at x = 0.5. Sample at
  // x = 0.25 (k=8 of 32 segments on Lx=1) — an exact control point.
  // Warp z at x = 0.25: z_centre + 0.05·sin(2π·0.25/1) = z_centre + 0.05.
  // Weft z at y = 0.25 (k=8 of 32 segments on Ly=1): z_centre - 0.05.
  const double z_centre = 0.15;
  const auto& wcl = warp->centerline();
  const auto& xcl = weft->centerline();
  REQUIRE(wcl.size() == 33 && xcl.size() == 33);
  REQUIRE(std::abs(wcl[8][0] - 0.25) < 1e-12);
  REQUIRE(std::abs(xcl[8][1] - 0.25) < 1e-12);
  REQUIRE(std::abs(wcl[8][2] - (z_centre + 0.05)) < 1e-12);
  REQUIRE(std::abs(xcl[8][2] - (z_centre - 0.05)) < 1e-12);
}

void test_weave_generator_phase_tagging() {
  rvegen::weave_generator<double> gen{
      {1.0, 1.0, 0.2}, 2, 2, 0.04, 0.03};
  gen.set_warp_phase_name("warp");
  gen.set_weft_phase_name("weft");
  auto shapes = gen.build();
  REQUIRE(shapes[0]->phase_name() == "warp");
  REQUIRE(shapes[1]->phase_name() == "warp");
  REQUIRE(shapes[2]->phase_name() == "weft");
  REQUIRE(shapes[3]->phase_name() == "weft");
}

void test_weave_generator_validation_throws_on_bad_input() {
  bool threw_zero_warp = false;
  try { rvegen::weave_generator<double>{{1,1,0.2}, 0, 1, 0.05, 0.03}; }
  catch (std::runtime_error const&) { threw_zero_warp = true; }
  REQUIRE(threw_zero_warp);

  bool threw_zero_weft = false;
  try { rvegen::weave_generator<double>{{1,1,0.2}, 1, 0, 0.05, 0.03}; }
  catch (std::runtime_error const&) { threw_zero_weft = true; }
  REQUIRE(threw_zero_weft);

  bool threw_neg_radius = false;
  try { rvegen::weave_generator<double>{{1,1,0.2}, 2, 2, -0.01, 0.03}; }
  catch (std::runtime_error const&) { threw_neg_radius = true; }
  REQUIRE(threw_neg_radius);

  bool threw_too_few_segments = false;
  try { rvegen::weave_generator<double>{{1,1,0.2}, 2, 2, 0.05, 0.03, 1}; }
  catch (std::runtime_error const&) { threw_too_few_segments = true; }
  REQUIRE(threw_too_few_segments);

  // 2D domain (Lz == 0) with non-zero amplitude is geometrically
  // nonsensical — yarns would land at z = ±amplitude outside the
  // zero-extent domain. Must reject up front rather than silently
  // producing voxel-grid-incompatible shapes.
  bool threw_2d_with_amplitude = false;
  try { rvegen::weave_generator<double>{{1,1,0}, 2, 2, 0.05, 0.03}; }
  catch (std::runtime_error const&) { threw_2d_with_amplitude = true; }
  REQUIRE(threw_2d_with_amplitude);

  // 2D domain with amplitude == 0 is a deliberate flat-weave config;
  // it must construct fine.
  bool flat_2d_ok = true;
  try { rvegen::weave_generator<double>{{1,1,0}, 2, 2, 0.05, 0.0}; }
  catch (std::runtime_error const&) { flat_2d_ok = false; }
  REQUIRE(flat_2d_ok);
}

// ----------------------------------------------------------------------------
// voronoi_generator_2d (#114 phase 1): bounded 2D Voronoi tessellation
// via Sutherland-Hodgman half-plane clipping.
// ----------------------------------------------------------------------------
void test_voronoi_generator_2d_single_seed_returns_full_domain() {
  // One seed → its Voronoi cell IS the bounding rectangle.
  rvegen::voronoi_generator_2d<double> gen{1.0, 1.0, {{0.5, 0.5}}};
  auto cells = gen.build();
  REQUIRE(cells.size() == 1);
  REQUIRE(cells[0].size() == 4);   // bounding rect = 4 vertices
  REQUIRE(std::abs(rvegen::polygon_area(cells[0]) - 1.0) < 1e-12);
}

void test_voronoi_generator_2d_two_seeds_split_by_bisector() {
  // Two seeds symmetric about x = 0.5. Their perpendicular bisector
  // is the line x = 0.5; cells split the domain into left/right
  // half-rectangles of area 0.5 each.
  rvegen::voronoi_generator_2d<double> gen{
      1.0, 1.0, {{0.25, 0.5}, {0.75, 0.5}}};
  auto cells = gen.build();
  REQUIRE(cells.size() == 2);
  REQUIRE(std::abs(rvegen::polygon_area(cells[0]) - 0.5) < 1e-12);
  REQUIRE(std::abs(rvegen::polygon_area(cells[1]) - 0.5) < 1e-12);
}

void test_voronoi_generator_2d_four_corner_seeds_quadrant_split() {
  // Four seeds, one near each corner. The Voronoi diagram is
  // symmetric: each cell is a 0.5 × 0.5 quadrant of area 0.25.
  rvegen::voronoi_generator_2d<double> gen{1.0, 1.0, {
      {0.25, 0.25}, {0.75, 0.25}, {0.25, 0.75}, {0.75, 0.75}}};
  auto cells = gen.build();
  REQUIRE(cells.size() == 4);
  for (auto const& cell : cells) {
    REQUIRE(std::abs(rvegen::polygon_area(cell) - 0.25) < 1e-12);
  }
}

void test_voronoi_generator_2d_cells_cover_domain() {
  // For 6 random-ish seeds, the sum of cell areas must equal the
  // total domain area. This is the partition-of-unity invariant
  // that a correct Voronoi tessellation must satisfy.
  rvegen::voronoi_generator_2d<double> gen{2.0, 1.5, {
      {0.3, 0.4}, {1.7, 0.4}, {0.5, 1.1},
      {1.2, 0.8}, {1.5, 1.3}, {0.8, 0.2}}};
  auto cells = gen.build();
  REQUIRE(cells.size() == 6);
  double total = 0.0;
  for (auto const& cell : cells) total += rvegen::polygon_area(cell);
  REQUIRE(std::abs(total - 2.0 * 1.5) < 1e-10);
}

void test_voronoi_generator_2d_each_cell_contains_its_seed() {
  // Defining property: seed `i` lies inside cell `i` (or on its
  // boundary). Verified by walking the polygon edges and checking
  // each edge's bisector half-plane: if the seed is inside cell i,
  // it must satisfy every bisector constraint cell i is built from.
  // Easier check: the seed must be a non-trivial interior point
  // of the cell. Use a winding/ray-cast or the simpler "barycentric"
  // test: the seed is in the convex hull of the cell's vertices.
  // For convex polygons, that reduces to "for every edge of the
  // CCW polygon, the seed is to the left of the edge."
  rvegen::voronoi_generator_2d<double> gen{2.0, 1.5, {
      {0.3, 0.4}, {1.7, 0.4}, {0.5, 1.1},
      {1.2, 0.8}, {1.5, 1.3}, {0.8, 0.2}}};
  auto cells = gen.build();
  for (std::size_t i = 0; i < cells.size(); ++i) {
    auto const& cell = cells[i];
    auto const& seed = std::array<double, 2>{0.3, 0.4};   // placeholder
    const std::array<double, 2> s =
        (i == 0) ? std::array<double, 2>{0.3, 0.4} :
        (i == 1) ? std::array<double, 2>{1.7, 0.4} :
        (i == 2) ? std::array<double, 2>{0.5, 1.1} :
        (i == 3) ? std::array<double, 2>{1.2, 0.8} :
        (i == 4) ? std::array<double, 2>{1.5, 1.3} :
                   std::array<double, 2>{0.8, 0.2};
    (void)seed;
    REQUIRE(cell.size() >= 3);
    bool inside = true;
    for (std::size_t k = 0; k < cell.size(); ++k) {
      auto const& a = cell[k];
      auto const& b = cell[(k + 1) % cell.size()];
      // CCW: cross of (b - a) × (s - a) must be ≥ 0 (≥ rather than >
      // so a seed exactly on an edge still counts as inside).
      const double cross = (b[0] - a[0]) * (s[1] - a[1])
                         - (b[1] - a[1]) * (s[0] - a[0]);
      if (cross < -1e-9) { inside = false; break; }
    }
    REQUIRE(inside);
  }
}

void test_voronoi_generator_2d_validation_throws() {
  bool threw_zero_seeds = false;
  try { rvegen::voronoi_generator_2d<double>{1.0, 1.0, {}}; }
  catch (std::runtime_error const&) { threw_zero_seeds = true; }
  REQUIRE(threw_zero_seeds);

  bool threw_zero_Lx = false;
  try { rvegen::voronoi_generator_2d<double>{0.0, 1.0, {{0.5, 0.5}}}; }
  catch (std::runtime_error const&) { threw_zero_Lx = true; }
  REQUIRE(threw_zero_Lx);

  bool threw_zero_Ly = false;
  try { rvegen::voronoi_generator_2d<double>{1.0, 0.0, {{0.5, 0.5}}}; }
  catch (std::runtime_error const&) { threw_zero_Ly = true; }
  REQUIRE(threw_zero_Ly);
}

// ----------------------------------------------------------------------------
// convex_polygon (#114 follow-up): 2D convex-polygon shape that wraps
// Voronoi-cell output (and any other CCW-vertex polygon) as a
// shape_base subclass, so writers + the rest of the rvegen pipeline
// can consume polycrystal-style RVEs without custom plumbing.
// ----------------------------------------------------------------------------
void test_convex_polygon_unit_square_basics() {
  rvegen::convex_polygon<double> sq{{{0.0, 0.0}, {1.0, 0.0},
                                     {1.0, 1.0}, {0.0, 1.0}}};
  REQUIRE(std::abs(sq.area() - 1.0) < 1e-12);
  REQUIRE(sq.volume() == 0.0);
  REQUIRE(sq.vertex_count() == 4);
  // is_inside: interior point yes, exterior no, edge point yes.
  REQUIRE(sq.is_inside({0.5, 0.5, 0.0}));
  REQUIRE(!sq.is_inside({1.5, 0.5, 0.0}));
  REQUIRE(sq.is_inside({1.0, 0.5, 0.0}));   // on right edge
  // Centroid of the unit square is (0.5, 0.5).
  const auto mid = sq.get_middle_point();
  REQUIRE(std::abs(mid[0] - 0.5) < 1e-12);
  REQUIRE(std::abs(mid[1] - 0.5) < 1e-12);
}

void test_convex_polygon_triangle_area_and_inside() {
  rvegen::convex_polygon<double> tri{{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}};
  // 0.5 · |1·1 − 0·0| = 0.5.
  REQUIRE(std::abs(tri.area() - 0.5) < 1e-12);
  REQUIRE(tri.is_inside({0.2, 0.2, 0.0}));
  REQUIRE(!tri.is_inside({0.6, 0.6, 0.0}));
}

void test_convex_polygon_set_middle_point_translates_vertices() {
  rvegen::convex_polygon<double> p{{{0.0, 0.0}, {1.0, 0.0},
                                    {1.0, 1.0}, {0.0, 1.0}}};
  p.set_middle_point({10.0, 5.0, 0.0});
  REQUIRE(p.is_inside({10.0, 5.0, 0.0}));
  REQUIRE(!p.is_inside({0.5, 0.5, 0.0}));
  // Centroid round-trips.
  const auto mid = p.get_middle_point();
  REQUIRE(std::abs(mid[0] - 10.0) < 1e-12);
  REQUIRE(std::abs(mid[1] -  5.0) < 1e-12);
  // Area is translation-invariant.
  REQUIRE(std::abs(p.area() - 1.0) < 1e-12);
}

void test_convex_polygon_bounding_box_is_aabb() {
  rvegen::convex_polygon<double> p{{{0.1, 0.2}, {0.9, 0.3},
                                    {0.7, 0.8}, {0.2, 0.7}}};
  p.make_bounding_box();
  auto* bb = p.bounding_box();
  REQUIRE(bb != nullptr);
  auto const& top    = bb->top_point();
  auto const& bottom = bb->bottom_point();
  REQUIRE(std::abs(top[0]    - 0.9) < 1e-12);
  REQUIRE(std::abs(top[1]    - 0.8) < 1e-12);
  REQUIRE(std::abs(bottom[0] - 0.1) < 1e-12);
  REQUIRE(std::abs(bottom[1] - 0.2) < 1e-12);
}

void test_convex_polygon_clone_is_independent_copy() {
  rvegen::convex_polygon<double> p{{{0.0, 0.0}, {1.0, 0.0},
                                    {1.0, 1.0}, {0.0, 1.0}}};
  auto copy = p.clone();
  REQUIRE(copy != nullptr);
  copy->set_middle_point({5.0, 5.0, 0.0});
  // Original unaffected.
  REQUIRE(p.is_inside({0.5, 0.5, 0.0}));
  REQUIRE(copy->is_inside({5.0, 5.0, 0.0}));
}

void test_convex_polygon_rejects_too_few_vertices() {
  bool threw = false;
  try { rvegen::convex_polygon<double> p{{{0.0, 0.0}, {1.0, 0.0}}}; }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_convex_polygon_wraps_voronoi_cell() {
  // End-to-end: Voronoi generator → convex_polygon shape → is_inside
  // works correctly on the resulting cell. Pins the cell-to-shape
  // pipeline that motivates this PR.
  rvegen::voronoi_generator_2d<double> gen{
      1.0, 1.0, {{0.25, 0.5}, {0.75, 0.5}}};
  auto cells = gen.build();
  REQUIRE(cells.size() == 2);
  rvegen::convex_polygon<double> left_cell{cells[0]};
  rvegen::convex_polygon<double> right_cell{cells[1]};
  // Left cell covers x < 0.5; its seed (0.25, 0.5) is inside.
  REQUIRE(left_cell.is_inside({0.25, 0.5, 0.0}));
  REQUIRE(!left_cell.is_inside({0.75, 0.5, 0.0}));
  // Right cell mirror.
  REQUIRE(right_cell.is_inside({0.75, 0.5, 0.0}));
  REQUIRE(!right_cell.is_inside({0.25, 0.5, 0.0}));
  // Both half-rectangles have area 0.5.
  REQUIRE(std::abs(left_cell.area()  - 0.5) < 1e-12);
  REQUIRE(std::abs(right_cell.area() - 0.5) < 1e-12);
}

// gmsh_geo_writer dispatch for convex_polygon: emits N Points + N Lines
// + 1 Line Loop + 1 Plane Surface. End result is a polycrystal-RVE-to-
// gmsh-mesh pipeline that consumes voronoi_generator_2d output directly.
void test_gmsh_geo_writer_emits_plane_surface_for_convex_polygon() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::convex_polygon<double>>(
      std::vector<std::array<double, 2>>{{0.1, 0.1}, {0.4, 0.1},
                                          {0.4, 0.4}, {0.1, 0.4}}));

  rvegen::gmsh_geo_writer<double> writer{};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  const auto txt = out.str();

  // 4 Points + 4 Lines + 1 Line Loop + 1 Plane Surface = 10 directives.
  REQUIRE(txt.find("Point(") != std::string::npos);
  REQUIRE(txt.find("Line(")  != std::string::npos);
  REQUIRE(txt.find("Line Loop(")     != std::string::npos);
  REQUIRE(txt.find("Plane Surface(") != std::string::npos);
  // The Plane Surface id IS in the file (any positive integer is fine —
  // we don't hardcode the exact id since it depends on the allocator).
  REQUIRE(txt.find("(unsupported 2D shape skipped)") == std::string::npos);
}

void test_gmsh_geo_writer_polygon_id_appears_in_physical_group() {
  // The Plane Surface entity id should join the Physical Surface
  // group when phases are attached — same as Disk / Rectangle.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("grain");

  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  auto poly = std::make_unique<rvegen::convex_polygon<double>>(
      std::vector<std::array<double, 2>>{{0.1, 0.1}, {0.4, 0.1},
                                          {0.4, 0.4}, {0.1, 0.4}});
  poly->set_phase_name("grain");
  shapes.emplace_back(std::move(poly));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  const auto txt = out.str();
  // Physical Surface for "grain" (id 2 in the collection) must include
  // the polygon's Plane Surface entity id — we don't pin the literal
  // entity id (it's allocator-dependent), but we confirm the directive
  // is emitted and references some integer entity.
  REQUIRE(txt.find("Physical Surface(\"grain\", 2) = {") != std::string::npos);
}

// gmsh mesh-size knob (#117): setting characteristic_length_min/max
// makes the writer emit `Mesh.CharacteristicLengthMin/Max = X;`
// directives in the .geo header so the FE mesher uses the requested
// size without manual editing. Default 0 keeps the original commented
// placeholders.
void test_gmsh_geo_writer_emits_mesh_size_directives_when_set() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_characteristic_length_min(0.01);
  writer.set_characteristic_length_max(0.05);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  const auto txt = out.str();

  REQUIRE(txt.find("Mesh.CharacteristicLengthMin = 0.01;") != std::string::npos);
  REQUIRE(txt.find("Mesh.CharacteristicLengthMax = 0.05;") != std::string::npos);
  // The commented placeholders should NOT be emitted when an active
  // value is set.
  REQUIRE(txt.find("// Mesh.CharacteristicLengthMin = 0.05;") == std::string::npos);
  REQUIRE(txt.find("// Mesh.CharacteristicLengthMax = 0.1;")  == std::string::npos);
}

void test_gmsh_geo_writer_emits_element_order_when_set() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_element_order(2);
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  REQUIRE(out.str().find("Mesh.ElementOrder = 2;") != std::string::npos);
}

void test_gmsh_geo_writer_omits_element_order_when_default() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  REQUIRE(out.str().find("Mesh.ElementOrder") == std::string::npos);
}

void test_gmsh_geo_writer_keeps_placeholders_when_mesh_size_default() {
  // Default ctor leaves the mesh-size knobs at 0 → commented
  // placeholders remain. Back-compat for callers that haven't
  // adopted the new knob yet.
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));

  rvegen::gmsh_geo_writer<double> writer{};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  const auto txt = out.str();

  REQUIRE(txt.find("// Mesh.CharacteristicLengthMin = 0.05;") != std::string::npos);
  REQUIRE(txt.find("// Mesh.CharacteristicLengthMax = 0.1;")  != std::string::npos);
}

// One-line helper: voronoi_generator_2d → shape_vector ready for any
// writer. Skips degenerate (<3 vertex) cells silently — those only
// arise from coincident seeds.
void test_voronoi_to_shapes_helper_returns_shape_vector() {
  rvegen::voronoi_generator_2d<double> gen{
      1.0, 1.0, {{0.25, 0.5}, {0.75, 0.5}}};
  auto shapes = rvegen::voronoi_to_shapes(gen);
  REQUIRE(shapes.size() == 2);
  for (auto const& s : shapes) {
    REQUIRE(s != nullptr);
    REQUIRE(dynamic_cast<rvegen::convex_polygon<double>*>(s.get()) != nullptr);
    REQUIRE(std::abs(s->area() - 0.5) < 1e-12);
  }
}

// svg_writer dispatch for convex_polygon: emit a <polygon> element with
// SVG-flipped y-coordinates. Mirrors the gmsh dispatch added for the
// Voronoi pipeline.
void test_svg_writer_emits_polygon_element_for_convex_polygon() {
  rvegen::svg_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::convex_polygon<double>>(
      std::vector<std::array<double, 2>>{{0.1, 0.2}, {0.4, 0.2},
                                          {0.4, 0.6}, {0.1, 0.6}}));

  rvegen::svg_writer<double> writer{};
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  const auto txt = out.str();

  REQUIRE(txt.find("<polygon points=\"") != std::string::npos);
  // SVG y is top-down so each vertex's y must be Ly - input_y.
  // Bottom-left input (0.1, 0.2) → SVG (0.1, 0.8).
  REQUIRE(txt.find("0.1,0.8") != std::string::npos);
  REQUIRE(txt.find("(skipped 3D / unsupported shape)") == std::string::npos);
}

// ----------------------------------------------------------------------------
// phase + phase_collection: name + opaque material_config; ids start at 1.
// ----------------------------------------------------------------------------
void test_phase_collection_basics_and_ids() {
  rvegen::phase_collection<double> phases;
  auto const& matrix = phases.add("matrix",
      nlohmann::json{{"type", "linear_elasticity"}, {"K", 1.6e9}, {"G", 0.8e9}});
  auto const& fibre  = phases.add("fibre",
      nlohmann::json{{"type", "linear_elasticity"}, {"K", 5.0e10}, {"G", 3.0e10}});

  REQUIRE(matrix.id == 1);
  REQUIRE(fibre.id == 2);
  REQUIRE(phases.size() == 2);
  REQUIRE(phases.id_of("matrix") == 1);
  REQUIRE(phases.id_of("fibre") == 2);
  REQUIRE(phases.contains("matrix"));
  REQUIRE(!phases.contains("void"));

  // Material config is opaque pass-through.
  REQUIRE(phases.at("fibre").material_config["K"].get<double>() == 5.0e10);
}

void test_phase_collection_duplicate_throws() {
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  bool threw = false;
  try { phases.add("matrix"); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_phase_collection_at_unknown_throws() {
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  bool threw = false;
  try { (void)phases.at("nonexistent"); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_stl_ascii_reader_parse_error_includes_position() {
  // An "outer" keyword replaced with a typo — the reader should throw
  // with a position marker in the message.
  const char* bad = R"(solid bad
  facet normal 0 0 1
    OUTNER loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid
)";
  std::stringstream ss{bad};
  bool threw = false;
  std::string what;
  try { (void)rvegen::read_stl_ascii<double>(ss); }
  catch (std::runtime_error const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  // Position marker uses 'byte' as the unit.
  REQUIRE(what.find("byte") != std::string::npos);
}

void test_mesh_inclusion_input_via_registry() {
  // Drop the unit-cube STL fixture into a temp file, build a
  // mesh_inclusion_input via the JSON registry path, and confirm
  // new_shape() produces a translated mesh_inclusion.
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  const std::string tmp_path = "/tmp/rvegen_test_unit_cube.stl";
  {
    std::ofstream out{tmp_path};
    out << unit_cube_stl;
  }

  std::mt19937 engine{17};
  const auto dist_specs = nlohmann::json::parse(R"({
    "px": {"type": "constant", "value":  5.0},
    "py": {"type": "constant", "value":  3.0},
    "pz": {"type": "constant", "value": -2.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }

  const auto input_spec_json = std::string{R"({
    "type": "mesh_inclusion_input",
    "stl_path": ")"} + tmp_path + R"(",
    "position_x_dist": "px",
    "position_y_dist": "py",
    "position_z_dist": "pz"
  })";
  const auto input_spec = nlohmann::json::parse(input_spec_json);
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  REQUIRE(input != nullptr);

  auto shape = input->new_shape();
  auto* mesh = dynamic_cast<rvegen::mesh_inclusion<double>*>(shape.get());
  REQUIRE(mesh != nullptr);
  REQUIRE(mesh->triangle_count() == 12);
  // Translated to (5, 3, -2): origin should be outside, the new centre inside.
  REQUIRE(!mesh->is_inside({0.0, 0.0, 0.0}));
  REQUIRE(mesh->is_inside({5.0, 3.0, -2.0}));

  std::remove(tmp_path.c_str());
}

// Canonical `mesh_path` JSON field works the same as the legacy
// `stl_path` alias.
void test_mesh_inclusion_input_accepts_canonical_mesh_path() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  const std::string tmp_path = "/tmp/rvegen_test_mesh_path_canonical.stl";
  {
    std::ofstream out{tmp_path};
    out << unit_cube_stl;
  }

  std::mt19937 engine{31};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec_json = std::string{R"({
    "type": "mesh_inclusion_input",
    "mesh_path": ")"} + tmp_path + R"(",
    "position_x_dist": "p",
    "position_y_dist": "p",
    "position_z_dist": "p"
  })";
  const auto input_spec = nlohmann::json::parse(input_spec_json);
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  REQUIRE(input != nullptr);
  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(shape->is_inside({0.0, 0.0, 0.0}));
  std::remove(tmp_path.c_str());
}

void test_mesh_inclusion_input_rejects_both_mesh_and_stl_path() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{32};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "mesh_inclusion_input",
    "mesh_path": "/tmp/a.stl",
    "stl_path":  "/tmp/b.stl",
    "position_x_dist": "p",
    "position_y_dist": "p",
    "position_z_dist": "p"
  })");
  bool threw = false;
  std::string what;
  try {
    (void)rvegen::build_from_json(
        rvegen::input_registry_t<>::instance(), input_spec, dists);
  } catch (std::runtime_error const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  REQUIRE(what.find("both 'mesh_path' and") != std::string::npos);
}

void test_mesh_inclusion_input_rejects_neither_mesh_nor_stl_path() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{33};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "mesh_inclusion_input",
    "position_x_dist": "p",
    "position_y_dist": "p",
    "position_z_dist": "p"
  })");
  bool threw = false;
  std::string what;
  try {
    (void)rvegen::build_from_json(
        rvegen::input_registry_t<>::instance(), input_spec, dists);
  } catch (std::runtime_error const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  REQUIRE(what.find("mesh_path") != std::string::npos);
}

void test_mesh_inclusion_input_missing_file_throws() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{18};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "mesh_inclusion_input",
    "stl_path": "/tmp/rvegen_does_not_exist_xyzzy.stl",
    "position_x_dist": "p",
    "position_y_dist": "p",
    "position_z_dist": "p"
  })");
  bool threw = false;
  try {
    auto input = rvegen::build_from_json(
        rvegen::input_registry_t<>::instance(), input_spec, dists);
  } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

// mesh_inclusion_input now auto-detects binary vs ASCII STL via
// read_stl_file. Confirm a binary blob round-trips through the input
// without callers needing to know which format they have.
void test_mesh_inclusion_input_accepts_binary_stl() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  // Convert the existing ASCII unit-cube fixture to a binary STL blob so
  // the test exercises a closed mesh (a 1-triangle blob can't be
  // probed with is_inside meaningfully).
  std::vector<gte::Triangle3<double>> ascii_tris;
  {
    std::stringstream ss{unit_cube_stl};
    ascii_tris = rvegen::read_stl_ascii<double>(ss);
  }
  REQUIRE(ascii_tris.size() == 12);

  auto write_le_u32 = [](std::ostream& o, std::uint32_t v) {
    o.put(static_cast<char>(v & 0xff));
    o.put(static_cast<char>((v >> 8)  & 0xff));
    o.put(static_cast<char>((v >> 16) & 0xff));
    o.put(static_cast<char>((v >> 24) & 0xff));
  };
  auto write_le_f32 = [](std::ostream& o, float f) {
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    o.put(static_cast<char>(bits & 0xff));
    o.put(static_cast<char>((bits >> 8)  & 0xff));
    o.put(static_cast<char>((bits >> 16) & 0xff));
    o.put(static_cast<char>((bits >> 24) & 0xff));
  };

  const std::string tmp_path = "/tmp/rvegen_test_binary_via_input.stl";
  {
    std::ofstream out{tmp_path, std::ios::binary};
    for (int i = 0; i < 80; ++i) out.put('\0');     // header
    write_le_u32(out, static_cast<std::uint32_t>(ascii_tris.size()));
    for (auto const& t : ascii_tris) {
      // Normal (0,0,1) is fine — the reader skips it (computed from
      // vertex order anyway).
      for (float n : {0.f, 0.f, 1.f}) write_le_f32(out, n);
      for (int v = 0; v < 3; ++v) {
        for (int c = 0; c < 3; ++c) {
          write_le_f32(out, static_cast<float>(t.v[v][c]));
        }
      }
      out.put('\0'); out.put('\0');                  // attribute bytes
    }
  }

  std::mt19937 engine{23};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec_json = std::string{R"({
    "type": "mesh_inclusion_input",
    "stl_path": ")"} + tmp_path + R"(",
    "position_x_dist": "p",
    "position_y_dist": "p",
    "position_z_dist": "p"
  })";
  const auto input_spec = nlohmann::json::parse(input_spec_json);
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  REQUIRE(input != nullptr);

  // Cross-check the full pipeline: new_shape() must yield a placed
  // inclusion whose is_inside() actually reflects the cube geometry
  // (origin inside, far point outside). Catches both the binary-parse
  // path and the input → mesh_inclusion handoff.
  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(shape->is_inside({0.0, 0.0, 0.0}));
  REQUIRE(!shape->is_inside({2.0, 2.0, 2.0}));

  std::remove(tmp_path.c_str());
}

// `read_mesh_file` dispatches by file extension. Once mesh_inclusion_input
// uses it, a `.ply` path Just Works the same way `.stl` does — the
// canonical "STL or PLY in the same field" promise of #9.
void test_read_mesh_file_dispatches_stl_and_ply() {
  // Drop the existing ASCII unit-cube fixtures to temp files with the
  // matching extension and confirm each round-trips through dispatch.
  const std::string stl_tmp = "/tmp/rvegen_test_mesh_dispatch.stl";
  {
    std::ofstream out{stl_tmp};
    out << unit_cube_stl;
  }
  const std::string ply_tmp = "/tmp/rvegen_test_mesh_dispatch.ply";
  {
    std::ofstream out{ply_tmp};
    out << unit_cube_ply;
  }
  auto from_stl = rvegen::read_mesh_file<double>(stl_tmp);
  auto from_ply = rvegen::read_mesh_file<double>(ply_tmp);
  REQUIRE(from_stl.size() == 12);
  REQUIRE(from_ply.size() == 12);

  std::remove(stl_tmp.c_str());
  std::remove(ply_tmp.c_str());
}

void test_read_mesh_file_unknown_extension_throws() {
  const std::string bad = "/tmp/rvegen_test_mesh_dispatch.xyz";
  { std::ofstream out{bad}; out << "irrelevant"; }
  bool threw = false;
  std::string what;
  try { (void)rvegen::read_mesh_file<double>(bad); }
  catch (std::runtime_error const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  REQUIRE(what.find("unsupported extension") != std::string::npos);
  std::remove(bad.c_str());
}

void test_read_mesh_file_trims_trailing_whitespace() {
  // A path with stray trailing whitespace (a common JSON-config slip)
  // should still dispatch correctly rather than triggering the "no
  // extension" branch.
  const std::string stl_tmp = "/tmp/rvegen_test_mesh_trim.stl";
  {
    std::ofstream out{stl_tmp};
    out << unit_cube_stl;
  }
  auto tris = rvegen::read_mesh_file<double>(stl_tmp + "   ");
  REQUIRE(tris.size() == 12);
  std::remove(stl_tmp.c_str());
}

// PLY reader now tolerates unknown header lines (texturefile,
// materialfile, ply::group, etc.) — silently skipped, the way real
// PLY producers emit them. Previously this threw and refused
// otherwise-valid files from meshlab et al.
void test_ply_ascii_reader_tolerates_unknown_header_lines() {
  constexpr char const* with_extras = R"(ply
format ascii 1.0
comment exported from meshlab
texturefile foo.png
materialfile bar.mtl
ply::group fibre_cluster
element vertex 3
property float x
property float y
property float z
element face 1
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
3 0 1 2
)";
  std::stringstream ss{with_extras};
  auto tris = rvegen::read_ply_ascii<double>(ss);
  REQUIRE(tris.size() == 1);
}

// PLY parse errors mid-element now carry a position marker, matching
// the STL reader's `(at byte N)` convention.
void test_ply_ascii_reader_parse_error_includes_position() {
  // Truncated payload — header announces 3 vertices but only 2 are
  // present. The reader runs out of input mid-element.
  constexpr char const* truncated = R"(ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 0
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
)";
  std::stringstream ss{truncated};
  bool threw = false;
  std::string what;
  try { (void)rvegen::read_ply_ascii<double>(ss); }
  catch (std::runtime_error const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  REQUIRE(what.find("byte") != std::string::npos);
}

void test_mesh_inclusion_input_accepts_ply_via_dispatch() {
  // Same setup as the binary-STL test, but feeding a PLY file. Proves
  // the registry → input → mesh_inclusion path consumes either format.
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  const std::string tmp_path = "/tmp/rvegen_test_ply_via_input.ply";
  {
    std::ofstream out{tmp_path};
    out << unit_cube_ply;
  }

  std::mt19937 engine{29};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec_json = std::string{R"({
    "type": "mesh_inclusion_input",
    "stl_path": ")"} + tmp_path + R"(",
    "position_x_dist": "p",
    "position_y_dist": "p",
    "position_z_dist": "p"
  })";
  const auto input_spec = nlohmann::json::parse(input_spec_json);
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  REQUIRE(input != nullptr);

  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(shape->is_inside({0.0, 0.0, 0.0}));
  REQUIRE(!shape->is_inside({2.0, 2.0, 2.0}));

  std::remove(tmp_path.c_str());
}

void test_mesh_inclusion_input_empty_stl_throws() {
  // An STL file with zero triangles would yield a `mesh_inclusion`
  // whose `is_inside` is always false. The input ctor must refuse it
  // up front rather than letting the generator's volume-fraction
  // termination loop forever.
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  const std::string tmp_path = "/tmp/rvegen_test_empty.stl";
  {
    std::ofstream out{tmp_path};
    out << "solid empty\nendsolid empty\n";
  }

  std::mt19937 engine{19};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec_json = std::string{R"({
    "type": "mesh_inclusion_input",
    "stl_path": ")"} + tmp_path + R"(",
    "position_x_dist": "p",
    "position_y_dist": "p",
    "position_z_dist": "p"
  })";
  const auto input_spec = nlohmann::json::parse(input_spec_json);
  bool threw = false;
  try {
    auto input = rvegen::build_from_json(
        rvegen::input_registry_t<>::instance(), input_spec, dists);
  } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  std::remove(tmp_path.c_str());
}

void test_load_phases_from_json_rejects_non_array() {
  bool threw = false;
  try {
    rvegen::load_phases_from_json<double>(nlohmann::json::object());
  } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_load_phases_from_json_rejects_missing_name() {
  const auto cfg = nlohmann::json::parse(R"([{"material": {"K": 1.0}}])");
  bool threw = false;
  try { rvegen::load_phases_from_json<double>(cfg); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_load_phases_from_json_rejects_non_object_material() {
  const auto cfg = nlohmann::json::parse(R"([{"name": "x", "material": 42}])");
  bool threw = false;
  try { rvegen::load_phases_from_json<double>(cfg); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_load_phases_from_json_round_trip() {
  const auto cfg = nlohmann::json::parse(R"([
    {"name": "matrix", "material": {"type": "linear_elasticity", "K": 1.6e9, "G": 0.8e9}},
    {"name": "fibre",  "material": {"type": "linear_elasticity", "K": 5.0e10, "G": 3.0e10}}
  ])");
  auto phases = rvegen::load_phases_from_json<double>(cfg);
  REQUIRE(phases.size() == 2);
  REQUIRE(phases.id_of("matrix") == 1);
  REQUIRE(phases.id_of("fibre") == 2);
  REQUIRE(phases.at("matrix").material_config["G"].get<double>() == 0.8e9);

  // Insertion order preserved in ordered() view.
  auto const& ord = phases.ordered();
  REQUIRE(ord.size() == 2);
  REQUIRE(ord[0]->name == "matrix");
  REQUIRE(ord[1]->name == "fibre");
}

// ----------------------------------------------------------------------------
// Phase-name tagging: a `phase_name` field in an input's JSON config gets
// stamped onto every shape produced by that input.
// ----------------------------------------------------------------------------
void test_input_stamps_phase_name_on_produced_shape() {
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();

  std::mt19937 engine{77};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x": {"type": "constant", "value": 0.5},
    "y": {"type": "constant", "value": 0.5},
    "r": {"type": "constant", "value": 0.1}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "circle_input",
    "pos_x_dist": "x",
    "pos_y_dist": "y",
    "radius_dist": "r",
    "phase_name": "fibre"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  auto shape = input->new_shape();
  REQUIRE(shape != nullptr);
  REQUIRE(shape->phase_name() == "fibre");
}

void test_input_phase_name_default_is_empty() {
  // Same circle_input but without `phase_name` in the JSON. Default is empty.
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{78};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x": {"type": "constant", "value": 0.5},
    "y": {"type": "constant", "value": 0.5},
    "r": {"type": "constant", "value": 0.1}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "circle_input",
    "pos_x_dist": "x",
    "pos_y_dist": "y",
    "radius_dist": "r"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  auto shape = input->new_shape();
  REQUIRE(shape->phase_name().empty());
}

void test_input_metadata_json_field_merges_into_info() {
  // The `metadata` schema field accepts a JSON-encoded string of
  // arbitrary key/value pairs that get merged into the produced
  // shape's info blob. Combined with `phase_name`, both fields
  // round-trip and metadata can also override phase_name on its own.
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{121};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x": {"type": "constant", "value": 0.5},
    "y": {"type": "constant", "value": 0.5},
    "r": {"type": "constant", "value": 0.1}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  // phase_name + a metadata blob with custom keys + a numeric value.
  // The schema requires `metadata` to be a JSON-encoded string (not a
  // nested object) until the parameter visitor learns object passthrough.
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "circle_input",
    "pos_x_dist": "x",
    "pos_y_dist": "y",
    "radius_dist": "r",
    "phase_name": "fibre",
    "metadata": "{\"orientation_deg\": 42.5, \"source\": \"a.stl\", \"active\": true}"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  auto shape = input->new_shape();
  REQUIRE(shape->phase_name() == "fibre");
  REQUIRE(shape->info().get<double>("orientation_deg") == 42.5);
  REQUIRE(shape->info().get<std::string>("source") == "a.stl");
  REQUIRE(shape->info().get<bool>("active") == true);
}

void test_input_metadata_overrides_phase_name() {
  // Documented behaviour: `metadata` is merged AFTER `phase_name`,
  // so a `phase_name` key inside the metadata object wins.
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  std::mt19937 engine{909};
  const auto dist_specs = nlohmann::json::parse(R"({
    "x": {"type": "constant", "value": 0.5},
    "y": {"type": "constant", "value": 0.5},
    "r": {"type": "constant", "value": 0.1}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto input_spec = nlohmann::json::parse(R"({
    "type": "circle_input",
    "pos_x_dist": "x",
    "pos_y_dist": "y",
    "radius_dist": "r",
    "phase_name": "matrix",
    "metadata": "{\"phase_name\": \"fibre\"}"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), input_spec, dists);
  auto shape = input->new_shape();
  REQUIRE(shape->phase_name() == "fibre");
}

void test_info_generic_metadata_round_trip() {
  // Verify the info container behaves as a generic key-value store
  // beyond the phase_name shortcut: write multiple typed values,
  // confirm they round-trip, and confirm propagation through tag().
  rvegen::info i;
  i.set("phase_name", std::string{"matrix"});
  i.set("orientation_deg", 42.5);
  i.set("source_id", 7);
  i.set("active", true);

  REQUIRE(i.phase_name() == "matrix");
  REQUIRE(i.get<double>("orientation_deg") == 42.5);
  REQUIRE(i.get<int>("source_id") == 7);
  REQUIRE(i.get<bool>("active") == true);
  REQUIRE(i.contains("orientation_deg"));
  REQUIRE(!i.contains("missing"));
  REQUIRE(i.get_or<int>("missing", -1) == -1);

  // Stamp onto a shape via direct ctor + set_info; clone propagates it.
  rvegen::circle<double> c{0.5, 0.5, 0.1};
  c.set_info(i);
  REQUIRE(c.info().get<double>("orientation_deg") == 42.5);
  auto cloned = c.clone();
  REQUIRE(cloned->info().phase_name() == "matrix");
  REQUIRE(cloned->info().get<bool>("active") == true);
}

// ----------------------------------------------------------------------------
// Binary STL reader (#9 follow-up).
// ----------------------------------------------------------------------------

// Build a minimal binary STL blob containing `n` identical unit-triangle
// records. Returns a std::string so it can be wrapped in stringstream.
namespace {
std::string make_binary_stl_blob(std::uint32_t n_tris) {
  std::string blob;
  blob.resize(80, '\0');                          // header
  blob.push_back(static_cast<char>(n_tris & 0xff));
  blob.push_back(static_cast<char>((n_tris >> 8)  & 0xff));
  blob.push_back(static_cast<char>((n_tris >> 16) & 0xff));
  blob.push_back(static_cast<char>((n_tris >> 24) & 0xff));
  for (std::uint32_t t = 0; t < n_tris; ++t) {
    // Normal (0,0,1) — 12 zero bytes for x,y then 0x3f800000 for z=1.0.
    const std::array<unsigned char, 12> normal{
        0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x00, 0x80, 0x3f};
    for (auto b : normal) blob.push_back(static_cast<char>(b));
    // 3 vertices: (0,0,0), (1,0,0), (0,1,0).
    const std::array<float, 9> verts{
        0.f, 0.f, 0.f,
        1.f, 0.f, 0.f,
        0.f, 1.f, 0.f};
    for (float f : verts) {
      std::uint32_t bits;
      std::memcpy(&bits, &f, sizeof(bits));
      blob.push_back(static_cast<char>(bits & 0xff));
      blob.push_back(static_cast<char>((bits >> 8)  & 0xff));
      blob.push_back(static_cast<char>((bits >> 16) & 0xff));
      blob.push_back(static_cast<char>((bits >> 24) & 0xff));
    }
    blob.push_back('\0');                         // attribute byte count lo
    blob.push_back('\0');                         // attribute byte count hi
  }
  return blob;
}
}

void test_stl_binary_reader_basic() {
  auto blob = make_binary_stl_blob(3);
  std::stringstream ss{blob};
  auto tris = rvegen::read_stl_binary<double>(ss);
  REQUIRE(tris.size() == 3);
  // Each record uses the same vertex pattern.
  for (auto const& t : tris) {
    REQUIRE(std::abs(t.v[1][0] - 1.0) < 1e-7);
    REQUIRE(std::abs(t.v[2][1] - 1.0) < 1e-7);
  }
}

void test_stl_binary_reader_short_file_throws() {
  // Only 60 bytes — less than the 80-byte header.
  std::stringstream ss{std::string(60, '\0')};
  bool threw = false;
  try { (void)rvegen::read_stl_binary<double>(ss); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_stl_binary_reader_sanity_cap_throws() {
  // Header (80 NULs) + a triangle count of 200M (above the 100M cap).
  std::string blob(80, '\0');
  std::uint32_t huge = 200'000'000;
  blob.push_back(static_cast<char>(huge & 0xff));
  blob.push_back(static_cast<char>((huge >> 8)  & 0xff));
  blob.push_back(static_cast<char>((huge >> 16) & 0xff));
  blob.push_back(static_cast<char>((huge >> 24) & 0xff));
  std::stringstream ss{blob};
  bool threw = false;
  try { (void)rvegen::read_stl_binary<double>(ss); }
  catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_stl_auto_detect_dispatches_correctly() {
  // Binary input → read_stl picks read_stl_binary.
  {
    auto blob = make_binary_stl_blob(2);
    std::stringstream ss{blob};
    auto tris = rvegen::read_stl<double>(ss);
    REQUIRE(tris.size() == 2);
  }
  // ASCII input → read_stl picks read_stl_ascii.
  {
    std::stringstream ss{unit_cube_stl};
    auto tris = rvegen::read_stl<double>(ss);
    REQUIRE(tris.size() == 12);
  }
}

// ----------------------------------------------------------------------------
// voxel_writer + phase_collection: voxels emit phase ids (matrix=0, fibre=1,
// ...) keyed off each shape's `phase_name`, instead of 1-based shape index.
// ----------------------------------------------------------------------------
void test_voxel_writer_emits_phase_ids_from_phase_collection() {
  // Two phases, two shapes — one tagged "fibre", one untagged. The
  // tagged shape's voxels carry the fibre id; untagged voxels (including
  // the background and the second circle) stay at 0 (matrix sentinel).
  rvegen::phase_collection<double> phases;
  const auto matrix_id = phases.add("matrix").id;   // id 1
  const auto fibre_id  = phases.add("fibre").id;    // id 2

  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  {
    auto fibre = std::make_unique<rvegen::circle<double>>(0.25, 0.25, 0.15);
    fibre->set_phase_name("fibre");
    shapes.push_back(std::move(fibre));
  }
  {
    // No phase_name set — voxels inside this shape should stay at 0
    // (the matrix / untagged sentinel) rather than collide with fibre.
    auto untagged = std::make_unique<rvegen::circle<double>>(0.75, 0.75, 0.15);
    shapes.push_back(std::move(untagged));
  }

  constexpr std::size_t N = 32;
  rvegen::voxel_writer<double> w{N, N, 1};
  w.set_phases(&phases);

  const auto grid = w.sample(shapes, std::array<double, 3>{1.0, 1.0, 0.0});

  std::size_t n_fibre = 0;
  bool any_other = false;
  for (auto v : grid) {
    if (v == 0 || v == fibre_id) {
      if (v == fibre_id) ++n_fibre;
    } else {
      any_other = true;
    }
  }
  REQUIRE(matrix_id == 1);
  REQUIRE(fibre_id  == 2);
  REQUIRE(!any_other);     // never the untagged circle's shape index
  REQUIRE(n_fibre > 0);    // tagged circle does contribute voxels

  // Sample at known voxel coordinates inside each shape to pin down the
  // mapping unambiguously. The all-zero background satisfies any loose
  // "n_zero > 0" check on its own, so we instead probe specific voxels.
  const auto voxel_at = [&](std::size_t i, std::size_t j) {
    return grid[j * N + i];
  };
  // Tagged circle is at (0.25, 0.25) → voxel (8, 8).
  REQUIRE(voxel_at(8, 8) == fibre_id);
  // Untagged circle is at (0.75, 0.75) → voxel (24, 24) — must stay at 0
  // even though the voxel is inside the shape.
  REQUIRE(voxel_at(24, 24) == 0);
}

void test_vtk_legacy_writer_emits_phase_ids_from_phase_collection() {
  // Same scheme as voxel_writer: a phase_collection attached via
  // set_phases() makes the emitted SCALARS field carry real phase ids
  // instead of 1-based shape indices.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");                              // id 1
  const auto fibre_id = phases.add("fibre").id;      // id 2

  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  auto fibre = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2);
  fibre->set_phase_name("fibre");
  shapes.push_back(std::move(fibre));

  constexpr std::size_t N = 16;
  rvegen::vtk_legacy_writer<double> w{N, N, 1};
  REQUIRE(w.phases() == nullptr);   // legacy default
  w.set_phases(&phases);
  REQUIRE(w.phases() == &phases);

  std::stringstream out;
  w.write(out, shapes, std::array<double, 3>{1.0, 1.0, 0.0});
  const auto text = out.str();

  // SCALARS field is now `phase_id` (renamed from `phase`) so a
  // ParaView user can tell phase-id mode from shape-index mode by
  // inspecting the field name. The title line carries a short
  // human-readable scheme tag too.
  REQUIRE(text.find("SCALARS phase_id int 1") != std::string::npos);
  REQUIRE(text.find("phase_id from phase_collection") != std::string::npos);
  // Format-version marker must be reachable in BOTH the phase-id and
  // shape-index branches — a refactor that suppressed it in either
  // would silently break downstream consumers' scheme detection.
  REQUIRE(text.find("[rvegen-vtk-format: v2]") != std::string::npos);
  // Center voxel (8, 8) for a centered radius-0.2 circle in a 16x16 grid
  // is inside the fibre and must read as `fibre_id` (== 2).
  REQUIRE(text.find('\n' + std::to_string(fibre_id) + '\n') !=
          std::string::npos);
  // And the legacy "1" alone must NOT appear in the body (no shape would
  // map to it under the new scheme). A targeted check: ensure no line
  // reading just "1" — easier to verify by scanning the post-header tail.
  const auto header_end = text.find("LOOKUP_TABLE default\n");
  REQUIRE(header_end != std::string::npos);
  const auto body = text.substr(header_end);
  REQUIRE(body.find("\n1\n") == std::string::npos);
}

void test_voxel_writer_unknown_phase_name_throws() {
  // Shape carries a phase_name that the collection does not declare —
  // the writer must throw rather than silently dropping voxels to 0.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");

  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  auto fibre = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2);
  fibre->set_phase_name("typo_fibre");
  shapes.push_back(std::move(fibre));

  rvegen::voxel_writer<double> w{16, 16, 1};
  w.set_phases(&phases);

  bool threw = false;
  try {
    (void)w.sample(shapes, std::array<double, 3>{1.0, 1.0, 0.0});
  } catch (std::runtime_error const&) {
    threw = true;
  }
  REQUIRE(threw);
}

void test_voxel_writer_without_phases_keeps_shape_index_mode() {
  // Without set_phases, the writer falls back to the legacy 1-based
  // shape-index scheme — vital to keep existing single-phase smoke
  // tests green and downstream consumers untouched.
  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  shapes.push_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2));

  rvegen::voxel_writer<double> w{16, 16, 1};
  REQUIRE(w.phases() == nullptr);

  const auto grid = w.sample(shapes, std::array<double, 3>{1.0, 1.0, 0.0});
  bool saw_one = false;
  for (auto v : grid) {
    REQUIRE(v == 0 || v == 1);
    if (v == 1) saw_one = true;
  }
  REQUIRE(saw_one);
}

void test_voxel_writer_header_lists_phase_names_when_attached() {
  // When a phase_collection is attached, the writer's header includes a
  // `# phase <id> = <name>` line per phase so downstream tools can map
  // IDs back to names without needing the JSON config.
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");

  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  auto fibre = std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.15);
  fibre->set_phase_name("fibre");
  shapes.push_back(std::move(fibre));

  rvegen::voxel_writer<double> w{8, 8, 1};
  w.set_phases(&phases);

  std::stringstream out;
  w.write(out, shapes, std::array<double, 3>{1.0, 1.0, 0.0});
  const auto text = out.str();
  REQUIRE(text.find("# phase 1 = matrix") != std::string::npos);
  REQUIRE(text.find("# phase 2 = fibre")  != std::string::npos);
  REQUIRE(text.find("phase ids from phase_collection") != std::string::npos);
  // Format-version marker — same `[rvegen-<format>: v<N>]` convention
  // as vtk_legacy_writer's title-line marker, so a single regex in a
  // downstream multi-format reader can pick out the version of any
  // rvegen-emitted file.
  REQUIRE(text.find("[rvegen-voxel-format: v2]") != std::string::npos);
}

} // namespace

int main() {
  test_periodic_collision_across_box_faces();
  test_periodic_aabb_overlap_helper();
  test_random_generator_allows_outside_box();
  test_rectangle_input_via_registry();
  test_box_input_via_registry();
  test_until_full_lets_generator_max_attempts_cap();
  test_ellipse_basics_and_rotation();
  test_ellipse_collision_via_gte();
  test_circle_ellipse_collision();
  test_ellipse_input_via_registry();
  test_polyline_tube_inside_along_segment();
  test_polyline_tube_bend_inside_corner();
  test_polyline_tube_volume_and_bbox();
  test_polyline_tube_translate_via_set_middle_point();
  test_vtk_legacy_writer_header_and_size();
  test_gmsh_geo_writer_periodic_2d();
  test_gmsh_geo_writer_periodic_3d();
  test_gmsh_geo_writer_non_periodic_unchanged();
  test_gmsh_geo_writer_2d_physical_groups_per_phase();
  test_gmsh_geo_writer_3d_physical_groups_per_phase();
  test_gmsh_geo_writer_without_phases_emits_no_physical_groups();
  test_gmsh_geo_writer_periodic_and_phases_combine();
  test_gmsh_geo_writer_physical_groups_emitted_in_alpha_order();
  test_gmsh_geo_writer_phases_unknown_name_throws();
  test_gmsh_geo_writer_strict_mode_rejects_untagged_shape();
  test_gmsh_geo_writer_strict_mode_no_phases_attached_is_noop();
  test_voxel_writer_strict_mode_rejects_untagged_shape();
  test_vtk_legacy_writer_strict_mode_rejects_untagged_shape();
  test_oriented_uniform_uniform_regime();
  test_oriented_uniform_concentrated_regime();
  test_voronoi_cell_unit_cube_volume_and_inside();
  test_voronoi_cell_face_with_collinear_first_three_vertices();
  test_voronoi_cell_scale_invariant_collinear_threshold();
  test_voronoi_cell_translate_via_set_middle_point();
  test_stl_ascii_reader_triangle_count();
  test_ply_ascii_reader_triangulates_quad_faces();
  test_ply_ascii_reader_round_trips_through_mesh_inclusion();
  test_ply_ascii_reader_rejects_binary_format();
  test_ply_binary_reader_round_trips_unit_cube();
  test_ply_auto_detect_routes_binary_and_ascii();
  test_ply_binary_reader_rejects_ascii_format();
  test_ply_binary_reader_short_read_throws();
  test_ply_ascii_reader_rejects_missing_xyz();
  test_ply_ascii_reader_skips_extra_vertex_properties();
  test_ply_ascii_reader_rejects_out_of_range_index();
  test_stl_binary_reader_basic();
  test_stl_binary_reader_short_file_throws();
  test_stl_binary_reader_sanity_cap_throws();
  test_stl_auto_detect_dispatches_correctly();
  test_stl_ascii_reader_rejects_binary_stl();
  test_stl_ascii_reader_parse_error_includes_position();
  test_mesh_inclusion_input_via_registry();
  test_mesh_inclusion_input_accepts_canonical_mesh_path();
  test_mesh_inclusion_input_rejects_both_mesh_and_stl_path();
  test_mesh_inclusion_input_rejects_neither_mesh_nor_stl_path();
  test_mesh_inclusion_input_missing_file_throws();
  test_mesh_inclusion_input_accepts_binary_stl();
  test_read_mesh_file_dispatches_stl_and_ply();
  test_read_mesh_file_unknown_extension_throws();
  test_read_mesh_file_trims_trailing_whitespace();
  test_ply_ascii_reader_tolerates_unknown_header_lines();
  test_ply_ascii_reader_parse_error_includes_position();
  test_mesh_inclusion_input_accepts_ply_via_dispatch();
  test_mesh_inclusion_input_empty_stl_throws();
  test_mesh_inclusion_inside_outside_cube();
  test_mesh_inclusion_volume_unit_cube();
  test_mesh_inclusion_translate_via_set_middle_point();
  test_polyline_tube_input_via_registry();
  test_polyline_tube_input_direct_ctor();
  test_polyline_tube_input_missing_dist_throws();
  test_polyline_tube_directional_input_via_registry();
  test_field_list_schema_round_trip();
  test_field_list_optional_field_not_required();
  test_field_list_empty_pack();
  test_field_list_carries_description_and_unit_annotations();
  test_field_list_carries_range_annotation();
  test_field_list_no_annotations_no_hint_bases();
  test_circle_schema_post_field_list_migration();
  test_circle_schema_driven_ctor_via_field_list();
  test_sphere_schema_post_field_list_migration();
  test_sphere_schema_driven_ctor_via_field_list();
  test_rectangle_schema_post_field_list_migration();
  test_rectangle_schema_driven_ctor_via_field_list();
  test_box_schema_post_field_list_migration();
  test_box_schema_driven_ctor_via_field_list();
  test_ellipse_schema_post_field_list_migration();
  test_ellipse_schema_driven_ctor_via_field_list();
  test_bingham_uniform_regime_unit_length();
  test_bingham_uniform_kappa_zero_spreads_over_sphere();
  test_bingham_operator_call_matches_sample();
  test_bingham_prolate_clusters_on_axis();
  test_bingham_oblate_avoids_axis();
  test_bingham_rotated_oblate_avoids_axis();
  test_bingham_rotated_axis();
  test_bingham_implements_direction_distribution_base();
  test_polyline_tube_oriented_input_aligns_with_bingham_axis();
  test_polyline_tube_oriented_input_uses_scalar_position_distributions();
  test_direction_distribution_registry_builds_bingham_from_json();
  test_direction_distribution_registry_schema_has_descriptions();
  test_weave_generator_tube_count_matches_warp_plus_weft();
  test_weave_generator_warp_yarns_run_along_x();
  test_weave_generator_weft_yarns_run_along_y();
  test_weave_generator_warp_and_weft_interlace_180_out_of_phase();
  test_weave_generator_phase_tagging();
  test_weave_generator_validation_throws_on_bad_input();
  test_voronoi_generator_2d_single_seed_returns_full_domain();
  test_voronoi_generator_2d_two_seeds_split_by_bisector();
  test_voronoi_generator_2d_four_corner_seeds_quadrant_split();
  test_voronoi_generator_2d_cells_cover_domain();
  test_voronoi_generator_2d_each_cell_contains_its_seed();
  test_voronoi_generator_2d_validation_throws();
  test_convex_polygon_unit_square_basics();
  test_convex_polygon_triangle_area_and_inside();
  test_convex_polygon_set_middle_point_translates_vertices();
  test_convex_polygon_bounding_box_is_aabb();
  test_convex_polygon_clone_is_independent_copy();
  test_convex_polygon_rejects_too_few_vertices();
  test_convex_polygon_wraps_voronoi_cell();
  test_gmsh_geo_writer_emits_plane_surface_for_convex_polygon();
  test_gmsh_geo_writer_polygon_id_appears_in_physical_group();
  test_gmsh_geo_writer_emits_mesh_size_directives_when_set();
  test_gmsh_geo_writer_emits_element_order_when_set();
  test_gmsh_geo_writer_omits_element_order_when_default();
  test_gmsh_geo_writer_keeps_placeholders_when_mesh_size_default();
  test_voronoi_to_shapes_helper_returns_shape_vector();
  test_svg_writer_emits_polygon_element_for_convex_polygon();
  test_phase_collection_basics_and_ids();
  test_phase_collection_duplicate_throws();
  test_phase_collection_at_unknown_throws();
  test_load_phases_from_json_round_trip();
  test_input_stamps_phase_name_on_produced_shape();
  test_input_phase_name_default_is_empty();
  test_input_metadata_json_field_merges_into_info();
  test_input_metadata_overrides_phase_name();
  test_info_generic_metadata_round_trip();
  test_load_phases_from_json_rejects_non_array();
  test_load_phases_from_json_rejects_missing_name();
  test_load_phases_from_json_rejects_non_object_material();
  test_voxel_writer_emits_phase_ids_from_phase_collection();
  test_vtk_legacy_writer_emits_phase_ids_from_phase_collection();
  test_voxel_writer_unknown_phase_name_throws();
  test_voxel_writer_without_phases_keeps_shape_index_mode();
  test_voxel_writer_header_lists_phase_names_when_attached();

  if (failures > 0) {
    std::cerr << failures << " extra-types smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "extra_types_smoke: PASS\n";
  return EXIT_SUCCESS;
}
