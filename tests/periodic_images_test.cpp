// Periodic copies are linked to their primary (#121): a copy carries its
// lattice shift and shares the primary's inclusion id, and everything that
// must count an inclusion once (terminations, statistics, shape-index
// output, phase assignment) goes by that link instead of by where a
// centre lies.

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/pipeline.h"

namespace {
int failures = 0;
#define REQUIRE(cond)                                                    \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << "REQUIRE failed: " #cond " at " << __FILE__ << ":"    \
                << __LINE__ << "\n";                                     \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

using shape_vector = std::vector<std::unique_ptr<rvegen::shape_base<double>>>;
using json = nlohmann::json;
constexpr double pi = std::numbers::pi;

// Appends `primary` and its periodic copies; returns the number of copies.
std::size_t add_with_copies(shape_vector& v,
                            std::unique_ptr<rvegen::shape_base<double>> primary,
                            std::array<double, 3> const& box) {
  auto copies = rvegen::generate_periodic_wraps(*primary, box);
  const std::size_t n = copies.size();
  v.push_back(std::move(primary));
  for (auto& c : copies) v.push_back(std::move(c));
  return n;
}

std::size_t primaries(shape_vector const& v) {
  std::size_t n = 0;
  for (auto const& s : v) n += s->is_primary() ? 1 : 0;
  return n;
}

void test_links() {
  const std::array<double, 3> box{1.0, 2.0, 3.0};
  // Inside: no copies, no id, still a primary.
  rvegen::sphere<double> inside{0.5, 1.0, 1.5, 0.1};
  REQUIRE(inside.is_primary());
  REQUIRE(rvegen::generate_periodic_wraps(inside, box).empty());
  REQUIRE(!inside.inclusion_id().has_value());

  // Across 1, 2 and 3 faces: 1, 3 and 7 copies.
  struct item { std::array<double, 3> c; std::size_t copies; };
  for (auto const& it : {item{{0.05, 1.0, 1.5}, 1}, item{{0.95, 1.95, 1.5}, 3},
                         item{{0.05, 0.05, 2.95}, 7}}) {
    rvegen::sphere<double> p{it.c[0], it.c[1], it.c[2], 0.1};
    auto copies = rvegen::generate_periodic_wraps(p, box);
    REQUIRE(copies.size() == it.copies);
    REQUIRE(p.is_primary());
    REQUIRE(p.inclusion_id().has_value());
    std::set<std::array<int, 3>> shifts;
    for (auto const& c : copies) {
      REQUIRE(!c->is_primary());
      REQUIRE(c->inclusion_id() == p.inclusion_id());
      const auto sh = c->image_shift();
      shifts.insert(sh);
      // The shift is the lattice translation from the primary.
      const auto m = c->get_middle_point();
      for (int k = 0; k < 3; ++k) {
        REQUIRE(std::abs(sh[k]) <= 1);
        REQUIRE(std::abs(m[k] - (it.c[k] + sh[k] * box[k])) < 1e-14);
      }
      // A clone of a copy is still that copy.
      const auto again = c->clone();
      REQUIRE(again->image_shift() == sh);
      REQUIRE(again->inclusion_id() == p.inclusion_id());
    }
    REQUIRE(shifts.size() == it.copies);   // all different
  }
  // Two inclusions never share an id.
  rvegen::sphere<double> a{0.05, 1.0, 1.5, 0.1}, b{0.05, 1.5, 1.5, 0.1};
  (void)rvegen::generate_periodic_wraps(a, box);
  (void)rvegen::generate_periodic_wraps(b, box);
  REQUIRE(a.inclusion_id() != b.inclusion_id());
  // Wrapping the same primary again keeps its id.
  const auto id = a.inclusion_id();
  const auto second = rvegen::generate_periodic_wraps(a, box);
  REQUIRE(a.inclusion_id() == id);
  REQUIRE(second.size() == 1 && second[0]->inclusion_id() == id);
}

// A shape centred exactly on a face, an edge or a corner is counted once.
void test_on_face_counted_once() {
  const std::array<double, 3> box2{1.0, 1.0, 0.0};
  const double r = 0.1;
  for (auto const& c : std::vector<std::array<double, 2>>{
           {0.0, 0.5}, {1.0, 0.5}, {0.5, 0.0}, {0.5, 1.0},
           {0.0, 0.0}, {1.0, 1.0}, {0.0, 1.0}, {1.0, 0.0}}) {
    shape_vector v;
    const auto n = add_with_copies(
        v, std::make_unique<rvegen::circle<double>>(c[0], c[1], r), box2);
    const bool corner = (c[0] == 0.0 || c[0] == 1.0) && (c[1] == 0.0 || c[1] == 1.0);
    REQUIRE(n == (corner ? 3u : 1u));
    REQUIRE(primaries(v) == 1);
    REQUIRE(rvegen::number_of_inclusions<double>{1}(v));
    REQUIRE(!rvegen::number_of_inclusions<double>{2}(v));
    const double phi = rvegen::current_volume_fraction<double>(v, box2);
    REQUIRE(std::abs(phi - pi * r * r) < 1e-15);
    REQUIRE(rvegen::volume_fraction_in_domain<double>(v, box2) == phi);
    // The voxel grid agrees: the pieces add up to one disc, with one id.
    const std::size_t n_vox = 200;
    const auto grid = rvegen::sample_voxel_grid(v, box2, n_vox, n_vox, 1);
    std::size_t filled = 0;
    std::set<rvegen::phase_id> ids;
    for (auto g : grid) if (g != 0) { ++filled; ids.insert(g); }
    REQUIRE(ids.size() == 1);
    REQUIRE(std::abs(double(filled) / double(n_vox * n_vox) - pi * r * r) < 2e-3);
  }
  const std::array<double, 3> box3{1.0, 1.0, 1.0};
  for (auto const& c : std::vector<std::array<double, 3>>{
           {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {0.0, 0.5, 1.0}, {0.5, 0.5, 0.0}}) {
    shape_vector v;
    (void)add_with_copies(
        v, std::make_unique<rvegen::sphere<double>>(c[0], c[1], c[2], r), box3);
    REQUIRE(primaries(v) == 1);
    REQUIRE(std::abs(rvegen::current_volume_fraction<double>(v, box3) -
                     4.0 / 3.0 * pi * r * r * r) < 1e-15);
  }
}

// Shape-index voxel and VTK output: as many distinct ids as primaries.
void test_distinct_ids_equal_primaries() {
  const std::array<double, 3> box{1.0, 1.0, 1.0};
  shape_vector v;
  (void)add_with_copies(v, std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.12), box);    // inside
  (void)add_with_copies(v, std::make_unique<rvegen::sphere<double>>(0.03, 0.5, 0.2, 0.1), box);    // 1 face
  (void)add_with_copies(v, std::make_unique<rvegen::sphere<double>>(0.97, 0.96, 0.6, 0.1), box);   // 2 faces
  (void)add_with_copies(v, std::make_unique<rvegen::sphere<double>>(0.02, 0.04, 0.97, 0.1), box);  // 3 faces
  REQUIRE(v.size() == 1 + 2 + 4 + 8);
  REQUIRE(primaries(v) == 4);

  const std::size_t n = 40;
  std::set<rvegen::phase_id> ids;
  for (auto g : rvegen::sample_voxel_grid(v, box, n, n, n)) if (g != 0) ids.insert(g);
  REQUIRE(ids.size() == 4);

  rvegen::vtk_legacy_writer<double> vtk{n, n, n};
  std::stringstream out;
  vtk.write(out, v, box);
  const std::string text = out.str();
  const std::string marker = "LOOKUP_TABLE default\n";
  const auto pos = text.find(marker);
  REQUIRE(pos != std::string::npos);
  std::istringstream body{text.substr(pos == std::string::npos ? 0 : pos + marker.size())};
  std::set<int> vtk_ids;
  for (int x = 0; body >> x;) if (x != 0) vtk_ids.insert(x);
  REQUIRE(vtk_ids.size() == 4);

  // The same inclusions with the copies appended last (as the Voronoi
  // generators do) give the same count.
  shape_vector tail;
  shape_vector late;
  for (auto const& c : std::vector<std::array<double, 3>>{
           {0.03, 0.5, 0.2}, {0.97, 0.96, 0.6}, {0.02, 0.04, 0.97}}) {
    auto p = std::make_unique<rvegen::sphere<double>>(c[0], c[1], c[2], 0.1);
    for (auto& w : rvegen::generate_periodic_wraps(*p, box)) late.push_back(std::move(w));
    tail.push_back(std::move(p));
  }
  for (auto& w : late) tail.push_back(std::move(w));
  ids.clear();
  for (auto g : rvegen::sample_voxel_grid(tail, box, n, n, n)) if (g != 0) ids.insert(g);
  REQUIRE(ids.size() == 3);
  REQUIRE(primaries(tail) == 3);
}

// The counting rule, case by case.
void test_counting_rule() {
  const std::array<double, 3> box{1.0, 1.0, 0.0};
  // A primary centred OUTSIDE the box whose copy holds the part inside
  // counts (once); under a centre test it was the copy that counted.
  shape_vector v;
  (void)add_with_copies(v, std::make_unique<rvegen::circle<double>>(-0.02, 0.5, 0.1), box);
  REQUIRE(v.size() == 2);
  REQUIRE(rvegen::is_counted_in_rve(*v[0], box));
  REQUIRE(!rvegen::is_counted_in_rve(*v[1], box));
  REQUIRE(std::abs(rvegen::current_volume_fraction<double>(v, box) - pi * 0.01) < 1e-15);
  // g(r) takes the centre into the box: two inclusions 0.125 apart across
  // the face are seen at that distance, not lost.
  (void)add_with_copies(v, std::make_unique<rvegen::circle<double>>(0.105, 0.5, 0.02), box);
  const auto g = rvegen::radial_distribution_2d(v, box, 10);   // bins of 0.05
  REQUIRE(g.size() == 10);
  REQUIRE(g[2].g > 0.0);                 // the pair, in bin [0.1, 0.15)
  for (std::size_t b = 0; b < g.size(); ++b) if (b != 2) REQUIRE(g[b].g == 0.0);

  // Shapes that cross a face WITHOUT copies (what random_generator
  // produces): the one centred inside counts in full, the one centred
  // outside not at all, which estimates the clipped area without bias.
  shape_vector clipped;
  clipped.push_back(std::make_unique<rvegen::circle<double>>(0.02, 0.5, 0.1));
  clipped.push_back(std::make_unique<rvegen::circle<double>>(-0.02, 0.2, 0.1));
  for (auto& s : clipped) s->make_bounding_box();
  REQUIRE(rvegen::is_counted_in_rve(*clipped[0], box));
  REQUIRE(!rvegen::is_counted_in_rve(*clipped[1], box));
  REQUIRE(std::abs(rvegen::current_volume_fraction<double>(clipped, box) - pi * 0.01) < 1e-15);
  // Both are inclusions all the same.
  REQUIRE(rvegen::number_of_inclusions<double>{2}(clipped));
  REQUIRE(rvegen::number_of_inclusions<double>{2}.is_met(clipped, true));
}

// End to end: the periodic generator links what it places.
void test_periodic_generator_links() {
  auto cfg = json::parse(R"({
    "domain_box": [1.0, 1.0, 1.0], "seed": 11,
    "distributions": {"x": {"type": "uniform_real", "a": 0.0, "b": 1.0},
                      "r": {"type": "uniform_real", "a": 0.08, "b": 0.14}},
    "shapes": [{"type": "sphere_input", "pos_x_dist": "x", "pos_y_dist": "x",
                "pos_z_dist": "x", "radius_dist": "r"}],
    "generator": {"type": "periodic", "max_attempts": 200000},
    "termination": {"type": "number_of_inclusions", "target": 25}
  })");
  const auto r = rvegen::generate(cfg);
  REQUIRE(primaries(r.shapes) == 25);
  REQUIRE(r.shapes.size() > 25);   // some cross a face
  double vol = 0.0;
  for (std::size_t i = 0; i < r.shapes.size(); ++i) {
    auto const& s = *r.shapes[i];
    if (s.is_primary()) { vol += s.volume(); continue; }
    // A copy follows its primary and shares its id and size.
    REQUIRE(s.inclusion_id().has_value());
    std::size_t p = i;
    while (!r.shapes[p]->is_primary()) --p;
    REQUIRE(r.shapes[p]->inclusion_id() == s.inclusion_id());
    REQUIRE(r.shapes[p]->volume() == s.volume());
  }
  REQUIRE(std::abs(rvegen::current_volume_fraction<double>(r.shapes, r.domain_box) - vol) < 1e-14);
  // Voxel fraction of the wrapped RVE = analytic volume of the primaries.
  const std::size_t n = 96;
  std::size_t filled = 0;
  std::set<rvegen::phase_id> ids;
  for (auto g : rvegen::sample_voxel_grid(r.shapes, r.domain_box, n, n, n))
    if (g != 0) { ++filled; ids.insert(g); }
  REQUIRE(ids.size() == 25);
  REQUIRE(std::abs(double(filled) / double(n * n * n) - vol) < 0.01 * vol + 1e-3);
}

// A centre distribution that lies entirely outside the box, several box
// lengths away: the periodic generator must still fill THE BOX (found in
// review: the copies reach one lattice step, so such shapes used to leave
// nothing inside).
void test_periodic_generator_far_centres() {
  auto make = [](double lo, double hi) {
    return json{
        {"domain_box", {1.0, 1.0, 0.0}}, {"seed", 6},
        {"distributions", {{"x", {{"type", "uniform_real"}, {"a", lo}, {"b", hi}}},
                           {"r", {{"type", "uniform_real"}, {"a", 0.05}, {"b", 0.1}}}}},
        {"shapes", json::array({json{{"type", "circle_input"}, {"pos_x_dist", "x"},
                                     {"pos_y_dist", "x"}, {"radius_dist", "r"}}})},
        {"generator", {{"type", "periodic"}, {"max_attempts", 100000}}},
        {"termination", {{"type", "volume_fraction"}, {"target_fraction", 0.15}}}};
  };
  for (auto const& range : std::vector<std::array<double, 2>>{
           {-5.0, -3.0}, {3.25, 7.5}, {-0.5, 1.5}, {0.0, 1.0}}) {
    const auto r = rvegen::generate(make(range[0], range[1]));
    double area = 0.0;
    for (auto const& s : r.shapes) {
      if (!s->is_primary()) continue;
      area += s->area();
      const auto c = s->get_middle_point();
      REQUIRE(c[0] >= 0.0 && c[0] < 1.0 && c[1] >= 0.0 && c[1] < 1.0);
    }
    REQUIRE(area >= 0.15);
    REQUIRE(std::abs(rvegen::current_volume_fraction<double>(r.shapes, r.domain_box) - area) < 1e-14);
    // What is actually inside the box is that much material.
    const std::size_t n = 400;
    std::size_t filled = 0;
    for (auto g : rvegen::sample_voxel_grid(r.shapes, r.domain_box, n, n, 1))
      filled += g != 0 ? 1 : 0;
    REQUIRE(std::abs(double(filled) / double(n * n) - area) < 3e-3);
  }
}

// Periodic Voronoi: every grain once, and the grains fill the box exactly.
void test_voronoi_pieces() {
  for (int dim : {2, 3}) {
    json cfg = {{"domain_box", dim == 2 ? json{1.0, 1.0, 0.0} : json{1.0, 1.0, 1.0}},
                {"seed", 4},
                {"generator", {{"type", dim == 2 ? "voronoi_2d" : "voronoi_3d"},
                               {"n_seeds", 12}, {"periodic", true},
                               {"backend", "in_house"}}}};
    if (dim == 2) cfg["generator"].erase("backend");
    const auto r = rvegen::generate(cfg);
    REQUIRE(primaries(r.shapes) == 12);
    REQUIRE(r.shapes.size() > 12);
    REQUIRE(std::abs(rvegen::current_volume_fraction<double>(r.shapes, r.domain_box) - 1.0) < 1e-9);
    for (auto const& s : r.shapes) {
      if (s->is_primary()) continue;
      REQUIRE(s->inclusion_id().has_value());
      // The copy belongs to the grain with the same grain_id.
      bool found = false;
      for (auto const& p : r.shapes) {
        if (p->is_primary() && p->inclusion_id() == s->inclusion_id()) {
          found = p->info().template get<std::size_t>("grain_id") ==
                  s->info().template get<std::size_t>("grain_id");
        }
      }
      REQUIRE(found);
    }
  }
}
}  // namespace

int main() {
  test_links();
  test_on_face_counted_once();
  test_distinct_ids_equal_primaries();
  test_counting_rule();
  test_periodic_generator_links();
  test_periodic_generator_far_centres();
  test_voronoi_pieces();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "periodic_images_test OK\n";
  return EXIT_SUCCESS;
}
