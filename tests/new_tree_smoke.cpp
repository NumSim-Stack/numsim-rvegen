// Phase 2 smoke test for the new include/rvegen/ tree.
//
// Validates: construction, shape_base contract (area/volume/expansion/middle
// point/bounding box), and the compile-time collision_details overload set.
//
// The new tree is JSON-naive; this test uses pure C++ APIs.

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>

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

template <typename T>
bool nearly_equal(T a, T b, T tol = T{1e-9}) {
  return std::fabs(a - b) <= tol;
}

void test_circle_basics() {
  rvegen::circle<double> c{1.0, 2.0, 3.0};
  REQUIRE(c(0) == 1.0);
  REQUIRE(c(1) == 2.0);
  REQUIRE(c.radius == 3.0);
  REQUIRE(nearly_equal(c.area(), 9.0 * std::numbers::pi_v<double>));
  REQUIRE(c.volume() == 0.0);

  auto exp = c.max_expansion();
  REQUIRE(exp[0] == 3.0 && exp[1] == 3.0 && exp[2] == 0.0);

  auto mp = c.get_middle_point();
  REQUIRE(mp[0] == 1.0 && mp[1] == 2.0 && mp[2] == 0.0);

  c.make_bounding_box();
  auto* bb = c.bounding_box();
  REQUIRE(bb != nullptr);
  REQUIRE(bb->top_point()[0] == 4.0);    // x + r
  REQUIRE(bb->bottom_point()[0] == -2.0); // x - r
}

void test_sphere_basics() {
  rvegen::sphere<double> s{0.0, 0.0, 0.0, 2.0};
  REQUIRE(s.area() == 0.0);
  // 4/3 * pi * r^3
  REQUIRE(nearly_equal(s.volume(),
                       4.0 / 3.0 * std::numbers::pi_v<double> * 8.0));

  auto exp = s.max_expansion();
  REQUIRE(exp[0] == 2.0 && exp[1] == 2.0 && exp[2] == 2.0);

  s.make_bounding_box();
  auto* bb = s.bounding_box();
  REQUIRE(bb != nullptr);
  REQUIRE(bb->top_point()[2] == 2.0);
  REQUIRE(bb->bottom_point()[2] == -2.0);
}

void test_collision_circle_circle() {
  rvegen::circle<double> a{0.0, 0.0, 1.0};
  rvegen::circle<double> overlapping{1.5, 0.0, 1.0};   // centres 1.5 apart, radii sum 2 → overlap
  rvegen::circle<double> separate{3.0, 0.0, 1.0};      // centres 3.0 apart, radii sum 2 → clear
  rvegen::circle<double> touching{2.0, 0.0, 1.0};      // centres 2.0 apart, radii sum 2 → exact touch

  REQUIRE(rvegen::collision_details(a, overlapping) == true);
  REQUIRE(rvegen::collision_details(a, separate) == false);
  REQUIRE(rvegen::collision_details(a, touching) == true); // >= so touching counts as collision
}

void test_collision_sphere_sphere() {
  rvegen::sphere<double> a{0.0, 0.0, 0.0, 1.0};
  rvegen::sphere<double> overlapping{0.0, 0.0, 1.5, 1.0};
  rvegen::sphere<double> separate{0.0, 0.0, 3.0, 1.0};

  REQUIRE(rvegen::collision_details(a, overlapping) == true);
  REQUIRE(rvegen::collision_details(a, separate) == false);
}

void test_rectangle_basics() {
  rvegen::rectangle<double> r{1.0, 2.0, 4.0, 6.0};
  REQUIRE(r(0) == 1.0 && r(1) == 2.0);
  REQUIRE(r.width() == 4.0 && r.height() == 6.0);
  REQUIRE(r.area() == 24.0);
  REQUIRE(r.volume() == 0.0);

  auto exp = r.max_expansion();
  REQUIRE(exp[0] == 2.0 && exp[1] == 3.0 && exp[2] == 0.0);

  r.make_bounding_box();
  auto* bb = r.bounding_box();
  REQUIRE(bb->top_point()[0] == 3.0);    // 1 + 2
  REQUIRE(bb->bottom_point()[1] == -1.0); // 2 - 3
}

void test_box_basics() {
  rvegen::box<double> b{0.0, 0.0, 0.0, 2.0, 4.0, 6.0};
  REQUIRE(b.area() == 0.0);
  REQUIRE(b.volume() == 48.0);

  auto exp = b.max_expansion();
  REQUIRE(exp[0] == 1.0 && exp[1] == 2.0 && exp[2] == 3.0);

  b.make_bounding_box();
  auto* bb = b.bounding_box();
  REQUIRE(bb->top_point()[2] == 3.0);
  REQUIRE(bb->bottom_point()[2] == -3.0);
}

void test_collision_rectangle_rectangle() {
  rvegen::rectangle<double> a{0.0, 0.0, 2.0, 2.0};       // [-1,1] × [-1,1]
  rvegen::rectangle<double> overlapping{1.5, 0.0, 2.0, 2.0};  // [0.5,2.5] × [-1,1]
  rvegen::rectangle<double> separate{5.0, 0.0, 2.0, 2.0};     // [4,6] × [-1,1]

  REQUIRE(rvegen::collision_details(a, overlapping) == true);
  REQUIRE(rvegen::collision_details(a, separate) == false);
}

void test_collision_box_box() {
  rvegen::box<double> a{0.0, 0.0, 0.0, 2.0, 2.0, 2.0};
  rvegen::box<double> overlapping{0.0, 0.0, 1.0, 2.0, 2.0, 2.0};
  rvegen::box<double> separate{5.0, 5.0, 5.0, 1.0, 1.0, 1.0};

  REQUIRE(rvegen::collision_details(a, overlapping) == true);
  REQUIRE(rvegen::collision_details(a, separate) == false);
}

void test_collision_circle_rectangle() {
  rvegen::rectangle<double> rect{0.0, 0.0, 4.0, 2.0};   // [-2,2] × [-1,1]
  rvegen::circle<double> inside{0.0, 0.0, 0.5};         // centre inside rect
  rvegen::circle<double> grazing{2.5, 0.0, 0.5};        // centre 0.5 away from edge → touches
  rvegen::circle<double> clear{4.0, 0.0, 0.5};          // far enough away

  REQUIRE(rvegen::collision_details(inside,  rect) == true);
  REQUIRE(rvegen::collision_details(grazing, rect) == true);
  REQUIRE(rvegen::collision_details(clear,   rect) == false);

  // Symmetry: rect × circle delegates to circle × rect.
  REQUIRE(rvegen::collision_details(rect, clear) == false);
}

void test_collision_sphere_box() {
  rvegen::box<double> b{0.0, 0.0, 0.0, 2.0, 2.0, 2.0};
  rvegen::sphere<double> overlapping{1.5, 0.0, 0.0, 0.6}; // poked into +x face
  rvegen::sphere<double> clear{5.0, 0.0, 0.0, 0.5};

  REQUIRE(rvegen::collision_details(overlapping, b) == true);
  REQUIRE(rvegen::collision_details(clear,       b) == false);
  // Symmetry.
  REQUIRE(rvegen::collision_details(b, overlapping) == true);
}

// Note: shapes are not literal types post-Phase-11 because they inherit from
// gte primitives whose default ctors aren't constexpr. The collision_details
// overloads for circle/sphere/rect/box are still cheap and fully inlinable
// at runtime.

void test_gmsh_geo_writer_2d() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.3, 0.4, 0.05));
  shapes.emplace_back(std::make_unique<rvegen::rectangle<double>>(0.7, 0.6, 0.1, 0.2));

  std::stringstream out;
  rvegen::gmsh_geo_writer<double>{}.write(out, shapes, {1.0, 1.0, 0.0});

  const auto txt = out.str();
  REQUIRE(txt.find("Rectangle(1)") != std::string::npos); // domain
  REQUIRE(txt.find("Disk(2)")      != std::string::npos); // circle inclusion
  REQUIRE(txt.find("Rectangle(3)") != std::string::npos); // rectangle inclusion
  REQUIRE(txt.find("Box(")         == std::string::npos); // no 3D entities
}

void test_is_inside() {
  rvegen::circle<double> c{0.5, 0.5, 0.1};
  REQUIRE(c.is_inside({0.5, 0.5, 0.0}) == true);   // centre
  REQUIRE(c.is_inside({0.55, 0.5, 0.0}) == true);  // within radius
  REQUIRE(c.is_inside({0.7, 0.5, 0.0}) == false);  // outside

  rvegen::sphere<double> s{0.0, 0.0, 0.0, 1.0};
  REQUIRE(s.is_inside({0.0, 0.0, 0.0}) == true);
  REQUIRE(s.is_inside({0.5, 0.5, 0.5}) == true);   // sqrt(0.75) < 1
  REQUIRE(s.is_inside({1.0, 1.0, 1.0}) == false);

  rvegen::rectangle<double> r{0.0, 0.0, 2.0, 4.0};
  REQUIRE(r.is_inside({0.0, 0.0, 0.0}) == true);
  REQUIRE(r.is_inside({1.0, 2.0, 0.0}) == true);   // corner (touches)
  REQUIRE(r.is_inside({1.5, 0.0, 0.0}) == false);
}

void test_voxel_writer_2d() {
  // 2D RVE: one circle inclusion at the centre of a unit box, sampled on a
  // 16×16 grid. Voxel at (0.5, 0.5) must be marked 1 (the only inclusion);
  // a corner voxel must be 0 (matrix).
  rvegen::voxel_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.2));

  rvegen::voxel_writer<double> writer{16, 16, 1};
  const auto grid = writer.sample(shapes, {1.0, 1.0, 0.0});
  REQUIRE(grid.size() == 16u * 16u);

  auto idx = [](std::size_t i, std::size_t j) { return j * 16 + i; };
  REQUIRE(grid[idx(8, 8)] == 1);   // centre — inside circle
  REQUIRE(grid[idx(0, 0)] == 0);   // corner — matrix
  REQUIRE(grid[idx(15, 15)] == 0); // opposite corner — matrix
}

void test_voxel_writer_3d() {
  // 3D RVE: one sphere at centre of unit cube, 8³ grid.
  rvegen::voxel_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.25));

  rvegen::voxel_writer<double> writer{8, 8, 8};
  const auto grid = writer.sample(shapes, {1.0, 1.0, 1.0});
  REQUIRE(grid.size() == 8u * 8u * 8u);

  auto idx = [](std::size_t i, std::size_t j, std::size_t k) { return (k * 8 + j) * 8 + i; };
  REQUIRE(grid[idx(4, 4, 4)] == 1); // centre voxel — inside sphere
  REQUIRE(grid[idx(0, 0, 0)] == 0); // corner — matrix
}

void test_gmsh_geo_writer_3d() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.1));
  shapes.emplace_back(std::make_unique<rvegen::box<double>>(0.2, 0.2, 0.2, 0.1, 0.1, 0.1));

  std::stringstream out;
  rvegen::gmsh_geo_writer<double>{}.write(out, shapes, {1.0, 1.0, 1.0});

  const auto txt = out.str();
  REQUIRE(txt.find("Box(1)")    != std::string::npos); // domain
  REQUIRE(txt.find("Sphere(2)") != std::string::npos);
  REQUIRE(txt.find("Box(3)")    != std::string::npos);
  REQUIRE(txt.find("Disk(")     == std::string::npos); // no 2D entities
}

} // namespace

int main() {
  test_circle_basics();
  test_sphere_basics();
  test_rectangle_basics();
  test_box_basics();
  test_collision_circle_circle();
  test_collision_sphere_sphere();
  test_collision_rectangle_rectangle();
  test_collision_box_box();
  test_collision_circle_rectangle();
  test_collision_sphere_box();
  test_is_inside();
  test_gmsh_geo_writer_2d();
  test_gmsh_geo_writer_3d();
  test_voxel_writer_2d();
  test_voxel_writer_3d();

  if (failures > 0) {
    std::cerr << failures << " new-tree smoke failure(s)\n";
    return EXIT_FAILURE;
  }
  std::cout << "new_tree_smoke: PASS\n";
  return EXIT_SUCCESS;
}
