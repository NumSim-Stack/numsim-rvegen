// Voronoi robustness regression tests (#79): scale-invariance,
// coincident-seed policy, boundary seeds, index alignment, shape guards.

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <tuple>
#include <stdexcept>
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

template <typename F> bool throws_runtime_error(F&& f) {
  try { f(); } catch (std::runtime_error const&) { return true; }
  return false;
}

std::vector<std::array<double, 3>> seeds3(std::size_t n, double L,
                                          unsigned seed) {
  std::mt19937 eng{seed};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  std::vector<std::array<double, 3>> s;
  for (std::size_t i = 0; i < n; ++i)
    s.push_back({u(eng) * L, u(eng) * L, u(eng) * L});
  return s;
}
std::vector<std::array<double, 2>> seeds2(std::size_t n, double L,
                                          unsigned seed) {
  std::mt19937 eng{seed};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  std::vector<std::array<double, 2>> s;
  for (std::size_t i = 0; i < n; ++i) s.push_back({u(eng) * L, u(eng) * L});
  return s;
}

double volume_sum(std::vector<std::array<double, 3>> const& s, double L) {
  rvegen::voronoi_generator_3d<double> gen{L, L, L, s};
  double total = 0.0;
  for (auto const& c : gen.build()) total += c->volume();
  return total / (L * L * L);
}
double area_sum(std::vector<std::array<double, 2>> const& s, double L) {
  rvegen::voronoi_generator_2d<double> gen{L, L, s};
  double total = 0.0;
  for (auto const& p : gen.build()) total += rvegen::polygon_area(p);
  return total / (L * L);
}

void test_partition_of_unity_across_scales() {
  for (double L : {1e-6, 1e-4, 1.0, 1e3}) {
    REQUIRE(std::abs(volume_sum(seeds3(15, L, 7), L) - 1.0) < 1e-8);
    REQUIRE(std::abs(area_sum(seeds2(40, L, 11), L) - 1.0) < 1e-10);
  }
}

void test_coincident_seeds_rejected() {
  REQUIRE(throws_runtime_error([] {
    rvegen::voronoi_generator_3d<double> g{
        1, 1, 1, {{.3, .3, .3}, {.3, .3, .3}, {.7, .7, .7}}};
  }));
  REQUIRE(throws_runtime_error([] {
    rvegen::voronoi_generator_2d<double> g{1, 1, {{.3, .3}, {.7, .7}, {.3, .3}}};
  }));
  // Same at physical scale: 1e-12 apart in a 1e-6 domain is coincident.
  REQUIRE(throws_runtime_error([] {
    rvegen::voronoi_generator_3d<double> g{
        1e-6, 1e-6, 1e-6, {{3e-7, 3e-7, 3e-7}, {3e-7 + 1e-16, 3e-7, 3e-7}}};
  }));
  // Outside the domain is rejected too.
  REQUIRE(throws_runtime_error([] {
    rvegen::voronoi_generator_2d<double> g{1, 1, {{.5, .5}, {1.5, .5}}};
  }));
}

void test_nearly_coincident_seeds_still_partition() {
  for (double L : {1e-6, 1.0, 1e3}) {
    const double eps = 1e-5 * L;   // far above the coincidence tolerance
    auto s3 = seeds3(6, L, 3);
    s3.push_back({s3[0][0] + eps, s3[0][1], s3[0][2]});
    REQUIRE(std::abs(volume_sum(s3, L) - 1.0) < 1e-6);
    auto s2 = seeds2(8, L, 5);
    s2.push_back({s2[0][0] + eps, s2[0][1] - eps});
    REQUIRE(std::abs(area_sum(s2, L) - 1.0) < 1e-8);
  }
}

void test_boundary_seeds() {
  for (double L : {1e-6, 1.0, 1e3}) {
    // Seeds on faces, edges and a corner.
    std::vector<std::array<double, 3>> s3{
        {0, .5 * L, .5 * L}, {L, .5 * L, .5 * L}, {.5 * L, 0, 0},
        {.5 * L, L, L},      {0, 0, 0},           {.4 * L, .5 * L, .6 * L}};
    REQUIRE(std::abs(volume_sum(s3, L) - 1.0) < 1e-8);
    std::vector<std::array<double, 2>> s2{
        {0, 0}, {L, L}, {0, L}, {L, 0}, {.5 * L, 0}, {.5 * L, .5 * L}};
    REQUIRE(std::abs(area_sum(s2, L) - 1.0) < 1e-10);
  }
}

void test_cell_index_matches_seed_index() {
  const double L = 1.0;
  const auto s = seeds2(12, L, 21);
  rvegen::voronoi_generator_2d<double> gen{L, L, s};
  auto shapes = rvegen::voronoi_to_shapes(gen);
  REQUIRE(shapes.size() == s.size());
  for (std::size_t i = 0; i < s.size(); ++i)
    REQUIRE(shapes[i]->is_inside({s[i][0], s[i][1], 0.0}));

  const auto s3 = seeds3(10, L, 4);
  rvegen::voronoi_generator_3d<double> gen3{L, L, L, s3};
  auto cells = gen3.build();
  REQUIRE(cells.size() == s3.size());
  for (std::size_t i = 0; i < s3.size(); ++i)
    REQUIRE(cells[i]->is_inside(s3[i]));
}

void test_voronoi_cell_rejects_bad_face_index() {
  gte::Vector<3, double> v;
  v[0] = 0; v[1] = 0; v[2] = 0;
  bool threw = false;
  try {
    rvegen::voronoi_cell<double> c{{v}, {{0, 1, 2}}};
  } catch (std::out_of_range const&) { threw = true; }
  REQUIRE(threw);
}

void test_empty_polygon_bounding_box_and_expansion() {
  rvegen::convex_polygon<double> p;   // default-constructed
  p.make_bounding_box();              // must not read _vertices[0]
  REQUIRE(p.max_expansion()[0] == 0.0);
  REQUIRE(!p.is_inside({0, 0, 0}));

  // max_expansion is the AABB half-extent (matches voronoi_cell).
  rvegen::convex_polygon<double> q{{{0, 0}, {4, 0}, {4, 2}, {0, 2}}};
  auto e = q.max_expansion();
  REQUIRE(std::abs(e[0] - 2.0) < 1e-12 && std::abs(e[1] - 1.0) < 1e-12);
}

void test_polygon_is_inside_scale_relative() {
  for (double L : {1e-6, 1.0, 1e3}) {
    rvegen::convex_polygon<double> q{{{0, 0}, {L, 0}, {L, L}, {0, L}}};
    REQUIRE(q.is_inside({.5 * L, .5 * L, 0}));
    REQUIRE(q.is_inside({.5 * L, 0, 0}));                  // on the edge
    REQUIRE(!q.is_inside({.5 * L, -1e-3 * L, 0}));         // clearly outside
  }
}

void test_voronoi_cell_volume_centroid() {
  // Wedge (triangular prism) whose volume centroid differs from the
  // vertex mean only when vertices are unevenly distributed: use a
  // pyramid with apex (0,0,3) over the unit square base [0,1]^2 at
  // z=0: volume centroid z = h/4 = 0.75, vertex mean z = 0.6.
  using V = gte::Vector<3, double>;
  auto mk = [](double x, double y, double z) {
    V v; v[0] = x; v[1] = y; v[2] = z; return v;
  };
  std::vector<V> verts{mk(0, 0, 0), mk(1, 0, 0), mk(1, 1, 0), mk(0, 1, 0),
                       mk(.5, .5, 3)};
  std::vector<std::vector<std::size_t>> faces{
      {0, 3, 2, 1}, {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}};
  rvegen::voronoi_cell<double> c{verts, faces};
  auto m = c.get_middle_point();
  REQUIRE(std::abs(m[0] - .5) < 1e-12);
  REQUIRE(std::abs(m[2] - 0.75) < 1e-12);
  c.set_middle_point({1, 1, 1});
  auto m2 = c.get_middle_point();
  REQUIRE(std::abs(m2[2] - 1.0) < 1e-12);
}

} // namespace

// Review follow-up (#93): tolerance scales with the SHORTEST edge, and
// near-degenerate (jittered lattice) inputs still partition the box.
void test_thin_slab_and_jittered_lattice() {
  // Thin slab, random seeds.
  for (double Lz : {1e-2, 1e-4}) {
    std::mt19937 eng{7};
    std::uniform_real_distribution<double> u{0.0, 1.0};
    std::vector<std::array<double, 3>> s;
    for (int i = 0; i < 12; ++i) s.push_back({u(eng), u(eng), u(eng) * Lz});
    rvegen::voronoi_generator_3d<double> gen{1.0, 1.0, Lz, s};
    double total = 0.0;
    for (auto const& c : gen.build()) total += c->volume();
    REQUIRE(std::abs(total / Lz - 1.0) < 1e-9);
  }
  // Regular lattice in a 1e-4 slab (exactly degenerate vertices).
  auto lattice = [](int m, double jit, double Lz) {
    std::mt19937 eng{1};
    std::uniform_real_distribution<double> nj{-1.0, 1.0};
    std::vector<std::array<double, 3>> s;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < m; ++j)
        for (int k = 0; k < m; ++k)
          s.push_back({(i + .5) / m + jit * nj(eng), (j + .5) / m + jit * nj(eng),
                       ((k + .5) / m + jit * nj(eng)) * Lz});
    return s;
  };
  for (auto const& [m, jit, Lz] :
       {std::tuple{3, 0.0, 1e-4}, std::tuple{4, 1e-9, 1.0},
        std::tuple{4, 1e-8, 1.0}, std::tuple{4, 1e-10, 1.0}}) {
    auto s = lattice(m, jit, Lz);
    rvegen::voronoi_generator_3d<double> gen{1.0, 1.0, Lz, s};
    auto cells = gen.build();
    double total = 0.0;
    for (std::size_t i = 0; i < cells.size(); ++i) {
      total += cells[i]->volume();
      REQUIRE(cells[i]->is_inside(s[i]));
    }
    REQUIRE(std::abs(total / Lz - 1.0) < 1e-6);
  }
}

// An empty/collapsed polygon placeholder must not emit `Line Loop() = {}`.
void test_gmsh_skips_empty_polygon() {
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.push_back(std::make_unique<rvegen::convex_polygon<double>>());
  shapes.push_back(std::make_unique<rvegen::convex_polygon<double>>(
      std::vector<std::array<double, 2>>{{0, 0}, {1, 0}, {1, 1}}));
  std::ostringstream out;
  rvegen::gmsh_geo_writer<double>{}.write(out, shapes, {1.0, 1.0, 0.0});
  REQUIRE(out.str().find("Line Loop(_ll) = {};") == std::string::npos);
  REQUIRE(out.str().find("Plane Surface") != std::string::npos);
}

void test_max_expansion_encloses_asymmetric_shapes() {
  // middle +/- max_expansion must contain every vertex even when the
  // centroid is far from the AABB centre (#79 review).
  const std::vector<std::array<double, 2>> tri{{0.1, 0.2}, {0.7, 0.5}, {0.1, 0.8}};
  const std::vector<std::array<double, 2>> sliver{{0, 0}, {10, 0}, {0.1, 0.1}};
  for (auto const& pts : {tri, sliver}) {
    rvegen::convex_polygon<double> p{pts};
    const auto m = p.get_middle_point();
    const auto e = p.max_expansion();
    for (auto const& v : pts) {
      REQUIRE(v[0] >= m[0] - e[0] - 1e-14 && v[0] <= m[0] + e[0] + 1e-14);
      REQUIRE(v[1] >= m[1] - e[1] - 1e-14 && v[1] <= m[1] + e[1] + 1e-14);
    }
    p.make_bounding_box();   // the bounding box stays the exact AABB
    auto const* bb = p.bounding_box();
    REQUIRE(bb != nullptr);
    double mnx = 1e300, mxx = -1e300;
    for (auto const& v : pts) { mnx = std::min(mnx, v[0]); mxx = std::max(mxx, v[0]); }
    REQUIRE(bb->bottom_point()[0] == mnx && bb->top_point()[0] == mxx);
  }

  // Strongly asymmetric 3D cell: a tetrahedron with a long spike along x.
  using vec = gte::Vector<3, double>;
  auto mk = [](double x, double y, double z) { vec v; v[0] = x; v[1] = y; v[2] = z; return v; };
  std::vector<vec> verts{mk(0, 0, 0), mk(10, 0, 0), mk(0, 1, 0), mk(0, 0, 1)};
  rvegen::voronoi_cell<double> c{verts, {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}};
  const auto m = c.get_middle_point();
  const auto e = c.max_expansion();
  for (auto const& v : verts)
    for (int k = 0; k < 3; ++k)
      REQUIRE(v[k] >= m[k] - e[k] - 1e-14 && v[k] <= m[k] + e[k] + 1e-14);
}

void test_gmsh_overlap_detected_for_asymmetric_polygons() {
  // Triangle apex at x = 0.7 overlaps a rectangle starting at x = 0.65; the
  // centroid-based extent used to miss the overlap.
  rvegen::gmsh_geo_writer<double>::shape_vector shapes;
  shapes.push_back(std::make_unique<rvegen::convex_polygon<double>>(
      std::vector<std::array<double, 2>>{{0.1, 0.2}, {0.7, 0.5}, {0.1, 0.8}}));
  shapes.push_back(std::make_unique<rvegen::convex_polygon<double>>(
      std::vector<std::array<double, 2>>{{0.65, 0.4}, {0.95, 0.4}, {0.95, 0.6}, {0.65, 0.6}}));
  std::ostringstream out;
  rvegen::gmsh_geo_writer<double>{}.write(out, shapes, {1.0, 1.0, 0.0});
  REQUIRE(out.str().find("_tool()") != std::string::npos);
}

int main() {
  test_partition_of_unity_across_scales();
  test_coincident_seeds_rejected();
  test_nearly_coincident_seeds_still_partition();
  test_boundary_seeds();
  test_cell_index_matches_seed_index();
  test_voronoi_cell_rejects_bad_face_index();
  test_empty_polygon_bounding_box_and_expansion();
  test_polygon_is_inside_scale_relative();
  test_voronoi_cell_volume_centroid();
  test_thin_slab_and_jittered_lattice();
  test_gmsh_skips_empty_polygon();
  test_max_expansion_encloses_asymmetric_shapes();
  test_gmsh_overlap_detected_for_asymmetric_polygons();
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "voronoi_robustness_test OK\n";
  return 0;
}
