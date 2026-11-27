// Hardening tests for the STL / PLY readers and mesh_inclusion
// validation (#80). Every malformed / hostile input must surface as a
// std::runtime_error with a useful message — never junk triangles,
// bad_alloc, or UB casts. All inputs are in-memory streams.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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

// Run `fn`; return the runtime_error message, or nullopt-ish "" with
// `threw == false` when nothing was thrown. Any other exception type
// counts as a failure (e.g. bad_alloc from an unchecked reserve).
template <typename F>
std::pair<bool, std::string> throws_runtime(F&& fn) {
  try {
    fn();
  } catch (std::runtime_error const& e) {
    return {true, e.what()};
  } catch (std::exception const& e) {
    std::cerr << "  unexpected exception type: " << e.what() << "\n";
    return {false, e.what()};
  }
  return {false, ""};
}

#define REQUIRE_THROWS_WITH(expr, needle)                                      \
  do {                                                                         \
    auto [threw_, what_] = throws_runtime([&] { (void)(expr); });              \
    REQUIRE(threw_);                                                           \
    if (threw_ && what_.find(needle) == std::string::npos) {                   \
      std::cerr << "  message was: " << what_ << "\n";                         \
      REQUIRE(what_.find(needle) != std::string::npos);                        \
    }                                                                          \
  } while (0)

using tri_list = std::vector<gte::Triangle3<double>>;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

constexpr char const* one_facet = R"(solid t
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
)";

// Unit cube [0,1]^3: 8 vertices, 12 outward-wound triangles.
constexpr std::array<std::array<double, 3>, 8> cube_v{{
    {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
    {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}}};
constexpr std::array<std::array<int, 3>, 12> cube_f{{
    {0, 2, 1}, {0, 3, 2},   // z = 0, normal -z
    {4, 5, 6}, {4, 6, 7},   // z = 1, normal +z
    {0, 1, 5}, {0, 5, 4},   // y = 0, normal -y
    {3, 7, 6}, {3, 6, 2},   // y = 1, normal +y
    {0, 4, 7}, {0, 7, 3},   // x = 0, normal -x
    {1, 2, 6}, {1, 6, 5}}}; // x = 1, normal +x

tri_list make_cube(double scale = 1.0, double shift = 0.0) {
  tri_list out;
  for (auto const& f : cube_f) {
    gte::Triangle3<double> t;
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) {
        t.v[c][k] = cube_v[f[c]][k] * scale + shift;
      }
    }
    out.push_back(t);
  }
  return out;
}

void flip(gte::Triangle3<double>& t) { std::swap(t.v[1], t.v[2]); }

void write_le_u32(std::ostream& o, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) o.put(static_cast<char>((v >> (8 * i)) & 0xff));
}
void write_le_f32(std::ostream& o, float v) {
  std::uint32_t bits;
  std::memcpy(&bits, &v, sizeof(bits));
  write_le_u32(o, bits);
}

constexpr char const* ply_tri_header_ascii =
    "ply\nformat ascii 1.0\n"
    "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
    "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
    "0 0 0\n1 0 0\n0 1 0\n";

// ---------------------------------------------------------------------------
// ASCII STL
// ---------------------------------------------------------------------------

void test_stl_valid_single_facet_parses() {
  std::stringstream ss{std::string{one_facet} + "endsolid t\n"};
  auto tris = rvegen::read_stl_ascii<double>(ss);
  REQUIRE(tris.size() == 1);
  REQUIRE(tris[0].v[1][0] == 1.0);
}

void test_stl_malformed_number_throws() {
  // Issue repro: comma decimal separator used to yield a junk triangle.
  std::stringstream ss{std::string{one_facet} + R"(
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 1,5 2
      vertex 0 1 0
    endloop
  endfacet
endsolid t
)"};
  REQUIRE_THROWS_WITH(rvegen::read_stl_ascii<double>(ss), "facet 1");
}

void test_stl_truncated_mid_vertex_throws() {
  // Issue repro: file ends after "vertex 1 0" — used to reuse stale z.
  std::stringstream ss{std::string{one_facet} + R"(
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0)"};
  REQUIRE_THROWS_WITH(rvegen::read_stl_ascii<double>(ss), "end of input");
}

void test_stl_truncated_before_endfacet_throws() {
  std::stringstream ss{R"(solid t
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
)"};
  REQUIRE_THROWS_WITH(rvegen::read_stl_ascii<double>(ss), "endfacet");
}

void test_stl_bad_closing_keyword_throws() {
  std::stringstream ss{R"(solid t
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
      vertex 1 1 0
    endloop
  endfacet
endsolid t
)"};
  REQUIRE_THROWS_WITH(rvegen::read_stl_ascii<double>(ss), "endloop");
}

void test_stl_mixed_case_keywords_parse() {
  std::stringstream ss{R"(SOLID Part
  Facet Normal 0 0 1
    OUTER Loop
      Vertex 0 0 0
      VERTEX 1 0 0
      vertex 0 1 0
    EndLoop
  ENDFACET
EndSolid Part
)"};
  auto tris = rvegen::read_stl_ascii<double>(ss);
  REQUIRE(tris.size() == 1);
  REQUIRE(tris[0].v[2][1] == 1.0);
}

void test_stl_binary_count_exceeds_data_throws() {
  std::ostringstream out;
  for (int i = 0; i < 80; ++i) out.put('\0');
  write_le_u32(out, 50'000'000u);           // ~2.5 GB advertised
  for (int i = 0; i < 50; ++i) out.put('\0');  // one record actually present
  std::stringstream ss{out.str()};
  REQUIRE_THROWS_WITH(rvegen::read_stl_binary<double>(ss), "advertises");
}

void test_stl_binary_exact_and_trailing_bytes_ok() {
  std::ostringstream out;
  for (int i = 0; i < 80; ++i) out.put('\0');
  write_le_u32(out, 1);
  for (int i = 0; i < 3; ++i) write_le_f32(out, 0.f);   // normal
  for (float c : {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f}) {
    write_le_f32(out, c);
  }
  out.put('\0'); out.put('\0');
  {
    std::stringstream ss{out.str()};
    REQUIRE(rvegen::read_stl_binary<double>(ss).size() == 1);
  }
  out << "trailing";   // >= 50·n policy: trailing bytes are tolerated
  {
    std::stringstream ss{out.str()};
    REQUIRE(rvegen::read_stl_binary<double>(ss).size() == 1);
  }
}

// ---------------------------------------------------------------------------
// PLY
// ---------------------------------------------------------------------------

void test_ply_huge_element_count_throws() {
  // Under the absolute cap but far beyond what the payload can hold.
  std::stringstream ss{
      "ply\nformat ascii 1.0\n"
      "element vertex 50000000\nproperty float x\nproperty float y\n"
      "property float z\n"
      "element face 0\nproperty list uchar int vertex_indices\nend_header\n"
      "0 0 0\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "payload");
}

void test_ply_header_element_flood_is_bounded() {
  // Many distinct empty elements: used to make the duplicate-name
  // check quadratic (40k elements took seconds). Now capped.
  std::string hdr = "ply\nformat ascii 1.0\n";
  for (int i = 0; i < 40000; ++i) hdr += "element e" + std::to_string(i) + " 0\n";
  hdr += "end_header\n";
  std::istringstream ss{hdr};
  REQUIRE_THROWS_WITH(rvegen::read_ply<double>(ss), "more than");
  std::istringstream dup{
      "ply\nformat ascii 1.0\nelement vertex 0\nelement vertex 0\nend_header\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply<double>(dup), "declared twice");
}

void test_ply_element_count_over_cap_throws() {
  std::stringstream ss{
      "ply\nformat ascii 1.0\n"
      "element vertex 1e12\nproperty float x\nproperty float y\n"
      "property float z\n"
      "element face 0\nproperty list uchar int vertex_indices\nend_header\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "sanity limit");
}

void test_ply_binary_huge_element_count_throws() {
  std::ostringstream out;
  out << "ply\nformat binary_little_endian 1.0\n"
         "element vertex 3\nproperty float x\nproperty float y\n"
         "property float z\n"
         "element face 90000000\nproperty list uchar int vertex_indices\n"
         "end_header\n";
  for (int i = 0; i < 9; ++i) write_le_f32(out, 0.f);
  std::stringstream ss{out.str()};
  REQUIRE_THROWS_WITH(rvegen::read_ply_binary<double>(ss), "payload");
}

void test_ply_huge_list_count_throws() {
  {
    std::stringstream ss{std::string{
        "ply\nformat ascii 1.0\n"
        "element vertex 3\nproperty float x\nproperty float y\n"
        "property float z\n"
        "element face 1\nproperty list uint int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"} + "4000000000 0 1 2\n"};
    REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "sanity limit");
  }
  {
    // Under the absolute cap but more than the remaining input.
    std::stringstream ss{std::string{
        "ply\nformat ascii 1.0\n"
        "element vertex 3\nproperty float x\nproperty float y\n"
        "property float z\n"
        "element face 1\nproperty list uint int vertex_indices\nend_header\n"
        "0 0 0\n1 0 0\n0 1 0\n"} + "500000 0 1 2\n"};
    REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "remaining");
  }
}

void test_ply_negative_index_throws() {
  std::stringstream ss{std::string{ply_tri_header_ascii} + "3 0 -1 2\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "invalid vertex index");
}

void test_ply_nan_index_throws() {
  std::stringstream ss{std::string{ply_tri_header_ascii} + "3 0 nan 2\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "invalid vertex index");
}

void test_ply_fractional_index_throws() {
  std::stringstream ss{std::string{ply_tri_header_ascii} + "3 0 1.5 2\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "invalid vertex index");
}

void test_ply_out_of_range_index_throws() {
  std::stringstream ss{std::string{ply_tri_header_ascii} + "3 0 1 3\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "invalid vertex index");
}

void test_ply_binary_negative_index_throws() {
  std::ostringstream out;
  out << "ply\nformat binary_little_endian 1.0\n"
         "element vertex 3\nproperty float x\nproperty float y\n"
         "property float z\n"
         "element face 1\nproperty list uchar int vertex_indices\n"
         "end_header\n";
  for (float c : {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f}) {
    write_le_f32(out, c);
  }
  out.put(3);
  write_le_u32(out, 0);
  write_le_u32(out, 0xFFFFFFFFu);   // int32 -1
  write_le_u32(out, 2);
  std::stringstream ss{out.str()};
  REQUIRE_THROWS_WITH(rvegen::read_ply_binary<double>(ss), "invalid vertex index");
}

void test_ply_malformed_number_throws() {
  std::stringstream ss{
      "ply\nformat ascii 1.0\n"
      "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
      "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
      "0 0 0\n1 0,5 0\n0 1 0\n3 0 1 2\n"};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(ss), "malformed number");
}

void test_ply_face_before_vertex_parses() {
  constexpr char const* face_first = R"(ply
format ascii 1.0
element face 2
property list uchar int vertex_indices
property uchar red
element vertex 4
property float x
property float y
property float z
end_header
3 0 1 2 255
3 0 2 3 128
0 0 0
1 0 0
1 1 0
0 1 0
)";
  std::stringstream ss{face_first};
  auto tris = rvegen::read_ply_ascii<double>(ss);
  REQUIRE(tris.size() == 2);
  REQUIRE(tris[0].v[1][0] == 1.0 && tris[0].v[1][1] == 0.0);
  REQUIRE(tris[0].v[2][0] == 1.0 && tris[0].v[2][1] == 1.0);
  REQUIRE(tris[1].v[2][0] == 0.0 && tris[1].v[2][1] == 1.0);

  // Same through the auto-detect entry point.
  std::stringstream ss2{face_first};
  REQUIRE(rvegen::read_ply<double>(ss2).size() == 2);
}

// ---------------------------------------------------------------------------
// mesh_inclusion validation
// ---------------------------------------------------------------------------

void test_mesh_inclusion_consistent_cube_volume() {
  rvegen::mesh_inclusion<double> mesh{make_cube()};
  REQUIRE(std::abs(mesh.volume() - 1.0) < 1e-12);
  REQUIRE(mesh.is_inside({0.5, 0.5, 0.5}));
}

void test_mesh_inclusion_inconsistent_winding_is_reoriented() {
  // Flip a few triangles: the old abs(sum) volume would be wrong
  // (each flipped triangle subtracts instead of adds).
  auto tris = make_cube();
  flip(tris[0]);
  flip(tris[5]);
  flip(tris[10]);
  rvegen::mesh_inclusion<double> mesh{tris};
  REQUIRE(std::abs(mesh.volume() - 1.0) < 1e-12);
  REQUIRE(mesh.is_inside({0.5, 0.5, 0.5}));

  // Fully inverted mesh too.
  auto inv = make_cube();
  for (auto& t : inv) flip(t);
  rvegen::mesh_inclusion<double> inv_mesh{inv};
  REQUIRE(std::abs(inv_mesh.volume() - 1.0) < 1e-12);

  // Shared-ptr ctor re-orients a private copy; the source is untouched.
  auto shared = std::make_shared<tri_list const>(tris);
  rvegen::mesh_inclusion<double> from_shared{shared};
  REQUIRE(std::abs(from_shared.volume() - 1.0) < 1e-12);
  REQUIRE(from_shared.base_triangles() != shared);
  REQUIRE((*shared)[0].v[1] == tris[0].v[1]);
}

void test_mesh_inclusion_consistent_shared_is_zero_copy() {
  auto shared = std::make_shared<tri_list const>(make_cube());
  rvegen::mesh_inclusion<double> mesh{shared};
  REQUIRE(mesh.base_triangles() == shared);
}

void test_mesh_inclusion_cavity_volume() {
  // Outer cube [0,3]^3 with an inner cube [1,2]^3 cavity; both given
  // with outward winding (a common producer mistake for cavities).
  // Parity is_inside treats the inner box as a hole, so the volume
  // must be 27 − 1 = 26.
  auto tris = make_cube(3.0);
  auto inner = make_cube(1.0, 1.0);
  tris.insert(tris.end(), inner.begin(), inner.end());
  rvegen::mesh_inclusion<double> mesh{tris};
  REQUIRE(std::abs(mesh.volume() - 26.0) < 1e-9);
  REQUIRE(mesh.is_inside({0.5, 0.5, 0.5}));
  REQUIRE(!mesh.is_inside({1.5, 1.5, 1.5}));

  // Two disjoint solids add up.
  auto two = make_cube();
  auto other = make_cube(1.0, 5.0);
  for (auto& t : other) flip(t);
  two.insert(two.end(), other.begin(), other.end());
  rvegen::mesh_inclusion<double> mesh2{two};
  REQUIRE(std::abs(mesh2.volume() - 2.0) < 1e-12);
}

void test_mesh_inclusion_cavity_touching_container() {
  // Inner cube [0,5]^3 shares a corner and three faces with the outer
  // [0,10]^3 (its largest-face probe lies on the container surface):
  // volume must still be 1000 - 125.
  for (double shift : {0.0, 5.0}) {
    auto tris = make_cube(10.0);
    auto inner = make_cube(5.0, shift);
    tris.insert(tris.end(), inner.begin(), inner.end());
    rvegen::mesh_inclusion<double> mesh{tris};
    REQUIRE(std::abs(mesh.volume() - 875.0) < 1e-9);
  }
}

void test_mesh_inclusion_open_mesh_throws() {
  auto tris = make_cube();
  tris.pop_back();
  REQUIRE_THROWS_WITH(rvegen::mesh_inclusion<double>{tris}, "not closed");

  std::stringstream ss{std::string{one_facet} + "endsolid t\n"};
  auto single = rvegen::read_stl_ascii<double>(ss);
  REQUIRE_THROWS_WITH(rvegen::mesh_inclusion<double>{single}, "not closed");
}

void test_mesh_inclusion_non_manifold_throws() {
  auto tris = make_cube();
  tris.push_back(tris[0]);   // edge now shared by 3 triangles
  REQUIRE_THROWS_WITH(rvegen::mesh_inclusion<double>{tris}, "shared by 3");
}

void test_mesh_inclusion_non_finite_vertex_throws() {
  auto tris = make_cube();
  tris[3].v[1][2] = std::nan("");
  REQUIRE_THROWS_WITH(rvegen::mesh_inclusion<double>{tris}, "non-finite");
  auto tris2 = make_cube();
  tris2[7].v[0][0] = INFINITY;
  REQUIRE_THROWS_WITH(rvegen::mesh_inclusion<double>{tris2}, "non-finite");
}

void test_mesh_inclusion_cached_bbox_tracks_offset() {
  rvegen::mesh_inclusion<double> mesh{make_cube(2.0)};
  auto e = mesh.max_expansion();
  REQUIRE(e[0] == 1.0 && e[1] == 1.0 && e[2] == 1.0);
  mesh.set_middle_point({10.0, 0.0, -5.0});
  mesh.make_bounding_box();
  auto const& bb = *mesh.bounding_box();
  REQUIRE(bb.bottom_point()[0] == 9.0 && bb.top_point()[0] == 11.0);
  REQUIRE(bb.bottom_point()[2] == -6.0 && bb.top_point()[2] == -4.0);
  auto copy = mesh.clone();
  REQUIRE(copy->volume() == mesh.volume());
}

// ---------------------------------------------------------------------------
// Review follow-ups
// ---------------------------------------------------------------------------

tri_list make_tet(double x, double y, double z, double s) {
  const std::array<std::array<double, 3>, 4> p{{
      {x, y, z}, {x + s, y, z}, {x, y + s, z}, {x, y, z + s}}};
  constexpr std::array<std::array<int, 3>, 4> f{{
      {0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}};
  tri_list out;
  for (auto const& t : f) {
    gte::Triangle3<double> tri;
    for (int c = 0; c < 3; ++c) {
      for (int k = 0; k < 3; ++k) tri.v[c][k] = p[t[c]][k];
    }
    out.push_back(tri);
  }
  return out;
}

// UV sphere of radius r: 2n(n-1) triangles, closed, outward wound.
tri_list make_sphere(int n, double r) {
  auto P = [&](int i, int j) {
    const double th = M_PI * i / n, ph = 2 * M_PI * (j % (2 * n)) / (2 * n);
    if (i == 0) return gte::Vector3<double>{0, 0, r};
    if (i == n) return gte::Vector3<double>{0, 0, -r};
    return gte::Vector3<double>{r * std::sin(th) * std::cos(ph),
                                r * std::sin(th) * std::sin(ph),
                                r * std::cos(th)};
  };
  tri_list out;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < 2 * n; ++j) {
      const auto a = P(i, j), b = P(i + 1, j), c = P(i + 1, j + 1), d = P(i, j + 1);
      if (i != 0) out.push_back({a, b, d});
      if (i != n - 1) out.push_back({b, c, d});
    }
  }
  return out;
}

double signed_volume(tri_list const& tris) {
  double v = 0;
  for (auto const& t : tris) v += gte::DotCross(t.v[0], t.v[1], t.v[2]);
  return v / 6;
}

void test_mesh_inclusion_corner_touching_cubes() {
  // Two unit cubes sharing only the corner (1,1,1). The old nesting
  // probe started ON that vertex and flipped the second cube (vol 0).
  for (bool flip_second : {false, true}) {
    auto tris = make_cube();
    auto other = make_cube(1.0, 1.0);
    if (flip_second) for (auto& t : other) flip(t);
    tris.insert(tris.end(), other.begin(), other.end());
    rvegen::mesh_inclusion<double> mesh{tris};
    REQUIRE(std::abs(mesh.volume() - 2.0) < 1e-12);
  }
}

void test_mesh_inclusion_many_cavities_and_parts() {
  // A finely tessellated sphere (exercises the binned ray path) with
  // small tetrahedral cavities inside, plus disjoint parts outside.
  auto tris = make_sphere(40, 1.0);
  const double sphere_vol = signed_volume(tris);
  const double s = 0.05;
  int n_cav = 0;
  for (double x = -0.4; x <= 0.41; x += 0.2) {
    for (double y = -0.4; y <= 0.41; y += 0.4) {
      auto t = make_tet(x, y, 0.1 * x, s);
      for (auto& tri : t) flip(tri);   // wrong winding on purpose
      tris.insert(tris.end(), t.begin(), t.end());
      ++n_cav;
    }
  }
  for (int k = 0; k < 20; ++k) {
    auto t = make_tet(3.0 + 2.0 * k, 0.0, 0.0, 1.0);
    tris.insert(tris.end(), t.begin(), t.end());
  }
  rvegen::mesh_inclusion<double> mesh{tris};
  const double tet_vol = s * s * s / 6;
  const double expected = sphere_vol - n_cav * tet_vol + 20.0 / 6.0;
  REQUIRE(std::abs(mesh.volume() - expected) < 1e-9);

  // Many disjoint parts in a row: near-linear nesting test.
  tri_list row;
  for (int k = 0; k < 20000; ++k) {
    auto t = make_tet(2.0 * k, 0.0, 0.0, 1.0);
    row.insert(row.end(), t.begin(), t.end());
  }
  rvegen::mesh_inclusion<double> row_mesh{row};
  REQUIRE(std::abs(row_mesh.volume() - 20000.0 / 6.0) < 1e-6);
}

void test_mesh_inclusion_zero_volume_throws() {
  // A triangle and its reverse: closed, manifold, encloses nothing.
  std::stringstream ss{R"(solid x
facet normal 0 0 0
outer loop
vertex 0 0 0
vertex 1 0 0
vertex 0 1 0
endloop
endfacet
facet normal 0 0 0
outer loop
vertex 0 0 0
vertex 0 1 0
vertex 1 0 0
endloop
endfacet
endsolid x
)"};
  auto tris = rvegen::read_stl_ascii<double>(ss);
  REQUIRE(tris.size() == 2);
  REQUIRE_THROWS_WITH(rvegen::mesh_inclusion<double>{tris}, "zero volume");
}

void test_ply_fan_triangle_limit() {
  // 4 faces, each a 200-gon over 200 uchar-indexed vertices: 792 fan
  // triangles. A lowered limit must throw; the default must not.
  std::ostringstream out;
  out << "ply\nformat binary_little_endian 1.0\n"
         "element vertex 200\nproperty uchar x\nproperty uchar y\n"
         "property uchar z\n"
         "element face 4\nproperty list uchar uchar vertex_indices\n"
         "end_header\n";
  for (int v = 0; v < 200; ++v) {
    out.put(static_cast<char>(v)); out.put(0); out.put(0);
  }
  for (int f = 0; f < 4; ++f) {
    out.put(static_cast<char>(200));
    for (int k = 0; k < 200; ++k) out.put(static_cast<char>(k));
  }
  const auto blob = out.str();
  {
    std::stringstream ss{blob};
    REQUIRE(rvegen::read_ply<double>(ss).size() == 4 * 198);
  }
  {
    std::stringstream ss{blob};
    REQUIRE_THROWS_WITH(
        rvegen::read_ply<double>(ss, rvegen::ply_read_limits{500}),
        "triangles");
  }
}

void test_ply_face_first_large_fan() {
  // Face-before-vertex path buffers uint32 triples; check a big fan.
  std::ostringstream out;
  out << "ply\nformat ascii 1.0\nelement face 1\n"
         "property list ushort int vertex_indices\n"
         "element vertex 1000\nproperty float x\nproperty float y\n"
         "property float z\nend_header\n1000";
  for (int k = 0; k < 1000; ++k) out << ' ' << k;
  out << '\n';
  for (int k = 0; k < 1000; ++k) out << k << " 0 0\n";
  std::stringstream ss{out.str()};
  auto tris = rvegen::read_ply_ascii<double>(ss);
  REQUIRE(tris.size() == 998);
  REQUIRE(tris[997].v[2][0] == 999.0);
}

void test_stl_printable_binary_header_detected_by_size() {
  // Binary STL with a "solid" header and every payload byte printable
  // (floats with bit pattern 0x41414141 = "AAAA"): only the exact
  // 84 + 50·n size rule identifies it as binary.
  std::string blob = "solid x";
  blob.resize(80, ' ');
  blob += std::string{"\x01\x00\x00\x00", 4};
  blob += std::string(50, 'A');
  REQUIRE(blob.size() == 134);
  {
    std::stringstream ss{blob};
    auto tris = rvegen::read_stl<double>(ss);
    REQUIRE(tris.size() == 1);
  }
  {
    std::stringstream ss{blob};
    REQUIRE_THROWS_WITH(rvegen::read_stl_ascii<double>(ss), "binary");
  }
}

void test_stl_ascii_without_facet_or_endsolid_throws() {
  // A binary-looking header whose count does not match the size falls
  // to the ASCII path; "solid" swallows the line and nothing follows.
  std::string blob = "solid x";
  blob.resize(80, ' ');
  blob += std::string{"\xff\xff\xff\x00", 4};
  std::stringstream ss{blob};
  REQUIRE_THROWS_WITH(rvegen::read_stl<double>(ss), "without any 'facet'");

  // An empty-but-terminated solid is still fine.
  std::stringstream ok{"solid empty\nendsolid empty\n"};
  REQUIRE(rvegen::read_stl_ascii<double>(ok).empty());
}

void test_parse_nits() {
  // Double sign.
  std::stringstream pm{std::string{
      "ply\nformat ascii 1.0\n"
      "element vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
      "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
      "+0 +0 +0\n1 0 0\n0 1 +-1\n3 0 1 2\n"}};
  REQUIRE_THROWS_WITH(rvegen::read_ply_ascii<double>(pm), "malformed number");

  // UTF-8 BOM, STL and PLY (direct and auto-detect).
  {
    std::stringstream ss{"\xEF\xBB\xBF" + std::string{one_facet} + "endsolid t\n"};
    REQUIRE(rvegen::read_stl<double>(ss).size() == 1);
  }
  for (int autodetect = 0; autodetect < 2; ++autodetect) {
    std::stringstream ss{"\xEF\xBB\xBF" + std::string{ply_tri_header_ascii} +
                         "3 0 1 2\n"};
    auto tris = autodetect ? rvegen::read_ply<double>(ss)
                           : rvegen::read_ply_ascii<double>(ss);
    REQUIRE(tris.size() == 1);
  }

  // ASCII-only case fold: a non-ASCII byte never matches a keyword.
  // (Long solid name keeps the 2 non-ASCII bytes under the 5 % sniff.)
  std::stringstream bad{"solid " + std::string(100, 't') +
                        "\nfacet normal 0 0 1\n\xC3\x9Cuter loop\n"};
  REQUIRE_THROWS_WITH(rvegen::read_stl_ascii<double>(bad), "expected 'outer'");
}

} // namespace

int main() {
  test_mesh_inclusion_corner_touching_cubes();
  test_mesh_inclusion_many_cavities_and_parts();
  test_mesh_inclusion_zero_volume_throws();
  test_ply_fan_triangle_limit();
  test_ply_face_first_large_fan();
  test_stl_printable_binary_header_detected_by_size();
  test_stl_ascii_without_facet_or_endsolid_throws();
  test_parse_nits();

  test_stl_valid_single_facet_parses();
  test_stl_malformed_number_throws();
  test_stl_truncated_mid_vertex_throws();
  test_stl_truncated_before_endfacet_throws();
  test_stl_bad_closing_keyword_throws();
  test_stl_mixed_case_keywords_parse();
  test_stl_binary_count_exceeds_data_throws();
  test_stl_binary_exact_and_trailing_bytes_ok();

  test_ply_huge_element_count_throws();
  test_ply_header_element_flood_is_bounded();
  test_ply_element_count_over_cap_throws();
  test_ply_binary_huge_element_count_throws();
  test_ply_huge_list_count_throws();
  test_ply_negative_index_throws();
  test_ply_nan_index_throws();
  test_ply_fractional_index_throws();
  test_ply_out_of_range_index_throws();
  test_ply_binary_negative_index_throws();
  test_ply_malformed_number_throws();
  test_ply_face_before_vertex_parses();

  test_mesh_inclusion_consistent_cube_volume();
  test_mesh_inclusion_inconsistent_winding_is_reoriented();
  test_mesh_inclusion_consistent_shared_is_zero_copy();
  test_mesh_inclusion_cavity_volume();
  test_mesh_inclusion_cavity_touching_container();
  test_mesh_inclusion_open_mesh_throws();
  test_mesh_inclusion_non_manifold_throws();
  test_mesh_inclusion_non_finite_vertex_throws();
  test_mesh_inclusion_cached_bbox_tracks_offset();

  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "mesh_io_test: all checks passed\n";
  return EXIT_SUCCESS;
}
