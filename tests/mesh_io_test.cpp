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

} // namespace

int main() {
  test_stl_valid_single_facet_parses();
  test_stl_malformed_number_throws();
  test_stl_truncated_mid_vertex_throws();
  test_stl_truncated_before_endfacet_throws();
  test_stl_bad_closing_keyword_throws();
  test_stl_mixed_case_keywords_parse();
  test_stl_binary_count_exceeds_data_throws();
  test_stl_binary_exact_and_trailing_bytes_ok();

  test_ply_huge_element_count_throws();
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
