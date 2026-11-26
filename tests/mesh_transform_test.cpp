// mesh_inclusion placement (#9): rotation + uniform scale on top of
// translation, the input's orientation / scale options, to_mesh +
// dispatcher registration, and the issue's acceptance test (voxelize a
// tetrahedron and compare with the analytic volume).

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
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

template <typename F>
bool throws_runtime(F&& fn) {
  try { fn(); } catch (std::runtime_error const&) { return true; }
  return false;
}

using tri_list = std::vector<gte::Triangle3<double>>;
using mesh_t = rvegen::mesh_inclusion<double>;
constexpr double pi = std::numbers::pi;

// Right-corner tetrahedron with edge a: vertices 0, a·e_x, a·e_y, a·e_z.
tri_list make_tet(double a) {
  const std::array<gte::Vector3<double>, 4> p{{
      {0, 0, 0}, {a, 0, 0}, {0, a, 0}, {0, 0, a}}};
  constexpr std::array<std::array<int, 3>, 4> f{{
      {0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}}};
  tri_list out;
  for (auto const& t : f) out.push_back({p[t[0]], p[t[1]], p[t[2]]});
  return out;
}

// Axis-aligned box [-hx,hx]×[-hy,hy]×[-hz,hz].
tri_list make_box(double hx, double hy, double hz) {
  std::array<gte::Vector3<double>, 8> v;
  for (int i = 0; i < 8; ++i) {
    v[i] = {(i & 1) ? hx : -hx, (i & 2) ? hy : -hy, (i & 4) ? hz : -hz};
  }
  constexpr std::array<std::array<int, 3>, 12> f{{
      {0, 2, 1}, {1, 2, 3}, {4, 5, 6}, {5, 7, 6}, {0, 1, 4}, {1, 5, 4},
      {2, 6, 3}, {3, 6, 7}, {0, 4, 2}, {2, 4, 6}, {1, 3, 5}, {3, 7, 5}}};
  tri_list out;
  for (auto const& t : f) out.push_back({v[t[0]], v[t[1]], v[t[2]]});
  return out;
}

double tet_surface_area(double a) {
  return a * a * (1.5 + std::sqrt(3.0) / 2.0);
}

bool near(std::array<double, 3> const& a, std::array<double, 3> const& b,
          double tol) {
  return std::abs(a[0] - b[0]) <= tol && std::abs(a[1] - b[1]) <= tol &&
         std::abs(a[2] - b[2]) <= tol;
}

std::array<double, 3> apply(gte::Matrix3x3<double> const& r,
                            std::array<double, 3> const& v) {
  const auto w = r * gte::Vector3<double>{v[0], v[1], v[2]};
  return {w[0], w[1], w[2]};
}

// Voxel-volume error bound. A voxel whose centre is classified
// differently from some point x inside it has x and the centre on
// opposite sides of the surface, so dist(x, ∂Ω) <= h·√3/2. The
// misclassified volume therefore lies in the tube of half-width h·√3/2
// around the surface: |V_voxel − V| <= A·h·√3 to leading order.
double voxel_tolerance(double area, double h) { return area * h * std::sqrt(3.0); }

double voxel_volume(std::unique_ptr<rvegen::shape_base<double>> shape,
                    double L, std::size_t n) {
  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  shapes.push_back(std::move(shape));
  const auto grid = rvegen::sample_voxel_grid(shapes, {L, L, L}, n, n, n);
  std::size_t inside = 0;
  for (auto id : grid) inside += (id != 0);
  const double h = L / static_cast<double>(n);
  return static_cast<double>(inside) * h * h * h;
}

// ---------------------------------------------------------------------------
// Acceptance: voxelized tetrahedron vs analytic volume
// ---------------------------------------------------------------------------

void test_voxelized_tetrahedron_matches_analytic() {
  const double a = 0.8, L = 1.0;
  const std::size_t n = 100;
  const double h = L / n;
  auto tet = std::make_unique<mesh_t>(make_tet(a));
  tet->set_middle_point({0.5, 0.5, 0.5});
  const double exact = a * a * a / 6.0;
  REQUIRE(std::abs(tet->volume() - exact) < 1e-14);
  const double vox = voxel_volume(std::move(tet), L, n);
  const double tol = voxel_tolerance(tet_surface_area(a), h);
  std::cout << "  tet: voxel " << vox << " exact " << exact << " |err| "
            << std::abs(vox - exact) << " tol " << tol << "\n";
  REQUIRE(std::abs(vox - exact) <= tol);
}

void test_voxelized_rotated_scaled_tetrahedron_matches_analytic() {
  const double a = 0.5, s = 1.4, L = 1.5;
  const std::size_t n = 120;
  const double h = L / n;
  auto tet = std::make_unique<mesh_t>(make_tet(a));
  tet->set_rotation_axis_angle({1.0, 2.0, 3.0}, 0.7);
  tet->set_scale(s);
  tet->set_middle_point({0.75, 0.75, 0.75});
  const double exact = std::pow(a * s, 3) / 6.0;
  REQUIRE(std::abs(tet->volume() - exact) < 1e-14);
  // The placed tet must sit inside the voxel domain for the comparison.
  tet->make_bounding_box();
  auto const& bb = *tet->bounding_box();
  for (int k = 0; k < 3; ++k) {
    REQUIRE(bb.bottom_point()[k] > 0.0 && bb.top_point()[k] < L);
  }
  const double vox = voxel_volume(std::move(tet), L, n);
  const double tol = voxel_tolerance(tet_surface_area(a * s), h);
  std::cout << "  rotated+scaled tet: voxel " << vox << " exact " << exact
            << " |err| " << std::abs(vox - exact) << " tol " << tol << "\n";
  REQUIRE(std::abs(vox - exact) <= tol);
}

// ---------------------------------------------------------------------------
// Rotation conventions, inside test, scale
// ---------------------------------------------------------------------------

void test_rotation_conventions_agree() {
  // +90° about z maps e_x → e_y under every entry point.
  const std::array<double, 3> ex{1, 0, 0}, ey{0, 1, 0};
  mesh_t m{make_tet(1.0)};
  m.set_rotation_axis_angle({0, 0, 1}, pi / 2);
  REQUIRE(near(apply(m.rotation(), ex), ey, 1e-12));
  m.set_rotation_euler_zyz(pi / 2, 0.0, 0.0);
  REQUIRE(near(apply(m.rotation(), ex), ey, 1e-12));
  m.set_rotation(gte::Quaternion<double>(0, 0, std::sin(pi / 4), std::cos(pi / 4)));
  REQUIRE(near(apply(m.rotation(), ex), ey, 1e-12));
  // z-y-z: Ry(beta) tilts e_z towards +x.
  m.set_rotation_euler_zyz(0.0, pi / 2, 0.0);
  REQUIRE(near(apply(m.rotation(), {0, 0, 1}), ex, 1e-12));
  // Composition order R = Rz(alpha)·Ry(beta): e_z → e_x → e_y.
  m.set_rotation_euler_zyz(pi / 2, pi / 2, 0.0);
  REQUIRE(near(apply(m.rotation(), {0, 0, 1}), ey, 1e-12));
}

void test_rotated_scaled_inside_and_volume() {
  // Slab 2 × 0.2 × 0.2 along x, rotated 90° about z → along y.
  mesh_t m{make_box(1.0, 0.1, 0.1)};
  m.set_middle_point({5.0, 5.0, 5.0});
  REQUIRE(m.is_inside({5.9, 5.0, 5.0}));
  REQUIRE(!m.is_inside({5.0, 5.9, 5.0}));
  m.set_rotation_axis_angle({0, 0, 1}, pi / 2);
  REQUIRE(!m.is_inside({5.9, 5.0, 5.0}));
  REQUIRE(m.is_inside({5.0, 5.9, 5.0}));
  REQUIRE(std::abs(m.volume() - 0.08) < 1e-14);
  m.set_scale(2.0);
  REQUIRE(m.is_inside({5.0, 6.9, 5.0}));
  REQUIRE(!m.is_inside({5.0, 7.1, 5.0}));
  REQUIRE(std::abs(m.volume() - 0.64) < 1e-13);
  auto e = m.max_expansion();
  REQUIRE(std::abs(e[0] - 0.2) < 1e-12 && std::abs(e[1] - 2.0) < 1e-12);
  // Middle point is unaffected by rotation / scale.
  REQUIRE(near(m.get_middle_point(), {5.0, 5.0, 5.0}, 1e-15));
  // Clones carry the placement.
  auto c = m.clone();
  REQUIRE(c->is_inside({5.0, 6.9, 5.0}));
  REQUIRE(std::abs(c->volume() - 0.64) < 1e-13);
}

void test_invalid_transforms_throw() {
  mesh_t m{make_tet(1.0)};
  REQUIRE(throws_runtime([&] { m.set_scale(0.0); }));
  REQUIRE(throws_runtime([&] { m.set_scale(-1.0); }));
  REQUIRE(throws_runtime([&] { m.set_scale(std::nan("")); }));
  gte::Matrix3x3<double> shear = gte::Matrix3x3<double>::Identity();
  shear(0, 1) = 0.5;
  REQUIRE(throws_runtime([&] { m.set_rotation(shear); }));
  gte::Matrix3x3<double> mirror = gte::Matrix3x3<double>::Identity();
  mirror(2, 2) = -1.0;
  REQUIRE(throws_runtime([&] { m.set_rotation(mirror); }));
  REQUIRE(throws_runtime([&] { m.set_rotation_axis_angle({0, 0, 0}, 1.0); }));
  REQUIRE(throws_runtime([&] { (void)rvegen::uniform_random_rotation(1.5, 0.2, 0.3); }));
}

// ---------------------------------------------------------------------------
// AABB of a randomly rotated mesh
// ---------------------------------------------------------------------------

void test_random_rotation_aabb_contains_transformed_vertices() {
  std::mt19937_64 rng{20261127};
  std::uniform_real_distribution<double> us{0.5, 2.0}, up{-3.0, 3.0};
  auto tris = make_tet(1.0);
  auto extra = make_box(0.3, 0.7, 0.2);   // irregular second component
  for (auto& t : extra) for (auto& v : t.v) v[0] += 3.0;
  tris.insert(tris.end(), extra.begin(), extra.end());
  mesh_t m{tris};
  for (int trial = 0; trial < 200; ++trial) {
    m.set_rotation(rvegen::uniform_random_rotation<double>(rng));
    m.set_scale(us(rng));
    const double mx = up(rng), my = up(rng), mz = up(rng);
    m.set_middle_point({mx, my, mz});
    m.make_bounding_box();
    auto const& bb = *m.bounding_box();
    const auto mesh = rvegen::to_mesh(m);
    std::array<double, 3> lo{1e300, 1e300, 1e300}, hi{-1e300, -1e300, -1e300};
    bool contained = true;
    for (auto const& v : mesh.verts) {
      for (int k = 0; k < 3; ++k) {
        contained = contained && v[k] >= bb.bottom_point()[k] - 1e-12 &&
                    v[k] <= bb.top_point()[k] + 1e-12;
        lo[k] = std::min(lo[k], v[k]);
        hi[k] = std::max(hi[k], v[k]);
      }
    }
    REQUIRE(contained);
    // ...and it is the tight AABB, and within the centred max_expansion box.
    const auto e = m.max_expansion();
    const std::array<double, 3> mid{mx, my, mz};
    for (int k = 0; k < 3; ++k) {
      REQUIRE(std::abs(lo[k] - bb.bottom_point()[k]) < 1e-12);
      REQUIRE(std::abs(hi[k] - bb.top_point()[k]) < 1e-12);
      REQUIRE(lo[k] >= mid[k] - e[k] - 1e-12 && hi[k] <= mid[k] + e[k] + 1e-12);
    }
  }
}

// ---------------------------------------------------------------------------
// Input: isotropic random orientation, Euler, scale
// ---------------------------------------------------------------------------

std::string write_tet_stl(std::string const& path) {
  std::ofstream out{path};
  out << "solid tet\n";
  for (auto const& t : make_tet(1.0)) {
    out << "facet normal 0 0 0\nouter loop\n";
    for (auto const& v : t.v) out << "vertex " << v[0] << ' ' << v[1] << ' ' << v[2] << '\n';
    out << "endloop\nendfacet\n";
  }
  out << "endsolid tet\n";
  return path;
}

void test_uniform_orientation_is_isotropic() {
  // Through the JSON registry path: orientation "uniform" + U(0,1).
  rvegen::register_all_distributions<>();
  rvegen::register_all_inputs<>();
  const auto path = write_tet_stl("/tmp/rvegen_test_mesh_transform_iso.stl");
  std::mt19937 engine{12345};
  const auto dist_specs = nlohmann::json::parse(R"({
    "p": {"type": "constant", "value": 0.0},
    "u": {"type": "uniform_real", "a": 0.0, "b": 1.0}
  })");
  rvegen::distribution_map_t<double> dists;
  for (auto const& [name, spec] : dist_specs.items()) {
    auto d = rvegen::build_from_json(
        rvegen::distribution_registry_t<>::instance(), spec, engine);
    dists.emplace(name, std::shared_ptr<rvegen::distribution_base<double>>{
                            std::move(d)});
  }
  const auto spec = nlohmann::json::parse(std::string{R"({
    "type": "mesh_inclusion_input", "mesh_path": ")"} + path + R"(",
    "position_x_dist": "p", "position_y_dist": "p", "position_z_dist": "p",
    "orientation": "uniform", "orientation_dist": "u"
  })");
  auto input = rvegen::build_from_json(
      rvegen::input_registry_t<>::instance(), spec, dists);
  REQUIRE(input != nullptr);

  constexpr int N = 20000;
  std::array<double, 3> mean{};
  std::array<std::array<double, 3>, 3> second{};
  for (int i = 0; i < N; ++i) {
    auto shape = input->new_shape();
    auto const* m = dynamic_cast<mesh_t const*>(shape.get());
    REQUIRE(m != nullptr);
    if (!m) break;
    const auto v = apply(m->rotation(), {0, 0, 1});
    for (int a = 0; a < 3; ++a) {
      mean[a] += v[a] / N;
      for (int b = 0; b < 3; ++b) second[a][b] += v[a] * v[b] / N;
    }
  }
  const double tol = 3.0 / std::sqrt(static_cast<double>(N));
  double worst_mean = 0, worst_second = 0;
  for (int a = 0; a < 3; ++a) {
    worst_mean = std::max(worst_mean, std::abs(mean[a]));
    for (int b = 0; b < 3; ++b) {
      worst_second = std::max(
          worst_second, std::abs(second[a][b] - (a == b ? 1.0 / 3.0 : 0.0)));
    }
  }
  std::cout << "  isotropy N=" << N << ": max|mean| " << worst_mean
            << ", max|S - I/3| " << worst_second << ", tol " << tol << "\n";
  REQUIRE(worst_mean <= tol);
  REQUIRE(worst_second <= tol);

  // Unknown mode / missing distribution are config errors.
  auto bad = spec;
  bad["orientation"] = "bogus";
  REQUIRE(throws_runtime([&] {
    (void)rvegen::build_from_json(rvegen::input_registry_t<>::instance(), bad, dists);
  }));
  auto missing = spec;
  missing.erase("orientation_dist");
  REQUIRE(throws_runtime([&] {
    (void)rvegen::build_from_json(rvegen::input_registry_t<>::instance(), missing, dists);
  }));
  std::remove(path.c_str());
}

void test_input_euler_and_scale() {
  const auto path = write_tet_stl("/tmp/rvegen_test_mesh_transform_euler.stl");
  rvegen::constant_distribution<double> zero{0.0}, quarter{pi / 2}, two{2.0};
  rvegen::mesh_inclusion_input<double> input{path, zero, zero, zero};
  input.set_euler_zyz_orientation(quarter, zero, zero);
  input.set_scale_distribution(two);
  auto shape = input.new_shape();
  auto const* m = dynamic_cast<mesh_t const*>(shape.get());
  REQUIRE(m != nullptr);
  if (m) {
    REQUIRE(near(apply(m->rotation(), {1, 0, 0}), {0, 1, 0}, 1e-12));
    REQUIRE(std::abs(m->volume() - 8.0 / 6.0) < 1e-12);
  }
  rvegen::constant_distribution<double> neg{-1.0};
  input.set_scale_distribution(neg);
  REQUIRE(throws_runtime([&] { (void)input.new_shape(); }));
  std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// to_mesh, dispatcher, writers
// ---------------------------------------------------------------------------

void test_to_mesh_dispatcher_and_writers() {
  auto shape = std::make_unique<mesh_t>(make_tet(1.0));
  shape->set_rotation_axis_angle({0, 1, 0}, 0.3);
  shape->set_scale(0.5);
  shape->set_middle_point({0.5, 0.5, 0.5});

  const auto direct = rvegen::to_mesh(*shape);
  REQUIRE(direct.verts.size() == 4);   // welded
  REQUIRE(direct.tris.size() == 4);
  // Outward winding preserved: signed volume of the placed mesh > 0.
  double sv = 0;
  for (auto const& t : direct.tris) {
    gte::Vector3<double> a, b, c;
    for (int k = 0; k < 3; ++k) {
      a[k] = direct.verts[t[0]][k]; b[k] = direct.verts[t[1]][k]; c[k] = direct.verts[t[2]][k];
    }
    sv += gte::DotCross(a, b, c) / 6;
  }
  REQUIRE(std::abs(sv - shape->volume()) < 1e-12);

  auto& d = rvegen::mesh_dispatcher<double>::instance();
  d.clear();
  REQUIRE(d(*shape).empty());
  rvegen::register_all_meshes<double>();
  const auto via = d(static_cast<rvegen::shape_base<double> const&>(*shape));
  REQUIRE(via.verts == direct.verts && via.tris == direct.tris);

  std::vector<std::unique_ptr<rvegen::shape_base<double>>> shapes;
  shapes.push_back(std::move(shape));
  {
    std::ostringstream out;
    rvegen::three_js_writer<double>{}.write(out, shapes, {1.0, 1.0, 1.0});
    REQUIRE(out.str().find("kind: 'mesh'") != std::string::npos);
  }
  {
    std::ostringstream out;
    rvegen::svg_3d_writer<double>{}.write(out, shapes, {1.0, 1.0, 1.0});
    REQUIRE(out.str().find("<polygon") != std::string::npos);
  }
}

} // namespace

int main() {
  test_voxelized_tetrahedron_matches_analytic();
  test_voxelized_rotated_scaled_tetrahedron_matches_analytic();
  test_rotation_conventions_agree();
  test_rotated_scaled_inside_and_volume();
  test_invalid_transforms_throw();
  test_random_rotation_aabb_contains_transformed_vertices();
  test_uniform_orientation_is_isotropic();
  test_input_euler_and_scale();
  test_to_mesh_dispatcher_and_writers();
  if (failures != 0) {
    std::cerr << failures << " check(s) failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "mesh_transform_test: all checks passed\n";
  return EXIT_SUCCESS;
}
