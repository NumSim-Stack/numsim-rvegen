// Round-trip test for gmsh_geo_writer (#74): writes .geo files from real
// generator output, meshes them with the gmsh executable found at configure
// time (RVEGEN_GMSH_EXECUTABLE), and checks the resulting mesh:
//
//   * gmsh exits 0 (the .geo is valid for the OpenCASCADE kernel),
//   * every node lies inside the RVE (wrap copies / overhang are clipped),
//   * element areas / volumes sum to the domain size (conforming
//     fragments: nothing is meshed twice, the matrix is not dropped),
//   * the inclusion phases cover the expected area / volume,
//   * periodic runs: boundary nodes on opposite faces match exactly.
//
// The test is only registered when CMake finds gmsh.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "rvegen/rvegen.h"

#ifndef RVEGEN_GMSH_EXECUTABLE
#error "RVEGEN_GMSH_EXECUTABLE must be defined"
#endif

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (false)

using shape_vector = std::vector<std::unique_ptr<rvegen::shape_base<double>>>;
using box_t = std::array<double, 3>;

struct mesh {
  std::vector<std::array<double, 3>> nodes;  // indexed by node id
  std::map<int, std::string> physical_names; // tag → name
  // Area (2D, triangles) or volume (3D, tetrahedra) per physical tag;
  // tag 0 for elements without a physical group.
  std::map<int, double> measure_by_tag;
  bool ok{false};
};

double tri_area(std::array<double, 3> const& a, std::array<double, 3> const& b,
                std::array<double, 3> const& c) {
  const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double x = u[1] * v[2] - u[2] * v[1];
  const double y = u[2] * v[0] - u[0] * v[2];
  const double z = u[0] * v[1] - u[1] * v[0];
  return 0.5 * std::sqrt(x * x + y * y + z * z);
}

double tet_volume(std::array<double, 3> const& a, std::array<double, 3> const& b,
                  std::array<double, 3> const& c, std::array<double, 3> const& d) {
  const double u[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  const double v[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double w[3] = {d[0] - a[0], d[1] - a[1], d[2] - a[2]};
  return std::abs(u[0] * (v[1] * w[2] - v[2] * w[1]) -
                  u[1] * (v[0] * w[2] - v[2] * w[0]) +
                  u[2] * (v[0] * w[1] - v[1] * w[0])) / 6.0;
}

// Minimal MSH 2.2 ASCII reader — nodes, physical names, and linear
// triangles (type 2) / tetrahedra (type 4).
mesh read_msh22(std::string const& path) {
  mesh m;
  std::ifstream in{path};
  if (!in) return m;
  std::string tok;
  while (in >> tok) {
    if (tok == "$PhysicalNames") {
      int n; in >> n;
      for (int i = 0; i < n; ++i) {
        int dim, tag; std::string name;
        in >> dim >> tag >> name;
        if (name.size() >= 2) name = name.substr(1, name.size() - 2);
        m.physical_names[tag] = name;
      }
    } else if (tok == "$Nodes") {
      std::size_t n; in >> n;
      for (std::size_t i = 0; i < n; ++i) {
        std::size_t id; double x, y, z;
        in >> id >> x >> y >> z;
        if (m.nodes.size() <= id) m.nodes.resize(id + 1);
        m.nodes[id] = {x, y, z};
      }
    } else if (tok == "$Elements") {
      std::size_t n; in >> n;
      std::string line;
      std::getline(in, line);
      for (std::size_t i = 0; i < n; ++i) {
        std::getline(in, line);
        std::istringstream ls{line};
        std::size_t id, type, ntags;
        ls >> id >> type >> ntags;
        std::vector<int> tags(ntags);
        for (auto& t : tags) ls >> t;
        const int phys = ntags > 0 ? tags[0] : 0;
        std::vector<std::size_t> nn;
        std::size_t v;
        while (ls >> v) nn.push_back(v);
        if (type == 2 && nn.size() == 3) {
          m.measure_by_tag[phys] +=
              tri_area(m.nodes[nn[0]], m.nodes[nn[1]], m.nodes[nn[2]]);
        } else if (type == 4 && nn.size() == 4) {
          m.measure_by_tag[phys] += tet_volume(m.nodes[nn[0]], m.nodes[nn[1]],
                                               m.nodes[nn[2]], m.nodes[nn[3]]);
        }
      }
      m.ok = true;
    }
  }
  return m;
}

// Removes <stem>.{geo,msh,log} on scope exit unless a check failed since
// construction (failed runs keep their files for inspection).
struct cleanup_on_success {
  explicit cleanup_on_success(std::string stem)
      : _stem{std::move(stem)}, _failures_at_start{failures} {}
  ~cleanup_on_success() {
    if (failures != _failures_at_start) return;
    for (auto const* ext : {".geo", ".msh", ".log"}) {
      std::remove((_stem + ext).c_str());
    }
  }
  std::string _stem;
  int _failures_at_start;
};

// Write `shapes` with `writer`, mesh with gmsh, return the parsed mesh.
mesh write_and_mesh(rvegen::gmsh_geo_writer<double> const& writer,
                    shape_vector const& shapes, box_t const& box,
                    std::string const& stem) {
  const std::string geo = stem + ".geo";
  const std::string msh = stem + ".msh";
  const std::string log = stem + ".log";
  {
    std::ostringstream buf;
    writer.write(buf, shapes, box);
    std::ofstream out{geo};
    out << buf.str();
  }
  std::remove(msh.c_str());
  const bool is_3d = box[2] > 0.0;
  const std::string cmd = std::string{"\""} + RVEGEN_GMSH_EXECUTABLE + "\" " +
                          (is_3d ? "-3 " : "-2 ") + geo +
                          " -format msh22 -o " + msh + " > " + log + " 2>&1";
  const int rc = std::system(cmd.c_str());
  if (rc != 0) {
    std::cerr << stem << ": gmsh failed (rc=" << rc << "), see " << log << "\n";
    ++failures;
    return {};
  }
  auto m = read_msh22(msh);
  REQUIRE(m.ok);
  return m;
}

void check_inside_domain(mesh const& m, box_t const& box) {
  const double eps = 1e-9 * std::max({box[0], box[1], box[2]});
  const int dim = box[2] > 0.0 ? 3 : 2;
  bool inside = true;
  for (std::size_t i = 1; i < m.nodes.size(); ++i) {
    for (int k = 0; k < dim; ++k) {
      if (m.nodes[i][k] < -eps || m.nodes[i][k] > box[k] + eps) inside = false;
    }
  }
  REQUIRE(inside);
}

// Nodes on the low face of each axis must match the nodes on the high
// face, translated by the box length.
void check_periodic_nodes(mesh const& m, box_t const& box) {
  const int dim = box[2] > 0.0 ? 3 : 2;
  const double eps = 1e-7 * std::max({box[0], box[1], box[2]});
  for (int a = 0; a < dim; ++a) {
    std::multiset<std::vector<long long>> lo, hi;
    for (std::size_t i = 1; i < m.nodes.size(); ++i) {
      auto const& p = m.nodes[i];
      std::vector<long long> key;
      for (int k = 0; k < dim; ++k) {
        if (k != a) key.push_back(std::llround(p[k] / eps));
      }
      if (std::abs(p[a]) < eps) lo.insert(key);
      else if (std::abs(p[a] - box[a]) < eps) hi.insert(key);
    }
    if (lo != hi || lo.empty()) {
      std::cerr << "axis " << a << ": " << lo.size() << " low-face nodes vs "
                << hi.size() << " high-face nodes do not match\n";
    }
    REQUIRE(!lo.empty());
    REQUIRE(lo == hi);
  }
}

double total_measure(mesh const& m) {
  double s = 0.0;
  for (auto const& [tag, v] : m.measure_by_tag) s += v;
  return s;
}

double measure_of(mesh const& m, std::string const& name) {
  for (auto const& [tag, n] : m.physical_names) {
    if (n == name) {
      auto it = m.measure_by_tag.find(tag);
      return it == m.measure_by_tag.end() ? 0.0 : it->second;
    }
  }
  return -1.0;
}

// Area / volume of the primaries (centre in the domain). Wrap copies are
// periodic images, so clipped primaries + clipped wraps tile exactly
// this much of the domain.
double primary_measure(shape_vector const& shapes, box_t const& box,
                       std::string const& phase) {
  double s = 0.0;
  for (auto const& sh : shapes) {
    if (sh->phase_name() != phase) continue;
    if (!rvegen::is_centre_in_domain(sh->get_middle_point(), box)) continue;
    s += box[2] > 0.0 ? sh->volume() : sh->area();
  }
  return s;
}

void test_periodic_2d_circles_and_ellipses() {
  const box_t box{1.0, 0.8, 0.0};
  std::mt19937 eng{11};
  rvegen::uniform_real_distribution<> px{0.0, 1.0, eng}, py{0.0, 0.8, eng};
  rvegen::constant_distribution<> rc{0.07, eng}, ra{0.09, eng}, rb{0.04, eng};
  rvegen::uniform_real_distribution<> rot{0.0, 3.0, eng};

  rvegen::rve_generator_base<double>::input_vector inputs;
  auto ci = std::make_unique<rvegen::circle_input<>>(px, py, rc);
  ci->set_phase_name("fibre");
  inputs.push_back(std::move(ci));
  auto ei = std::make_unique<rvegen::ellipse_input<>>(px, py, ra, rb, rot);
  ei->set_phase_name("particle");
  inputs.push_back(std::move(ei));

  rvegen::periodic_generator<> gen{20000};
  rvegen::volume_fraction<> term{0.25, box[0] * box[1]};
  auto shapes = gen.compute(inputs, term, box);
  REQUIRE(shapes.size() > 10);
  bool any_wrap = false;
  for (auto const& s : shapes) {
    if (!rvegen::is_centre_in_domain(s->get_middle_point(), box)) any_wrap = true;
  }
  REQUIRE(any_wrap);  // the interesting case: inclusions crossing the boundary

  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");
  phases.add("particle");
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_periodic(true);
  writer.set_phases(&phases);
  writer.set_characteristic_length_max(0.01);

  const cleanup_on_success cleanup{"gmsh_rt_periodic_2d"};
  const auto m = write_and_mesh(writer, shapes, box, "gmsh_rt_periodic_2d");
  if (!m.ok) return;
  check_inside_domain(m, box);
  check_periodic_nodes(m, box);
  const double domain = box[0] * box[1];
  REQUIRE(std::abs(total_measure(m) - domain) < 1e-9 * domain);
  REQUIRE(m.measure_by_tag.count(0) == 0);  // every element is in a group
  REQUIRE(measure_of(m, "matrix") > 0.0);
  for (auto const* phase : {"fibre", "particle"}) {
    const double expect = primary_measure(shapes, box, phase);
    const double got = measure_of(m, phase);
    // Inscribed polygonal boundaries lose a little area (≈0.3% at
    // h = 0.01 on r = 0.07); a missing clipped piece would lose more.
    REQUIRE(got > 0.99 * expect && got <= expect * (1.0 + 1e-9));
  }
}

void test_periodic_3d_spheres_and_boxes() {
  const box_t box{1.0, 1.2, 0.9};
  std::mt19937 eng{5};
  rvegen::uniform_real_distribution<> px{0.0, 1.0, eng}, py{0.0, 1.2, eng},
      pz{0.0, 0.9, eng};
  rvegen::constant_distribution<> rs{0.16, eng}, edge{0.18, eng};

  rvegen::rve_generator_base<double>::input_vector inputs;
  auto si = std::make_unique<rvegen::sphere_input<>>(px, py, pz, rs);
  si->set_phase_name("particle");
  inputs.push_back(std::move(si));
  auto bi = std::make_unique<rvegen::box_input<>>(px, py, pz, edge, edge, edge);
  bi->set_phase_name("particle");
  inputs.push_back(std::move(bi));

  rvegen::periodic_generator<> gen{5000};
  rvegen::number_of_inclusions<> term{6};
  auto shapes = gen.compute(inputs, term, box);
  bool any_wrap = false;
  for (auto const& s : shapes) {
    if (!rvegen::is_centre_in_domain(s->get_middle_point(), box)) any_wrap = true;
  }
  REQUIRE(any_wrap);

  rvegen::phase_collection<double> phases;
  phases.add("particle");  // no declared "matrix": gmsh assigns its tag
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_periodic(true);
  writer.set_phases(&phases);
  writer.set_characteristic_length_max(0.06);

  const cleanup_on_success cleanup{"gmsh_rt_periodic_3d"};
  const auto m = write_and_mesh(writer, shapes, box, "gmsh_rt_periodic_3d");
  if (!m.ok) return;
  check_inside_domain(m, box);
  check_periodic_nodes(m, box);
  const double domain = box[0] * box[1] * box[2];
  REQUIRE(std::abs(total_measure(m) - domain) < 1e-9 * domain);
  REQUIRE(m.measure_by_tag.count(0) == 0);
  REQUIRE(measure_of(m, "matrix") > 0.0);
  const double expect = primary_measure(shapes, box, "particle");
  const double got = measure_of(m, "particle");
  // Inscribed faceted spheres lose ≈3.5% at h = 0.06 on r = 0.16
  // (converges to 1 with refinement: 0.965 at h = 0.06, 0.990 at 0.03).
  REQUIRE(got > 0.94 * expect && got <= expect * (1.0 + 1e-9));
}

// Non-periodic, no phases: overhanging shapes (random generator) and a
// convex polygon. Without Physical groups gmsh saves every element, so
// the area sum detects overlap (double meshing) and untrimmed overhang.
void test_non_periodic_overhang_and_polygon() {
  const box_t box{1.0, 1.0, 0.0};
  shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.15));
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(1.02, 0.3, 0.1));  // overhang
  shapes.emplace_back(std::make_unique<rvegen::rectangle<double>>(0.1, 0.95, 0.15, 0.2));  // overhang
  shapes.emplace_back(std::make_unique<rvegen::convex_polygon<double>>(
      std::vector<std::array<double, 2>>{{0.7, 0.7}, {0.9, 0.7}, {0.8, 0.9}}));
  for (auto& s : shapes) s->make_bounding_box();

  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_characteristic_length_max(0.05);
  const cleanup_on_success cleanup{"gmsh_rt_non_periodic_2d"};
  const auto m = write_and_mesh(writer, shapes, box, "gmsh_rt_non_periodic_2d");
  if (!m.ok) return;
  check_inside_domain(m, box);
  REQUIRE(std::abs(total_measure(m) - 1.0) < 1e-9);
}

// First-wins expectation on a fine grid of cell centres: each point is
// owned by the first shape containing it (voxel_grid semantics). Works for
// 2D (box[2] == 0) and 3D domains.
std::map<std::string, double> first_wins_areas(shape_vector const& shapes,
                                               box_t const& box) {
  const bool is_3d = box[2] > 0.0;
  const int nx = is_3d ? 120 : 1000;
  const int nz = is_3d ? 120 : 1;
  std::map<std::string, double> area;
  const double cell = box[0] * box[1] * (is_3d ? box[2] : 1.0) /
                      (double(nx) * nx * nz);
  for (int ix = 0; ix < nx; ++ix) {
    for (int iy = 0; iy < nx; ++iy) {
      for (int iz = 0; iz < nz; ++iz) {
        const std::array<double, 3> p{(ix + 0.5) * box[0] / nx,
                                      (iy + 0.5) * box[1] / nx,
                                      is_3d ? (iz + 0.5) * box[2] / nz : 0.0};
        std::string owner = "matrix";
        for (auto const& sh : shapes) {
          if (sh->is_inside(p)) { owner = sh->phase_name(); break; }
        }
        area[owner] += cell;
      }
    }
  }
  return area;
}

shape_vector make_shapes(std::vector<std::pair<std::unique_ptr<rvegen::shape_base<double>>,
                                               std::string>> items) {
  shape_vector v;
  for (auto& [sh, phase] : items) {
    sh->set_phase_name(phase);
    sh->make_bounding_box();
    v.push_back(std::move(sh));
  }
  return v;
}

using item = std::pair<std::unique_ptr<rvegen::shape_base<double>>, std::string>;
template <typename S, typename... A>
item mk(std::string phase, A... a) {
  return {std::make_unique<S>(a...), std::move(phase)};
}

// Overlapping / nested / full-domain inclusions: the geometry must be
// partitioned so the first shape owns its full area, later ones only
// the remainder, and everything else is matrix. `expect_shapes` (default:
// `shapes`) gives the order the expectation is computed in.
void check_first_wins(shape_vector const& shapes, std::string const& stem,
                      bool periodic = false, box_t const& box = {1.0, 1.0, 0.0},
                      double h = 0.01,
                      shape_vector const* expect_shapes = nullptr) {
  const cleanup_on_success cleanup{stem};
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("a");
  phases.add("b");
  phases.add("c");
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  writer.set_periodic(periodic);
  writer.set_characteristic_length_max(h);
  const auto m = write_and_mesh(writer, shapes, box, stem);
  if (!m.ok) return;
  check_inside_domain(m, box);
  if (periodic) check_periodic_nodes(m, box);
  const double domain = box[0] * box[1] * (box[2] > 0.0 ? box[2] : 1.0);
  REQUIRE(std::abs(total_measure(m) - domain) < 1e-9 * domain);
  REQUIRE(m.measure_by_tag.count(0) == 0);
  auto const expect = first_wins_areas(expect_shapes ? *expect_shapes : shapes, box);
  for (auto const* name : {"matrix", "a", "b", "c"}) {
    const auto it = expect.find(name);
    const double e = it == expect.end() ? 0.0 : it->second;
    double g = measure_of(m, name);
    if (g < 0.0) g = 0.0;  // phase without any piece may have no group
    // Faceted boundaries shrink inclusions by a few percent in 3D at
    // h = 0.05 (measured up to 3.7% of the phase) and by < 0.3% in 2D at
    // h = 0.01; the grid sampling error is far below the absolute term.
    const double rel = box[2] > 0.0 ? 0.05 : 0.005;
    if (std::abs(g - e) > rel * e + 1e-4 * domain) {
      std::cerr << stem << ": phase " << name << " measure " << g
                << " expected " << e << "\n";
      ++failures;
    }
    if (std::getenv("RVEGEN_GMSH_TEST_VERBOSE")) {
      std::cerr << stem << " " << name << " got " << g << " expect " << e
                << " diff " << g - e << "\n";
    }
  }
}

void test_first_wins_nested() {
  using C = rvegen::circle<double>;
  // big first: the small circle is entirely swallowed
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 0.5, 0.5, 0.3)); v.push_back(mk<C>("b", 0.5, 0.5, 0.1));
      return v; }()), "gmsh_rt_nested_big_first");
  // small first: the big one becomes an annulus
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 0.5, 0.5, 0.1)); v.push_back(mk<C>("b", 0.5, 0.5, 0.3));
      return v; }()), "gmsh_rt_nested_small_first");
}

void test_first_wins_overlap() {
  using C = rvegen::circle<double>;
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 0.4, 0.5, 0.2)); v.push_back(mk<C>("b", 0.6, 0.5, 0.2));
      return v; }()), "gmsh_rt_overlap_two");
  // three-way overlap, plus a disjoint circle that must stay untouched
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 0.4, 0.4, 0.22)); v.push_back(mk<C>("b", 0.6, 0.4, 0.22));
      v.push_back(mk<C>("c", 0.5, 0.6, 0.22)); v.push_back(mk<C>("a", 0.9, 0.9, 0.05));
      return v; }()), "gmsh_rt_overlap_three");
}

void test_first_wins_full_domain() {
  using C = rvegen::circle<double>;
  using R = rvegen::rectangle<double>;
  // full-domain rectangle first: it owns everything, matrix is empty
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<R>("a", 0.5, 0.5, 1.0, 1.0)); v.push_back(mk<C>("b", 0.5, 0.5, 0.2));
      return v; }()), "gmsh_rt_full_first");
  // full-domain rectangle last: it owns what the circle leaves
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("b", 0.5, 0.5, 0.2)); v.push_back(mk<R>("a", 0.5, 0.5, 1.0, 1.0));
      return v; }()), "gmsh_rt_full_last");
}

// 3D first-wins: nested, overlapping and a disjoint sphere, plus a box.
void test_first_wins_3d() {
  using S = rvegen::sphere<double>;
  using B = rvegen::box<double>;
  const box_t box{1.0, 1.0, 1.0};
  // nested, small first: the big sphere becomes a shell
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<S>("a", 0.5, 0.5, 0.5, 0.15));
      v.push_back(mk<S>("b", 0.5, 0.5, 0.5, 0.3)); return v; }()),
      "gmsh_rt_3d_nested_small_first", false, box, 0.05);
  // nested, big first: the small one is swallowed
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<S>("a", 0.5, 0.5, 0.5, 0.3));
      v.push_back(mk<S>("b", 0.5, 0.5, 0.5, 0.15)); return v; }()),
      "gmsh_rt_3d_nested_big_first", false, box, 0.05);
  // overlapping spheres + a box cutting both + a disjoint sphere
  check_first_wins(make_shapes([] { std::vector<item> v;
      v.push_back(mk<S>("a", 0.4, 0.5, 0.5, 0.25));
      v.push_back(mk<S>("b", 0.6, 0.5, 0.5, 0.25));
      v.push_back(mk<B>("c", 0.5, 0.5, 0.5, 0.2, 0.9, 0.2));
      v.push_back(mk<S>("a", 0.85, 0.15, 0.15, 0.1)); return v; }()),
      "gmsh_rt_3d_overlap", false, box, 0.05);
}

// First wins must be translation-consistent with periodic=true: two
// particles overlapping across the boundary, listed A, B, A', B' (wrap
// copies after both originals). Previously B' beat A' on the high side but
// A beat B on the low side, so the faces could not be paired.
void test_first_wins_periodic_order() {
  using C = rvegen::circle<double>;
  auto build = [](bool interleaved) {
    std::vector<item> v;
    v.push_back(mk<C>("a", 0.03, 0.5, 0.1));
    v.push_back(mk<C>("b", 0.97, 0.5, 0.1));
    if (interleaved) {  // A, B, A', B'
      v.push_back(mk<C>("a", 1.03, 0.5, 0.1));
      v.push_back(mk<C>("b", -0.03, 0.5, 0.1));
    } else {            // A, A', B, B'
      v.insert(v.begin() + 1, mk<C>("a", 1.03, 0.5, 0.1));
      v.push_back(mk<C>("b", -0.03, 0.5, 0.1));
    }
    return make_shapes(std::move(v));
  };
  // Expectation: A (and its wrap copy) owns the overlap on both sides.
  const auto expected = build(false);
  const auto shapes = build(true);
  check_first_wins(shapes, "gmsh_rt_periodic_order", true, {1.0, 1.0, 0.0}, 0.01,
                   &expected);
}

// A sphere cut by periodic faces whose plane passes close to the sphere's
// pole axis (centre 0.0024 from the low-x face) made gmsh fail with
// "curve N has 5 vertices, whereas correspondant M has 2" at coarse mesh
// sizes; the writer now moves such spheres' poles to another axis.
void test_periodic_sphere_cut_near_pole() {
  using S = rvegen::sphere<double>;
  const box_t box{1.0, 1.0, 1.0};
  const double r = 0.082939374200207416;
  auto shapes = make_shapes([&] { std::vector<item> v;
      v.push_back(mk<S>("p", -0.0024221518456235547, 0.28226640679118864,
                        0.41985503044439154, r));
      v.push_back(mk<S>("p", 0.99757784815437645, 0.28226640679118864,
                        0.41985503044439154, r));
      return v; }());
  rvegen::phase_collection<double> phases;
  phases.add("p");
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_periodic(true);
  writer.set_phases(&phases);
  writer.set_characteristic_length_max(0.2);
  const cleanup_on_success cleanup{"gmsh_rt_pole"};
  const auto m = write_and_mesh(writer, shapes, box, "gmsh_rt_pole");
  if (!m.ok) return;
  check_inside_domain(m, box);
  check_periodic_nodes(m, box);
  REQUIRE(std::abs(total_measure(m) - 1.0) < 1e-9);
  REQUIRE(measure_of(m, "p") > 0.0);
}

// An unmatched boundary-cut inclusion with periodic=true must make gmsh
// fail loudly instead of silently producing a non-periodic mesh.
void test_periodic_unmatched_errors() {
  shape_vector shapes = make_shapes([] { std::vector<item> v;
      v.push_back(mk<rvegen::circle<double>>("a", 1.0, 0.5, 0.2)); return v; }());
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_periodic(true);
  const std::string stem = "gmsh_rt_unmatched_periodic";
  {
    std::ostringstream buf;
    writer.write(buf, shapes, {1.0, 1.0, 0.0});
    std::ofstream out{stem + ".geo"};
    out << buf.str();
  }
  const std::string cmd = std::string{"\""} + RVEGEN_GMSH_EXECUTABLE + "\" -2 " +
                          stem + ".geo -format msh22 -o " + stem + ".msh > " +
                          stem + ".log 2>&1";
  const int rc = std::system(cmd.c_str());
  std::ifstream log{stem + ".log"};
  std::stringstream ss; ss << log.rdbuf();
  const bool reported = ss.str().find("no periodic partner") != std::string::npos;
  REQUIRE(reported);
  if (!reported || rc == 0) {
    if (rc == 0) std::cerr << "unmatched periodic: gmsh exited 0\n";
  }
  if (reported) {
    for (auto const* ext : {".geo", ".msh", ".log"}) std::remove((stem + ext).c_str());
  }
}

// --- scale independence and empty clips -------------------------------

std::string script_of(rvegen::gmsh_geo_writer<double> const& writer,
                      shape_vector const& shapes, box_t const& box) {
  std::ostringstream buf;
  writer.write(buf, shapes, box);
  return buf.str();
}

// A 2D RVE of edge `L` (in absolute units) with a fully inside circle A and
// a circle B centred on the x = L face, plus (periodic) its wrap image on
// x = 0. OpenCASCADE's absolute tolerances used to drop B at L = 1e-5 and
// fail outright at 1e-6; the script is box-relative now. B must cover pi r^2
// / 2 (pi r^2 with the image), the mesh must come out in absolute units.
void check_scaled_boundary_cut_2d(double L, bool periodic) {
  using C = rvegen::circle<double>;
  const double r = 0.1 * L;
  std::vector<item> v;
  v.push_back(mk<C>("a", 0.2 * L, 0.3 * L, r));
  v.push_back(mk<C>("b", L, 0.3 * L, r));
  if (periodic) v.push_back(mk<C>("b", 0.0, 0.3 * L, r));
  const auto shapes = make_shapes(std::move(v));
  const box_t box{L, L, 0.0};
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("a");
  phases.add("b");
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  writer.set_periodic(periodic);
  writer.set_characteristic_length_max(0.02 * L);
  const std::string stem = std::string{"gmsh_rt_scale2d_"} +
                           (periodic ? "per_" : "non_") + std::to_string(L);
  const cleanup_on_success cleanup{stem};
  const auto m = write_and_mesh(writer, shapes, box, stem);
  if (!m.ok) return;
  check_inside_domain(m, box);
  if (periodic) check_periodic_nodes(m, box);
  const double pi_r2 = std::acos(-1.0) * r * r;
  REQUIRE(std::abs(total_measure(m) - L * L) < 1e-9 * L * L);
  const double a = measure_of(m, "a");
  const double b = measure_of(m, "b");
  // Inscribed polygons lose ~1% at this mesh size; a missing piece loses 50%+.
  REQUIRE(a > 0.97 * pi_r2 && a <= pi_r2 * (1.0 + 1e-9));
  const double b_expect = periodic ? pi_r2 : 0.5 * pi_r2;
  REQUIRE(b > 0.97 * b_expect && b <= b_expect * (1.0 + 1e-9));
  // saved in absolute units: the far node sits at L
  double xmax = 0.0;
  for (std::size_t i = 1; i < m.nodes.size(); ++i) xmax = std::max(xmax, m.nodes[i][0]);
  REQUIRE(std::abs(xmax - L) < 1e-9 * L);
}

void test_scale_independence_2d() {
  for (double L : {1e-9, 1e-5, 1e5, 1e9}) {
    check_scaled_boundary_cut_2d(L, false);
    check_scaled_boundary_cut_2d(L, true);
  }
}

void test_scale_independence_3d() {
  using S = rvegen::sphere<double>;
  for (double L : {1e-6, 1e6}) {
    for (bool periodic : {false, true}) {
      const double r = 0.15 * L;
      std::vector<item> v;
      v.push_back(mk<S>("a", 0.3 * L, 0.5 * L, 0.5 * L, r));
      v.push_back(mk<S>("b", L, 0.5 * L, 0.5 * L, r));
      if (periodic) v.push_back(mk<S>("b", 0.0, 0.5 * L, 0.5 * L, r));
      const auto shapes = make_shapes(std::move(v));
      const box_t box{L, L, L};
      rvegen::phase_collection<double> phases;
      phases.add("matrix");
      phases.add("a");
      phases.add("b");
      rvegen::gmsh_geo_writer<double> writer{};
      writer.set_phases(&phases);
      writer.set_periodic(periodic);
      writer.set_characteristic_length_max(0.1 * L);
      const std::string stem = std::string{"gmsh_rt_scale3d_"} +
                               (periodic ? "per_" : "non_") + std::to_string(L);
      const cleanup_on_success cleanup{stem};
      const auto m = write_and_mesh(writer, shapes, box, stem);
      if (!m.ok) continue;
      check_inside_domain(m, box);
      if (periodic) check_periodic_nodes(m, box);
      const double ball = 4.0 / 3.0 * std::acos(-1.0) * r * r * r;
      REQUIRE(std::abs(total_measure(m) - L * L * L) < 1e-8 * L * L * L);
      const double b_expect = periodic ? ball : 0.5 * ball;
      REQUIRE(measure_of(m, "b") > 0.7 * b_expect);
      REQUIRE(measure_of(m, "b") <= b_expect * (1.0 + 1e-9));
    }
  }
}

// Unit boxes keep the old script (no scaling lines); other boxes say what
// was done in the header.
void test_scale_header() {
  using C = rvegen::circle<double>;
  rvegen::gmsh_geo_writer<double> writer{};
  auto const unit = script_of(writer, make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 0.5, 0.5, 0.1)); return v; }()), {1.0, 1.0, 0.0});
  REQUIRE(unit.find("ScalingFactor") == std::string::npos);
  auto const big = script_of(writer, make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 5.0, 2.0, 1.0)); return v; }()), {10.0, 4.0, 0.0});
  REQUIRE(big.find("Mesh.ScalingFactor = 10;") != std::string::npos);
  REQUIRE(big.find("Rectangle(1) = {0, 0, 0, 1, 0.4, 0};") != std::string::npos);
  REQUIRE(big.find("Disk(2) = {0.5, 0.2, 0, 0.1, 0.1};") != std::string::npos);
}

// Mesh `shapes` periodically and require a clean exit plus tiling
// (total measure == domain, periodic nodes match, every shape's phase
// present). The wrap images that miss the cell must not disturb anything.
void check_periodic_images(shape_vector const& shapes, box_t const& box,
                           std::string const& stem, double h) {
  const cleanup_on_success cleanup{stem};
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("p");
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_phases(&phases);
  writer.set_periodic(true);
  writer.set_characteristic_length_max(h);
  const auto m = write_and_mesh(writer, shapes, box, stem);
  if (!m.ok) return;
  check_inside_domain(m, box);
  check_periodic_nodes(m, box);
  const double domain = box[0] * box[1] * (box[2] > 0.0 ? box[2] : 1.0);
  REQUIRE(std::abs(total_measure(m) - domain) < 1e-8 * domain);
  REQUIRE(measure_of(m, "matrix") > 0.5 * domain);
  REQUIRE(measure_of(m, "p") > 0.0);
}

// gmsh 4.14.0 answers an empty BooleanIntersection with the tool's tag
// (the domain), which deleted the matrix. Wrap images that miss the cell
// must be harmless: the exact circle / sphere test drops them before they
// are emitted, and every other shape's result list is guarded.
void test_periodic_images_missing_the_cell() {
  using S = rvegen::sphere<double>;
  using C = rvegen::circle<double>;
  using P = rvegen::convex_polygon<double>;
  // The spheres of examples/periodic_spheres_gmsh_3d that crossed two faces:
  // image 7 (centre (0.7015, 1.1465, -0.0736), r 0.1549) is 0.164 from the box.
  {
    std::vector<item> v;
    const double x = 0.7015279153230102, y = 0.14652468207184144;
    const double z = 0.9264383943468313, r = 0.15488354350926983;
    v.push_back(mk<S>("p", x, y, z, r));
    v.push_back(mk<S>("p", x, y + 1.0, z, r));
    v.push_back(mk<S>("p", x, y, z - 1.0, r));
    v.push_back(mk<S>("p", x, y + 1.0, z - 1.0, r));
    const auto shapes = make_shapes(std::move(v));
    rvegen::gmsh_geo_writer<double> writer{};
    writer.set_periodic(true);
    const auto txt = script_of(writer, shapes, {1.0, 1.0, 1.0});
    REQUIRE(txt.find("Sphere(4)") != std::string::npos);
    REQUIRE(txt.find("Sphere(5)") == std::string::npos);  // misses the cell
    check_periodic_images(shapes, {1.0, 1.0, 1.0}, "gmsh_rt_missing_sphere", 0.06);
  }
  // 2D: a circle crossing two faces misses the corner cell.
  {
    std::vector<item> v;
    v.push_back(mk<C>("p", 0.92, 0.92, 0.12));
    v.push_back(mk<C>("p", -0.08, 0.92, 0.12));
    v.push_back(mk<C>("p", 0.92, -0.08, 0.12));
    v.push_back(mk<C>("p", -0.08, -0.08, 0.12));  // reaches the cell: dist 0 < r
    v.push_back(mk<C>("p", 1.92, 0.5, 0.12));     // far outside
    const auto shapes = make_shapes(std::move(v));
    check_periodic_images(shapes, {1.0, 1.0, 0.0}, "gmsh_rt_missing_circle", 0.02);
  }
  // A shape whose bounding box overlaps the cell but which does not: the
  // clip is empty and must not be mistaken for the domain.
  {
    std::vector<item> v;
    v.push_back(mk<C>("p", 0.3, 0.3, 0.1));
    v.push_back(mk<P>("p", std::vector<typename P::point_type>{{0.9, 1.2}, {1.2, 0.9}, {1.2, 1.2}}));
    const auto shapes = make_shapes(std::move(v));
    rvegen::gmsh_geo_writer<double> writer{};
    const auto txt = script_of(writer, shapes, {1.0, 1.0, 0.0});
    // A polygon is pre-clipped to the domain in C++: one that misses the
    // box is skipped outright; any other shape keeps the guarded list.
    REQUIRE(txt.find("!= 1") != std::string::npos ||
            txt.find("polygon skipped") != std::string::npos);
    // Not periodic: the polygon-only shape exercises the guard on its own.
    const std::string stem = "gmsh_rt_missing_polygon";
    const cleanup_on_success cleanup{stem};
    rvegen::phase_collection<double> phases;
    phases.add("matrix");
    phases.add("p");
    writer.set_phases(&phases);
    writer.set_characteristic_length_max(0.03);
    const auto m = write_and_mesh(writer, shapes, {1.0, 1.0, 0.0}, stem);
    if (m.ok) {
      REQUIRE(std::abs(total_measure(m) - 1.0) < 1e-9);
      REQUIRE(measure_of(m, "matrix") > 0.9);
    }
  }
}

// A shape that clearly overlaps the domain but clips to nothing is an
// error in the script, not a silent "vanished". Text check only: a real
// OpenCASCADE failure cannot be provoked on purpose.
void test_clip_error_guard_text() {
  using C = rvegen::circle<double>;
  rvegen::gmsh_geo_writer<double> writer{};
  auto const overlap = script_of(writer, make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 1.0, 0.5, 0.2)); return v; }()), {1.0, 1.0, 0.0});
  REQUIRE(overlap.find("clearly overlaps the domain") != std::string::npos);
  // inside the domain: nothing is clipped, nothing to guard
  auto const inside = script_of(writer, make_shapes([] { std::vector<item> v;
      v.push_back(mk<C>("a", 0.5, 0.5, 0.2)); return v; }()), {1.0, 1.0, 0.0});
  REQUIRE(inside.find("Error(") == std::string::npos);
}

// A circle / sphere exactly tangent to a periodic face used to make the
// periodic output fail (one-sided point contact).
void test_periodic_tangent() {
  using C = rvegen::circle<double>;
  using S = rvegen::sphere<double>;
  {
    std::vector<item> v;
    v.push_back(mk<C>("p", 0.1, 0.5, 0.1));
    v.push_back(mk<C>("p", 1.1, 0.5, 0.1));
    check_periodic_images(make_shapes(std::move(v)), {1.0, 1.0, 0.0},
                          "gmsh_rt_tangent_circle", 0.05);
  }
  {
    std::vector<item> v;
    v.push_back(mk<S>("p", 0.2, 0.5, 0.5, 0.2));
    v.push_back(mk<S>("p", 1.2, 0.5, 0.5, 0.2));
    check_periodic_images(make_shapes(std::move(v)), {1.0, 1.0, 1.0},
                          "gmsh_rt_tangent_sphere", 0.08);
  }
}

// Rejected input must not truncate an existing file, and non-finite
// numbers must be rejected.
void test_writer_rejects_bad_input() {
  const std::string path = "gmsh_rt_reject.geo";
  { std::ofstream o{path}; o << "KEEP\n"; }
  shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, 0.1));
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.1));
  rvegen::gmsh_geo_writer<double> writer{path};
  bool threw = false;
  try { writer.run(shapes, {1.0, 1.0, 0.0}); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  std::ifstream in{path};
  std::string line; std::getline(in, line);
  REQUIRE(line == "KEEP");

  shape_vector nan_shapes;
  nan_shapes.emplace_back(std::make_unique<rvegen::circle<double>>(0.5, 0.5, std::nan("")));
  std::ostringstream buf;
  threw = false;
  try { writer.write(buf, nan_shapes, {1.0, 1.0, 0.0}); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  REQUIRE(buf.str().empty());
  threw = false;
  try { writer.write(buf, {}, {1.0, std::nan(""), 0.0}); } catch (std::invalid_argument const&) { threw = true; }
  REQUIRE(threw);
  REQUIRE(buf.str().empty());
  if (threw) std::remove(path.c_str());
}

// ---- tessellations (#8) ---------------------------------------------------

// Seeds with a minimum pairwise distance (keeps the tessellation clear of
// the precision problems of near-degenerate cells).
template <std::size_t D>
std::vector<std::array<double, D>> separated_seeds(std::size_t n, box_t const& box,
                                                   double min_dist, unsigned rng) {
  std::mt19937 eng{rng};
  std::vector<std::array<double, D>> out;
  while (out.size() < n) {
    std::array<double, D> p{};
    for (std::size_t k = 0; k < D; ++k) {
      p[k] = std::uniform_real_distribution<double>{0.05 * box[k], 0.95 * box[k]}(eng);
    }
    bool ok = true;
    for (auto const& q : out) {
      double d2 = 0.0;
      for (std::size_t k = 0; k < D; ++k) d2 += (p[k] - q[k]) * (p[k] - q[k]);
      if (d2 < min_dist * min_dist) { ok = false; break; }
    }
    if (ok) out.push_back(p);
  }
  return out;
}

// No two nodes may coincide: duplicates would mean the shared faces were
// meshed twice, i.e. the mesh is not conforming.
void check_no_duplicate_nodes(mesh const& m, double scale = 1.0) {
  std::set<std::array<long long, 3>> seen;
  std::size_t dup = 0;
  for (std::size_t i = 1; i < m.nodes.size(); ++i) {
    std::array<long long, 3> k{};
    for (int a = 0; a < 3; ++a) k[a] = std::llround(m.nodes[i][a] / (scale * 1e-8));
    if (!seen.insert(k).second) ++dup;
  }
  if (dup != 0) std::cerr << dup << " duplicate nodes\n";
  REQUIRE(dup == 0);
}

void check_tessellation(shape_vector const& shapes, box_t const& box,
                        std::string const& stem, double h, double tol) {
  const cleanup_on_success cleanup{stem};
  rvegen::gmsh_geo_writer<double> writer{};
  writer.set_tessellation(true);
  writer.set_characteristic_length_max(h);
  const auto m = write_and_mesh(writer, shapes, box, stem);
  if (!m.ok) return;
  check_inside_domain(m, box);
  check_no_duplicate_nodes(m, std::max({box[0], box[1], box[2]}));
  const bool is_3d = box[2] > 0.0;
  const double domain = is_3d ? box[0] * box[1] * box[2] : box[0] * box[1];
  double sum = 0.0;
  for (std::size_t i = 0; i < shapes.size(); ++i) {
    const double expect = is_3d ? shapes[i]->volume() : shapes[i]->area();
    const double got = measure_of(m, "grain_" + std::to_string(i));
    // Planar faces: the mesh reproduces the polyhedron exactly.
    if (!(std::abs(got - expect) <= tol * domain)) {
      std::cerr << stem << ": grain " << i << " measure " << got << " expected "
                << expect << "\n";
      ++failures;
    }
    sum += got;
  }
  REQUIRE(std::abs(sum - domain) < 1e-9 * domain);
  std::cout << stem << ": " << shapes.size() << " grains, nodes " << m.nodes.size() - 1 << ", sum " << sum << " / " << domain << "\n";
  if (is_3d) {
    // Boundary physical surface = the box surface.
    const double surf = 2.0 * (box[0] * box[1] + box[1] * box[2] + box[0] * box[2]);
    const double got = measure_of(m, "boundary");
    REQUIRE(std::abs(got - surf) < 1e-9 * surf);
  }
}

void test_tessellation_3d() {
  const box_t box{1.0, 1.2, 0.9};
  rvegen::voronoi_generator_3d<double> gen{
      box[0], box[1], box[2], separated_seeds<3>(20, box, 0.2, 3)};
  auto shapes = gen.build();
  REQUIRE(shapes.size() == 20);
  check_tessellation(shapes, box, "gmsh_rt_tess_3d", 0.15, 1e-9);
}

void test_tessellation_2d() {
  const box_t box{1.0, 0.8, 0.0};
  rvegen::voronoi_generator_2d<double> gen{
      box[0], box[1], separated_seeds<2>(30, box, 0.09, 4)};
  auto shapes = rvegen::voronoi_to_shapes(gen);
  REQUIRE(shapes.size() == 30);
  check_tessellation(shapes, box, "gmsh_rt_tess_2d", 0.05, 1e-9);
}

void test_tessellation_rejects_mixing() {
  const box_t box{1.0, 1.0, 1.0};
  rvegen::voronoi_generator_3d<double> gen{
      1.0, 1.0, 1.0, separated_seeds<3>(4, box, 0.2, 1)};
  auto shapes = gen.build();
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.1));
  rvegen::gmsh_geo_writer<double> writer{};
  std::ostringstream buf;
  bool threw = false;
  try { writer.write(buf, shapes, box); } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
  REQUIRE(buf.str().empty());
}

std::vector<std::array<double, 3>> jittered_lattice(int n, double jit, unsigned seed) {
  std::mt19937 g{seed};
  std::uniform_real_distribution<double> u{-1.0, 1.0};
  std::vector<std::array<double, 3>> s;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      for (int k = 0; k < n; ++k)
        s.push_back({(i + 0.5 + jit * u(g)) / n, (j + 0.5 + jit * u(g)) / n,
                     (k + 0.5 + jit * u(g)) / n});
  return s;
}

// Lattice cells meet in 8-fold degenerate vertices; jittering the seeds
// turns them into tiny edges / faces. Above the merge tolerance the mesh
// is regular, just below it the features are collapsed.
void test_tessellation_jittered_lattice() {
  const box_t box{1.0, 1.0, 1.0};
  for (double jit : {1e-3, 1e-4, 3e-6}) {
    rvegen::voronoi_generator_3d<double> gen{1.0, 1.0, 1.0,
                                            jittered_lattice(3, jit, 1)};
    auto shapes = gen.build();
    REQUIRE(shapes.size() == 27);
    check_tessellation(shapes, box, "gmsh_rt_tess_jitter", 0.3, 1e-4);
  }
}

// Features far below what gmsh can handle used to be written as-is and made
// gmsh segfault (rc 139) or fail with PLC errors. Now the writer must either
// emit geometry that gmsh meshes or reject the cells with a clear error.
void test_tessellation_near_degenerate_never_crashes() {
  const box_t box{1.0, 1.0, 1.0};
  int rejected = 0;
  // Lattices jittered by jit, and (negative jit = -d) a lattice with one
  // seed duplicated at distance d. A robust generator may itself reject
  // the input: coincident seeds (std::runtime_error "seeds i and j are
  // coincident"), or a near-lattice set whose cells do not fill the box
  // ("cell volumes sum to ..."). Both count as a rejection, the same as
  // the writer's own.
  for (double jit : {1e-6, 1e-7, 1e-8, 1e-9, 1e-11, -1e-12, -1e-10}) {
    for (unsigned seed : {1u, 2u, 3u}) {
      shape_vector shapes;
      rvegen::gmsh_geo_writer<double> writer{};
      writer.set_characteristic_length_max(0.3);
      bool threw = false;
      try {
        auto seeds = jittered_lattice(2, jit > 0.0 ? jit : 0.0, seed);
        if (jit < 0.0) {
          seeds.push_back(seeds[3]);
          seeds.back()[0] += -jit;
        }
        rvegen::voronoi_generator_3d<double> gen{1.0, 1.0, 1.0, std::move(seeds)};
        shapes = gen.build();
        std::ostringstream buf;
        writer.write(buf, shapes, box);
      } catch (std::runtime_error const& e) {
        threw = true;
        ++rejected;
        const std::string what{e.what()};
        REQUIRE(what.find("seeds") != std::string::npos ||
                what.find("cell volumes sum to") != std::string::npos);
      }
      if (!threw) {
        const cleanup_on_success cleanup{"gmsh_rt_degenerate"};
        const auto m = write_and_mesh(writer, shapes, box, "gmsh_rt_degenerate");
        // Merged features move vertices by <= 1e-5 of the box.
        if (m.ok) {
          double vol = 0.0;
          for (std::size_t i = 0; i < shapes.size(); ++i) {
            vol += std::max(0.0, measure_of(m, "grain_" + std::to_string(i)));
          }
          REQUIRE(std::abs(vol - 1.0) < 1e-4);
        }
      }
    }
  }
  REQUIRE(rejected > 0);  // the degenerate inputs are not silently passed on
}

std::vector<gte::Vector<3, double>> cube_vertices(double x0, double x1, double y0,
                                                  double y1, double z0, double z1) {
  std::vector<gte::Vector<3, double>> v;
  for (int k = 0; k < 2; ++k)
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        gte::Vector<3, double> p;
        p[0] = i ? x1 : x0;
        p[1] = j ? y1 : y0;
        p[2] = k ? z1 : z0;
        v.push_back(p);
      }
  return v;
}

std::unique_ptr<rvegen::shape_base<double>> cube_cell(double x0, double x1, double y0,
                                                      double y1, double z0, double z1) {
  return std::make_unique<rvegen::voronoi_cell<double>>(
      cube_vertices(x0, x1, y0, y1, z0, z1),
      std::vector<std::vector<std::size_t>>{{0, 2, 3, 1}, {0, 4, 6, 2}, {1, 3, 7, 5},
                                            {0, 1, 5, 4}, {2, 6, 7, 3}, {4, 5, 7, 6}});
}

// Cells of a unit-box tessellation scaled by L (the generator is built at
// unit scale so this test does not depend on how it copes with tiny boxes).
shape_vector scaled_cells(shape_vector const& shapes, double L) {
  shape_vector out;
  for (auto const& sh : shapes) {
    if (auto const* c = dynamic_cast<rvegen::voronoi_cell<double> const*>(sh.get())) {
      auto v = c->vertices();
      for (auto& p : v) { p[0] *= L; p[1] *= L; p[2] *= L; }
      out.push_back(std::make_unique<rvegen::voronoi_cell<double>>(std::move(v), c->faces()));
    } else if (auto const* q = dynamic_cast<rvegen::convex_polygon<double> const*>(sh.get())) {
      auto v = q->vertices();
      for (auto& p : v) { p[0] *= L; p[1] *= L; }
      out.push_back(std::make_unique<rvegen::convex_polygon<double>>(std::move(v)));
    }
  }
  return out;
}

// Tessellations of boxes that are tiny / huge in absolute units: written
// box-relative, saved in absolute units, grains tile the box.
void test_tessellation_scale_independence() {
  for (double L : {1e-5, 1e5}) {
    {
      const box_t unit{1.0, 1.2, 0.9};
      rvegen::voronoi_generator_3d<double> gen{
          unit[0], unit[1], unit[2], separated_seeds<3>(12, unit, 0.25, 3)};
      auto base = gen.build();
      REQUIRE(base.size() == 12);
      auto shapes = scaled_cells(base, L);
      const box_t box{L * unit[0], L * unit[1], L * unit[2]};
      check_tessellation(shapes, box, "gmsh_rt_tess_scale3d_" + std::to_string(L),
                         0.15 * L, 1e-9);
    }
    {
      const box_t unit{1.0, 0.8, 0.0};
      rvegen::voronoi_generator_2d<double> gen{
          unit[0], unit[1], separated_seeds<2>(15, unit, 0.15, 4)};
      auto base = rvegen::voronoi_to_shapes(gen);
      REQUIRE(base.size() == 15);
      auto shapes = scaled_cells(base, L);
      const box_t box{L * unit[0], L * unit[1], 0.0};
      check_tessellation(shapes, box, "gmsh_rt_tess_scale2d_" + std::to_string(L),
                         0.08 * L, 1e-9);
    }
  }
}

void test_tessellation_duplicate_seeds_and_holes() {
  const box_t box{1.0, 1.0, 1.0};
  rvegen::gmsh_geo_writer<double> writer{};
  {  // coincident seeds: rejected by the generator or, failing that, by the writer
    std::string msg;
    try {
      rvegen::voronoi_generator_3d<double> gen{
          1.0, 1.0, 1.0,
          {{0.3, 0.3, 0.3}, {0.3, 0.3, 0.3}, {0.7, 0.7, 0.7}}};
      auto shapes = gen.build();
      std::ostringstream buf;
      writer.write(buf, shapes, box);
    } catch (std::runtime_error const& e) { msg = e.what(); }
    REQUIRE(!msg.empty());
    REQUIRE(msg.find("seeds") != std::string::npos);
  }
  // The writer's own checks, fed hand-built cells so they stay covered
  // whether or not the generator already refuses such seeds.
  {  // three identical cells: the same faces claimed three times
    shape_vector shapes;
    for (int i = 0; i < 3; ++i) shapes.push_back(cube_cell(0, 1, 0, 1, 0, 1));
    std::string msg;
    try { std::ostringstream buf; writer.write(buf, shapes, box); }
    catch (std::runtime_error const& e) { msg = e.what(); }
    REQUIRE(msg.find("more than two cells") != std::string::npos);
    REQUIRE(msg.find("seeds") != std::string::npos);
  }
  {  // a cell that collapses below the merge tolerance (coincident seeds)
    shape_vector shapes;
    shapes.push_back(cube_cell(0, 1, 0, 1, 0, 1));
    shapes.push_back(cube_cell(0.5, 0.5 + 1e-8, 0.5, 0.5 + 1e-8, 0.5, 0.5 + 1e-8));
    std::string msg;
    try { std::ostringstream buf; writer.write(buf, shapes, box); }
    catch (std::runtime_error const& e) { msg = e.what(); }
    REQUIRE(msg.find("collapsed") != std::string::npos);
    REQUIRE(msg.find("seeds") != std::string::npos);
  }
  {  // a cell with a face missing is not a closed polyhedron
    auto open = std::make_unique<rvegen::voronoi_cell<double>>(
        cube_vertices(0, 1, 0, 1, 0, 1),
        std::vector<std::vector<std::size_t>>{
            {0, 2, 3, 1}, {0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}});
    shape_vector shapes;
    shapes.push_back(std::move(open));
    std::string msg;
    try { std::ostringstream buf; writer.write(buf, shapes, box); }
    catch (std::runtime_error const& e) { msg = e.what(); }
    REQUIRE(msg.find("not a closed polyhedron") != std::string::npos);
  }
  {  // phases would be silently ignored: rejected
    rvegen::voronoi_generator_3d<double> gen{
        1.0, 1.0, 1.0, {{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}}};
    auto shapes = gen.build();
    rvegen::phase_collection<double> phases;
    phases.add("a");
    rvegen::gmsh_geo_writer<double> pw{};
    pw.set_phases(&phases);
    std::string msg;
    try { std::ostringstream buf; pw.write(buf, shapes, box); }
    catch (std::runtime_error const& e) { msg = e.what(); }
    REQUIRE(msg.find("phases are not supported") != std::string::npos);
  }
  {  // a missing neighbour leaves an interior face owned by one cell
    rvegen::voronoi_generator_3d<double> gen{
        1.0, 1.0, 1.0, {{0.25, 0.5, 0.5}, {0.75, 0.5, 0.5}}};
    auto shapes = gen.build();
    REQUIRE(shapes.size() == 2);
    shapes.pop_back();
    std::string msg;
    try { std::ostringstream buf; writer.write(buf, shapes, box); }
    catch (std::runtime_error const& e) { msg = e.what(); }
    REQUIRE(msg.find("not watertight") != std::string::npos);
  }
}

} // namespace

int main() {
  test_periodic_2d_circles_and_ellipses();
  test_periodic_3d_spheres_and_boxes();
  test_non_periodic_overhang_and_polygon();
  test_first_wins_nested();
  test_first_wins_overlap();
  test_first_wins_full_domain();
  test_first_wins_3d();
  test_first_wins_periodic_order();
  test_periodic_sphere_cut_near_pole();
  test_periodic_unmatched_errors();
  test_scale_independence_2d();
  test_scale_independence_3d();
  test_scale_header();
  test_periodic_images_missing_the_cell();
  test_clip_error_guard_text();
  test_periodic_tangent();
  test_writer_rejects_bad_input();
  test_tessellation_3d();
  test_tessellation_2d();
  test_tessellation_rejects_mixing();
  test_tessellation_jittered_lattice();
  test_tessellation_scale_independence();
  test_tessellation_near_degenerate_never_crashes();
  test_tessellation_duplicate_seeds_and_holes();
  if (failures == 0) {
    std::cout << "gmsh_geo_roundtrip_test: PASS\n";
    return 0;
  }
  std::cerr << failures << " gmsh round-trip failure(s)\n";
  return 1;
}
