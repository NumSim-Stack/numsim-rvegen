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
  return 0.5 * std::abs((b[0] - a[0]) * (c[1] - a[1]) -
                        (c[0] - a[0]) * (b[1] - a[1]));
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

// Write `shapes` with `writer`, mesh with gmsh, return the parsed mesh.
mesh write_and_mesh(rvegen::gmsh_geo_writer<double> const& writer,
                    shape_vector const& shapes, box_t const& box,
                    std::string const& stem) {
  const std::string geo = stem + ".geo";
  const std::string msh = stem + ".msh";
  const std::string log = stem + ".log";
  {
    std::ofstream out{geo};
    writer.write(out, shapes, box);
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
  const auto m = write_and_mesh(writer, shapes, box, "gmsh_rt_non_periodic_2d");
  if (!m.ok) return;
  check_inside_domain(m, box);
  REQUIRE(std::abs(total_measure(m) - 1.0) < 1e-9);
}

} // namespace

int main() {
  test_periodic_2d_circles_and_ellipses();
  test_periodic_3d_spheres_and_boxes();
  test_non_periodic_overhang_and_polygon();
  if (failures == 0) {
    std::cout << "gmsh_geo_roundtrip_test: PASS\n";
    return 0;
  }
  std::cerr << failures << " gmsh round-trip failure(s)\n";
  return 1;
}
