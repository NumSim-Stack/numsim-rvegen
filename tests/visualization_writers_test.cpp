// Issue #82: tube winding, three.js camera framing vs. domain scale, and
// the "no silently dropped shapes" policy of the visualization writers.

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numbers>
#include <sstream>
#include <string>
#include <thread>
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

using shapes_t = rvegen::three_js_writer<double>::shape_vector;

// Value of `key: <number>` in the emitted JS (first occurrence after `view`).
double js_number(std::string const& txt, std::string const& key) {
  const auto v = txt.find("const view");
  const auto p = txt.find(key + ": ", v);
  return std::stod(txt.substr(p + key.size() + 2));
}
std::array<double, 3> js_vec(std::string const& txt, std::string const& key) {
  const auto v = txt.find("const view");
  auto p = txt.find(key + ": [", v) + key.size() + 3;
  std::array<double, 3> r{};
  for (auto& x : r) {
    std::size_t used = 0;
    x = std::stod(txt.substr(p), &used);
    p += used + 1;
  }
  return r;
}

std::size_t count_of(std::string const& s, std::string const& what) {
  std::size_t n = 0, p = 0;
  while ((p = s.find(what, p)) != std::string::npos) { ++n; p += what.size(); }
  return n;
}

void test_tube_winding_outward() {
  // Straight and bent centerlines; closed mesh => signed volume > 0 and
  // every side triangle's normal points away from the axis.
  for (int bent = 0; bent < 2; ++bent) {
    std::vector<std::array<double, 3>> cl{{0, 0, 0}, {0.5, 0, 0}, {1, 0, 0}};
    if (bent) cl = {{0, 0, 0}, {0.5, 0, 0}, {0.8, 0.3, 0.1}};
    rvegen::polyline_tube<double> tube{cl, 0.05};
    const auto m = rvegen::to_mesh(tube, 16);
    double vol = 0;
    const std::size_t side_tris = (cl.size() - 1) * 2 * 16;
    std::size_t bad = 0, i = 0;
    for (auto const& t : m.tris) {
      auto const& a = m.verts[t[0]];
      auto const& b = m.verts[t[1]];
      auto const& c = m.verts[t[2]];
      vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) -
              a[1] * (b[0] * c[2] - b[2] * c[0]) +
              a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
      if (i++ < side_tris) {
        const std::array<double, 3> e1{b[0]-a[0], b[1]-a[1], b[2]-a[2]};
        const std::array<double, 3> e2{c[0]-a[0], c[1]-a[1], c[2]-a[2]};
        const std::array<double, 3> n{e1[1]*e2[2]-e1[2]*e2[1],
                                      e1[2]*e2[0]-e1[0]*e2[2],
                                      e1[0]*e2[1]-e1[1]*e2[0]};
        auto const& vn = m.normals[t[0]];  // outward radial vertex normal
        if (n[0]*vn[0] + n[1]*vn[1] + n[2]*vn[2] <= 0) ++bad;
      }
    }
    REQUIRE(bad == 0);
    REQUIRE(vol > 0);
    if (!bent) {
      const double expected = std::numbers::pi * 0.05 * 0.05 * 1.0;
      REQUIRE(std::abs(vol - expected) / expected < 0.05);
    }
  }
}

void test_camera_scales_with_domain() {
  for (double L : {1e-6, 1e-3, 1.0, 1e3}) {
    shapes_t shapes;
    shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(
        0.5 * L, 0.5 * L, 0.5 * L, 0.1 * L));
    std::ostringstream os;
    rvegen::three_js_writer<double>{}.write(os, shapes, {L, 2 * L, 0.5 * L});
    const auto txt = os.str();
    const double half_diag = 0.5 * L * std::sqrt(1 + 4 + 0.25);
    const auto c = js_vec(txt, "center");
    const auto pos = js_vec(txt, "position");
    const double dist = js_number(txt, "distance");
    const double near_ = js_number(txt, "near");
    const double far_ = js_number(txt, "far");
    const double rel = 1e-6;
    REQUIRE(std::abs(c[0] - 0.5 * L) < rel * L);
    REQUIRE(std::abs(c[1] - L) < rel * L);
    REQUIRE(std::abs(c[2] - 0.25 * L) < rel * L);
    double d2 = 0;
    for (int k = 0; k < 3; ++k) d2 += (pos[k] - c[k]) * (pos[k] - c[k]);
    REQUIRE(std::abs(std::sqrt(d2) - dist) < 1e-6 * dist);
    REQUIRE(dist > half_diag * 1.5 && dist < half_diag * 5);   // frames the box
    REQUIRE(near_ > 0 && near_ < dist - half_diag);            // nothing clipped
    REQUIRE(far_ > dist + half_diag);
    REQUIRE(far_ / near_ < 1e4);                               // depth precision
    REQUIRE(txt.find("GridHelper(view.maxDim") != std::string::npos);
  }
}

void test_tube_rendered_via_dispatcher() {
  rvegen::mesh_dispatcher<double>::instance().clear();
  rvegen::reset_unrendered_warnings();
  shapes_t shapes;
  shapes.emplace_back(std::make_unique<rvegen::polyline_tube<double>>(
      std::vector<std::array<double, 3>>{{0.1, 0.5, 0.5}, {0.9, 0.5, 0.5}},
      0.05));
  std::ostringstream cerr_buf;
  auto* old = std::cerr.rdbuf(cerr_buf.rdbuf());
  std::ostringstream js, svg;
  rvegen::three_js_writer<double>{}.write(js, shapes, {1, 1, 1});
  rvegen::svg_3d_writer<double>{600, ""}.write(svg, shapes, {1, 1, 1});
  std::cerr.rdbuf(old);
  REQUIRE(js.str().find("kind: 'mesh'") != std::string::npos);
  REQUIRE(svg.str().find("<polygon") != std::string::npos);
  REQUIRE(cerr_buf.str().empty());   // rendered, nothing warned
}

// Shape with no mesh registered anywhere (id outside every table).
struct dummy_shape final : rvegen::shape_base<double> {
  double area() const override { return 0; }
  double volume() const override { return 0; }
  std::array<double, 3> max_expansion() const override { return {0, 0, 0}; }
  std::array<double, 3> get_middle_point() const override { return {0, 0, 0}; }
  void set_middle_point(std::array<double, 3>) override {}
  bool is_inside(std::array<double, 3> const&) const override { return false; }
  void make_bounding_box() override {}
  std::unique_ptr<rvegen::shape_base<double>> clone() const override { return nullptr; }
  numsim_core::type_id shape_id() const noexcept override { return 100000; }
};

void test_unrenderable_shape_warns_once() {
  // A shape nobody registered a mesh for (built-ins are auto-registered).
  auto& d = rvegen::mesh_dispatcher<double>::instance();
  d.clear();
  d.register_shape<rvegen::sphere<double>>();
  rvegen::reset_unrendered_warnings();

  shapes_t shapes;
  shapes.emplace_back(std::make_unique<dummy_shape>());
  shapes.emplace_back(std::make_unique<dummy_shape>());

  std::ostringstream cerr_buf;
  auto* old = std::cerr.rdbuf(cerr_buf.rdbuf());
  std::ostringstream o;
  rvegen::three_js_writer<double>{}.write(o, shapes, {1, 1, 1});
  rvegen::three_js_writer<double>{}.write(o, shapes, {1, 1, 1});
  const auto after3js = cerr_buf.str();
  rvegen::svg_3d_writer<double>{600, ""}.write(o, shapes, {1, 1, 1});
  std::cerr.rdbuf(old);

  REQUIRE(count_of(after3js, "three_js_writer: skipping") == 1);  // once/type
  REQUIRE(count_of(cerr_buf.str(), "svg_3d_writer: skipping") == 1);

  // 2D writer: a sphere is dropped with a warning too.
  shapes_t s3;
  s3.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.1));
  std::ostringstream eb;
  old = std::cerr.rdbuf(eb.rdbuf());
  rvegen::svg_writer<double>{}.write(o, s3, {1, 1, 0});
  std::cerr.rdbuf(old);
  REQUIRE(count_of(eb.str(), "svg_writer: skipping") == 1);
  d.clear();
}
void test_degenerate_domain_and_thin_zoom() {
  shapes_t shapes;
  std::ostringstream os;
  rvegen::three_js_writer<double>{}.write(os, shapes, {0, 0, 0});
  const auto z = os.str();
  REQUIRE(z.find("nan") == std::string::npos && z.find("inf") == std::string::npos);
  REQUIRE(js_number(z, "distance") > 0 && js_number(z, "near") > 0);
  REQUIRE(js_number(z, "far") > js_number(z, "near"));

  // Anisotropic 1000 x 1 x 1: can zoom to the thin dimension, not only to
  // 5 % of the framing distance.
  std::ostringstream os2;
  rvegen::three_js_writer<double>{}.write(os2, shapes, {1000, 1, 1});
  const auto t = os2.str();
  REQUIRE(js_number(t, "minDistance") < 0.02 * js_number(t, "distance"));
  REQUIRE(js_number(t, "minDistance") >= 1.5 * js_number(t, "near"));
  // Portrait-window refit is emitted in the page script.
  REQUIRE(t.find("const fit") != std::string::npos);
}

void test_custom_registration_keeps_builtins() {
  auto& d = rvegen::mesh_dispatcher<double>::instance();
  d.clear();
  rvegen::reset_unrendered_warnings();
  // Host registers only one shape; the tube must still be meshed.
  d.register_shape<rvegen::sphere<double>>();
  shapes_t shapes;
  shapes.emplace_back(std::make_unique<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.1));
  shapes.emplace_back(std::make_unique<rvegen::polyline_tube<double>>(
      std::vector<std::array<double, 3>>{{0.1, 0.5, 0.5}, {0.9, 0.5, 0.5}}, 0.05));
  std::ostringstream js;
  rvegen::three_js_writer<double>{}.write(js, shapes, {1, 1, 1});
  REQUIRE(js.str().find("kind: 'mesh'") != std::string::npos);
  d.clear();
}

void test_warning_uses_readable_type_name() {
  rvegen::mesh_dispatcher<double>::instance().clear();
  rvegen::reset_unrendered_warnings();
  shapes_t shapes;
  shapes.emplace_back(std::make_unique<dummy_shape>());
  std::ostringstream eb;
  auto* old = std::cerr.rdbuf(eb.rdbuf());
  std::ostringstream js;
  rvegen::three_js_writer<double>{}.write(js, shapes, {1, 1, 1});
  std::cerr.rdbuf(old);
  REQUIRE(eb.str().find("dummy_shape") != std::string::npos);
  REQUIRE(eb.str().find("Z") == std::string::npos || eb.str().find("::") != std::string::npos);
  rvegen::mesh_dispatcher<double>::instance().clear();
}

void test_concurrent_writers_auto_register() {
  // Empty dispatcher + many writers at once: the lazy built-in
  // registration must not race (checked under -fsanitize=thread).
  rvegen::mesh_dispatcher<double>::instance().clear();
  std::vector<std::thread> th;
  std::atomic<int> bad{0};
  for (int i = 0; i < 8; ++i) {
    th.emplace_back([&] {
      for (int k = 0; k < 50; ++k) {
        shapes_t s;
        s.emplace_back(std::make_unique<rvegen::polyline_tube<double>>(
            std::vector<std::array<double, 3>>{{0.1, 0.5, 0.5}, {0.9, 0.5, 0.5}},
            0.05));
        std::ostringstream js;
        rvegen::three_js_writer<double>{}.write(js, s, {1, 1, 1});
        if (js.str().find("kind: 'mesh'") == std::string::npos) ++bad;
      }
    });
  }
  for (auto& t : th) t.join();
  REQUIRE(bad == 0);
}
} // namespace

int main() {
  test_degenerate_domain_and_thin_zoom();
  test_custom_registration_keeps_builtins();
  test_warning_uses_readable_type_name();
  test_concurrent_writers_auto_register();
  test_tube_winding_outward();
  test_camera_scales_with_domain();
  test_tube_rendered_via_dispatcher();
  test_unrenderable_shape_warns_once();
  if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
  std::cout << "visualization_writers_test OK\n";
  return 0;
}
