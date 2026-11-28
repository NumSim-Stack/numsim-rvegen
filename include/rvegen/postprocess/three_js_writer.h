#pragma once

// 3D HTML output for RVE visualization. Emits a single self-contained
// .html file that loads Three.js from a CDN and renders the inclusions
// with orbit-camera controls. Open in any modern browser; rotate, zoom,
// and pan with the mouse.
//
// Domain box is shown as a wireframe — the "transparent matrix" effect
// you'd otherwise get with translucent material rendering, without
// the depth-sorting headaches.
//
// Shape dispatch is by dynamic_cast (output is one-shot, not on a hot
// loop). sphere and box use native Three.js geometry; every other shape
// (tube, polygon, Voronoi cell, 2D shapes at z=0, ...) is meshed through
// mesh_dispatcher and sent as a BufferGeometry. Shapes with no mesh are
// skipped with ONE warning per type on std::cerr (mesh_fallback.h).
//
// Camera, clipping planes, orbit limits, grid and axes are computed here in
// C++ from the domain box (emitted as `const view = {...}`), so the scene is
// framed for any domain scale.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <Mathematics/Vector3.h>
#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
#include "../visualization/mesh_fallback.h"
#include "../shapes/box.h"
#include "../shapes/sphere.h"
#include "../types.h"

namespace rvegen {

template <typename T = double>
class three_js_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  three_js_writer() = default;

  explicit three_js_writer(std::string output_path)
      : _output_path{std::move(output_path)} {}

  // The HTML is responsive to window size, so the only schema parameter
  // is the destination output_path.
  explicit three_js_writer(parameter_handler_t const& handler)
      : _output_path{handler.template get<std::string>("output_path")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("output_path")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"destination path for the self-contained Three.js HTML viewer">>();
    return s;
  }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  void validate(domain<value_type> const& /*domain_box*/) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "three_js_writer: output_path not set; cannot run() as post-process"};
    }
  }

  void run(shape_vector const& shapes,
           domain<value_type> const& domain_box) const override {
    validate(domain_box);
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "three_js_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  void write(std::ostream& out,
             shape_vector const& shapes,
             domain<value_type> const& domain_box) const {
    const auto Lx = domain_box[0];
    const auto Ly = domain_box[1];
    const auto Lz = domain_box[2];

    // Frame the domain's bounding sphere with a perspective camera.
    constexpr double fov_deg = 45.0;
    // A degenerate (all-zero / non-finite) domain would put the camera at
    // its own target with near == far == 0; frame a unit cube instead.
    double Fx = Lx, Fy = Ly, Fz = Lz;
    if (!(std::isfinite(Fx) && std::isfinite(Fy) && std::isfinite(Fz)) ||
        !(Fx > 0 || Fy > 0 || Fz > 0)) {
      Fx = Fy = Fz = 1.0;
    }
    const double half_diag =
        0.5 * std::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
    const double max_dim = std::max({Fx, Fy, Fz});
    // Smallest non-zero extent: zooming may go down to half of it.
    double min_dim = max_dim;
    for (double e : {Fx, Fy, Fz}) if (e > 0) min_dim = std::min(min_dim, e);
    const double dist = 1.15 * half_diag /
        std::sin(0.5 * fov_deg * std::numbers::pi / 180.0);
    gte::Vector3<double> dir{1.0, 0.8, 1.0};
    gte::Normalize(dir);
    const double cx0 = 0.5 * Fx, cy0 = 0.5 * Fy, cz0 = 0.5 * Fz;
    const double near_d = dist * 2e-3, far_d = dist * 10.0;
    const double min_distance =
        std::max(1.5 * near_d, std::min(0.05 * dist, 0.5 * min_dim));

    const auto old_prec = out.precision(10);
    struct prec_restore {
      std::ostream& o; std::streamsize p;
      ~prec_restore() { o.precision(p); }
    } restore{out, old_prec};

    out << R"(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>rvegen RVE</title>
<style>
  html, body { margin: 0; height: 100%; overflow: hidden; background: #f4f6f8; }
  #hud { position: absolute; top: 10px; left: 12px; padding: 8px 12px;
         background: rgba(255,255,255,0.85); border: 1px solid #ddd;
         border-radius: 4px; font: 13px/1.4 -apple-system, sans-serif;
         color: #333; pointer-events: none; }
  #hud b { color: #111; }
</style>
</head>
<body>
<div id="hud">
  <b>rvegen RVE</b><br>
  drag to rotate · scroll to zoom · shift+drag to pan
</div>
<script type="importmap">
{
  "imports": {
    "three":           "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/":   "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const domain = [)" << Lx << ", " << Ly << ", " << Lz << R"(];
// Camera framing derived from the domain size (see three_js_writer.h).
const view = {
  center: [)" << cx0 << ", " << cy0 << ", " << cz0 << R"(],
  position: [)" << cx0 + dir[0] * dist << ", " << cy0 + dir[1] * dist << ", "
        << cz0 + dir[2] * dist << R"(],
  distance: )" << dist << R"(,
  fov: )" << fov_deg << R"(,
  near: )" << near_d << R"(,
  far: )" << far_d << R"(,
  minDistance: )" << min_distance << R"(,
  maxDim: )" << max_dim << R"(
};
const shapes = [
)";

    // Tasteful palette (Tableau 10).
    static constexpr char const* palette[] = {
        "0x1f77b4", "0xff7f0e", "0x2ca02c", "0xd62728", "0x9467bd",
        "0x8c564b", "0xe377c2", "0x7f7f7f", "0xbcbd22", "0x17becf"
    };
    constexpr std::size_t palette_size = std::size(palette);

    // ~2.5 KB per mesh shape: warn above ~50 MB (20000 shapes).
    warn_large_output("three_js_writer", shapes.size(), 20000, 2500);

    std::size_t idx = 0;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      char const* color = palette[idx % palette_size];

      if (auto const* s = dynamic_cast<sphere<value_type> const*>(raw); s) {
        out << "  { kind: 'sphere', x: " << (*s)(0)
            << ", y: " << (*s)(1) << ", z: " << (*s)(2)
            << ", r: " << s->radius
            << ", color: " << color << " },\n";
      } else if (auto const* b = dynamic_cast<box<value_type> const*>(raw); b) {
        const auto cx = ((*b).max[0] + (*b).min[0]) * value_type{0.5};
        const auto cy = ((*b).max[1] + (*b).min[1]) * value_type{0.5};
        const auto cz = ((*b).max[2] + (*b).min[2]) * value_type{0.5};
        const auto w  = (*b).max[0] - (*b).min[0];
        const auto h  = (*b).max[1] - (*b).min[1];
        const auto d  = (*b).max[2] - (*b).min[2];
        out << "  { kind: 'box', x: " << cx << ", y: " << cy << ", z: " << cz
            << ", w: " << w << ", h: " << h << ", d: " << d
            << ", color: " << color << " },\n";
      } else {
        auto const mesh = mesh_or_warn(*raw, "three_js_writer");
        if (!mesh.empty()) {
          out << "  { kind: 'mesh', color: " << color << ", pos: [";
          bool first = true;
          for (auto const& v : mesh.verts) {
            out << (first ? "" : ",") << v[0] << ',' << v[1] << ',' << v[2];
            first = false;
          }
          out << "], idx: [";
          first = true;
          for (auto const& t : mesh.tris) {
            out << (first ? "" : ",") << t[0] << ',' << t[1] << ',' << t[2];
            first = false;
          }
          out << "] },\n";
        }
      }
      ++idx;
    }

    out << R"(];

const [Lx, Ly, Lz] = domain;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0xf4f6f8);

// three.js fov is VERTICAL; in a portrait window the horizontal field is
// narrower, so pull the camera back (fit >= 1) until the domain's bounding
// sphere fits both ways. Applied once at load; the user's orbit is kept on
// resize.
const halfV = 0.5 * view.fov * Math.PI / 180;
const halfEff = Math.atan(Math.tan(halfV) * Math.min(1, window.innerWidth / window.innerHeight));
const fit = Math.sin(halfV) / Math.sin(halfEff);
const camera = new THREE.PerspectiveCamera(
    view.fov, window.innerWidth/window.innerHeight, view.near, view.far * fit);
camera.position.set(...view.position.map((p, i) => view.center[i] + (p - view.center[i]) * fit));

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Orbit camera around the box centre.
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(...view.center);
controls.minDistance = view.minDistance;
controls.maxDistance = view.distance * 5 * fit;
controls.update();

// Lighting — directional + ambient for solid shading.
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.85);
key.position.set(Lx*1.5, Ly*2.0, Lz*1.5);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.35);
fill.position.set(-Lx*0.5, Ly*0.8, -Lz*0.5);
scene.add(fill);

// Domain box — wireframe only (the "transparent matrix").
{
  // Unit box scaled afterwards: EdgesGeometry hashes vertices at 1e-4
  // precision and would collapse a tiny (e.g. 1e-6) domain.
  const geom = new THREE.BoxGeometry(1, 1, 1);
  const edges = new THREE.EdgesGeometry(geom);
  const line = new THREE.LineSegments(
      edges, new THREE.LineBasicMaterial({ color: 0x444444, linewidth: 1.5 }));
  line.scale.set(Lx, Ly, Lz);
  line.position.set(Lx*0.5, Ly*0.5, Lz*0.5);
  scene.add(line);
}

// Grid (floor, y = 0) and axes, both scaled with the domain.
{
  const grid = new THREE.GridHelper(view.maxDim * 2, 10, 0x999999, 0xcccccc);
  grid.position.set(view.center[0], 0, view.center[2]);
  scene.add(grid);
  scene.add(new THREE.AxesHelper(view.maxDim * 0.6));
}

// Inclusions — solid colour with subtle shading.
for (const s of shapes) {
  let geom;
  if (s.kind === 'sphere') {
    geom = new THREE.SphereGeometry(s.r, 28, 20);
  } else if (s.kind === 'box') {
    geom = new THREE.BoxGeometry(s.w, s.h, s.d);
  } else if (s.kind === 'mesh') {
    geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(s.pos, 3));
    geom.setIndex(s.idx);
    geom.computeVertexNormals();
  } else continue;

  const mat = new THREE.MeshStandardMaterial({
    color: s.color, roughness: 0.55, metalness: 0.0,
    transparent: false, opacity: 1.0,
    side: s.kind === 'mesh' ? THREE.DoubleSide : THREE.FrontSide
  });
  const mesh = new THREE.Mesh(geom, mat);
  if (s.kind !== 'mesh') mesh.position.set(s.x, s.y, s.z);
  scene.add(mesh);
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

(function loop(){
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
})();
</script>
</body>
</html>
)";
  }

private:
  std::string _output_path{};
};

} // namespace rvegen
