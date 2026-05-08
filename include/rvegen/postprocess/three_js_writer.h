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
// loop). Supported: sphere, box. 2D shapes (circle, rectangle, ellipse)
// are skipped — use svg_writer for those.

#include <array>
#include <cstddef>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
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
        .description("destination path for the self-contained Three.js HTML viewer");
    return s;
  }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "three_js_writer: output_path not set; cannot run() as post-process"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "three_js_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& domain_box) const {
    const auto Lx = domain_box[0];
    const auto Ly = domain_box[1];
    const auto Lz = domain_box[2];

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
const shapes = [
)";

    // Tasteful palette (Tableau 10).
    static constexpr char const* palette[] = {
        "0x1f77b4", "0xff7f0e", "0x2ca02c", "0xd62728", "0x9467bd",
        "0x8c564b", "0xe377c2", "0x7f7f7f", "0xbcbd22", "0x17becf"
    };
    constexpr std::size_t palette_size = std::size(palette);

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
      }
      ++idx;
    }

    out << R"(];

const [Lx, Ly, Lz] = domain;

const scene  = new THREE.Scene();
scene.background = new THREE.Color(0xf4f6f8);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth/window.innerHeight, 0.01, 100);
const max_dim = Math.max(Lx, Ly, Lz);
camera.position.set(Lx*1.7, Ly*1.7, Lz*1.7);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Orbit camera around the box centre.
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(Lx*0.5, Ly*0.5, Lz*0.5);
controls.update();

// Lighting — directional + ambient for solid shading.
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 0.85);
key.position.set(Lx*2, Ly*2.5, Lz*2);
scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.35);
fill.position.set(-Lx, Ly*0.3, -Lz);
scene.add(fill);

// Domain box — wireframe only (the "transparent matrix").
{
  const geom = new THREE.BoxGeometry(Lx, Ly, Lz);
  const edges = new THREE.EdgesGeometry(geom);
  const line = new THREE.LineSegments(
      edges, new THREE.LineBasicMaterial({ color: 0x444444, linewidth: 1.5 }));
  line.position.set(Lx*0.5, Ly*0.5, Lz*0.5);
  scene.add(line);
}

// Inclusions — solid colour with subtle shading.
for (const s of shapes) {
  let geom;
  if (s.kind === 'sphere') {
    geom = new THREE.SphereGeometry(s.r, 28, 20);
  } else if (s.kind === 'box') {
    geom = new THREE.BoxGeometry(s.w, s.h, s.d);
  } else continue;

  const mat = new THREE.MeshStandardMaterial({
    color: s.color, roughness: 0.55, metalness: 0.0,
    transparent: false, opacity: 1.0
  });
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(s.x, s.y, s.z);
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
