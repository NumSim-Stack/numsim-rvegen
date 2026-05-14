// Minimal install-smoke consumer. If we got here, the CMake config
// resolved, the headers are reachable, and rvegen's public surface
// compiles + links. The actual logic just exercises a few primitives
// and a JSON-aware writer to confirm the transitive deps (Eigen, Boost,
// nlohmann_json) propagated correctly via the install config.

#include <iostream>
#include <sstream>
#include <vector>
#include <memory>

#include <rvegen/shapes/circle.h>
#include <rvegen/shapes/sphere.h>
#include <rvegen/postprocess/svg_writer.h>

int main() {
  rvegen::circle<double> c{0.5, 0.5, 0.1};
  if (!c.is_inside({0.55, 0.55, 0.0})) return 1;

  rvegen::sphere<double> s{0.0, 0.0, 0.0, 0.5};
  if (s.volume() <= 0.0) return 2;

  // Exercise nlohmann_json transitively via svg_writer (uses it for
  // its parameter handler / postprocess base).
  rvegen::svg_writer<double>::shape_vector shapes;
  shapes.emplace_back(std::make_unique<rvegen::circle<double>>(c));
  rvegen::svg_writer<double> writer{};   // default ctor — no output_path
  std::stringstream out;
  writer.write(out, shapes, {1.0, 1.0, 0.0});
  if (out.str().empty()) return 3;

  std::cout << "install_smoke: PASS\n";
  return 0;
}
