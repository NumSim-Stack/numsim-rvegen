#pragma once

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
#include "../shapes/circle.h"
#include "../shapes/rectangle.h"
#include "../shapes/sphere.h"
#include "../types.h"

namespace rvegen {

// Writes an RVE as a gmsh .geo script. Gmsh consumes this file and produces
// the FEM mesh.
//
// Output structure:
//   * 2D RVE: a Rectangle entity for the domain, then per-shape 2D entities
//   * 3D RVE: a Box entity for the domain, then per-shape 3D entities
//
// Shape dispatch uses dynamic_cast — output is one-shot, not on the inner
// generation loop, so the cost is negligible and the dispatch is localised.
//
// Supported shapes: circle, sphere, rectangle, box. Adding a shape: extend
// the dispatch chain in write().
template <typename T = double>
class gmsh_geo_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  // Default ctor leaves _output_path empty — only stream-based write() is
  // usable in that state. Kept so tests / library users can capture the
  // emitted text into a stringstream without needing a file.
  gmsh_geo_writer() = default;

  explicit gmsh_geo_writer(std::string output_path)
      : _output_path{std::move(output_path)} {}

  explicit gmsh_geo_writer(parameter_handler_t const& handler)
      : _output_path{handler.template get<std::string>("output_path")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("output_path").template add<numsim_core::is_required>();
    return s;
  }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "gmsh_geo_writer: output_path not set; cannot run() as post-process"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "gmsh_geo_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  // Stream-based emit. Kept public so tests and embedders can capture the
  // output to a stringstream without round-tripping through a file.
  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& box) const {
    const bool is_3d = box[2] > value_type{0};
    if (is_3d) {
      write_3d(out, shapes, box);
    } else {
      write_2d(out, shapes, box);
    }
  }

private:
  static void write_header(std::ostream& out) {
    out << "// rvegen — generated gmsh .geo script\n"
        << "// Mesh.CharacteristicLengthMin = 0.05;\n"
        << "// Mesh.CharacteristicLengthMax = 0.1;\n\n";
  }

  void write_2d(std::ostream& out,
                shape_vector const& shapes,
                std::array<value_type, 3> const& domain_box) const {
    write_header(out);
    out << "Rectangle(1) = {0, 0, 0, "
        << domain_box[0] << ", " << domain_box[1] << ", 0};\n\n";

    std::size_t entity_id = 2;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      if (auto const* c = dynamic_cast<circle<value_type> const*>(raw); c) {
        // gmsh: Circle(id) = {x, y, z, r}; via Disk surface for filled inclusion.
        out << "Disk(" << entity_id << ") = {"
            << (*c)(0) << ", " << (*c)(1) << ", 0, "
            << c->radius << ", " << c->radius << "};\n";
        ++entity_id;
      } else if (auto const* r = dynamic_cast<rectangle<value_type> const*>(raw); r) {
        const auto half_w = r->width()  * value_type{0.5};
        const auto half_h = r->height() * value_type{0.5};
        out << "Rectangle(" << entity_id << ") = {"
            << (*r)(0) - half_w << ", " << (*r)(1) - half_h << ", 0, "
            << r->width() << ", " << r->height() << ", 0};\n";
        ++entity_id;
      } else {
        out << "// (unsupported 2D shape skipped)\n";
      }
    }
  }

  void write_3d(std::ostream& out,
                shape_vector const& shapes,
                std::array<value_type, 3> const& domain_box) const {
    write_header(out);
    out << "Box(1) = {0, 0, 0, "
        << domain_box[0] << ", " << domain_box[1] << ", " << domain_box[2]
        << "};\n\n";

    std::size_t entity_id = 2;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      if (auto const* s = dynamic_cast<sphere<value_type> const*>(raw); s) {
        out << "Sphere(" << entity_id << ") = {"
            << (*s)(0) << ", " << (*s)(1) << ", " << (*s)(2) << ", "
            << s->radius << "};\n";
        ++entity_id;
      } else if (auto const* b = dynamic_cast<box<value_type> const*>(raw); b) {
        const auto hx = b->width()  * value_type{0.5};
        const auto hy = b->height() * value_type{0.5};
        const auto hz = b->depth()  * value_type{0.5};
        out << "Box(" << entity_id << ") = {"
            << (*b)(0) - hx << ", " << (*b)(1) - hy << ", " << (*b)(2) - hz << ", "
            << b->width() << ", " << b->height() << ", " << b->depth() << "};\n";
        ++entity_id;
      } else {
        out << "// (unsupported 3D shape skipped)\n";
      }
    }
  }

  std::string _output_path;
};

} // namespace rvegen
