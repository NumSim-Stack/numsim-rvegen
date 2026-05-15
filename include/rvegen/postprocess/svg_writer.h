#pragma once

// 2D SVG output for RVE visualization. Transparent background, colored
// inclusions, thin stroke for definition, periodic-correct because every
// shape (primary + wrap) in the accepted vector is rendered.
//
// SVG y-axis points DOWN, while RVE coordinates have +y pointing up; we
// flip y at render time so the picture matches the typical convention
// (origin at bottom-left).
//
// Shape dispatch is by dynamic_cast (output is one-shot, not a hot loop).
// Supported: circle, rectangle, ellipse. Sphere/box are 3D and skipped.

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <numbers>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
#include "../shapes/circle.h"
#include "../shapes/convex_polygon.h"
#include "../shapes/ellipse.h"
#include "../shapes/rectangle.h"
#include "../types.h"

namespace rvegen {

template <typename T = double>
class svg_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  svg_writer() = default;

  // Schema-driven ctor — width gives the rendered canvas size in CSS
  // pixels (height is derived from the domain box aspect ratio).
  explicit svg_writer(parameter_handler_t const& handler)
      : _canvas_width{handler.template get<value_type>("canvas_width")},
        _output_path{handler.template get<std::string>("output_path")} {}

  svg_writer(value_type canvas_width, std::string output_path)
      : _canvas_width{canvas_width}, _output_path{std::move(output_path)} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<value_type>("canvas_width")
        .template add<numsim_core::is_required>()
        .template add<min_only<value_type{1}>>()
        .template add<numsim_core::unit_label<"px">>()
        .template add<numsim_core::description_label<"rendered SVG canvas width in CSS pixels (height derived from aspect)">>();
    s.template insert<std::string>("output_path")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"destination path for the SVG file">>();
    return s;
  }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "svg_writer: output_path not set; cannot run() as post-process"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "svg_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& domain_box) const {
    const auto Lx = domain_box[0];
    const auto Ly = domain_box[1];
    const auto canvas_h = _canvas_width * (Ly / Lx);
    const auto stroke_w = Lx * value_type{0.003};

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << _canvas_width << "\" "
        << "height=\"" << canvas_h << "\" "
        << "viewBox=\"0 0 " << Lx << ' ' << Ly << "\">\n"
        << "  <!-- rvegen RVE; transparent background -->\n";

    // Domain border — thin grey, transparent fill. Drawn first so
    // inclusions sit on top.
    out << "  <rect x=\"0\" y=\"0\" "
        << "width=\"" << Lx << "\" height=\"" << Ly << "\" "
        << "fill=\"none\" stroke=\"#888\" "
        << "stroke-width=\"" << stroke_w << "\"/>\n";

    // Inclusions. Cycle through a tasteful palette.
    static constexpr char const* palette[] = {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    };
    constexpr std::size_t palette_size = std::size(palette);

    std::size_t idx = 0;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      char const* fill = palette[idx % palette_size];

      if (auto const* c = dynamic_cast<circle<value_type> const*>(raw); c) {
        out << "  <circle "
            << "cx=\"" << (*c)(0) << "\" "
            << "cy=\"" << (Ly - (*c)(1)) << "\" "
            << "r=\""  << c->radius << "\" "
            << "fill=\""   << fill   << "\" fill-opacity=\"0.7\" "
            << "stroke=\"#222\" stroke-width=\"" << stroke_w << "\"/>\n";
      } else if (auto const* r = dynamic_cast<rectangle<value_type> const*>(raw); r) {
        const auto x = r->min[0];
        const auto y = Ly - r->max[1];  // flip y for SVG
        const auto w = r->max[0] - r->min[0];
        const auto h = r->max[1] - r->min[1];
        out << "  <rect "
            << "x=\"" << x << "\" y=\"" << y << "\" "
            << "width=\"" << w << "\" height=\"" << h << "\" "
            << "fill=\""   << fill   << "\" fill-opacity=\"0.7\" "
            << "stroke=\"#222\" stroke-width=\"" << stroke_w << "\"/>\n";
      } else if (auto const* e = dynamic_cast<ellipse<value_type> const*>(raw); e) {
        const auto cx = (*e)(0);
        const auto cy = Ly - (*e)(1);
        const auto rx = e->radius_a();
        const auto ry = e->radius_b();
        // Convert rotation rad → deg; flip sign because SVG y is inverted.
        const auto deg = -e->rotation() * value_type{180}
                       / std::numbers::pi_v<value_type>;
        out << "  <ellipse "
            << "cx=\"" << cx << "\" cy=\"" << cy << "\" "
            << "rx=\"" << rx << "\" ry=\"" << ry << "\" "
            << "transform=\"rotate(" << deg << ' ' << cx << ' ' << cy << ")\" "
            << "fill=\""   << fill   << "\" fill-opacity=\"0.7\" "
            << "stroke=\"#222\" stroke-width=\"" << stroke_w << "\"/>\n";
      } else if (auto const* p = dynamic_cast<convex_polygon<value_type> const*>(raw); p) {
        // SVG <polygon> takes a space-separated points list. Flip y
        // for each vertex to keep SVG's top-down y-axis convention.
        out << "  <polygon points=\"";
        bool first = true;
        for (auto const& v : p->vertices()) {
          if (!first) out << ' ';
          out << v[0] << ',' << (Ly - v[1]);
          first = false;
        }
        out << "\" fill=\""   << fill   << "\" fill-opacity=\"0.7\" "
            << "stroke=\"#222\" stroke-width=\"" << stroke_w << "\"/>\n";
      } else {
        out << "  <!-- (skipped 3D / unsupported shape) -->\n";
      }
      ++idx;
    }
    out << "</svg>\n";
  }

  [[nodiscard]] value_type canvas_width() const noexcept {
    return _canvas_width;
  }

private:
  value_type _canvas_width{800};
  std::string _output_path{};
};

} // namespace rvegen
