#pragma once

// Static 3D RVE picture. Orthographic axonometric projection (camera at
// (1, 1, 1) direction normalized). Spheres render as circles (parallel
// projection of a sphere is a circle of the same world radius). Boxes
// render as 6 quad faces with painters' algorithm depth sort. Domain
// box drawn as a wireframe — the "transparent matrix" effect without
// alpha-blending headaches.
//
// Self-contained SVG, viewable in any browser, image viewer, or chat.
//
// 2D shapes (circle, rectangle, ellipse) are skipped — use svg_writer.

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
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
#include "../shapes/box.h"
#include "../shapes/sphere.h"
#include "../types.h"

namespace rvegen {

template <typename T = double>
class svg_3d_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  svg_3d_writer() = default;

  svg_3d_writer(value_type canvas_width, std::string output_path)
      : _canvas_width{canvas_width}, _output_path{std::move(output_path)} {}

  explicit svg_3d_writer(parameter_handler_t const& handler)
      : _canvas_width{handler.template get<value_type>("canvas_width")},
        _output_path{handler.template get<std::string>("output_path")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<value_type>("canvas_width")
        .template add<numsim_core::is_required>();
    s.template insert<std::string>("output_path")
        .template add<numsim_core::is_required>();
    return s;
  }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "svg_3d_writer: output_path not set; cannot run() as post-process"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "svg_3d_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& domain_box) const {
    using std::sqrt;
    const auto Lx = domain_box[0];
    const auto Ly = domain_box[1];
    const auto Lz = domain_box[2];

    // Axonometric orthographic projection from direction (1, 1, 1).
    // Screen-right basis:  R = (1, 0, -1) / sqrt(2)
    // Screen-up    basis:  U = (-1, 2, -1) / sqrt(6)
    // Depth axis (view direction toward camera): V = (1, 1, 1) / sqrt(3)
    const auto inv_s2 = value_type{1} / std::sqrt(value_type{2});
    const auto inv_s6 = value_type{1} / std::sqrt(value_type{6});
    const auto inv_s3 = value_type{1} / std::sqrt(value_type{3});

    auto project = [&](value_type x, value_type y, value_type z) {
      // SVG y-axis points down → flip sign on screen-up.
      return std::pair{(x - z) * inv_s2, -(value_type{-1} * x
                                            + value_type{2} * y
                                            - z) * inv_s6};
    };
    auto depth = [&](value_type x, value_type y, value_type z) {
      return (x + y + z) * inv_s3;  // larger = closer to camera
    };

    // viewBox from projected domain corners.
    value_type minX = std::numeric_limits<value_type>::infinity();
    value_type minY = std::numeric_limits<value_type>::infinity();
    value_type maxX = -std::numeric_limits<value_type>::infinity();
    value_type maxY = -std::numeric_limits<value_type>::infinity();
    for (int i = 0; i < 8; ++i) {
      auto [px, py] = project(((i & 1) ? Lx : value_type{0}),
                               ((i & 2) ? Ly : value_type{0}),
                               ((i & 4) ? Lz : value_type{0}));
      minX = std::min(minX, px); maxX = std::max(maxX, px);
      minY = std::min(minY, py); maxY = std::max(maxY, py);
    }
    const auto pad = std::max(Lx, std::max(Ly, Lz)) * value_type{0.10};
    minX -= pad; maxX += pad; minY -= pad; maxY += pad;
    const auto vbW = maxX - minX;
    const auto vbH = maxY - minY;

    // Build the draw list with depths so we can sort painters-style.
    struct draw_item {
      value_type depth;
      std::string xml;
    };
    std::vector<draw_item> items;
    items.reserve(shapes.size() * 6);  // boxes contribute up to 6 faces

    static constexpr char const* palette[] = {
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    };
    constexpr std::size_t palette_size = std::size(palette);

    auto emit = [&](value_type d, std::string xml) {
      items.push_back({d, std::move(xml)});
    };

    auto fmt = [](double v) {
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.5g", v);
      return std::string{buf};
    };

    std::size_t idx = 0;
    for (auto const& shape : shapes) {
      auto const* raw = shape.get();
      char const* color = palette[idx % palette_size];
      const auto stroke_w = std::max(Lx, std::max(Ly, Lz)) * value_type{0.003};

      if (auto const* s = dynamic_cast<sphere<value_type> const*>(raw); s) {
        auto [px, py] = project((*s)(0), (*s)(1), (*s)(2));
        auto d = depth((*s)(0), (*s)(1), (*s)(2));
        std::string x = "<circle cx=\"" + fmt(px) +
                        "\" cy=\"" + fmt(py) +
                        "\" r=\"" + fmt(s->radius) +
                        "\" fill=\"" + color +
                        "\" stroke=\"#222\" stroke-width=\"" +
                        fmt(stroke_w) + "\"/>";
        emit(d, std::move(x));
      } else if (auto const* b = dynamic_cast<box<value_type> const*>(raw); b) {
        // Project 8 corners, emit 6 faces with their centroid depths.
        std::array<std::pair<value_type, value_type>, 8> p2;
        std::array<std::array<value_type, 3>, 8> p3;
        for (int i = 0; i < 8; ++i) {
          p3[i] = {(i & 1) ? b->max[0] : b->min[0],
                   (i & 2) ? b->max[1] : b->min[1],
                   (i & 4) ? b->max[2] : b->min[2]};
          p2[i] = project(p3[i][0], p3[i][1], p3[i][2]);
        }
        // Cube faces by corner indices (CCW from outside; not strictly
        // needed since we draw filled polygons).
        constexpr int faces[6][4] = {
            {0, 1, 3, 2},  // -z
            {4, 6, 7, 5},  // +z
            {0, 4, 5, 1},  // -y
            {2, 3, 7, 6},  // +y
            {0, 2, 6, 4},  // -x
            {1, 5, 7, 3},  // +x
        };
        for (int f = 0; f < 6; ++f) {
          // Centroid depth.
          value_type cd = 0, cx = 0, cy = 0, cz = 0;
          for (int k = 0; k < 4; ++k) {
            int corner = faces[f][k];
            cx += p3[corner][0]; cy += p3[corner][1]; cz += p3[corner][2];
          }
          cx *= value_type{0.25}; cy *= value_type{0.25}; cz *= value_type{0.25};
          cd = depth(cx, cy, cz);
          std::string pts;
          for (int k = 0; k < 4; ++k) {
            int corner = faces[f][k];
            pts += fmt(p2[corner].first) + "," + fmt(p2[corner].second) + " ";
          }
          // Subtle face shading: top a touch lighter, bottom a touch darker.
          [[maybe_unused]] double shade = (f == 1) ? 1.10 : (f == 0) ? 0.85 : 1.0;
          std::string x = "<polygon points=\"" + pts +
                          "\" fill=\"" + color +
                          "\" fill-opacity=\"0.85\" "
                          "stroke=\"#222\" stroke-width=\"" +
                          fmt(stroke_w) + "\"/>";
          emit(cd, std::move(x));
        }
      }
      ++idx;
    }

    // Sort back-to-front (smallest depth first).
    std::sort(items.begin(), items.end(),
              [](auto const& a, auto const& b) { return a.depth < b.depth; });

    // Domain wireframe — draw last so it sits on top of everything for
    // visual context. Project all 12 edges.
    constexpr int edges[12][2] = {
        {0,1},{2,3},{4,5},{6,7},  // x-aligned
        {0,2},{1,3},{4,6},{5,7},  // y-aligned
        {0,4},{1,5},{2,6},{3,7}   // z-aligned
    };
    std::array<std::pair<value_type, value_type>, 8> dp;
    for (int i = 0; i < 8; ++i) {
      dp[i] = project(((i & 1) ? Lx : value_type{0}),
                       ((i & 2) ? Ly : value_type{0}),
                       ((i & 4) ? Lz : value_type{0}));
    }

    out << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        << "<svg xmlns=\"http://www.w3.org/2000/svg\" "
        << "width=\"" << _canvas_width << "\" "
        << "height=\"" << (_canvas_width * vbH / vbW) << "\" "
        << "viewBox=\"" << minX << ' ' << minY
        << ' ' << vbW << ' ' << vbH << "\">\n"
        << "  <!-- rvegen 3D RVE; orthographic axonometric projection -->\n";

    // Inclusions, sorted back-to-front.
    for (auto const& it : items) out << "  " << it.xml << "\n";

    // Domain wireframe.
    const auto sw = std::max(Lx, std::max(Ly, Lz)) * value_type{0.005};
    for (int e = 0; e < 12; ++e) {
      auto a = dp[edges[e][0]], b = dp[edges[e][1]];
      out << "  <line x1=\"" << a.first << "\" y1=\"" << a.second
          << "\" x2=\"" << b.first << "\" y2=\"" << b.second
          << "\" stroke=\"#444\" stroke-width=\"" << sw << "\"/>\n";
    }
    out << "</svg>\n";
  }

private:
  value_type _canvas_width{900};
  std::string _output_path{};
};

} // namespace rvegen
