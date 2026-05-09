#pragma once

// Statistics post-process — emits CSV files with the standard
// microstructure descriptors (volume_fraction, S2(r), RDF g(r)) so a
// paper-writing user gets paper-ready columns from a single JSON config
// entry. Reuses the helpers in rvegen/statistics/microstructure_stats.h.
//
// JSON usage:
//   { "type": "statistics",
//     "output_path": "stats.csv",
//     "nx": 128, "ny": 128,
//     "nbins": 32 }

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "../statistics/microstructure_stats.h"
#include "../types.h"
#include "post_process_base.h"

namespace rvegen {

template <typename T = double>
class statistics_post_process final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  statistics_post_process() = default;

  explicit statistics_post_process(parameter_handler_t const& handler)
      : _output_path{handler.template get<std::string>("output_path")},
        _nx{handler.template get<std::size_t>("nx")},
        _ny{handler.template get<std::size_t>("ny")},
        _nbins{handler.template get<std::size_t>("nbins")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("output_path")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"destination CSV path">>();
    s.template insert<std::size_t>("nx")
        .template add<numsim_core::is_required>()
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::unit_label<"cells">>()
        .template add<numsim_core::description_label<"voxel-grid resolution along x for S2 sampling">>();
    s.template insert<std::size_t>("ny")
        .template add<numsim_core::is_required>()
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::unit_label<"cells">>()
        .template add<numsim_core::description_label<"voxel-grid resolution along y for S2 sampling">>();
    s.template insert<std::size_t>("nbins")
        .template add<numsim_core::is_required>()
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"number of radial bins for S2(r) and g(r)">>();
    return s;
  }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "statistics_post_process: output_path not set"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "statistics_post_process: cannot open '" + _output_path + "'"};
    }

    const double phi = volume_fraction_in_domain(shapes, domain_box);
    auto s2 = two_point_correlation_2d(shapes, domain_box, _nx, _ny, _nbins);
    auto rdf = radial_distribution_2d(shapes, domain_box, _nbins);

    out << "# rvegen microstructure statistics\n"
        << "# domain_box = " << domain_box[0] << ' ' << domain_box[1]
        << ' ' << domain_box[2] << '\n'
        << "# volume_fraction = " << phi << '\n'
        << "# columns: r, S2(r), g(r)\n";

    const std::size_t N = std::min(s2.size(), rdf.size());
    for (std::size_t i = 0; i < N; ++i) {
      out << s2[i].r << ',' << s2[i].s2 << ',' << rdf[i].g << '\n';
    }
  }

private:
  std::string _output_path;
  std::size_t _nx{128};
  std::size_t _ny{128};
  std::size_t _nbins{32};
};

} // namespace rvegen
