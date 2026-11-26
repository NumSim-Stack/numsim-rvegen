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
//     "nz": 128,            // required for 3D domains, ignored in 2D
//     "nbins": 32 }
//
// Dispatches on the domain: a 2D domain (domain_box[2] == 0) uses
// two_point_correlation_2d / radial_distribution_2d; a 3D domain uses
// two_point_correlation_3d (S2 over an nx × ny × nz grid) and
// radial_distribution_3d (spherical shells, density N/V). `nz` is ignored
// in 2D. The 3D S2 voxelization costs O(nx·ny·nz · shapes) is_inside
// calls (128³ ≈ 2.1 M voxels), so `nz` has no default: a 3D domain
// without it throws from run(), making the user pick the z resolution and
// therefore the cost. Any implicit default would either hide that cost
// (nz = nx) or silently give anisotropic voxels (a voxel budget).

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
        _nz{handler.contains("nz")
                ? handler.template get<std::size_t>("nz")
                : std::size_t{0}},
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
    // Not schema-required so existing 2D configs stay valid; run() throws
    // for a 3D domain when it is missing.
    s.template insert<std::size_t>("nz")
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::unit_label<"cells">>()
        .template add<numsim_core::description_label<"voxel-grid resolution along z for 3D S2 sampling (required for 3D RVEs, ignored for 2D). 3D cost scales with nx*ny*nz">>();
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
    const bool is_3d = domain_box[2] > value_type{0};
    if (is_3d && _nz == 0) {
      throw std::runtime_error{
          "statistics_post_process: 3D domain needs \"nz\" (voxel-grid "
          "resolution along z for S2). Set it explicitly; 3D S2 cost scales "
          "with nx*ny*nz."};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "statistics_post_process: cannot open '" + _output_path + "'"};
    }

    const double phi = volume_fraction_in_domain(shapes, domain_box);
    auto s2 = is_3d
        ? two_point_correlation_3d(shapes, domain_box, _nx, _ny, _nz, _nbins)
        : two_point_correlation_2d(shapes, domain_box, _nx, _ny, _nbins);
    auto rdf = is_3d ? radial_distribution_3d(shapes, domain_box, _nbins)
                     : radial_distribution_2d(shapes, domain_box, _nbins);

    out << "# rvegen microstructure statistics\n"
        << "# domain_box = " << domain_box[0] << ' ' << domain_box[1]
        << ' ' << domain_box[2] << '\n'
        << "# dimension = " << (is_3d ? "3D" : "2D") << '\n'
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
  std::size_t _nz{0};  // 0 = not set; required for 3D domains
  std::size_t _nbins{32};
};

} // namespace rvegen
