#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
#include "../types.h"
#include "voxel_grid.h"

namespace rvegen {

// Voxel-grid writer for the FFT homogenization pipeline.
//
// Subdivides the RVE box into nx × ny × nz voxels, samples each voxel's
// centre point with shape_base::is_inside, and writes a phase-id grid:
//   0       — matrix (no inclusion)
//   1..N    — index (1-based) of the inclusion the voxel falls inside
//
// 2D RVEs (domain_box[2] == 0) collapse to a single z-slice (nz = 1).
//
// Output format: ASCII header + flat array, voxels in (k, j, i) order
// (z-slice fastest-varying-axis is k). The format is intentionally
// human-readable and trivial to parse from any FFT solver.
template <typename T = double>
class voxel_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  voxel_writer() = default;

  // Numeric-only ctor — for stream-based use (tests / embedders that hand
  // the writer their own ostream). _output_path stays empty; run() will
  // refuse until a path is provided.
  voxel_writer(std::size_t nx, std::size_t ny, std::size_t nz) noexcept
      : _nx{nx}, _ny{ny}, _nz{nz} {}

  voxel_writer(std::size_t nx, std::size_t ny, std::size_t nz,
               std::string output_path)
      : _nx{nx}, _ny{ny}, _nz{nz}, _output_path{std::move(output_path)} {}

  explicit voxel_writer(parameter_handler_t const& handler)
      : _nx{handler.template get<std::size_t>("nx")},
        _ny{handler.template get<std::size_t>("ny")},
        _nz{handler.template get<std::size_t>("nz")},
        _output_path{handler.template get<std::string>("output_path")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::size_t>("nx").template add<numsim_core::is_required>()
        .min(1.0).units("cells").description("voxel-grid resolution along x");
    s.template insert<std::size_t>("ny").template add<numsim_core::is_required>()
        .min(1.0).units("cells").description("voxel-grid resolution along y");
    s.template insert<std::size_t>("nz").template add<numsim_core::is_required>()
        .min(1.0).units("cells")
        .description("voxel-grid resolution along z (collapses to 1 for 2D RVEs)");
    s.template insert<std::string>("output_path").template add<numsim_core::is_required>()
        .description("destination path for the voxel-grid file");
    return s;
  }

  void resolution(std::size_t nx, std::size_t ny, std::size_t nz) noexcept {
    _nx = nx; _ny = ny; _nz = nz;
  }

  [[nodiscard]] std::size_t nx() const noexcept { return _nx; }
  [[nodiscard]] std::size_t ny() const noexcept { return _ny; }
  [[nodiscard]] std::size_t nz() const noexcept { return _nz; }
  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "voxel_writer: output_path not set; cannot run() as post-process"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "voxel_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  // Stream-based emit. Public so tests and embedders can capture the output
  // to a stringstream without round-tripping through a file.
  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& domain_box) const {
    const auto grid = sample(shapes, domain_box);
    const std::size_t nz_eff = effective_nz(domain_box, _nz);

    out << "# rvegen voxel grid\n"
        << "# nx ny nz: " << _nx << ' ' << _ny << ' ' << nz_eff << '\n'
        << "# Lx Ly Lz: " << domain_box[0] << ' ' << domain_box[1] << ' '
        << domain_box[2] << '\n'
        << "# 0 = matrix; 1..N = inclusion index (insertion order)\n";

    for (std::size_t k = 0; k < nz_eff; ++k) {
      for (std::size_t j = 0; j < _ny; ++j) {
        for (std::size_t i = 0; i < _nx; ++i) {
          out << grid[(k * _ny + j) * _nx + i] << '\n';
        }
      }
    }
  }

  // Computed grid as a flat phase_id array, exposed for tests and any
  // downstream tool that wants the data without parsing the file format.
  [[nodiscard]] std::vector<phase_id>
  sample(shape_vector const& shapes,
         std::array<value_type, 3> const& domain_box) const {
    return sample_voxel_grid(shapes, domain_box, _nx, _ny, _nz);
  }

private:
  std::size_t _nx{32};
  std::size_t _ny{32};
  std::size_t _nz{32};
  std::string _output_path{};
};

} // namespace rvegen
