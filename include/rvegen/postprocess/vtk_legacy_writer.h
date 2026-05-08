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
#include "../types.h"
#include "voxel_grid.h"

namespace rvegen {

// ParaView-friendly legacy VTK file writer. Emits an ASCII
// STRUCTURED_POINTS dataset that ParaView opens directly, with one scalar
// "phase" per voxel (0 = matrix, 1..N = inclusion index).
//
// This writer does NOT depend on the VTK library — it's pure text output,
// the format is a few-line header plus the flat phase grid. The file
// extension is conventionally ".vtk".
//
// Sampling is shared with voxel_writer via sample_voxel_grid.
template <typename T = double>
class vtk_legacy_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  vtk_legacy_writer() = default;

  vtk_legacy_writer(std::size_t nx, std::size_t ny, std::size_t nz) noexcept
      : _nx{nx}, _ny{ny}, _nz{nz} {}

  vtk_legacy_writer(std::size_t nx, std::size_t ny, std::size_t nz,
                    std::string output_path)
      : _nx{nx}, _ny{ny}, _nz{nz}, _output_path{std::move(output_path)} {}

  explicit vtk_legacy_writer(parameter_handler_t const& handler)
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
        .description("destination path for the legacy VTK file (.vtk)");
    return s;
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
          "vtk_legacy_writer: output_path not set; cannot run() as post-process"};
    }
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "vtk_legacy_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
  }

  // Stream-based emit, kept public for tests / embedders.
  void write(std::ostream& out,
             shape_vector const& shapes,
             std::array<value_type, 3> const& domain_box) const {
    const auto grid = sample_voxel_grid(shapes, domain_box, _nx, _ny, _nz);
    const std::size_t nz_eff = effective_nz(domain_box, _nz);

    const auto dx = domain_box[0] / static_cast<value_type>(_nx);
    const auto dy = domain_box[1] / static_cast<value_type>(_ny);
    const auto dz = (nz_eff > 1)
        ? domain_box[2] / static_cast<value_type>(nz_eff)
        : value_type{1};

    out << "# vtk DataFile Version 3.0\n"
        << "rvegen voxel grid\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << _nx << ' ' << _ny << ' ' << nz_eff << '\n'
        << "ORIGIN 0 0 0\n"
        << "SPACING " << dx << ' ' << dy << ' ' << dz << '\n'
        << "POINT_DATA " << (_nx * _ny * nz_eff) << '\n'
        << "SCALARS phase int 1\n"
        << "LOOKUP_TABLE default\n";

    for (std::size_t k = 0; k < nz_eff; ++k) {
      for (std::size_t j = 0; j < _ny; ++j) {
        for (std::size_t i = 0; i < _nx; ++i) {
          out << grid[(k * _ny + j) * _nx + i] << '\n';
        }
      }
    }
  }

private:
  std::size_t _nx{32};
  std::size_t _ny{32};
  std::size_t _nz{32};
  std::string _output_path{};
};

} // namespace rvegen
