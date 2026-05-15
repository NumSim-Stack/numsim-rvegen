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
#include "../phase/phase.h"
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
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::unit_label<"cells">>()
        .template add<numsim_core::description_label<"voxel-grid resolution along x">>();
    s.template insert<std::size_t>("ny").template add<numsim_core::is_required>()
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::unit_label<"cells">>()
        .template add<numsim_core::description_label<"voxel-grid resolution along y">>();
    s.template insert<std::size_t>("nz").template add<numsim_core::is_required>()
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::unit_label<"cells">>()
        .template add<numsim_core::description_label<"voxel-grid resolution along z (collapses to 1 for 2D RVEs)">>();
    s.template insert<std::string>("output_path").template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"destination path for the legacy VTK file (.vtk)">>();
    return s;
  }

  [[nodiscard]] std::size_t nx() const noexcept { return _nx; }
  [[nodiscard]] std::size_t ny() const noexcept { return _ny; }
  [[nodiscard]] std::size_t nz() const noexcept { return _nz; }
  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

  // Same phase-collection plumbing as voxel_writer: when attached, the
  // emitted `phase` scalar is the real phase id (matrix=0 / untagged,
  // 1..N from `phase_collection`) instead of the legacy 1-based shape
  // index. Caller retains ownership; pointer must outlive the next
  // run()/write() call. Pass nullptr to clear.
  void set_phases(phase_collection<value_type> const* phases) noexcept {
    _phases = phases;
  }
  [[nodiscard]] phase_collection<value_type> const* phases() const noexcept {
    return _phases;
  }

  // Strict mode for phase tagging — mirrors voxel_writer's. With
  // phases attached AND strict==true, an untagged shape (empty
  // `phase_name`) makes `write()` throw. Default is the lenient
  // "untagged → 0" fallback. When no phase_collection is attached,
  // strict mode is silently a no-op.
  void set_phase_strict(bool strict) noexcept { _phase_strict = strict; }
  [[nodiscard]] bool phase_strict() const noexcept { return _phase_strict; }

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
    const auto grid = _phases
        ? sample_voxel_grid(shapes, domain_box, _nx, _ny, _nz,
                            *_phases, _phase_strict)
        : sample_voxel_grid(shapes, domain_box, _nx, _ny, _nz);
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
  phase_collection<value_type> const* _phases{nullptr};
  bool _phase_strict{false};
};

} // namespace rvegen
