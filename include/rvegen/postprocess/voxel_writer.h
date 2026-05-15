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
#include "../phase/phase.h"
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
        .template add<numsim_core::description_label<"destination path for the voxel-grid file">>();
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

  // Attach a phase_collection so the writer emits real phase ids (matrix
  // = 0, fibre = 1, ...) keyed off each shape's `phase_name()` instead of
  // the 1-based shape index. Caller retains ownership; the pointer must
  // outlive the writer's next `run()` / `write()` / `sample()` call.
  // Passing nullptr clears the attachment and restores shape-index mode.
  void set_phases(phase_collection<value_type> const* phases) noexcept {
    _phases = phases;
  }
  [[nodiscard]] phase_collection<value_type> const* phases() const noexcept {
    return _phases;
  }

  // Strict mode for phase tagging — mirrors gmsh_geo_writer's. With
  // phases attached AND strict==true, an untagged shape (empty
  // `phase_name`) makes `sample()` / `write()` throw instead of
  // mapping the shape's voxels to 0. Default is the lenient
  // "untagged → 0" fallback. When no phase_collection is attached,
  // strict mode is silently a no-op.
  void set_phase_strict(bool strict) noexcept { _phase_strict = strict; }
  [[nodiscard]] bool phase_strict() const noexcept { return _phase_strict; }

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

    // File format convention: any number of `#`-prefixed comment lines
    // (variable count — depends on whether phases are attached), then a
    // flat sequence of integer voxel values. Parsers MUST skip all
    // leading `#` lines rather than counting fixed offsets, because the
    // count changes when `set_phases()` is used.
    //
    // The version-marker token `[rvegen-voxel-format: v2]` matches the
    // convention used by `vtk_legacy_writer` (and any future writer)
    // — bracketed `[rvegen-<format>: v<N>]` is searchable anywhere in
    // the comment header and explicitly identifies the file format.
    out << "# rvegen voxel grid [rvegen-voxel-format: v2]\n"
        << "# nx ny nz: " << _nx << ' ' << _ny << ' ' << nz_eff << '\n'
        << "# Lx Ly Lz: " << domain_box[0] << ' ' << domain_box[1] << ' '
        << domain_box[2] << '\n';
    if (_phases) {
      out << "# id scheme: phase ids from phase_collection (0 = matrix / untagged)\n";
      for (auto const* p : _phases->ordered()) {
        out << "# phase " << p->id << " = " << p->name << '\n';
      }
    } else {
      out << "# id scheme: 1-based shape index (0 = matrix)\n";
    }

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
    if (_phases) {
      return sample_voxel_grid(shapes, domain_box, _nx, _ny, _nz,
                               *_phases, _phase_strict);
    }
    return sample_voxel_grid(shapes, domain_box, _nx, _ny, _nz);
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
