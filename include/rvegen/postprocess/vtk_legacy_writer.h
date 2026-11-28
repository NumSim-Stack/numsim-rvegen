#pragma once

#include <array>
#include <cstddef>
#include <fstream>
#include <ios>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

#include <numsim-core/input_parameter_controller.h>

#include "post_process_base.h"
#include "../phase/phase.h"
#include "../types.h"
#include "../util/format_number.h"
#include "phase_table.h"
#include "voxel_grid.h"

namespace rvegen {

// ParaView-friendly legacy VTK file writer. Emits an ASCII
// STRUCTURED_POINTS dataset that ParaView opens directly, with one scalar
// per voxel (0 = matrix; remaining values follow the active id scheme,
// see below).
//
// Layout (rvegen-vtk-format v3): each voxel is one VTK *cell*.
// DIMENSIONS are the point counts (nx+1, ny+1, nz+1), ORIGIN is 0 0 0,
// SPACING is the voxel size (Lx/nx, Ly/ny, Lz/nz), and the scalars are
// CELL_DATA with nx·ny·nz values. Cell (i, j, k) therefore spans
// [i·dx, (i+1)·dx] × ... and its value is the sample taken at the cell
// centre, so ParaView shows the field exactly over [0, L] with no
// half-voxel shift. 2D RVEs keep a single z point (DIMENSIONS nx+1 ny+1 1),
// which VTK treats as nx·ny pixel cells.
//
// This writer does NOT depend on the VTK library — it's pure text output,
// the format is a few-line header plus the flat phase grid. The file
// extension is conventionally ".vtk".
//
// Sampling is shared with voxel_writer via sample_voxel_grid.
//
// Output is NOT bit-stable for a given shape list: the SCALARS field
// name and the title-line suffix depend on whether `set_phases()` was
// called and which collection is attached. The title line carries
// `[rvegen-vtk-format: v3]` so downstream parsers can detect the
// scheme change; snapshot-diff tests should expect the title to vary
// across writer-state changes.
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
  void set_phases(phase_collection<value_type> const* phases) noexcept override {
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
    // Reject an unusable grid before the output file is created.
    (void)checked_voxel_count(_nx, _ny, effective_nz(domain_box, _nz));
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "vtk_legacy_writer: cannot open '" + _output_path + "' for writing"};
    }
    write(out, shapes, domain_box);
    out.close();
    if (_phases) write_phase_table_sidecar(_output_path, *_phases);
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
    const bool is_3d = domain_box[2] > value_type{0};
    const auto dz = is_3d
        ? domain_box[2] / static_cast<value_type>(nz_eff)
        : value_type{1};
    // Point dimensions: one more than the cell count along every sampled
    // axis; a 2D RVE stays flat with a single z point.
    const std::size_t pz = is_3d ? nz_eff + 1 : std::size_t{1};

    // Field name distinguishes id schemes so a ParaView user looking
    // at a .vtk file can tell whether the integers are real phase ids
    // (set_phases attached) or 1-based shape indices (legacy fallback)
    // without re-reading the rvegen config that generated the file.
    // The VTK legacy format's title-line is also pressed into service
    // as a short scheme note for the same reason.
    //
    // The title line carries a "[rvegen-vtk-format: vN]" token so a
    // downstream parser can detect schema changes unambiguously.
    //   v1: single fixed SCALARS name ("phase").
    //   v2: SCALARS renamed to "phase_id" / "shape_index" depending on
    //       writer state.
    //   v3 (current): voxels moved from POINT_DATA on an
    // nx × ny × nz point lattice (which ParaView drew half a voxel off,
    // with extent (n−1)·d) to CELL_DATA on an (nx+1) × (ny+1) × (nz+1)
    // point lattice, so each voxel occupies its true box in [0, L].
    const char* const field_name  = _phases ? "phase_id" : "shape_index";
    const char* const title_suffix = _phases
        ? (_phases->has_matrix()
               ? " (phase_id from phase_collection; uncovered cells = matrix phase id, 0 = untagged; "
                 "name table in <output>.phases.json)"
               : " (phase_id from phase_collection, 0 = matrix/untagged; "
                 "name table in <output>.phases.json)")
        : " (1-based inclusion index, wraps share their primary's id, 0 = matrix)";

    // SPACING is written with format_number (reads back to the same
    // value), so L = n · spacing holds in the reader.
    out << "# vtk DataFile Version 3.0\n"
        << "rvegen voxel grid [rvegen-vtk-format: v3]" << title_suffix << "\n"
        << "ASCII\n"
        << "DATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << (_nx + 1) << ' ' << (_ny + 1) << ' ' << pz << '\n'
        << "ORIGIN 0 0 0\n"
        << "SPACING " << format_number(dx) << ' ' << format_number(dy)
        << ' ' << format_number(dz) << '\n'
        << "CELL_DATA " << checked_voxel_count(_nx, _ny, nz_eff) << '\n'
        << "SCALARS " << field_name << " int 1\n"
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
