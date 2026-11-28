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
#include "../util/format_number.h"
#include "phase_table.h"
#include "voxel_fractions.h"
#include "voxel_grid.h"

namespace rvegen {

// Voxel-grid writer (regular phase-id grid, e.g. for FFT solvers).
//
// Subdivides the RVE box into nx × ny × nz voxels, samples each voxel's
// centre point with shape_base::is_inside, and writes a phase-id grid:
//   0       — matrix (no inclusion)
//   1..N    — index (1-based) of the inclusion the voxel falls inside
//
// 2D RVEs collapse to a single z-slice (nz = 1).
// For a 3D RVE written with nz = 1 the single slice is the mid-plane
// z = Lz/2 (voxel centres, as along x and y).
//
// Output format: ASCII header + flat array, voxels in (k, j, i) order
// (z-slice fastest-varying-axis is k). The format is intentionally
// human-readable and trivial to parse from any FFT solver.
//
// Partial volumes (#115): with `subsamples` k > 1 every voxel is sampled
// at k points per axis instead of its centre. The value of a voxel is then
// the id with the largest share, and the voxels that hold more than one id
// are listed after the grid with the share of each (format v4, see
// write()). With k = 1 the file is the v3 file, byte for byte.
//
// run() also reports, through the log of post_process_base, the volume
// fraction the grid ended up with next to the one of the shapes, and warns
// when a shape is fewer than `min_voxels_per_feature` voxels across.
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
        _output_path{handler.template get<std::string>("output_path")},
        _subsamples{handler.template get<std::size_t>("subsamples")},
        _min_voxels{handler.template get<value_type>("min_voxels_per_feature")} {}

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
    add_voxel_sampling_parameters<value_type>(s);
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

  // Sample points per voxel and axis (1: the voxel centre).
  void set_subsamples(std::size_t subsamples) noexcept {
    _subsamples = subsamples;
  }
  [[nodiscard]] std::size_t subsamples() const noexcept { return _subsamples; }
  // Voxels a shape should be across before run() warns; 0: no check.
  void set_min_voxels_per_feature(value_type voxels) noexcept {
    _min_voxels = voxels;
  }
  [[nodiscard]] value_type min_voxels_per_feature() const noexcept {
    return _min_voxels;
  }

  // Attach a phase_collection so the writer emits real phase ids keyed
  // off each shape's `phase_name()` instead of the 1-based shape index.
  // Voxels no shape covers carry the declared matrix phase's id
  // (phase_collection::set_matrix), or 0 when none is declared. run()
  // also writes the `<output_path>.phases.json` name table. Caller retains ownership; the pointer must
  // outlive the writer's next `run()` / `write()` / `sample()` call.
  // Passing nullptr clears the attachment and restores shape-index mode.
  void set_run_info(nlohmann::json const* info) noexcept override {
    _info = info;
  }

  void set_phases(phase_collection<value_type> const* phases) noexcept override {
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

  void validate(domain<value_type> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "voxel_writer: output_path not set; cannot run() as post-process"};
    }
    // Reject an unusable grid before the output file is created.
    validate_voxel_sampling(domain_box, _nx, _ny, _nz, _subsamples);
  }

  void run(shape_vector const& shapes,
           domain<value_type> const& domain_box) const override {
    validate(domain_box);
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "voxel_writer: cannot open '" + _output_path + "' for writing"};
    }
    const auto samples_of_id = write_counted(out, shapes, domain_box);
    out.close();
    if (_phases) write_phase_table_sidecar(_output_path, *_phases);
    if (this->has_log()) {
      report_voxel_sampling<value_type>(
          shapes, domain_box, _nx, _ny, _nz, _subsamples,
          static_cast<double>(_min_voxels), _phases, _phase_strict,
          samples_of_id,
          [this](std::string const& line) { this->log_info(line); },
          [this](std::string const& line) { this->log_warn(line); });
    }
  }

  // Stream-based emit. Public so tests and embedders can capture the output
  // to a stringstream without round-tripping through a file.
  void write(std::ostream& out,
             shape_vector const& shapes,
             domain<value_type> const& domain_box) const {
    (void)write_counted(out, shapes, domain_box);
  }

  // The partial-volume grid this writer would write (any `subsamples`).
  [[nodiscard]] voxel_fractions
  sample_fractions(shape_vector const& shapes,
                   domain<value_type> const& domain_box) const {
    if (_phases) {
      return sample_voxel_fractions(shapes, domain_box, _nx, _ny, _nz,
                                    _subsamples, *_phases, _phase_strict);
    }
    return sample_voxel_fractions(shapes, domain_box, _nx, _ny, _nz,
                                  _subsamples);
  }

private:
  // write(), returning the sample points per id of what was written.
  [[nodiscard]] std::vector<std::uint64_t>
  write_counted(std::ostream& out, shape_vector const& shapes,
                domain<value_type> const& domain_box) const {
    if (_subsamples != 1) return write_fractions(out, shapes, domain_box);
    const auto grid = sample(shapes, domain_box);
    const std::size_t nz_eff = effective_nz(domain_box, _nz);

    // File format convention: any number of `#`-prefixed comment lines
    // (variable count — depends on whether phases are attached), then a
    // flat sequence of integer voxel values. Parsers MUST skip all
    // leading `#` lines rather than counting fixed offsets, because the
    // count changes when `set_phases()` is used.
    //
    // The version-marker token `[rvegen-voxel-format: v3]` matches the
    // convention used by `vtk_legacy_writer` (and any future writer)
    // — bracketed `[rvegen-<format>: v<N>]` is searchable anywhere in
    // the comment header and explicitly identifies the file format.
    // v3: a 3D RVE written with nz = 1 samples the mid-plane z = Lz/2
    // (v2 sampled the bottom face z = 0).
    // The extents are written with format_number (as the VTK SPACING
    // line), so h = L/n is recoverable exactly; the layout, and so the
    // format version, is unchanged.
    out << "# rvegen voxel grid [rvegen-voxel-format: v3]\n"
        << "# nx ny nz: " << _nx << ' ' << _ny << ' ' << nz_eff << '\n'
        << "# Lx Ly Lz: " << format_number(domain_box[0]) << ' '
        << format_number(domain_box[1]) << ' '
        << format_number(domain_box[2]) << '\n';
    write_id_scheme(out);

    for (std::size_t k = 0; k < nz_eff; ++k) {
      for (std::size_t j = 0; j < _ny; ++j) {
        for (std::size_t i = 0; i < _nx; ++i) {
          out << grid[(k * _ny + j) * _nx + i] << '\n';
        }
      }
    }
    return count_voxel_ids(grid);
  }

  // Format v4, written for subsamples > 1. The header gains two lines
  // (the subsamples, and the number M of composite voxels with the layout
  // of their lines); the N = nx * ny * nz values are the dominant id of
  // each voxel; then M lines follow, one per composite voxel:
  //   <voxel index (k * ny + j) * nx + i> <id> <samples> <id> <samples> ...
  // with the ids ascending and the samples adding up to the sample points
  // per voxel, so <samples> / <sample points per voxel> is the exact share.
  // A voxel that is not listed belongs to its id entirely.
  [[nodiscard]] std::vector<std::uint64_t>
  write_fractions(std::ostream& out, shape_vector const& shapes,
                  domain<value_type> const& domain_box) const {
    const auto f = sample_fractions(shapes, domain_box);
    out << "# rvegen voxel grid [rvegen-voxel-format: v4]\n"
        << "# nx ny nz: " << f.nx << ' ' << f.ny << ' ' << f.nz << '\n'
        << "# Lx Ly Lz: " << format_number(domain_box[0]) << ' '
        << format_number(domain_box[1]) << ' '
        << format_number(domain_box[2]) << '\n'
        << "# subsamples: " << f.subsamples << " (" << f.samples_per_voxel
        << " sample points per voxel)\n";
    write_id_scheme(out);
    out << "# values: the id with the largest share of each voxel (the "
           "smallest id on a tie)\n"
        << "# composite voxels: " << f.mixed.size() << " (after the "
        << f.dominant.size() << " values, one line each: voxel index "
           "(k*ny + j)*nx + i, then pairs \"id samples\" adding up to "
        << f.samples_per_voxel << ")\n";
    for (const phase_id id : f.dominant) out << id << '\n';
    for (std::size_t m = 0; m < f.mixed.size(); ++m) {
      out << f.mixed[m];
      for (std::size_t e = f.first[m]; e < f.first[m + 1]; ++e)
        out << ' ' << f.shares[e].id << ' ' << f.shares[e].samples;
      out << '\n';
    }
    return f.samples_of_id;
  }

  // The comment lines on the run's gaps and on what the ids mean.
  void write_id_scheme(std::ostream& out) const {
    // Only when the run kept minimum gaps (#111): one more comment line.
    if (auto const note = this->placement_gap_note(_info); !note.empty()) {
      out << "# placement: " << note << '\n';
    }
    if (_phases) {
      if (_phases->has_matrix()) {
        out << "# id scheme: phase ids from phase_collection (uncovered "
               "voxels carry the matrix phase id; 0 = untagged shape)\n";
      } else {
        out << "# id scheme: phase ids from phase_collection (0 = matrix / untagged)\n";
      }
      for (auto const* p : _phases->ordered()) {
        out << "# phase " << p->id << " = " << p->name << '\n';
      }
      if (_phases->has_matrix()) {
        out << "# matrix phase: " << _phases->matrix_name() << " (id "
            << _phases->matrix_id() << ")\n";
      }
    } else {
      out << "# id scheme: 1-based inclusion index, periodic wraps share "
             "their primary's id (0 = matrix)\n";
    }
  }

public:
  // Computed grid as a flat phase_id array, exposed for tests and any
  // downstream tool that wants the data without parsing the file format.
  // One sample at the voxel centre, whatever `subsamples` is.
  [[nodiscard]] std::vector<phase_id>
  sample(shape_vector const& shapes,
         domain<value_type> const& domain_box) const {
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
  std::size_t _subsamples{1};
  value_type _min_voxels{8};
  phase_collection<value_type> const* _phases{nullptr};
  nlohmann::json const* _info{nullptr};
  bool _phase_strict{false};
};

} // namespace rvegen
