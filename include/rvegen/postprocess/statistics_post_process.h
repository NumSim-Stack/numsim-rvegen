#pragma once

// Statistics post-process — a CSV file with the standard geometric
// descriptors of the RVE: the volume fraction, the two-point probability
// S2 and the radial distribution g(r) of the centres. Reuses the helpers
// in rvegen/statistics/microstructure_stats.h.
//
// JSON usage:
//   { "type": "statistics",
//     "output_path": "stats.csv",
//     "nx": 128, "ny": 128,
//     "nz": 128,            // required for 3D domains, ignored in 2D
//     "nbins": 32,
//     "periodic": true,     // optional, see below
//     "phase_correlations": true }   // optional
//
// S2 is exact (#127): the pair counts of the nx x ny (x nz) voxel grid,
// computed by FFT, with no random sampling. The file has one row for
// r = 0, where S2 is the volume fraction of the grid, and one per bin
// (r is the bin centre, bins up to half the smallest box edge):
//
//   r, S2(r), g(r), S2_x(r), S2_y(r)[, S2_z(r)]
//
// S2(r) is the average over all directions, S2_x / S2_y / S2_z take the
// lags along one axis only: they agree for an isotropic structure and
// differ for aligned fibres or a weave. A field is empty where a bin
// holds no lag (bins narrower than a voxel) and where g(r) is undefined.
//
// "periodic" says how the box ends. true: lags wrap around, right for an
// RVE that was generated periodic. false: only pairs inside the box are
// counted, right for everything else. Left out, it follows the generator
// ("periodic" generator, or a generator with "periodic": true), and is
// false when the post-process runs outside the pipeline.
//
// With phases declared, the same four columns follow for every phase
// (S2[fibre]...) and every pair of phases (S2[fibre|matrix]: one point in
// each), unless "phase_correlations" is false or there are more than
// `max_correlated_phases`.
//
// The voxelization costs O(nx·ny·nz · shapes) is_inside calls (128³ ≈
// 2.1 M voxels), so `nz` has no default: a 3D domain without it throws
// from run(), making the user pick the z resolution and therefore the
// cost. The FFTs need 16 bytes per voxel and stored phase; a box that is
// not periodic is padded to about 1.5 times the voxels per axis.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
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
        _nbins{handler.template get<std::size_t>("nbins")},
        _periodic{handler.contains("periodic")
                      ? std::optional<bool>{handler.template get<bool>("periodic")}
                      : std::nullopt},
        _phase_correlations{
            handler.template get<bool>("phase_correlations")} {}

  // More phases than this and the columns per phase and pair of phases
  // are left out: n phases are n (n + 1) / 2 correlations of four columns.
  static constexpr std::size_t max_correlated_phases = 12;

  void set_grid(std::size_t nx, std::size_t ny, std::size_t nz = 0) noexcept {
    _nx = nx;
    _ny = ny;
    _nz = nz;
  }
  void set_nbins(std::size_t nbins) noexcept { _nbins = nbins; }
  void set_output_path(std::string path) { _output_path = std::move(path); }
  // nullopt: follow the generator of the run (see the header comment).
  void set_periodic(std::optional<bool> periodic) noexcept { _periodic = periodic; }
  void set_phase_correlations(bool on) noexcept { _phase_correlations = on; }

  void set_phases(phase_collection<value_type> const* phases) noexcept override {
    _phases = phases;
  }
  void set_run_info(nlohmann::json const* info) noexcept override {
    _info = info;
  }

  // How the box ends for S2: as set, else as the run was generated.
  [[nodiscard]] bool periodic() const {
    if (_periodic) return *_periodic;
    if (_info == nullptr || !_info->is_object()) return false;
    auto const config = _info->find("config");
    if (config == _info->end() || !config->is_object()) return false;
    auto const generator = config->find("generator");
    if (generator == config->end() || !generator->is_object()) return false;
    auto const type = generator->find("type");
    if (type != generator->end() && type->is_string() &&
        type->template get<std::string>() == "periodic") {
      return true;
    }
    auto const flag = generator->find("periodic");
    return flag != generator->end() && flag->is_boolean() &&
           flag->template get<bool>();
  }

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
    // Not required and without a default: left out, it follows the generator.
    s.template insert<bool>("periodic")
        .template add<numsim_core::description_label<"S2: true wraps lags around the box (a periodic RVE), false counts only pairs inside it; left out, as the generator of the run">>();
    s.template optional<bool>("phase_correlations", true)
        .template add<numsim_core::description_label<"with phases: S2 columns for every phase and pair of phases">>();
    return s;
  }

  void validate(domain<value_type> const& domain_box) const override {
    if (_output_path.empty()) {
      throw std::runtime_error{
          "statistics_post_process: output_path not set"};
    }
    if (domain_box.is_3d() && _nz == 0) {
      throw std::runtime_error{
          "statistics_post_process: 3D domain needs \"nz\" (voxel-grid "
          "resolution along z for S2). Set it explicitly; 3D S2 cost scales "
          "with nx*ny*nz."};
    }
  }

  void run(shape_vector const& shapes,
           domain<value_type> const& domain_box) const override {
    validate(domain_box);
    const bool is_3d = domain_box.is_3d();
    const double phi = volume_fraction_in_domain(shapes, domain_box);
    const bool wrap = periodic();
    const bool with_phases = _phase_correlations && _phases != nullptr &&
                             !_phases->empty() &&
                             _phases->size() <= max_correlated_phases;
    const auto s2 = two_point_correlations(
        shapes, domain_box, _nx, _ny, is_3d ? _nz : std::size_t{1}, _nbins,
        wrap ? two_point_boundary::periodic : two_point_boundary::bounded,
        with_phases ? _phases : nullptr);
    auto rdf = is_3d ? radial_distribution_3d(shapes, domain_box, _nbins)
                     : radial_distribution_2d(shapes, domain_box, _nbins);

    // A name inside a column title: nothing that ends a CSV field.
    auto plain = [](std::string name) {
      for (char& c : name)
        if (c == ',' || c == '\n' || c == '\r' || c == '|') c = '_';
      return name;
    };
    const std::size_t axes = is_3d ? 3 : 2;
    constexpr std::array<char const*, 3> axis_name{"x", "y", "z"};
    std::string columns = "r, S2(r), g(r)";
    for (std::size_t a = 0; a < axes; ++a)
      columns += std::string{", S2_"} + axis_name[a] + "(r)";
    for (auto const& c : s2.phases) {
      const std::string of =
          c.a == c.b ? "[" + plain(c.a) + "]"
                     : "[" + plain(c.a) + "|" + plain(c.b) + "]";
      columns += ", S2" + of + "(r)";
      for (std::size_t a = 0; a < axes; ++a)
        columns += std::string{", S2_"} + axis_name[a] + of + "(r)";
    }

    // Everything is computed: a grid that cannot be used has thrown by
    // now, before the file exists.
    std::ofstream out{_output_path};
    if (!out) {
      throw std::runtime_error{
          "statistics_post_process: cannot open '" + _output_path + "'"};
    }
    out << std::setprecision(12);
    out << "# rvegen microstructure statistics\n"
        << "# domain_box = " << domain_box[0] << ' ' << domain_box[1]
        << ' ' << domain_box[2] << '\n'
        << "# dimension = " << (is_3d ? "3D" : "2D") << '\n'
        << "# volume_fraction = " << phi << '\n'
        << "# S2 = pair counts of the " << s2.grid[0] << " x " << s2.grid[1];
    if (is_3d) out << " x " << s2.grid[2];
    out << " voxel grid, exact (FFT); "
        << (wrap ? "periodic box, lags wrap around"
                 : "bounded box, pairs inside the box only")
        << '\n'
        << "# voxel_volume_fraction = " << s2.inclusion.zero << '\n';
    if (_phase_correlations && _phases != nullptr && !_phases->empty() &&
        !with_phases) {
      out << "# phase correlations left out: " << _phases->size()
          << " phases are more than " << max_correlated_phases << '\n';
    }
    out << "# columns: " << columns << '\n';
    // g(r) is undefined for fewer than two inclusions (empty rdf). S2 is
    // still valid, so every S2 bin gets a row and the g field stays empty.
    if (rdf.size() < _nbins) {
      out << "# g(r) undefined (fewer than 2 inclusions): g column left empty\n";
    }

    // A value, or nothing where a bin holds no lag.
    auto field = [&](double v) {
      out << ',';
      if (!std::isnan(v)) out << v;
    };
    auto along = [&](two_point_profile const& p, std::size_t bin, bool zero) {
      for (std::size_t a = 0; a < axes; ++a) {
        if (zero) field(p.zero);
        else if (bin < p.axis[a].size()) field(p.axis[a][bin]);
        else out << ',';   // one voxel along this axis: no lag
      }
    };
    auto row = [&](std::size_t bin, bool zero) {
      out << (zero ? 0.0 : s2.r(bin));
      field(zero ? s2.inclusion.zero : s2.inclusion.radial[bin]);
      out << ',';
      if (!zero && bin < rdf.size()) out << rdf[bin].g;
      along(s2.inclusion, bin, zero);
      for (auto const& c : s2.phases) {
        field(zero ? c.profile.zero : c.profile.radial[bin]);
        along(c.profile, bin, zero);
      }
      out << '\n';
    };
    row(0, true);
    for (std::size_t b = 0; b < _nbins; ++b) row(b, false);
    out.flush();
    if (!out) {
      throw std::runtime_error{
          "statistics_post_process: writing '" + _output_path + "' failed"};
    }
  }

private:
  std::string _output_path;
  std::size_t _nx{128};
  std::size_t _ny{128};
  std::size_t _nz{0};  // 0 = not set; required for 3D domains
  std::size_t _nbins{32};
  std::optional<bool> _periodic;
  bool _phase_correlations{true};
  phase_collection<value_type> const* _phases{nullptr};
  nlohmann::json const* _info{nullptr};
};

} // namespace rvegen
