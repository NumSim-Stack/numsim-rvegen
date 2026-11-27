#pragma once

// damask_material_config — writes a DAMASK (v3) `material.yaml` with one
// material entry per rvegen phase (#7).
//
// Layout:
//
//   homogenization: { SX: { N_constituents: 1, mechanical: {type: pass} } }
//   phase:          { "<name>": { lattice: cF, mechanical: {...} }, ... }
//   material:       [ { constituents: [{phase: "<name>", v: 1.0, O: [1,0,0,0]}],
//                       homogenization: SX }, ... ]
//
// Material index k (0-based, as DAMASK resolves the grid's material ids)
// equals the voxel / rvegen phase id k:
//   * k = 0 is the MATRIX (voxel cells no inclusion covers). It uses the
//     phase named by the `matrix_phase` parameter, defaulting to a phase
//     called "matrix"; the CLI forwards phase_assignment's "matrix" key
//     here. If no matrix phase exists, write() throws -- an inclusion RVE
//     cannot run in DAMASK without a matrix material.
//   * k >= 1 is rvegen phase id k (insertion order), so the matrix phase,
//     if declared as a regular phase, appears twice (as 0 and as its id).
//
// Per-phase constitutive data comes from the phase's opaque
// `material_config` JSON:
//   * `"damask": {...}`  -> emitted verbatim as the phase's `mechanical`
//     block (JSON is valid YAML flow syntax); full control for plastic /
//     anisotropic models.
//   * isotropic elasticity from any of {K,G}, {E,nu}, {lambda,mu} (top
//     level of material_config) -> `elastic: {type: Hooke, C_11, C_12,
//     C_44 (+C_13, C_33, C_66 for hP/tI)}`. The constants are validated
//     (K>0, G>0, i.e. E>0 and -1<nu<0.5); impossible values throw.
//   * `"lattice"` (material_config, default cF): one of cF, cI, hP, tI.
//   * otherwise the run throws, naming the phase.
//
// Orientations: every constituent gets the identity quaternion [1,0,0,0].
// Per-grain orientation for Voronoi polycrystals is NOT written: voronoi
// cells carry no orientation, so there is nothing to export.
//
// Names are written as JSON strings (valid YAML); the homogenization name
// must match [A-Za-z0-9_.-]+.

#include <array>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>
#include <numsim-core/input_parameter_controller.h>

#include "../phase/phase.h"
#include "../types.h"
#include "post_process_base.h"

namespace rvegen {

template <typename T = double>
class damask_material_config final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  damask_material_config() = default;

  explicit damask_material_config(parameter_handler_t const& handler)
      : _output_path{handler.template get<std::string>("output_path")},
        _homogenization{handler.template get<std::string>("homogenization")},
        _matrix_phase{handler.template get<std::string>("matrix_phase")} {
    check_homogenization_name(_homogenization);
  }

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<std::string>("output_path")
        .template add<numsim_core::is_required>()
        .template add<numsim_core::description_label<"destination path for the DAMASK material.yaml">>();
    s.template insert<std::string>("homogenization")
        .template add<numsim_core::set_default>(std::string{"SX"})
        .template add<numsim_core::description_label<"name of the single-constituent homogenization scheme (default SX)">>();
    s.template insert<std::string>("matrix_phase")
        .template add<numsim_core::set_default>(std::string{})
        .template add<numsim_core::description_label<"name of the phase used for voxel id 0 (the matrix); default: the phase called 'matrix'">>();
    return s;
  }

  void set_phases(phase_collection<T> const* phases) noexcept override {
    _phases = phases;
  }
  [[nodiscard]] phase_collection<T> const* phases() const noexcept {
    return _phases;
  }

  void run(shape_vector const& /*shapes*/,
           std::array<value_type, 3> const& /*domain_box*/) const override {
    if (_output_path.empty())
      throw std::runtime_error{"damask_material_config: output_path not set"};
    std::ofstream out{_output_path};
    if (!out)
      throw std::runtime_error{"damask_material_config: cannot open '" +
                               _output_path + "' for writing"};
    write(out);
  }

  void write(std::ostream& out) const {
    if (!_phases || _phases->empty())
      throw std::runtime_error{
          "damask_material_config: needs a non-empty \"phases\" array"};
    check_homogenization_name(_homogenization);
    std::string const matrix_name =
        _matrix_phase.empty() ? std::string{"matrix"} : _matrix_phase;
    if (!_phases->contains(matrix_name))
      throw std::runtime_error{
          "damask_material_config: no matrix phase '" + matrix_name +
          "' (voxel id 0 needs a material): declare a phase named \"matrix\" "
          "or set \"matrix_phase\" / phase_assignment.matrix"};
    out << "# material.yaml generated by rvegen; material index k = voxel / phase id k "
           "(0 = matrix)\n";
    out << "homogenization:\n  " << _homogenization
        << ":\n    N_constituents: 1\n    mechanical: {type: pass}\n";
    out << "phase:\n";
    for (auto const* p : _phases->ordered()) {
      out << "  " << quote(p->name) << ":\n    lattice: " << lattice(*p)
          << "\n    mechanical: " << mechanical_block(*p) << '\n';
    }
    out << "material:\n";
    auto entry = [&](std::string const& comment, std::string const& name) {
      out << "  - # " << comment << "\n"
          << "    constituents:\n"
          << "      - phase: " << quote(name) << "\n"
          << "        v: 1.0\n"
          << "        O: [1.0, 0.0, 0.0, 0.0]\n"
          << "    homogenization: " << _homogenization << '\n';
    };
    entry("voxel id 0 (matrix)", matrix_name);
    for (auto const* p : _phases->ordered())
      entry("phase id " + std::to_string(p->id) + " " + quote(p->name), p->name);
  }

private:
  static std::string quote(std::string const& s) {
    return nlohmann::json(s).dump();   // JSON string == valid YAML scalar
  }

  static void check_homogenization_name(std::string const& n) {
    if (n.empty() ||
        !std::all_of(n.begin(), n.end(), [](unsigned char c) {
          return std::isalnum(c) || c == '_' || c == '.' || c == '-';
        }))
      throw std::runtime_error{
          "damask_material_config: homogenization name must match "
          "[A-Za-z0-9_.-]+, got '" + n + "'"};
  }

  static std::string lattice(phase<T> const& p) {
    auto const& m = p.material_config;
    if (!m.contains("lattice")) return "cF";
    if (!m["lattice"].is_string())
      throw std::runtime_error{"damask_material_config: phase '" + p.name +
                               "': lattice must be a string"};
    auto l = m["lattice"].template get<std::string>();
    if (l != "cF" && l != "cI" && l != "hP" && l != "tI")
      throw std::runtime_error{
          "damask_material_config: phase '" + p.name + "': unsupported "
          "lattice '" + l + "' (isotropic constants support cF, cI, hP, tI; "
          "use a \"damask\" block for others)"};
    return l;
  }

  static std::string mechanical_block(phase<T> const& p) {
    auto const& m = p.material_config;
    if (m.contains("damask") && m["damask"].is_object())
      return m["damask"].dump();

    auto num = [&](char const* k) -> std::optional<double> {
      if (m.contains(k) && m[k].is_number()) return m[k].template get<double>();
      return std::nullopt;
    };
    auto bad = [&](std::string const& why) {
      return std::runtime_error{"damask_material_config: phase '" + p.name +
                                "': " + why};
    };
    std::optional<double> K, G;
    if (auto k = num("K"), g = num("G"); k && g) { K = k; G = g; }
    else if (auto e = num("E"), nu = num("nu"); e && nu) {
      if (!(std::isfinite(*e) && *e > 0.0))
        throw bad("Young's modulus E must be finite and > 0");
      if (!(std::isfinite(*nu) && *nu > -1.0 && *nu < 0.5))
        throw bad("Poisson ratio nu must be in (-1, 0.5)");
      K = *e / (3.0 * (1.0 - 2.0 * *nu));
      G = *e / (2.0 * (1.0 + *nu));
    } else if (auto l = num("lambda"), mu = num("mu"); l && mu) {
      K = *l + 2.0 * *mu / 3.0;
      G = *mu;
    }
    if (!K)
      throw std::runtime_error{
          "damask_material_config: phase '" + p.name +
          "' has no usable material (need K+G, E+nu, lambda+mu or a "
          "\"damask\" block in its material config)"};
    if (!(std::isfinite(*K) && *K > 0.0 && std::isfinite(*G) && *G > 0.0))
      throw bad("elastic constants must give finite K > 0 and G > 0");
    const double c11 = *K + 4.0 * *G / 3.0;
    const double c12 = *K - 2.0 * *G / 3.0;
    if (!std::isfinite(c11) || !std::isfinite(c12))
      throw bad("elastic constants overflow");
    nlohmann::json elastic = {{"type", "Hooke"},
                              {"C_11", c11}, {"C_12", c12}, {"C_44", *G}};
    auto const lat = lattice(p);
    if (lat == "hP" || lat == "tI") {
      elastic["C_13"] = c12;
      elastic["C_33"] = c11;
      if (lat == "tI") elastic["C_66"] = *G;
    }
    nlohmann::json mech = {{"output", {"F", "P"}}, {"elastic", elastic}};
    return mech.dump();
  }

  std::string _output_path;
  std::string _homogenization{"SX"};
  std::string _matrix_phase;
  phase_collection<T> const* _phases{nullptr};
};

} // namespace rvegen
