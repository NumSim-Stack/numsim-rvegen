#pragma once

// Run summary (#119): one JSON file that says what a run produced and how,
// written like any other output through a "postprocess" entry:
//
//   { "type": "run_summary", "output_path": "run.json" }
//
// Contents:
//   rvegen_version   version of the library that wrote the file
//   domain_box       [Lx, Ly, Lz]
//   n_shapes         shapes in the result, periodic copies included
//   n_inclusions     inclusions (each counted once)
//   volume_fraction  inclusion volume (3D) or area (2D) fraction
// and, when the writer runs inside the pipeline (set_run_info):
//   seed             the seed the run used
//   report           the generation report: termination, target, achieved,
//                    attempts, max_attempts, stop_reason, target_reached
//   config           the config as it was given
// so a study over many seeds or box sizes can collect one file per run.

#include <array>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
#include <numsim-core/input_parameter_controller.h>

#include "../generators/rve_generator_base.h"
#include "../types.h"
#include "../version.h"
#include "post_process_base.h"

namespace rvegen {

template <typename T = double>
class run_summary_writer final : public post_process_base<T> {
public:
  using value_type = T;
  using shape_vector = typename post_process_base<T>::shape_vector;

  explicit run_summary_writer(std::string output_path)
      : _output_path{std::move(output_path)} {}
  explicit run_summary_writer(parameter_handler_t const& handler)
      : _output_path{handler.template get<std::string>("output_path")} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t p;
    p.template required<std::string>("output_path")
        .template add<numsim_core::description_label<"destination JSON path of the run summary (version, seed, config, generation report, shape count, volume fraction)">>();
    return p;
  }

  void set_run_info(nlohmann::json const* info) noexcept override {
    _info = info;
  }

  void validate(std::array<value_type, 3> const& /*domain_box*/) const override {
    if (_output_path.empty())
      throw std::runtime_error{"run_summary: output_path is empty"};
  }

  // The summary as JSON (what run() writes).
  [[nodiscard]] nlohmann::json
  summary(shape_vector const& shapes,
          std::array<value_type, 3> const& domain_box) const {
    nlohmann::json j = nlohmann::json::object();
    j["rvegen_version"] = version;
    j["domain_box"] = domain_box;
    j["n_shapes"] = shapes.size();
    std::size_t inclusions = 0;
    for (auto const& s : shapes)
      if (s && s->is_primary()) ++inclusions;
    j["n_inclusions"] = inclusions;
    j["volume_fraction"] = current_volume_fraction<T>(shapes, domain_box);
    if (_info != nullptr && _info->is_object())
      for (auto const& [key, value] : _info->items()) j[key] = value;
    return j;
  }

  void run(shape_vector const& shapes,
           std::array<value_type, 3> const& domain_box) const override {
    validate(domain_box);
    std::ofstream out{_output_path};
    if (!out)
      throw std::runtime_error{"run_summary: cannot open '" + _output_path +
                               "' for writing"};
    out << summary(shapes, domain_box).dump(2) << '\n';
    if (!out)
      throw std::runtime_error{"run_summary: writing '" + _output_path +
                               "' failed"};
  }

  [[nodiscard]] std::string const& output_path() const noexcept {
    return _output_path;
  }

private:
  std::string _output_path;
  nlohmann::json const* _info{nullptr};
};

} // namespace rvegen
