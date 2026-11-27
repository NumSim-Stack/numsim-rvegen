#pragma once

// rvegen pipeline — the JSON-config -> shapes -> post-processes flow shared
// by the CLI (apps/rvegen_cli.cpp) and the Python bindings.
//
//   auto cfg = nlohmann::json::parse(...);
//   rvegen::pipeline_options opts;               // seed override, hooks
//   auto result = rvegen::run_pipeline(cfg, opts);
//   // result.shapes, result.domain_box, result.seed, ...
//
// Steps: register types -> domain box / seed -> distributions (scalar +
// direction) -> inputs -> generator + termination -> compute ->
// phases / phase_assignment -> post-processes.  `generate()` stops after
// the phase step; `run_post_processes()` runs the writers.  Messages the
// CLI prints ("rvegen: generated N shapes", ...) go through
// `pipeline_options::info` / `warn` so embedders stay silent by default.

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen.h"

namespace rvegen {

// Idempotent, thread-safe registration of every built-in type in every
// registry. Cheap to call repeatedly.
inline void register_all_types() {
  static std::once_flag once;
  std::call_once(once, [] {
    register_all_distributions<>();
    register_all_direction_distributions<>();
    register_all_terminations<>();
    register_all_generators<>();
    register_all_inputs<>();
    register_all_collision_pairs<>();   // precise dispatch in generators
    register_all_post_processes<>();
  });
}

struct pipeline_options {
  // Overrides the config's "seed" key when set (default 42 if neither).
  std::optional<std::uint64_t> seed;
  progress_options progress;
  // Run the config's "postprocess" entries (writers with output paths).
  bool run_postprocess = true;
  // Line sinks. `info` gets CLI stdout lines (no trailing newline),
  // `warn` the stderr warnings. Both default to no-ops.
  std::function<void(std::string const&)> info = [](std::string const&) {};
  std::function<void(std::string const&)> warn = [](std::string const&) {};
};

struct pipeline_result {
  using shape_vector = rve_generator_base<double>::shape_vector;

  shape_vector shapes;
  std::array<double, 3> domain_box{1.0, 1.0, 0.0};
  std::uint64_t seed{42};
  std::optional<phase_collection<double>> phases;
  std::vector<std::string> post_processes_run;   // "type" of each executed step
};

// Steps 1-7b: everything up to (and including) phase assignment.
[[nodiscard]] inline pipeline_result
generate(nlohmann::json const& config, pipeline_options const& opts = {}) {
  register_all_types();
  pipeline_result out;

  if (config.contains("domain_box")) {
    auto const& db = config.at("domain_box");
    out.domain_box[0] = db.at(0).get<double>();
    out.domain_box[1] = db.at(1).get<double>();
    out.domain_box[2] = db.size() >= 3 ? db.at(2).get<double>() : 0.0;
  }
  // Warn about unknown top-level keys (typos would otherwise be ignored).
  for (auto const& [key, value] : config.items()) {
    static constexpr std::array<char const*, 10> known{
        "domain_box", "seed", "distributions", "direction_distributions",
        "shapes", "generator", "termination", "postprocess", "phases",
        "phase_assignment"};
    if (key.starts_with('_')) continue;   // "_comment" style annotations
    if (std::find_if(known.begin(), known.end(), [&](char const* k) {
          return key == k;
        }) == known.end()) {
      opts.warn("rvegen: warning - unknown top-level key '" + key +
                "' is ignored");
    }
  }
  // The sampling engine is a 32-bit mt19937: reject seeds it would silently
  // truncate (or wrap, for negative JSON values) instead of mis-reporting.
  if (opts.seed) {
    out.seed = *opts.seed;
  } else if (config.contains("seed")) {
    auto const& sj = config.at("seed");
    if (!sj.is_number_integer() ||
        (!sj.is_number_unsigned() && sj.get<std::int64_t>() < 0)) {
      throw std::runtime_error{
          "'seed' must be an integer in [0, 4294967295] (32-bit engine)"};
    }
    out.seed = sj.get<std::uint64_t>();
  }
  if (out.seed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::runtime_error{
        "'seed' must be an integer in [0, 4294967295] (32-bit engine)"};
  }
  std::mt19937 engine{static_cast<std::uint32_t>(out.seed)};

  // Distributions.
  distribution_map_t<double> dists;
  if (config.contains("distributions")) {
    for (auto const& [name, spec] : config.at("distributions").items()) {
      auto dist = build_from_json(distribution_registry_t<>::instance(), spec,
                                  engine);
      dists.emplace(name, std::shared_ptr<distribution_base<double>>{
                              std::move(dist)});
    }
  }

  // Direction distributions (e.g. bingham) share the name space of the
  // scalar map: each is stored as a direction_distribution_holder that
  // direction-aware inputs (polyline_tube_oriented_input) unwrap.
  if (config.contains("direction_distributions")) {
    for (auto const& [name, spec] :
         config.at("direction_distributions").items()) {
      if (dists.contains(name)) {
        throw std::runtime_error{"duplicate distribution name '" + name +
                                 "' (direction_distributions share the "
                                 "distributions name space)"};
      }
      auto dir = build_from_json(
          direction_distribution_registry_t<>::instance(), spec, engine);
      dists.emplace(
          name, std::make_shared<direction_distribution_holder<double>>(
                    std::shared_ptr<direction_distribution_base<double>>{
                        std::move(dir)}));
    }
  }

  // Inputs. "shapes" is optional: deterministic generators (weave,
  // tow_weave, voronoi_*) build their shapes from the generator params.
  rve_generator_base<double>::input_vector inputs;
  for (auto const& spec : config.value("shapes", nlohmann::json::array())) {
    // A direction distribution in a scalar slot would otherwise only fail
    // at the first new_shape(); catch it here and name the key.
    if (spec.is_object()) {
      for (auto const& [key, value] : spec.items()) {
        if (key == "direction_dist" || !value.is_string() ||
            !key.ends_with("_dist"))
          continue;
        auto const it = dists.find(value.get<std::string>());
        if (it != dists.end() &&
            dynamic_cast<direction_distribution_holder<double>*>(
                it->second.get()) != nullptr) {
          throw std::runtime_error{
              "shape input '" + spec.value("type", std::string{"?"}) +
              "': key '" + key + "' refers to direction distribution '" +
              value.get<std::string>() +
              "', but a scalar distribution is required"};
        }
      }
    }
    inputs.push_back(
        build_from_json(input_registry_t<>::instance(), spec, dists));
  }

  // Generator + termination. Deterministic generators ignore the
  // termination predicate (#73), so the section is optional for them; a
  // placeholder keeps the compute() signature uniform.
  auto generator = build_from_json_checked(
      generator_registry_t<>::instance(), config.at("generator"),
      [&](std::string const& type, std::string const& key) {
        opts.warn("rvegen: warning - unknown key '" + key +
                  "' for generator '" + type + "' is ignored");
      });
  const bool deterministic =
      dynamic_cast<deterministic_generator<double>*>(generator.get()) !=
      nullptr;
  std::unique_ptr<termination_base<double>> termination;
  if (deterministic) {
    auto const gtype = config.at("generator").value("type", std::string{"?"});
    if (config.contains("shapes") && !config.at("shapes").empty())
      opts.warn("rvegen: warning - generator '" + gtype +
                "' is deterministic: \"shapes\" is ignored");
    if (config.contains("termination"))
      opts.warn("rvegen: warning - generator '" + gtype +
                "' is deterministic: \"termination\" is ignored");
    termination = std::make_unique<number_of_inclusions<double>>(std::size_t{1});
  } else {
    termination = build_from_json(termination_registry_t<>::instance(),
                                  config.at("termination"), out.domain_box);
  }

  out.shapes = generator->compute(inputs, *termination, out.domain_box,
                                  opts.progress);
  opts.info("rvegen: generated " + std::to_string(out.shapes.size()) +
            " shape" + (out.shapes.size() == 1 ? "" : "s"));

  // Phases: material blobs + shape -> phase assignment. The collection is
  // handed to every post-process through post_process_base::set_phases.
  if (config.contains("phases")) {
    out.phases.emplace(load_phases_from_json<double>(config.at("phases")));
  }
  if (config.contains("phase_assignment")) {
    if (!out.phases) {
      throw std::runtime_error{
          "\"phase_assignment\" given without a \"phases\" array"};
    }
    const auto assigned = phase_assignment<double>::from_json(
                              config.at("phase_assignment"), *out.phases)
                              .apply(out.shapes, out.domain_box);
    opts.info("rvegen: phase_assignment tagged " + std::to_string(assigned) +
              " shape" + (assigned == 1 ? "" : "s"));
  }
  return out;
}

// Step 8: the config's "postprocess" entries, each owning its sink.
inline void run_post_processes(nlohmann::json const& config,
                               pipeline_result& result,
                               pipeline_options const& opts = {}) {
  register_all_types();
  if (!config.contains("postprocess")) {
    opts.warn("rvegen: warning — no \"postprocess\" entries in config; "
              "nothing was written.");
    return;
  }
  for (auto const& spec : config.at("postprocess")) {
    // phase_assignment's "matrix" names the matrix phase; forward it to
    // the DAMASK writer unless the writer config already picks one.
    auto pp_spec = spec;
    if (config.contains("phase_assignment") &&
        config.at("phase_assignment").contains("matrix") &&
        spec.value("type", std::string{}) == "damask_material_config" &&
        !pp_spec.contains("matrix_phase")) {
      pp_spec["matrix_phase"] = config.at("phase_assignment").at("matrix");
    }
    auto pp = build_from_json(post_process_registry_t<>::instance(), pp_spec);
    if (result.phases) pp->set_phases(&*result.phases);
    pp->run(result.shapes, result.domain_box);
    auto type = spec.at("type").get<std::string>();
    opts.info("rvegen: post-process '" + type + "' done");
    result.post_processes_run.push_back(std::move(type));
  }
}

[[nodiscard]] inline pipeline_result
run_pipeline(nlohmann::json const& config, pipeline_options const& opts = {}) {
  auto result = generate(config, opts);
  if (opts.run_postprocess) run_post_processes(config, result, opts);
  return result;
}

} // namespace rvegen
