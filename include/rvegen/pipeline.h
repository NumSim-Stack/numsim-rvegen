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

#include <array>
#include <cstdint>
#include <functional>
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
  out.seed = opts.seed ? *opts.seed
                       : config.value<std::uint64_t>("seed", 42);
  std::mt19937 engine{out.seed};

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
    inputs.push_back(
        build_from_json(input_registry_t<>::instance(), spec, dists));
  }

  // Generator + termination. Deterministic generators ignore the
  // termination predicate (#73), so the section is optional for them; a
  // placeholder keeps the compute() signature uniform.
  auto generator = build_from_json(generator_registry_t<>::instance(),
                                   config.at("generator"));
  const bool deterministic =
      dynamic_cast<deterministic_generator<double>*>(generator.get()) !=
      nullptr;
  std::unique_ptr<termination_base<double>> termination;
  if (deterministic && !config.contains("termination")) {
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
                              .apply(out.shapes);
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
    auto pp = build_from_json(post_process_registry_t<>::instance(), spec);
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
