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
// direction) -> inputs -> post-processes built and phases parsed (so bad
// writer / phase blocks fail before any shape is placed) -> generator +
// termination -> compute -> phase_assignment applied -> post-processes run.
// `generate()` stops after the phase step; `run_post_processes()` runs the
// writers built by generate().  Messages the
// CLI prints ("rvegen: generated N shapes", ...) go through
// `pipeline_options::info` / `warn` so embedders stay silent by default.

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "json/parameter_visitor_nlohmann.h"
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

// Thrown by generate() when `progress.cancel` reports true after the
// generator returned: the (partial) result is dropped and no writer runs.
struct pipeline_cancelled : std::runtime_error {
  pipeline_cancelled() : std::runtime_error{"rvegen: run cancelled"} {}
};

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
  // Writers built (and schema-validated) by generate() BEFORE the shapes
  // are placed, so a bad "postprocess" block fails before any work or any
  // output file; run_post_processes() executes them.
  std::vector<std::unique_ptr<post_process_base<double>>> pending_post_processes;
  bool post_processes_built = false;
};

// Builds the config's "postprocess" writers into `result` (no output yet).
inline void build_post_processes(nlohmann::json const& config,
                                 pipeline_result& result) {
  result.pending_post_processes.clear();
  result.post_processes_built = true;
  if (!config.contains("postprocess")) return;
  for (auto const& spec : config.at("postprocess"))
    result.pending_post_processes.push_back(
        build_from_json(post_process_registry_t<>::instance(), spec));
}

// Steps 1-7b: everything up to (and including) phase assignment.
[[nodiscard]] inline pipeline_result
generate(nlohmann::json const& config, pipeline_options const& opts = {}) {
  register_all_types();
  pipeline_result out;

  if (config.contains("domain_box")) {
    auto const& db = config.at("domain_box");
    auto& box = out.domain_box;
    if (!db.is_array() || (db.size() != 2 && db.size() != 3)) {
      throw std::invalid_argument(
          "domain_box must be an array of 2 or 3 numbers, got " + db.dump());
    }
    box[2] = 0.0;
    for (std::size_t i = 0; i < db.size(); ++i) {
      if (!db[i].is_number()) {
        throw std::invalid_argument(
            "domain_box entries must be numbers, got " + db.dump());
      }
      box[i] = db[i].get<double>();
      if (!std::isfinite(box[i])) {
        throw std::invalid_argument(
            "domain_box entries must be finite, got " + db.dump());
      }
    }
    if (!(box[0] > 0.0) || !(box[1] > 0.0)) {
      throw std::invalid_argument(
          "domain_box x and y extents must be > 0, got " + db.dump());
    }
    if (box[2] < 0.0) {
      throw std::invalid_argument(
          "domain_box z extent must be >= 0 (0 means 2D), got " + db.dump());
    }
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
  // A config seed goes through the same strict integer reader as the
  // schema keys (rejects negative, fractional and non-numeric values by
  // key). The engine is the 32-bit std::mt19937; a wider seed would be
  // truncated silently, so reject it instead.
  if (opts.seed) {
    out.seed = *opts.seed;
  } else if (config.contains("seed")) {
    parameter_visitor_nlohmann seed_reader{config};
    out.seed = std::any_cast<std::uint64_t>(
        seed_reader.read("seed", typeid(std::uint64_t)));
  }
  if (!std::in_range<std::uint32_t>(out.seed)) {
    throw std::invalid_argument(
        "seed must be <= 4294967295 (the engine is 32-bit), got " +
        std::to_string(out.seed));
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

  // Build every post-process up front: the writers are schema-validated
  // here, so a bad block fails before generation instead of after it.
  // (set_phases happens at run time: `out` may still be moved.)
  if (opts.run_postprocess) {
    build_post_processes(config, out);
  }

  // Phases and phase_assignment are parsed (validated) before generation
  // too; the assignment is applied once the shapes exist.
  if (config.contains("phases")) {
    out.phases.emplace(load_phases_from_json<double>(config.at("phases")));
  }
  std::optional<phase_assignment<double>> assignment;
  if (config.contains("phase_assignment")) {
    if (!out.phases) {
      throw std::runtime_error{
          "\"phase_assignment\" given without a \"phases\" array"};
    }
    assignment.emplace(phase_assignment<double>::from_json(
        config.at("phase_assignment"), *out.phases));
    if (assignment->matrix()) out.phases->set_matrix(*assignment->matrix());
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
  auto* const det_gen =
      dynamic_cast<deterministic_generator<double>*>(generator.get());
  const bool deterministic = det_gen != nullptr;
  // Voronoi generators draw their seed points from the pipeline seed.
  if (det_gen) det_gen->set_pipeline_seed(out.seed);
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
  if (opts.progress.cancel()) throw pipeline_cancelled{};
  opts.info("rvegen: generated " + std::to_string(out.shapes.size()) +
            " shape" + (out.shapes.size() == 1 ? "" : "s"));
  // A generator also returns when it exhausts max_attempts. Detect that
  // by evaluating the termination predicate on the result (random_generator
  // evaluates it with count_all = true, the others with the primaries-only
  // default).
  const bool counts_all =
      config.at("generator").value("type", std::string{}) == "random";
  if (!termination->is_met(out.shapes, counts_all)) {
    opts.warn("rvegen: warning — stopped at max_attempts before reaching "
              "the termination target (" + std::to_string(out.shapes.size()) +
              " shapes placed).");
  }

  // Shape -> phase assignment. The collection is handed to every
  // post-process through post_process_base::set_phases.
  if (assignment) {
    const auto assigned = assignment->apply(out.shapes, out.domain_box);
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
  if (!result.post_processes_built) build_post_processes(config, result);
  auto writers = std::move(result.pending_post_processes);
  result.pending_post_processes.clear();
  std::size_t n = 0;
  for (auto const& spec : config.at("postprocess")) {
    auto& pp = writers.at(n++);
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
