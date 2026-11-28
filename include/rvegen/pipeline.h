#pragma once

// rvegen pipeline — the JSON-config -> shapes -> post-processes flow shared
// by the CLI (apps/rvegen_cli.cpp) and the Python bindings.
//
//   auto cfg = nlohmann::json::parse(...);
//   rvegen::pipeline_options opts;               // seed override, hooks
//   auto result = rvegen::run_pipeline(cfg, opts);
//   // result.shapes, result.domain_box, result.seed, ...
//
// Steps: register types -> top-level keys / domain box / seed ->
// distributions (scalar + direction) -> inputs -> post-processes built,
// output directories checked and phases parsed -> generator + termination
// -> compute -> phase_assignment applied -> post-processes run.
// Everything before `compute` is validation: a typo in any block fails
// before a shape is placed or a file is written (#120). `check_config()`
// runs only that part. `generate()` stops after the phase step;
// `run_post_processes()` runs the writers built by generate().  Messages the
// CLI prints ("rvegen: generated N shapes", ...) go through
// `pipeline_options::info` / `warn` so embedders stay silent by default.

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <typeinfo>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

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
  // An unknown top-level config key is an error: "post_process" instead of
  // "postprocess" would otherwise run the whole generation and write
  // nothing. false downgrades it to a warning. Keys starting with '_'
  // ("_comment") are always allowed.
  bool strict_keys = true;
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

namespace detail {

inline constexpr std::array<char const*, 10> top_level_keys{
    "domain_box", "seed", "distributions", "direction_distributions",
    "shapes", "generator", "termination", "postprocess", "phases",
    "phase_assignment"};

// Levenshtein distance (two rows; the keys are a few characters long).
[[nodiscard]] inline std::size_t edit_distance(std::string_view a,
                                               std::string_view b) {
  std::vector<std::size_t> prev(b.size() + 1), cur(b.size() + 1);
  for (std::size_t j = 0; j <= b.size(); ++j) prev[j] = j;
  for (std::size_t i = 1; i <= a.size(); ++i) {
    cur[0] = i;
    for (std::size_t j = 1; j <= b.size(); ++j) {
      cur[j] = std::min({prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] == b[j - 1] ? 0u : 1u)});
    }
    std::swap(prev, cur);
  }
  return prev[b.size()];
}

// The valid top-level key a typo most likely meant: the closest one within
// about a third of its length ("post_process" -> "postprocess", "shape" ->
// "shapes"); empty when nothing is close ("bogus").
[[nodiscard]] inline std::string nearest_top_level_key(std::string_view key) {
  std::string best;
  std::size_t best_d = std::numeric_limits<std::size_t>::max();
  for (char const* k : top_level_keys) {
    const std::size_t d = edit_distance(key, k);
    if (d < best_d) { best_d = d; best = k; }
  }
  const std::size_t limit =
      std::max<std::size_t>(2, std::max(key.size(), best.size()) / 3);
  return best_d <= limit ? best : std::string{};
}

// A section of the wrong JSON kind would otherwise surface as a library
// exception that does not name the section.
inline void require_section_kind(nlohmann::json const& config, char const* key,
                                 bool array) {
  if (!config.contains(key)) return;
  auto const& v = config.at(key);
  if (array ? v.is_array() : v.is_object()) return;
  throw std::invalid_argument{std::string{"\""} + key + "\" must be " +
                              (array ? "an array" : "an object") + ", got " +
                              v.type_name()};
}

// "postprocess[2] ('svg'): " - names one writer block in a message. Never
// throws: the entry may be anything (a number, an object whose "type" is
// not a string).
[[nodiscard]] inline std::string
post_process_label(nlohmann::json const& spec, std::size_t index) {
  std::string type = "?";
  if (spec.is_object()) {
    if (auto const it = spec.find("type");
        it != spec.end() && it->is_string())
      type = it->get<std::string>();
  }
  return "postprocess[" + std::to_string(index) + "] ('" + type + "'): ";
}

// Each writer's "output_path" must be writable before any shape is placed:
// it names a file, its directory exists and accepts new files, and the path
// itself is neither a directory nor a link to nowhere. Nothing is created
// or truncated here. (An empty or missing path is the writer's own
// validate() to reject: a custom post-process may not write a file.)
inline void check_output_paths(nlohmann::json const& config) {
  if (!config.contains("postprocess")) return;
  std::size_t index = 0;
  for (auto const& spec : config.at("postprocess")) {
    const std::size_t i = index++;
    if (!spec.is_object() || !spec.contains("output_path") ||
        !spec.at("output_path").is_string())
      continue;
    const auto text = spec.at("output_path").get<std::string>();
    if (text.empty()) continue;
    const std::string where =
        post_process_label(spec, i) + "output_path '" + text + "'";
    const std::filesystem::path path{text};
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
      throw std::runtime_error{where + " is a directory, not a file"};
    }
    if (!path.has_filename()) {   // "out/": a separator at the end
      throw std::runtime_error{where + " names a directory, not a file"};
    }
    // exists() follows links: a link whose target is missing would pass as
    // "new file" although opening it fails.
    if (std::filesystem::is_symlink(path, ec) &&
        !std::filesystem::exists(path, ec)) {
      throw std::runtime_error{
          where + " is a symbolic link whose target does not exist"};
    }
    auto dir = path.parent_path();
    if (dir.empty()) dir = ".";
    if (!std::filesystem::exists(dir, ec)) {
      throw std::runtime_error{where + ": directory '" + dir.string() +
                               "' does not exist"};
    }
    if (!std::filesystem::is_directory(dir, ec)) {
      throw std::runtime_error{where + ": '" + dir.string() +
                               "' is not a directory"};
    }
#if defined(__unix__) || defined(__APPLE__)
    const bool file_exists = std::filesystem::exists(path, ec);
    // A new file needs write + search permission on the directory, an
    // existing one write permission on the file itself.
    if (file_exists ? ::access(path.c_str(), W_OK) != 0
                    : ::access(dir.c_str(), W_OK | X_OK) != 0) {
      throw std::runtime_error{
          where + (file_exists ? " is not writable"
                               : ": directory '" + dir.string() +
                                     "' is not writable")};
    }
#endif
  }
}

// Runs `f`; an exception leaving it is rethrown with `where` in front of
// its message (with several writers "missing required parameter" alone does
// not say which block is wrong). The common categories keep their type, so
// callers and the Python bindings (ValueError / IndexError / RuntimeError)
// still see what kind of mistake it was.
template <typename F>
void with_context(std::string const& where, F&& f) {
  try {
    f();
  } catch (std::out_of_range const& e) {
    throw std::out_of_range{where + e.what()};
  } catch (std::domain_error const& e) {
    throw std::domain_error{where + e.what()};
  } catch (std::length_error const& e) {
    throw std::length_error{where + e.what()};
  } catch (std::invalid_argument const& e) {
    throw std::invalid_argument{where + e.what()};
  } catch (std::overflow_error const& e) {
    throw std::overflow_error{where + e.what()};
  } catch (std::exception const& e) {
    throw std::runtime_error{where + e.what()};
  }
}

} // namespace detail

// Builds the config's "postprocess" writers into `result` (no output yet).
inline void build_post_processes(nlohmann::json const& config,
                                 pipeline_result& result) {
  result.pending_post_processes.clear();
  result.post_processes_built = true;
  if (!config.contains("postprocess")) return;
  std::size_t index = 0;
  for (auto const& spec : config.at("postprocess")) {
    const auto where = detail::post_process_label(spec, index);
    if (!spec.is_object()) {
      throw std::invalid_argument{where + "each entry must be an object, got " +
                                  spec.type_name()};
    }
    if (!spec.contains("type") || !spec.at("type").is_string()) {
      throw std::invalid_argument{
          where + "needs a \"type\" string naming the post-process"};
    }
    detail::with_context(where, [&] {
      result.pending_post_processes.push_back(
          build_from_json(post_process_registry_t<>::instance(), spec));
    });
    ++index;
  }
}

namespace detail {

// Steps 1-7b. With `check_only` it returns after the validation part,
// before compute(): nothing is generated.
[[nodiscard]] inline pipeline_result
generate_impl(nlohmann::json const& config, pipeline_options const& opts,
              bool check_only) {
  register_all_types();
  pipeline_result out;

  if (!config.is_object()) {
    throw std::invalid_argument{
        std::string{"the config must be a JSON object, got "} +
        config.type_name()};
  }
  // Unknown top-level keys first: a misspelt section name is the mistake
  // that silently drops a whole part of the run.
  for (auto const& [key, value] : config.items()) {
    if (key.starts_with('_')) continue;   // "_comment" style annotations
    if (std::find_if(top_level_keys.begin(), top_level_keys.end(),
                     [&](char const* k) { return key == k; }) !=
        top_level_keys.end())
      continue;
    const auto hint = nearest_top_level_key(key);
    if (opts.strict_keys) {
      throw std::invalid_argument{
          "unknown top-level key '" + key + "'" +
          (hint.empty() ? std::string{} : " (did you mean '" + hint + "'?)") +
          "; keys starting with '_' are ignored"};
    }
    opts.warn("rvegen: warning - unknown top-level key '" + key +
              "' is ignored" +
              (hint.empty() ? std::string{} : " (did you mean '" + hint + "'?)"));
  }
  require_section_kind(config, "distributions", false);
  require_section_kind(config, "direction_distributions", false);
  require_section_kind(config, "shapes", true);
  require_section_kind(config, "postprocess", true);
  require_section_kind(config, "generator", false);
  require_section_kind(config, "termination", false);
  if (!config.contains("generator")) {
    throw std::invalid_argument{"the config has no \"generator\" section"};
  }

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
  // here and their output directories checked, so a bad block fails before
  // generation instead of after it. (set_phases happens at run time: `out`
  // may still be moved.)
  if (opts.run_postprocess || check_only) {
    build_post_processes(config, out);
    check_output_paths(config);
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

  // Each writer's own config-only checks (post_process_base::validate): a
  // missing output path, a voxel grid that is too large, a 3D box without
  // nz. They see the phases as at run time; the pointer is detached again
  // because `out` is still going to be moved.
  for (std::size_t i = 0; i < out.pending_post_processes.size(); ++i) {
    auto& pp = *out.pending_post_processes[i];
    detail::with_context(
        post_process_label(config.at("postprocess").at(i), i), [&] {
          if (out.phases) pp.set_phases(&*out.phases);
          struct detach {
            post_process_base<double>& p;
            ~detach() { p.set_phases(nullptr); }
          } guard{pp};
          pp.validate(out.domain_box);
        });
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
    if (!config.contains("termination")) {
      throw std::invalid_argument{
          "the config has no \"termination\" section (generator '" +
          config.at("generator").value("type", std::string{"?"}) +
          "' needs one)"};
    }
    termination = build_from_json(termination_registry_t<>::instance(),
                                  config.at("termination"), out.domain_box);
  }
  if (check_only) return out;

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

} // namespace detail

// Steps 1-7b: everything up to (and including) phase assignment.
[[nodiscard]] inline pipeline_result
generate(nlohmann::json const& config, pipeline_options const& opts = {}) {
  return detail::generate_impl(config, opts, false);
}

// Validation only (#120): builds every distribution, input, writer, the
// generator and the termination, parses phases and the phase assignment and
// checks the output directories, exactly as a run does, then stops. Places
// no shape and writes no file. Throws on the first problem; warnings go to
// `opts.warn`. Writers are validated whatever `opts.run_postprocess` says.
inline void check_config(nlohmann::json const& config,
                         pipeline_options const& opts = {}) {
  (void)detail::generate_impl(config, opts, true);
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
