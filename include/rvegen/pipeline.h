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
  // Detail lines for a verbose run: where the seed came from, which
  // parameters of a block took their default, how a relative path was
  // resolved. Default: no-op.
  std::function<void(std::string const&)> debug = [](std::string const&) {};
  // File paths in the config. A string parameter of a block whose key ends
  // in "_path" names a file: "output_path" of a post-process is written,
  // any other one ("mesh_path") is read.
  //   * A relative INPUT path is resolved against `input_dir`. The CLI and
  //     rvegen.run(<file>) pass the directory of the config file, so a
  //     config and the files it refers to can be moved together and run
  //     from anywhere.
  //   * A relative OUTPUT path is resolved against `output_dir`.
  // An empty directory (the default) means the working directory. Absolute
  // paths are never changed.
  std::filesystem::path input_dir;
  std::filesystem::path output_dir;
  // Transition aid for callers that used to read every path relative to
  // the working directory (the CLI, rvegen.run(<file>)): a relative input
  // that does not exist in `input_dir` but does exist relative to the
  // working directory is used from there, with a warning. Off by default,
  // and off whenever the caller named the input directory explicitly: a
  // stale file in the working directory must not replace a missing input.
  bool input_cwd_fallback = false;
  // An unknown top-level config key is an error: "post_process" instead of
  // "postprocess" would otherwise run the whole generation and write
  // nothing. false downgrades it to a warning. Keys starting with '_'
  // ("_comment") are always allowed.
  bool strict_keys = true;
  // A generator also returns when it runs out of attempts. By default a
  // missed termination target is a warning; with strict_targets (or
  // "strict": true in the config) it is an error: generate() throws
  // target_not_reached and no post-process runs.
  bool strict_targets = false;
};

// Thrown by generate() when the termination target was missed and the run
// is strict (pipeline_options::strict_targets, "strict": true).
struct target_not_reached : std::runtime_error {
  explicit target_not_reached(generation_report r)
      : std::runtime_error{r.summary()}, report{std::move(r)} {}
  generation_report report;
};

// The generation report as JSON (Python summary, run_summary writer).
[[nodiscard]] inline nlohmann::json report_to_json(generation_report const& r) {
  return {{"termination", r.termination},
          {"has_target", r.has_target},
          {"target", r.target},
          {"achieved", r.achieved},
          {"attempts", r.attempts},
          {"max_attempts", r.max_attempts},
          {"min_gap", r.min_gap},
          {"min_gap_relative", r.min_gap_relative},
          {"wall_gap", r.wall_gap},
          {"stop_reason", to_string(r.reason)},
          {"target_reached", !r.target_missed()}};
}

struct pipeline_result {
  using shape_vector = rve_generator_base<double>::shape_vector;

  shape_vector shapes;
  domain<double> domain_box{};   // default: the unit square
  std::uint64_t seed{42};
  std::optional<phase_collection<double>> phases;
  std::vector<std::string> post_processes_run;   // "type" of each executed step
  // What the generator achieved against the termination target (#119).
  generation_report report;
  // "seed", "report", "config": handed to every post-process
  // (post_process_base::set_run_info) by run_post_processes().
  nlohmann::json run_info;
  // Writers built (and schema-validated) by generate() BEFORE the shapes
  // are placed, so a bad "postprocess" block fails before any work or any
  // output file; run_post_processes() executes them.
  std::vector<std::unique_ptr<post_process_base<double>>> pending_post_processes;
  bool post_processes_built = false;
};

namespace detail {

inline constexpr std::array<char const*, 11> top_level_keys{
    "domain_box", "seed", "distributions", "direction_distributions",
    "shapes", "generator", "termination", "postprocess", "phases",
    "phase_assignment", "strict"};

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

// "<where> ('<type>'): " - names one block in a message. Never throws: the
// entry may be anything (a number, an object whose "type" is not a string).
[[nodiscard]] inline std::string
block_label(std::string const& where, nlohmann::json const& spec) {
  std::string type = "?";
  if (spec.is_object()) {
    if (auto const it = spec.find("type");
        it != spec.end() && it->is_string())
      type = it->get<std::string>();
  }
  return where + " ('" + type + "'): ";
}

// "postprocess[2] ('svg'): "
[[nodiscard]] inline std::string
post_process_label(nlohmann::json const& spec, std::size_t index) {
  return block_label("postprocess[" + std::to_string(index) + "]", spec);
}

// "rvegen: <label>default used for 'a', 'b'" (the verbose line of a block).
[[nodiscard]] inline std::string
defaults_line(std::string const& label, std::vector<std::string> const& keys) {
  std::string line = "rvegen: " + label + "default used for ";
  char const* sep = "";
  for (auto const& k : keys) {
    line += sep;
    line += "'" + k + "'";
    sep = ", ";
  }
  return line;
}

// Resolves the "*_path" parameters of one block in place (see
// pipeline_options::input_dir / output_dir). `log` = false: no lines.
inline void resolve_block_paths(nlohmann::json& block, std::string const& label,
                                bool is_post_process,
                                pipeline_options const& opts, bool log) {
  if (!block.is_object()) return;
  for (auto& [key, value] : block.items()) {
    if (!value.is_string() || !key.ends_with("_path")) continue;
    const bool output = is_post_process && key == "output_path";
    auto const& dir = output ? opts.output_dir : opts.input_dir;
    if (dir.empty()) continue;
    const auto text = value.template get<std::string>();
    if (text.empty()) continue;   // the block's own check rejects it
    const std::filesystem::path given{text};
    if (given.is_absolute()) continue;
    const auto resolved = dir / given;
    if (!output && opts.input_cwd_fallback) {
      // Before input_dir existed every path was relative to the working
      // directory. Keep such a setup running, but say so.
      std::error_code ec;
      if (!std::filesystem::exists(resolved, ec) &&
          std::filesystem::exists(given, ec)) {
        if (log)
          opts.warn("rvegen: warning - " + label + key + " '" + text +
                    "' was not found in '" + dir.string() +
                    "' (relative input paths are resolved against the "
                    "directory of the config file) but exists relative to "
                    "the working directory; using that file");
        continue;
      }
    }
    value = resolved.string();
    if (log)
      opts.debug("rvegen: " + label + key + " '" + text + "' -> '" +
                 resolved.string() + "'");
  }
}

// The config with its relative file paths resolved. Sections of the wrong
// kind are left alone (they are rejected right after).
[[nodiscard]] inline nlohmann::json
resolve_paths(nlohmann::json const& config, pipeline_options const& opts,
              bool log) {
  nlohmann::json out = config;
  if (!out.is_object()) return out;
  for (char const* section : {"distributions", "direction_distributions"}) {
    if (auto const it = out.find(section); it != out.end() && it->is_object())
      for (auto& [name, spec] : it->items())
        resolve_block_paths(spec,
                            block_label(std::string{section} + "." + name, spec),
                            false, opts, log);
  }
  for (char const* section : {"shapes", "postprocess"}) {
    auto const it = out.find(section);
    if (it == out.end() || !it->is_array()) continue;
    const bool pp = std::string_view{section} == "postprocess";
    std::size_t i = 0;
    for (auto& spec : *it) {
      resolve_block_paths(
          spec,
          block_label(std::string{section} + "[" + std::to_string(i) + "]", spec),
          pp, opts, log);
      ++i;
    }
  }
  for (char const* section : {"generator", "termination"}) {
    if (auto const it = out.find(section); it != out.end())
      resolve_block_paths(*it, block_label(section, *it), false, opts, log);
  }
  return out;
}

// Builds one block and sends its "default used for ..." line to opts.debug.
template <typename Registry, typename... Args>
auto build_block(Registry& registry, nlohmann::json const& spec,
                 std::string const& label, pipeline_options const& opts,
                 Args&... extra_ctor_args) {
  return build_from_json_reported(
      registry, spec, [](std::string const&, std::string const&) {},
      [&](std::string const&, std::vector<std::string> const& keys) {
        opts.debug(defaults_line(label, keys));
      },
      extra_ctor_args...);
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
// The paths in `config` are used as they are; `opts` only receives the
// verbose lines.
inline void build_post_processes(nlohmann::json const& config,
                                 pipeline_result& result,
                                 pipeline_options const& opts = {}) {
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
      result.pending_post_processes.push_back(detail::build_block(
          post_process_registry_t<>::instance(), spec, where, opts));
    });
    ++index;
  }
}

namespace detail {

// Steps 1-7b. With `check_only` it returns after the validation part,
// before compute(): nothing is generated.
[[nodiscard]] inline pipeline_result
generate_impl(nlohmann::json const& given_config, pipeline_options const& opts,
              bool check_only) {
  register_all_types();
  pipeline_result out;

  if (!given_config.is_object()) {
    throw std::invalid_argument{
        std::string{"the config must be a JSON object, got "} +
        given_config.type_name()};
  }
  // Relative file paths -> input_dir / output_dir. Only then is the config
  // copied; everything below reads `config`.
  std::optional<nlohmann::json> resolved;
  if (!opts.input_dir.empty() || !opts.output_dir.empty())
    resolved.emplace(resolve_paths(given_config, opts, true));
  nlohmann::json const& config = resolved ? *resolved : given_config;
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
  bool strict_targets = opts.strict_targets;
  if (auto const it = config.find("strict"); it != config.end()) {
    if (!it->is_boolean()) {
      throw std::invalid_argument{"\"strict\" must be true or false, got " +
                                  it->dump()};
    }
    strict_targets = strict_targets || it->template get<bool>();
  }

  // The one place where the dimension is decided: [Lx, Ly] is 2D,
  // [Lx, Ly, Lz] is 3D ([Lx, Ly, 0] is the older spelling of 2D).
  if (config.contains("domain_box"))
    out.domain_box = domain<double>::from_json(config.at("domain_box"));
  // A config seed goes through the same strict integer reader as the
  // schema keys (rejects negative, fractional and non-numeric values by
  // key). The engine is the 32-bit std::mt19937; a wider seed would be
  // truncated silently, so reject it instead.
  char const* seed_source = "default";
  if (opts.seed) {
    out.seed = *opts.seed;
    seed_source = "given by the caller";
  } else if (config.contains("seed")) {
    parameter_visitor_nlohmann seed_reader{config};
    out.seed = std::any_cast<std::uint64_t>(
        seed_reader.read("seed", typeid(std::uint64_t)));
    seed_source = "from the config";
  }
  if (!std::in_range<std::uint32_t>(out.seed)) {
    throw std::invalid_argument(
        "seed must be <= 4294967295 (the engine is 32-bit), got " +
        std::to_string(out.seed));
  }
  std::mt19937 engine{static_cast<std::uint32_t>(out.seed)};
  opts.debug("rvegen: seed " + std::to_string(out.seed) + " (" + seed_source +
             ")");

  // Distributions.
  distribution_map_t<double> dists;
  if (config.contains("distributions")) {
    for (auto const& [name, spec] : config.at("distributions").items()) {
      auto dist = build_block(distribution_registry_t<>::instance(), spec,
                              block_label("distributions." + name, spec), opts,
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
      auto dir = build_block(
          direction_distribution_registry_t<>::instance(), spec,
          block_label("direction_distributions." + name, spec), opts, engine);
      dists.emplace(
          name, std::make_shared<direction_distribution_holder<double>>(
                    std::shared_ptr<direction_distribution_base<double>>{
                        std::move(dir)}));
    }
  }

  // Inputs. "shapes" is optional: deterministic generators (weave,
  // tow_weave, voronoi_*) build their shapes from the generator params.
  rve_generator_base<double>::input_vector inputs;
  std::size_t shape_index = 0;
  std::string dimension_mismatch;   // first "shapes" entry of the wrong dimension
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
    inputs.push_back(build_block(
        input_registry_t<>::instance(), spec,
        block_label("shapes[" + std::to_string(shape_index) + "]", spec), opts,
        dists));
    // A planar shape in a 3D box (or a solid in a 2D one) has no meaning:
    // its area and the box volume do not add up to a fraction. Reported
    // below, once it is known that the generator uses "shapes" at all.
    if (int const d = inputs.back()->dimension();
        d != out.domain_box.dimension() && dimension_mismatch.empty()) {
      dimension_mismatch =
          block_label("shapes[" + std::to_string(shape_index) + "]", spec) +
          "produces " + std::to_string(d) + "D shapes, but domain_box " +
          nlohmann::json(out.domain_box.extents()).dump() + " is " +
          std::to_string(out.domain_box.dimension()) +
          "D (2D: \"domain_box\": [Lx, Ly]; 3D: [Lx, Ly, Lz])";
    }
    ++shape_index;
  }

  // Build every post-process up front: the writers are schema-validated
  // here and their output directories checked, so a bad block fails before
  // generation instead of after it. (set_phases happens at run time: `out`
  // may still be moved.)
  if (opts.run_postprocess || check_only) {
    build_post_processes(config, out, opts);
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
  auto generator = build_from_json_reported(
      generator_registry_t<>::instance(), config.at("generator"),
      [&](std::string const& type, std::string const& key) {
        opts.warn("rvegen: warning - unknown key '" + key +
                  "' for generator '" + type + "' is ignored");
      },
      [&](std::string const&, std::vector<std::string> const& keys) {
        opts.debug(defaults_line(
            block_label("generator", config.at("generator")), keys));
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
    if (!dimension_mismatch.empty())
      throw std::invalid_argument{dimension_mismatch};
    if (!config.contains("termination")) {
      throw std::invalid_argument{
          "the config has no \"termination\" section (generator '" +
          config.at("generator").value("type", std::string{"?"}) +
          "' needs one)"};
    }
    termination = build_block(
        termination_registry_t<>::instance(), config.at("termination"),
        block_label("termination", config.at("termination")), opts,
        out.domain_box);
  }
  if (check_only) return out;

  out.shapes = generator->compute(inputs, *termination, out.domain_box,
                                  opts.progress);
  if (opts.progress.cancel()) throw pipeline_cancelled{};
  opts.info("rvegen: generated " + std::to_string(out.shapes.size()) +
            " shape" + (out.shapes.size() == 1 ? "" : "s"));
  // A generator also returns when it exhausts max_attempts: the report
  // says what was reached. A generator that records none (a custom one)
  // is measured here with the termination predicate itself.
  if (auto const& recorded = generator->last_report()) {
    out.report = *recorded;
  } else if (!deterministic) {
    out.report.termination = termination->kind();
    out.report.has_target = termination->has_target();
    out.report.target = termination->target_value();
    out.report.achieved = termination->achieved_value(out.shapes, false);
    out.report.reason = (*termination)(out.shapes) ? stop_reason::reached
                                                   : stop_reason::max_attempts;
  }
  if (auto const line = out.report.summary(); !line.empty()) {
    if (!out.report.target_missed()) {
      opts.info("rvegen: " + line);
    } else if (strict_targets) {
      throw target_not_reached{out.report};
    } else {
      opts.warn("rvegen: warning - " + line);
    }
  }

  // Shape -> phase assignment. The collection is handed to every
  // post-process through post_process_base::set_phases.
  if (assignment) {
    const auto assigned = assignment->apply(out.shapes);
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
  // Built by generate() as a rule. Otherwise the paths have not been
  // resolved yet either (no lines: generate() has reported them).
  if (!result.post_processes_built) {
    if (opts.input_dir.empty() && opts.output_dir.empty())
      build_post_processes(config, result, opts);
    else
      build_post_processes(detail::resolve_paths(config, opts, false), result,
                           opts);
  }
  auto writers = std::move(result.pending_post_processes);
  result.pending_post_processes.clear();
  result.run_info = {{"seed", result.seed},
                     {"report", report_to_json(result.report)},
                     {"config", config}};
  std::size_t n = 0;
  for (auto const& spec : config.at("postprocess")) {
    // A cancel request (Ctrl-C, a log callback that raised) also stops
    // between writers: no further file is started.
    if (opts.progress.cancel()) throw pipeline_cancelled{};
    auto& pp = writers.at(n++);
    if (result.phases) pp->set_phases(&*result.phases);
    pp->set_run_info(&result.run_info);
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
