// rvegen — JSON-driven RVE (Representative Volume Element) generator.
//
// Usage:
//   rvegen <config.json>
//
// The config file specifies (by registered name) which distributions, shape
// inputs, generator strategy, termination predicate, and one or more
// post-processing steps (writers, voxelizers, ...) to apply. Each post-
// process owns its own output destination via the output_path field in its
// schema, so a single run can fan out to any combination of formats —
// .geo for FEM, .vtk / .vox for FFT, .svg / .html for visualization.

#include <array>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"

namespace {

constexpr int exit_usage  = 64;  // EX_USAGE
constexpr int exit_input  = 66;  // EX_NOINPUT
constexpr int exit_config = 78;  // EX_CONFIG

void print_usage(char const* argv0) {
  std::cerr
      << "usage: " << argv0 << " <config.json> [--progress]\n\n"
      << "  config.json   JSON file describing the RVE generation pipeline.\n"
      << "                Each entry under \"postprocess\" carries its own\n"
      << "                output_path.\n\n"
      << "  --progress    Emit live progress lines to stderr every 50\n"
      << "                accepted shapes — \"[progress] placed=X / N | φ=…\".\n"
      << "                Off by default so scripted output stays stable.\n";
}

int run(int argc, char** argv) {
  if (argc < 2) {
    print_usage(argv[0]);
    return exit_usage;
  }

  // ---- 1. Load JSON config -----------------------------------------------
  std::ifstream config_file{argv[1]};
  if (!config_file) {
    std::cerr << "rvegen: cannot open config file: " << argv[1] << '\n';
    return exit_input;
  }

  nlohmann::json config;
  try {
    config_file >> config;
  } catch (std::exception const& e) {
    std::cerr << "rvegen: parse error in " << argv[1] << ": " << e.what() << '\n';
    return exit_config;
  }

  // ---- 2. Register every known type in every registry --------------------
  rvegen::register_all_distributions<>();
  rvegen::register_all_terminations<>();
  rvegen::register_all_generators<>();
  rvegen::register_all_inputs<>();
  rvegen::register_all_collision_pairs<>();   // precise dispatch in generators
  rvegen::register_all_post_processes<>();

  // ---- 3. Domain box + RNG seed (top-level config keys) ------------------
  std::array<double, 3> domain_box{1.0, 1.0, 0.0};
  if (config.contains("domain_box")) {
    auto const& db = config.at("domain_box");
    domain_box[0] = db.at(0).get<double>();
    domain_box[1] = db.at(1).get<double>();
    domain_box[2] = db.size() >= 3 ? db.at(2).get<double>() : 0.0;
  }
  const auto seed = config.value<std::uint64_t>("seed", 42);
  std::mt19937 engine{seed};

  // ---- 4. Build distributions -------------------------------------------
  rvegen::distribution_map_t<double> dists;
  if (config.contains("distributions")) {
    for (auto const& [name, spec] : config.at("distributions").items()) {
      auto dist = rvegen::build_from_json(
          rvegen::distribution_registry_t<>::instance(), spec, engine);
      dists.emplace(name,
                    std::shared_ptr<rvegen::distribution_base<double>>{
                        std::move(dist)});
    }
  }

  // ---- 5. Build inputs (cross-reference distributions by name) ----------
  rvegen::rve_generator_base<double>::input_vector inputs;
  for (auto const& spec : config.at("shapes")) {
    auto input = rvegen::build_from_json(
        rvegen::input_registry_t<>::instance(), spec, dists);
    inputs.push_back(std::move(input));
  }

  // ---- 6. Generator + termination ---------------------------------------
  auto generator = rvegen::build_from_json(
      rvegen::generator_registry_t<>::instance(), config.at("generator"));
  auto termination = rvegen::build_from_json(
      rvegen::termination_registry_t<>::instance(),
      config.at("termination"), domain_box);

  // ---- 7. Run the pipeline ----------------------------------------------
  // Install a progress hook that prints "[N/T] φ = X.XX" lines to stderr
  // every emit_every accepted shapes. The hook is opt-in via the
  // `--progress` flag (disabled by default to keep CLI output stable for
  // scripting). Quiet runs go through the legacy single "generated N
  // shapes" line at the end.
  rvegen::progress_options progress_opts;
  bool show_progress = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string{argv[i]} == "--progress") show_progress = true;
  }
  if (show_progress) {
    progress_opts.on_progress = [](rvegen::progress_info info) {
      std::cerr << "[progress] placed=" << info.shapes_placed;
      if (info.shapes_target > 0) {
        std::cerr << " / " << info.shapes_target;
      }
      if (info.volume_fraction > 0.0) {
        std::cerr << " | φ=" << info.volume_fraction;
      }
      std::cerr << '\n';
    };
    // 25 is a compromise: fires often enough to show progress on the
    // small example RVEs (~10–50 shapes) while not spamming the
    // terminal on a 1000-shape run. The user can re-tune via library
    // API if they're embedding rvegen.
    progress_opts.emit_every = 25;
  }
  auto shapes = generator->compute(inputs, *termination, domain_box,
                                   progress_opts);
  std::cout << "rvegen: generated " << shapes.size() << " shape"
            << (shapes.size() == 1 ? "" : "s") << '\n';

  // ---- 8. Post-processing steps (each owns its sink) --------------------
  if (!config.contains("postprocess")) {
    std::cerr << "rvegen: warning — no \"postprocess\" entries in config; "
                 "nothing was written.\n";
  } else {
    for (auto const& spec : config.at("postprocess")) {
      auto pp = rvegen::build_from_json(
          rvegen::post_process_registry_t<>::instance(), spec);
      pp->run(shapes, domain_box);
      std::cout << "rvegen: post-process '"
                << spec.at("type").get<std::string>() << "' done\n";
    }
  }

  return EXIT_SUCCESS;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (std::exception const& e) {
    std::cerr << "rvegen: error: " << e.what() << '\n';
    return exit_config;
  }
}
