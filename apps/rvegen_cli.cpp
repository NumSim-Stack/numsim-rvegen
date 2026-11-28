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

#include <any>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>

#include <nlohmann/json.hpp>

#include "rvegen/pipeline.h"

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

  // ---- 2. Run the shared pipeline (include/rvegen/pipeline.h) ------------
  // Registration, distributions, inputs, generator + termination, phase
  // assignment and post-processing all live in the library so the Python
  // bindings run exactly the same code.
  rvegen::pipeline_options opts;
  opts.info = [](std::string const& line) { std::cout << line << '\n'; };
  opts.warn = [](std::string const& line) { std::cerr << line << '\n'; };

  // Optional `--progress` hook: "[progress] placed=X / N | φ=…" lines on
  // stderr, off by default to keep CLI output stable for scripting.
  bool show_progress = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string{argv[i]} == "--progress") show_progress = true;
  }
  if (show_progress) {
    opts.progress.on_progress = [](rvegen::progress_info info) {
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
    // terminal on a 1000-shape run.
    opts.progress.emit_every = 25;
  }

  (void)rvegen::run_pipeline(config, opts);

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
