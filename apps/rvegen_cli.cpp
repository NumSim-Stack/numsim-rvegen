// rvegen — JSON-driven RVE (Representative Volume Element) generator.
//
// Usage:
//   rvegen <config.json> [--progress]
//   rvegen --list-types
//
// The config file specifies (by registered name) which distributions, shape
// inputs, generator strategy, termination predicate, and one or more
// post-processing steps (writers, voxelizers, ...) to apply. Each post-
// process owns its own output destination via the output_path field in its
// schema, so a single run can fan out to any combination of formats —
// .geo for FEM, .vtk / .vox for FFT, .svg / .html for visualization.

#include <algorithm>
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
      << "usage: " << argv0 << " [options] <config.json>\n"
      << "       " << argv0 << " --check <config.json>\n"
      << "       " << argv0 << " --list-types\n\n"
      << "  config.json   JSON file describing the RVE generation pipeline.\n"
      << "                Each entry under \"postprocess\" carries its own\n"
      << "                output_path.\n\n"
      << "  --check       Validate the config and exit without generating:\n"
      << "                every block is built, phases are parsed and the\n"
      << "                output directories are checked, exactly as a run\n"
      << "                does before it places the first shape. No file is\n"
      << "                written. Exit status 0 means the config is valid.\n\n"
      << "  --allow-unknown-keys\n"
      << "                An unknown top-level key is an error (it is usually\n"
      << "                a misspelt section, which would be skipped); with\n"
      << "                this option it is only a warning. Keys starting\n"
      << "                with '_' are always allowed.\n\n"
      << "  --progress    Emit live progress lines to stderr every 50\n"
      << "                accepted shapes — \"[progress] placed=X / N | φ=…\".\n"
      << "                Off by default so scripted output stays stable.\n\n"
      << "  --list-types  Print every registered JSON \"type\" per registry as\n"
      << "                Markdown (README.md embeds this) and exit.\n\n"
      << "  --            End of options: the next argument is the config\n"
      << "                file even if its name starts with \"--\".\n";
}

template <typename Registry>
void print_registry(char const* title, Registry const& reg) {
  auto names = reg.registered_types();
  std::sort(names.begin(), names.end());
  std::cout << "- **" << title << "**:";
  char const* sep = " ";
  for (auto const& n : names) {
    std::cout << sep << '`' << n << '`';
    sep = ", ";
  }
  std::cout << '\n';
}

int list_types() {
  rvegen::register_all_types();
  print_registry("distributions (`distributions`)",
                 rvegen::distribution_registry_t<>::instance());
  print_registry("direction distributions (`direction_distributions`)",
                 rvegen::direction_distribution_registry_t<>::instance());
  print_registry("shapes (`shapes`)", rvegen::input_registry_t<>::instance());
  print_registry("generators (`generator`)",
                 rvegen::generator_registry_t<>::instance());
  print_registry("terminations (`termination`)",
                 rvegen::termination_registry_t<>::instance());
  print_registry("post-processes (`postprocess`)",
                 rvegen::post_process_registry_t<>::instance());
  return EXIT_SUCCESS;
}

int run(int argc, char** argv) {
  // Options may come before or after the config file.
  bool show_progress = false;
  bool check_only = false;
  bool allow_unknown_keys = false;
  char const* config_path = nullptr;
  bool options_ended = false;   // after "--" everything is a file name
  for (int i = 1; i < argc; ++i) {
    const std::string arg{argv[i]};
    if (!options_ended && arg == "--") {
      options_ended = true;
      continue;
    }
    if (options_ended) {
      if (config_path != nullptr) {
        std::cerr << "rvegen: more than one config file given ('"
                  << config_path << "' and '" << arg << "')\n\n";
        print_usage(argv[0]);
        return exit_usage;
      }
      config_path = argv[i];
      continue;
    }
    if (arg == "--list-types") return list_types();
    if (arg == "--progress") {
      show_progress = true;
    } else if (arg == "--check") {
      check_only = true;
    } else if (arg == "--allow-unknown-keys") {
      allow_unknown_keys = true;
    } else if (arg.starts_with("--")) {
      std::cerr << "rvegen: unknown option '" << arg << "'\n\n";
      print_usage(argv[0]);
      return exit_usage;
    } else if (config_path != nullptr) {
      std::cerr << "rvegen: more than one config file given ('" << config_path
                << "' and '" << arg << "')\n\n";
      print_usage(argv[0]);
      return exit_usage;
    } else {
      config_path = argv[i];
    }
  }
  if (config_path == nullptr) {
    print_usage(argv[0]);
    return exit_usage;
  }

  // ---- 1. Load JSON config -----------------------------------------------
  std::ifstream config_file{config_path};
  if (!config_file) {
    std::cerr << "rvegen: cannot open config file: " << config_path << '\n';
    return exit_input;
  }

  nlohmann::json config;
  try {
    config_file >> config;
  } catch (std::exception const& e) {
    std::cerr << "rvegen: parse error in " << config_path << ": " << e.what() << '\n';
    return exit_config;
  }

  // ---- 2. Run the shared pipeline (include/rvegen/pipeline.h) ------------
  // Registration, distributions, inputs, generator + termination, phase
  // assignment and post-processing all live in the library so the Python
  // bindings run exactly the same code.
  rvegen::pipeline_options opts;
  opts.info = [](std::string const& line) { std::cout << line << '\n'; };
  opts.warn = [](std::string const& line) { std::cerr << line << '\n'; };

  opts.strict_keys = !allow_unknown_keys;

  if (check_only) {
    rvegen::check_config(config, opts);
    std::cout << "rvegen: config OK: " << config_path << '\n';
    return EXIT_SUCCESS;
  }

  // Optional `--progress` hook: "[progress] placed=X / N | φ=…" lines on
  // stderr, off by default to keep CLI output stable for scripting.
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
