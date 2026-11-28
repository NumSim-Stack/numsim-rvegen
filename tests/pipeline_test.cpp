// rvegen::run_pipeline / generate (the library entry shared by the CLI and
// the Python bindings) and schema_to_json (#4).

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

#include "rvegen/pipeline.h"
#include "rvegen/schema/schema_to_json.h"

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

using json = nlohmann::json;

json circles_config() {
  return json::parse(R"({
    "domain_box": [1.0, 1.0, 0.0],
    "seed": 5,
    "distributions": {
      "x": {"type": "uniform_real", "a": 0.1, "b": 0.9},
      "r": {"type": "uniform_real", "a": 0.03, "b": 0.05}
    },
    "shapes": [{"type": "circle_input", "pos_x_dist": "x",
                "pos_y_dist": "x", "radius_dist": "r"}],
    "generator": {"type": "only_inside", "max_attempts": 100000},
    "termination": {"type": "number_of_inclusions", "target": 6},
    "phases": [{"name": "fibre"}],
    "phase_assignment": {"all_inclusions": "fibre"}
  })");
}

void test_generate() {
  std::vector<std::string> lines;
  rvegen::pipeline_options opts;
  opts.info = [&](std::string const& l) { lines.push_back(l); };
  auto r = rvegen::run_pipeline(circles_config(), opts);
  REQUIRE(r.shapes.size() == 6);
  REQUIRE(r.seed == 5);
  REQUIRE(r.phases && r.phases->size() == 1);
  REQUIRE(r.shapes[0]->phase_name() == "fibre");
  REQUIRE(lines.size() >= 2);
  REQUIRE(lines[0] == "rvegen: generated 6 shapes");
  REQUIRE(lines[1] == "rvegen: phase_assignment tagged 6 shapes");

  // Seed override changes the layout; same seed reproduces it.
  rvegen::pipeline_options o2;
  o2.seed = 9;
  auto a = rvegen::generate(circles_config(), o2);
  auto b = rvegen::generate(circles_config(), o2);
  REQUIRE(a.seed == 9);
  REQUIRE(a.shapes[0]->get_middle_point() == b.shapes[0]->get_middle_point());
  REQUIRE(a.shapes[0]->get_middle_point() != r.shapes[0]->get_middle_point());
}

void test_postprocess_toggle() {
  auto cfg = circles_config();
  cfg["postprocess"] = json::array({json{{"type", "svg"},
                                         {"canvas_width", 200},
                                         {"output_path", "pipeline_test.svg"}}});
  rvegen::pipeline_options off;
  off.run_postprocess = false;
  REQUIRE(rvegen::run_pipeline(cfg, off).post_processes_run.empty());
  auto on = rvegen::run_pipeline(cfg);
  REQUIRE(on.post_processes_run.size() == 1);
  REQUIRE(on.post_processes_run[0] == "svg");
}

void test_seed_and_warnings() {
  // Seeds the 32-bit engine would truncate, or that are not non-negative
  // integers, are rejected with the key named (std::invalid_argument, as
  // for every other mistyped config value).
  auto rejected = [](nlohmann::json const& cfg, rvegen::pipeline_options const& o,
                     char const* needle) {
    try { (void)rvegen::generate(cfg, o); }
    catch (std::invalid_argument const& e) {
      return std::string{e.what()}.find(needle) != std::string::npos;
    }
    return false;
  };
  auto cfg = circles_config();
  cfg["seed"] = 4294967296ULL;            // would silently truncate in mt19937
  REQUIRE(rejected(cfg, {}, "seed must be <= 4294967295"));
  cfg["seed"] = -3;
  REQUIRE(rejected(cfg, {}, "'seed'"));
  cfg["seed"] = 1.5;
  REQUIRE(rejected(cfg, {}, "'seed'"));
  rvegen::pipeline_options big;
  big.seed = 1ULL << 40;                  // override path is checked too
  REQUIRE(rejected(circles_config(), big, "seed must be <= 4294967295"));
  // domain_box is validated before anything is built.
  for (auto const& bad : {json::array({0, 1}), json::array({1, 1, -1}),
                          json::array({1}), json::array({1, "a"})}) {
    auto c = circles_config();
    c["domain_box"] = bad;
    REQUIRE(rejected(c, {}, "domain_box"));
  }

  // Deterministic generator + shapes/termination + unknown key -> warnings.
  auto det = circles_config();
  det["generator"] = json{{"type", "voronoi_2d"}, {"n_seeds", 3}, {"seed", 1}};
  std::vector<std::string> warns;
  rvegen::pipeline_options o;
  o.warn = [&](std::string const& w) { warns.push_back(w); };
  (void)rvegen::generate(det, o);
  auto has = [&](char const* t) {
    for (auto const& w : warns) if (w.find(t) != std::string::npos) return true;
    return false;
  };
  REQUIRE(has("\"shapes\" is ignored"));
  REQUIRE(has("\"termination\" is ignored"));
  // An unknown top-level key is an error by default (#120), a warning with
  // strict_keys = false; both carry a hint when a valid key is close.
  REQUIRE(!has("unknown top-level key"));
  det["bogus"] = 1;
  REQUIRE(rejected(det, o, "unknown top-level key 'bogus'; keys starting"));
  o.strict_keys = false;
  warns.clear();
  (void)rvegen::generate(det, o);
  REQUIRE(has("unknown top-level key 'bogus' is ignored"));
  REQUIRE(!has("did you mean"));
  det.erase("bogus");
  det["post_process"] = json::array();
  warns.clear();
  (void)rvegen::generate(det, o);
  REQUIRE(has("unknown top-level key 'post_process' is ignored (did you mean 'postprocess'?)"));
  det.erase("post_process");
  det["generator"]["n_seedz"] = 5;
  warns.clear();
  (void)rvegen::generate(det, o);
  REQUIRE(has("unknown key 'n_seedz' for generator 'voronoi_2d'"));
}

// A bad "postprocess" block must fail BEFORE any shape is placed and before
// earlier writers produce output.
void test_postprocess_validated_before_generation() {
  auto cfg = circles_config();
  std::filesystem::remove("pipeline_early.svg");
  cfg["postprocess"] = json::array(
      {json{{"type", "svg"}, {"canvas_width", 200},
            {"output_path", "pipeline_early.svg"}},
       json{{"type", "voxel"}, {"nx", 4}, {"ny", 4}, {"nz", 1}}});   // no output_path
  std::vector<std::string> lines;
  rvegen::pipeline_options opts;
  opts.info = [&](std::string const& l) { lines.push_back(l); };
  bool threw = false;
  std::string what;
  try { (void)rvegen::run_pipeline(cfg, opts); }
  catch (std::exception const& e) { threw = true; what = e.what(); }
  REQUIRE(threw);
  REQUIRE(what.find("output_path") != std::string::npos);
  REQUIRE(lines.empty());                                   // no "generated N shapes"
  REQUIRE(!std::filesystem::exists("pipeline_early.svg"));  // no output from writer 1

  // Same for an unknown writer type and for a bad phase_assignment.
  cfg["postprocess"] = json::array({json{{"type", "no_such_writer"}}});
  threw = false;
  try { (void)rvegen::run_pipeline(cfg, opts); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw && lines.empty());
  cfg = circles_config();
  cfg["phase_assignment"] = json{{"all_inclusions", "nope"}};
  threw = false;
  try { (void)rvegen::run_pipeline(cfg, opts); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw && lines.empty());

  // generate() with run_postprocess = false does not build writers.
  cfg = circles_config();
  cfg["postprocess"] = json::array({json{{"type", "voxel"}}});
  rvegen::pipeline_options off;
  off.run_postprocess = false;
  REQUIRE(rvegen::run_pipeline(cfg, off).shapes.size() == 6);
}

// #120: the whole config is validated before the first shape is placed, and
// check_config() runs only that validation.
void test_check_config() {
  namespace fs = std::filesystem;
  // A private directory under the system temp dir, removed on every exit
  // path (with permissions restored first, so a failed REQUIRE cannot leave
  // an undeletable tree behind for the next run).
  const fs::path dir =
      fs::temp_directory_path() /
      ("rvegen_pipeline_check_" + std::to_string(
#ifndef _WIN32
                                      static_cast<long long>(::getpid())
#else
                                      0LL
#endif
                                      ));
  struct cleanup {
    fs::path d;
    ~cleanup() {
      std::error_code ec;
      for (fs::recursive_directory_iterator it{d, ec}, end; !ec && it != end;
           it.increment(ec)) {
        if (!it->is_symlink(ec))
          fs::permissions(it->path(), fs::perms::owner_all, ec);
      }
      fs::remove_all(d, ec);
    }
  } guard{dir};
  fs::remove_all(dir);
  fs::create_directory(dir);
  auto message = [](json const& cfg, bool check_only,
                    std::vector<std::string>* info = nullptr) {
    rvegen::pipeline_options o;
    if (info) o.info = [info](std::string const& l) { info->push_back(l); };
    try {
      if (check_only) rvegen::check_config(cfg, o);
      else (void)rvegen::run_pipeline(cfg, o);
    } catch (std::exception const& e) { return std::string{e.what()}; }
    return std::string{};
  };
  auto has = [](std::string const& text, char const* needle) {
    return text.find(needle) != std::string::npos;
  };
  auto dir_empty = [&] { return fs::directory_iterator{dir} == fs::directory_iterator{}; };

  // The acceptance case: three writers, the third lacks a required key.
  // Nothing is generated and neither of the first two files appears.
  auto three = circles_config();
  three["postprocess"] = json::array(
      {json{{"type", "svg"}, {"canvas_width", 200},
            {"output_path", (dir / "a.svg").string()}},
       json{{"type", "voxel"}, {"nx", 4}, {"ny", 4}, {"nz", 1},
            {"output_path", (dir / "b.vox").string()}},
       json{{"type", "svg"}, {"output_path", (dir / "c.svg").string()}}});
  for (bool check_only : {false, true}) {
    std::vector<std::string> info;
    const auto what = message(three, check_only, &info);
    REQUIRE(has(what, "postprocess[2] ('svg')"));
    REQUIRE(has(what, "canvas_width"));
    REQUIRE(info.empty());        // no "generated N shapes"
    REQUIRE(dir_empty());
  }

  // A valid config: check_config() accepts it, places nothing, writes nothing.
  three["postprocess"][2]["canvas_width"] = 100;
  {
    std::vector<std::string> info;
    REQUIRE(message(three, true, &info).empty());
    REQUIRE(info.empty());
    REQUIRE(dir_empty());
  }
  // ...and the same config then runs and writes all three files.
  REQUIRE(message(three, false).empty());
  REQUIRE(fs::exists(dir / "a.svg") && fs::exists(dir / "b.vox") &&
          fs::exists(dir / "c.svg"));
  // An existing output file is fine for a re-run (it gets overwritten).
  REQUIRE(message(three, true).empty());

  // A misspelt section is rejected with the section it most likely meant.
  auto typo = circles_config();
  typo["post_process"] = three["postprocess"];
  for (bool check_only : {false, true}) {
    const auto what = message(typo, check_only);
    REQUIRE(has(what, "unknown top-level key 'post_process' (did you mean 'postprocess'?)"));
  }
  for (auto const& [wrong, right] :
       std::vector<std::pair<std::string, std::string>>{
           {"shape", "shapes"}, {"generators", "generator"},
           {"distribution", "distributions"}, {"domain-box", "domain_box"},
           {"phase_assignments", "phase_assignment"}, {"Seed", "seed"}}) {
    auto c = circles_config();
    c[wrong] = 1;
    REQUIRE(has(message(c, true), ("did you mean '" + right + "'?").c_str()));
  }
  // Far from every key: no misleading hint. "_"-keys stay allowed.
  auto far = circles_config();
  far["xyzzy"] = 1;
  REQUIRE(has(message(far, true), "unknown top-level key 'xyzzy'"));
  REQUIRE(!has(message(far, true), "did you mean"));
  auto note = circles_config();
  note["_comment"] = "free text";
  note["_units"] = json{{"length", "mm"}};
  REQUIRE(message(note, true).empty());

  // Output locations are checked before generation.
  auto out = circles_config();
  out["postprocess"] = json::array({json{
      {"type", "svg"}, {"canvas_width", 200},
      {"output_path", (dir / "missing" / "x.svg").string()}}});
  for (bool check_only : {false, true}) {
    std::vector<std::string> info;
    const auto what = message(out, check_only, &info);
    REQUIRE(has(what, "postprocess[0] ('svg')"));
    REQUIRE(has(what, "does not exist"));
    REQUIRE(info.empty());
  }
  out["postprocess"][0]["output_path"] = dir.string();          // a directory
  REQUIRE(has(message(out, true), "is a directory, not a file"));
  out["postprocess"][0]["output_path"] = (dir / "a.svg" / "x.svg").string();
  REQUIRE(has(message(out, true), "is not a directory"));       // parent is a file
#ifndef _WIN32
  if (::geteuid() != 0) {   // root may write anywhere
    const fs::path locked = dir / "locked";
    fs::create_directory(locked);
    fs::permissions(locked, fs::perms::owner_read | fs::perms::owner_exec);
    out["postprocess"][0]["output_path"] = (locked / "x.svg").string();
    REQUIRE(has(message(out, true), "is not writable"));
    fs::permissions(locked, fs::perms::owner_all);
    // A read-only existing file is caught too.
    const fs::path ro = dir / "readonly.svg";
    { std::ofstream f{ro}; f << "x"; }
    fs::permissions(ro, fs::perms::owner_read);
    out["postprocess"][0]["output_path"] = ro.string();
    REQUIRE(has(message(out, true), "is not writable"));
    fs::permissions(ro, fs::perms::owner_all);
  }
#endif

  // A path that names no file, and a link to nowhere.
  out["postprocess"][0]["output_path"] = (dir / "sub").string() + "/";
  REQUIRE(has(message(out, true), "names a directory, not a file"));
#ifndef _WIN32
  {
    const fs::path link = dir / "dangling.svg";
    fs::create_symlink(dir / "nowhere" / "target.svg", link);
    out["postprocess"][0]["output_path"] = link.string();
    REQUIRE(has(message(out, true), "symbolic link whose target does not exist"));
    fs::remove(link);
  }
#endif

  // Writer checks that need only the config and the box run before
  // generation too (post_process_base::validate), each naming its block.
  auto late = [&](json writer, std::array<double, 3> box, char const* needle) {
    auto c = circles_config();
    c["domain_box"] = box;
    c["postprocess"] = json::array(
        {json{{"type", "svg"}, {"canvas_width", 100},
              {"output_path", (dir / "first.svg").string()}},
         std::move(writer)});
    bool ok = true;
    for (bool check_only : {true, false}) {
      std::vector<std::string> info;
      const auto what = message(c, check_only, &info);
      ok = ok && has(what, "postprocess[1] (") && has(what, needle) &&
           info.empty() && !fs::exists(dir / "first.svg");
      if (!ok) std::cerr << "  got: " << what << "\n";
    }
    return ok;
  };
  // A grid far beyond the voxel limit.
  REQUIRE(late(json{{"type", "voxel"}, {"nx", 100000}, {"ny", 100000}, {"nz", 1},
                    {"output_path", (dir / "big.vox").string()}},
               {1.0, 1.0, 0.0}, "voxel grid"));
  REQUIRE(late(json{{"type", "vtk_legacy"}, {"nx", 100000}, {"ny", 100000},
                    {"nz", 1}, {"output_path", (dir / "big.vtk").string()}},
               {1.0, 1.0, 0.0}, "voxel grid"));
  // Statistics in a 3D box need nz.
  REQUIRE(late(json{{"type", "statistics"}, {"nx", 16}, {"ny", 16}, {"nbins", 8},
                    {"output_path", (dir / "s.csv").string()}},
               {1.0, 1.0, 1.0}, "3D domain needs \"nz\""));
  // Forced tessellation output cannot be periodic.
  REQUIRE(late(json{{"type", "gmsh_geo"}, {"tessellation", true},
                    {"periodic", true},
                    {"output_path", (dir / "t.geo").string()}},
               {1.0, 1.0, 0.0}, "periodic output is not supported"));
  // An output path that is missing or empty.
  REQUIRE(late(json{{"type", "svg"}, {"canvas_width", 100}}, {1.0, 1.0, 0.0},
               "output_path"));
  REQUIRE(late(json{{"type", "svg"}, {"canvas_width", 100}, {"output_path", ""}},
               {1.0, 1.0, 0.0}, "output_path not set"));
  // An entry that is not a writer block at all.
  REQUIRE(late(json{{"type", 5}, {"output_path", "x"}}, {1.0, 1.0, 0.0},
               "needs a \"type\" string"));
  REQUIRE(late(json(1), {1.0, 1.0, 0.0}, "each entry must be an object"));
  REQUIRE(late(json{{"type", "no_such_writer"}}, {1.0, 1.0, 0.0}, "no_such_writer"));

  // Missing or mistyped sections are named.
  auto c = circles_config();
  c.erase("generator");
  REQUIRE(has(message(c, true), "no \"generator\" section"));
  c = circles_config();
  c.erase("termination");
  REQUIRE(has(message(c, true), "no \"termination\" section"));
  c = circles_config();
  c["shapes"] = json{{"type", "circle_input"}};                 // object, not array
  REQUIRE(has(message(c, true), "\"shapes\" must be an array"));
  c = circles_config();
  c["distributions"] = json::array();
  REQUIRE(has(message(c, true), "\"distributions\" must be an object"));
  REQUIRE(has(message(json::array(), true), "must be a JSON object"));
  // Every other block is built by the check as well.
  c = circles_config();
  c["distributions"]["x"]["type"] = "no_such_distribution";
  REQUIRE(!message(c, true).empty());
  c = circles_config();
  c["shapes"][0]["radius_dist"] = "undefined_name";
  REQUIRE(!message(c, true).empty());
  c = circles_config();
  c["termination"]["target"] = -1;
  REQUIRE(!message(c, true).empty());
  c = circles_config();
  c["phase_assignment"] = json{{"all_inclusions", "nope"}};
  REQUIRE(!message(c, true).empty());

  // check_config validates writers even when the caller skips them at run
  // time; generate() with run_postprocess = false still does not.
  auto skip = circles_config();
  skip["postprocess"] = json::array({json{{"type", "voxel"}}});
  rvegen::pipeline_options off;
  off.run_postprocess = false;
  bool threw = false;
  try { rvegen::check_config(skip, off); } catch (std::exception const&) { threw = true; }
  REQUIRE(threw);
  REQUIRE(rvegen::generate(skip, off).shapes.size() == 6);
}

// progress.cancel stops the run and no writer runs afterwards.
void test_cancel_stops_pipeline() {
  auto cfg = circles_config();
  std::filesystem::remove("pipeline_cancel.svg");
  cfg["postprocess"] = json::array({json{{"type", "svg"}, {"canvas_width", 200},
                                         {"output_path", "pipeline_cancel.svg"}}});
  rvegen::pipeline_options opts;
  opts.progress.cancel = [] { return true; };
  bool cancelled = false;
  try { (void)rvegen::run_pipeline(cfg, opts); }
  catch (rvegen::pipeline_cancelled const&) { cancelled = true; }
  REQUIRE(cancelled);
  REQUIRE(!std::filesystem::exists("pipeline_cancel.svg"));
}

// Voronoi generators follow the pipeline seed; an explicit generator seed
// overrides it.
void test_voronoi_follows_pipeline_seed() {
  json cfg = {{"domain_box", {1.0, 1.0, 0.0}},
              {"generator", {{"type", "voronoi_2d"}, {"n_seeds", 6}}}};
  auto first_area = [&](std::uint64_t seed) {
    rvegen::pipeline_options o;
    o.seed = seed;
    return rvegen::generate(cfg, o).shapes[0]->area();
  };
  REQUIRE(first_area(1) != first_area(2));
  REQUIRE(first_area(2) == first_area(2));
  cfg["seed"] = 3;   // top-level seed from the config works as well
  auto a = rvegen::generate(cfg).shapes[0]->area();
  cfg["seed"] = 4;
  REQUIRE(a != rvegen::generate(cfg).shapes[0]->area());
  cfg["generator"]["seed"] = 11;   // explicit override: top-level seed irrelevant
  auto b = rvegen::generate(cfg).shapes[0]->area();
  cfg["seed"] = 3;
  REQUIRE(b == rvegen::generate(cfg).shapes[0]->area());
}

// numsim-core must not print "using default" lines on its own (stdout is
// captured at file-descriptor level).
void test_no_stdout_noise() {
#ifndef _WIN32
  std::fflush(stdout);
  const int saved = dup(1);
  const int fd = open("pipeline_stdout.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
  REQUIRE(saved >= 0 && fd >= 0);
  dup2(fd, 1);
  close(fd);
  (void)rvegen::generate(circles_config());   // many defaulted keys
  std::fflush(stdout);
  dup2(saved, 1);
  close(saved);
  std::ifstream in{"pipeline_stdout.txt"};
  std::string all((std::istreambuf_iterator<char>(in)), {});
  REQUIRE(all.find("using default") == std::string::npos);
  REQUIRE(all.empty());
#endif
}

void test_schema_to_json() {
  rvegen::register_all_types();
  auto j = rvegen::schema_to_json(
      rvegen::generator_registry_t<>::instance().schema("weave"));
  REQUIRE(j.at("n_warp_yarns").at("type") == "size_t");
  REQUIRE(j.at("n_warp_yarns").at("required") == true);
  REQUIRE(j.at("n_warp_yarns").at("min") == 1);
  REQUIRE(j.at("yarn_radius").at("type") == "double");
  REQUIRE(j.at("n_segments_per_yarn").at("has_default") == true);
  REQUIRE(j.at("n_segments_per_yarn").at("required") == false);
  REQUIRE(!j.at("amplitude").at("description").get<std::string>().empty());
}

} // namespace

int main() {
  test_generate();
  test_postprocess_toggle();
  test_seed_and_warnings();
  test_postprocess_validated_before_generation();
  test_check_config();
  test_cancel_stops_pipeline();
  test_voronoi_follows_pipeline_seed();
  test_no_stdout_noise();
  test_schema_to_json();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "pipeline_test OK\n";
  return EXIT_SUCCESS;
}
