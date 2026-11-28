// Generation report (#119): every generator says what it reached against
// its termination target, the pipeline turns a missed target into a
// warning or (strict) an error, and the run_summary writer records it.

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "rvegen/pipeline.h"
#include "rvegen/version.h"

namespace {
int failures = 0;
#define REQUIRE(cond)                                                    \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << "REQUIRE failed: " #cond " at " << __FILE__ << ":"    \
                << __LINE__ << "\n";                                     \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

using json = nlohmann::json;
using rvegen::stop_reason;

json config(std::string const& generator, json termination,
            std::size_t max_attempts) {
  const bool inside = generator == "only_inside";
  return json{
      {"domain_box", {1.0, 1.0, 0.0}},
      {"seed", 17},
      {"distributions",
       {{"x", {{"type", "uniform_real"}, {"a", inside ? 0.1 : 0.0},
               {"b", inside ? 0.9 : 1.0}}},
        {"r", {{"type", "uniform_real"}, {"a", 0.05}, {"b", 0.08}}}}},
      {"shapes", json::array({json{{"type", "circle_input"}, {"pos_x_dist", "x"},
                                   {"pos_y_dist", "x"}, {"radius_dist", "r"}}})},
      {"generator", {{"type", generator}, {"max_attempts", max_attempts}}},
      {"termination", std::move(termination)}};
}

struct sinks {
  std::vector<std::string> info, warn;
  [[nodiscard]] rvegen::pipeline_options options() {
    rvegen::pipeline_options o;
    o.info = [this](std::string const& s) { info.push_back(s); };
    o.warn = [this](std::string const& s) { warn.push_back(s); };
    return o;
  }
  [[nodiscard]] static bool has(std::vector<std::string> const& v,
                                std::string_view part) {
    for (auto const& l : v)
      if (l.find(part) != std::string::npos) return true;
    return false;
  }
};

std::size_t primaries(rvegen::pipeline_result const& r) {
  std::size_t n = 0;
  for (auto const& s : r.shapes) n += s->is_primary() ? 1 : 0;
  return n;
}

// The three placement generators x both termination types, target reached
// and target missed.
void test_report_matrix() {
  for (std::string const generator : {"only_inside", "random", "periodic"}) {
    // ---- number_of_inclusions, reached --------------------------------
    {
      sinks s;
      const auto r = rvegen::generate(
          config(generator, {{"type", "number_of_inclusions"}, {"target", 8}},
                 100000),
          s.options());
      auto const& rep = r.report;
      REQUIRE(rep.termination == "number_of_inclusions");
      REQUIRE(rep.has_target && !rep.target_missed());
      REQUIRE(rep.reason == stop_reason::reached);
      REQUIRE(rep.target == 8.0 && rep.achieved == 8.0);
      REQUIRE(primaries(r) == 8);
      REQUIRE(rep.attempts >= 8 && rep.attempts < 100000);
      REQUIRE(rep.max_attempts == 100000);
      REQUIRE(s.warn.empty());
      REQUIRE(sinks::has(s.info, "rvegen: number_of_inclusions: reached 8 of 8 "
                                 "after " + std::to_string(rep.attempts) +
                                     " attempts"));
      REQUIRE(!sinks::has(s.info, "max_attempts"));
    }
    // ---- number_of_inclusions, missed ---------------------------------
    {
      sinks s;
      const auto r = rvegen::generate(
          config(generator, {{"type", "number_of_inclusions"}, {"target", 1000}},
                 300),
          s.options());
      auto const& rep = r.report;
      REQUIRE(rep.reason == stop_reason::max_attempts);
      REQUIRE(rep.target_missed());
      REQUIRE(rep.attempts == 300 && rep.max_attempts == 300);
      REQUIRE(rep.target == 1000.0);
      REQUIRE(rep.achieved == static_cast<double>(primaries(r)));
      REQUIRE(rep.achieved > 0.0 && rep.achieved < 1000.0);
      REQUIRE(s.warn.size() == 1);
      REQUIRE(sinks::has(
          s.warn, "rvegen: warning - number_of_inclusions: reached " +
                      std::to_string(primaries(r)) +
                      " of 1000 after 300 attempts (max_attempts)"));
      REQUIRE(!sinks::has(s.info, "number_of_inclusions:"));
    }
    // ---- volume_fraction, reached -------------------------------------
    {
      sinks s;
      const auto r = rvegen::generate(
          config(generator,
                 {{"type", "volume_fraction"}, {"target_fraction", 0.2}}, 100000),
          s.options());
      auto const& rep = r.report;
      REQUIRE(rep.termination == "volume_fraction");
      REQUIRE(rep.reason == stop_reason::reached && !rep.target_missed());
      REQUIRE(rep.target == 0.2);
      REQUIRE(rep.achieved >= 0.2 && rep.achieved < 0.25);
      // The same number the rest of the library reports.
      REQUIRE(std::abs(rep.achieved - rvegen::current_volume_fraction<double>(
                                          r.shapes, r.domain_box)) < 1e-12);
      REQUIRE(rep.attempts > 0 && rep.attempts < 100000);
      REQUIRE(s.warn.empty());
      REQUIRE(sinks::has(s.info, "rvegen: volume_fraction: reached 0."));
    }
    // ---- volume_fraction, missed --------------------------------------
    {
      sinks s;
      const auto r = rvegen::generate(
          config(generator,
                 {{"type", "volume_fraction"}, {"target_fraction", 0.9}}, 400),
          s.options());
      auto const& rep = r.report;
      REQUIRE(rep.reason == stop_reason::max_attempts && rep.target_missed());
      REQUIRE(rep.attempts == 400);
      REQUIRE(rep.target == 0.9);
      REQUIRE(rep.achieved > 0.05 && rep.achieved < 0.9);
      REQUIRE(std::abs(rep.achieved - rvegen::current_volume_fraction<double>(
                                          r.shapes, r.domain_box)) < 1e-12);
      REQUIRE(s.warn.size() == 1);
      REQUIRE(sinks::has(s.warn, " of 0.9 after 400 attempts (max_attempts)"));
    }
  }
}

// The generators themselves: cancelled, no inputs, and the report of a
// second compute() replaces the first.
void test_generator_reports() {
  rvegen::register_all_types();
  std::mt19937 engine{3};
  rvegen::uniform_real_distribution<double> x{0.1, 0.9, engine};
  rvegen::uniform_real_distribution<double> rad{0.02, 0.03, engine};
  using input_vector = rvegen::rve_generator_base<double>::input_vector;
  const std::array<double, 3> box{1.0, 1.0, 0.0};
  auto inputs = [&] {
    input_vector v;
    v.push_back(std::make_unique<rvegen::circle_input<double>>(x, x, rad));
    return v;
  };

  auto check = [&](rvegen::rve_generator_base<double>& gen) {
    REQUIRE(!gen.last_report().has_value());   // nothing computed yet
    rvegen::number_of_inclusions<double> twenty{20};

    // Cancelled after 5 accepted shapes.
    {
      auto in = inputs();
      std::size_t placed = 0;
      rvegen::progress_options opts;
      opts.emit_every = 1;
      opts.on_progress = [&](rvegen::progress_info i) { placed = i.shapes_placed; };
      opts.cancel = [&] { return placed >= 5; };
      const auto shapes = gen.compute(in, twenty, box, opts);
      REQUIRE(gen.last_report().has_value());
      auto const& rep = *gen.last_report();
      REQUIRE(rep.reason == stop_reason::cancelled);
      REQUIRE(rep.target_missed());
      REQUIRE(rep.target == 20.0);
      REQUIRE(rep.achieved >= 5.0 && rep.achieved < 20.0);
      REQUIRE(rep.attempts >= 5);
      REQUIRE(rep.summary().find("(cancelled)") != std::string::npos);
    }
    // No inputs.
    {
      input_vector none;
      const auto shapes = gen.compute(none, twenty, box);
      REQUIRE(shapes.empty());
      auto const& rep = *gen.last_report();
      REQUIRE(rep.reason == stop_reason::no_inputs);
      REQUIRE(rep.attempts == 0 && rep.achieved == 0.0);
      REQUIRE(rep.summary() ==
              "number_of_inclusions: reached 0 of 20 after 0 attempts (no_inputs)");
    }
    // A normal run replaces the report.
    {
      auto in = inputs();
      const auto shapes = gen.compute(in, twenty, box);
      auto const& rep = *gen.last_report();
      REQUIRE(rep.reason == stop_reason::reached && rep.achieved == 20.0);
    }
    // until_full never stops by itself: running out of attempts is the
    // expected end, not a missed target.
    {
      auto in = inputs();
      rvegen::until_full<double> full;
      const auto shapes = gen.compute(in, full, box);
      auto const& rep = *gen.last_report();
      REQUIRE(rep.termination == "until_full");
      REQUIRE(!rep.has_target && !rep.target_missed());
      REQUIRE(rep.reason == stop_reason::max_attempts);
      REQUIRE(rep.attempts == 500);
      REQUIRE(rep.summary() == "until_full: stopped after 500 attempts (max_attempts)");
    }
  };
  rvegen::only_inside_generator<double> a;
  a.set_max_attempts(500);
  check(a);
  rvegen::random_generator<double> b;
  b.set_max_attempts(500);
  check(b);
  rvegen::periodic_generator<double> c;
  c.set_max_attempts(500);
  check(c);
}

// A generator that records no report (and so a custom one written before
// reports existed): the pipeline measures the result itself. One that
// throws leaves no report of an earlier compute() behind.
template <typename T>
class silent_generator final : public rvegen::rve_generator_base<T> {
public:
  using shape_vector = typename rvegen::rve_generator_base<T>::shape_vector;
  using input_vector = typename rvegen::rve_generator_base<T>::input_vector;
  explicit silent_generator(rvegen::parameter_handler_t const& h)
      : _count{h.template get<std::size_t>("count")} {}
  [[nodiscard]] static rvegen::parameter_controller_t parameters() {
    rvegen::parameter_controller_t p;
    p.template required<std::size_t>("count");
    return p;
  }
  [[nodiscard]] shape_vector
  compute(input_vector& inputs, rvegen::termination_base<T> const&,
          rvegen::domain<T> const&, rvegen::progress_options const& = {}) override {
    shape_vector out;
    for (std::size_t i = 0; i < _count; ++i) {
      auto s = inputs.at(0)->new_shape();
      s->set_middle_point({T(0.1) * T(i + 1), T(0.5), T(0)});
      out.push_back(std::move(s));
    }
    return out;
  }

private:
  std::size_t _count;
};

void test_unrecorded_and_stale_reports() {
  rvegen::register_all_types();
  rvegen::generator_registry_t<>::instance()
      .template register_type<silent_generator<double>>("test_silent");
  auto cfg = config("only_inside", {{"type", "number_of_inclusions"}, {"target", 5}}, 10);
  cfg["distributions"]["r"] = {{"type", "uniform_real"}, {"a", 0.01}, {"b", 0.02}};
  cfg["generator"] = {{"type", "test_silent"}, {"count", 3}};
  {
    sinks s;
    const auto r = rvegen::generate(cfg, s.options());
    REQUIRE(r.shapes.size() == 3);
    REQUIRE(r.report.termination == "number_of_inclusions");
    REQUIRE(r.report.target == 5.0 && r.report.achieved == 3.0);
    REQUIRE(r.report.reason == stop_reason::max_attempts && r.report.target_missed());
    REQUIRE(r.report.attempts == 0);   // not known
    REQUIRE(sinks::has(s.warn, "rvegen: warning - number_of_inclusions: reached 3 of 5 "
                               "(max_attempts)"));
    auto o = s.options();
    o.strict_targets = true;
    bool thrown = false;
    try { (void)rvegen::generate(cfg, o); }
    catch (rvegen::target_not_reached const&) { thrown = true; }
    REQUIRE(thrown);
  }
  cfg["generator"]["count"] = 5;
  {
    sinks s;
    const auto r = rvegen::generate(cfg, s.options());
    REQUIRE(r.report.reason == stop_reason::reached && r.report.achieved == 5.0);
    REQUIRE(s.warn.empty());
    REQUIRE(sinks::has(s.info, "rvegen: number_of_inclusions: reached 5 of 5"));
  }

  // An input that throws on its third shape: the report of the earlier,
  // successful compute() must not survive the failed one.
  struct failing_input final : rvegen::shape_input_base<double> {
    std::size_t made{0};
    std::size_t fail_at{0};
    [[nodiscard]] int dimension() const noexcept override { return 2; }
    [[nodiscard]] std::unique_ptr<rvegen::shape_base<double>> new_shape() override {
      if (fail_at != 0 && ++made >= fail_at) throw std::runtime_error{"input failed"};
      const double c = 0.1 + 0.07 * static_cast<double>(made % 11);
      return std::make_unique<rvegen::circle<double>>(c, c, 0.01);
    }
  };
  const std::array<double, 3> box{1.0, 1.0, 0.0};
  rvegen::number_of_inclusions<double> two{2};
  auto run = [&](rvegen::rve_generator_base<double>& gen) {
    rvegen::rve_generator_base<double>::input_vector in;
    in.push_back(std::make_unique<failing_input>());
    auto* input = static_cast<failing_input*>(in.back().get());
    input->fail_at = 0;
    for (std::size_t i = 0; i < 2; ++i) ++input->made;   // distinct centres
    (void)gen.compute(in, two, box);
    REQUIRE(gen.last_report().has_value());
    input->made = 0;
    input->fail_at = 1;
    bool thrown = false;
    try { (void)gen.compute(in, two, box); } catch (std::runtime_error const&) { thrown = true; }
    REQUIRE(thrown);
    REQUIRE(!gen.last_report().has_value());
  };
  rvegen::only_inside_generator<double> a;
  run(a);
  rvegen::random_generator<double> b;
  run(b);
  rvegen::periodic_generator<double> c;
  run(c);
}

void test_summary_text() {
  rvegen::generation_report r;
  REQUIRE(r.summary().empty());   // no termination: nothing to say
  r.termination = "volume_fraction";
  r.has_target = true;
  r.target = 0.35;
  r.achieved = 0.2231234;
  r.attempts = 50;
  r.reason = stop_reason::max_attempts;
  REQUIRE(r.summary() ==
          "volume_fraction: reached 0.2231 of 0.35 after 50 attempts (max_attempts)");
  r.achieved = 0.3502;
  r.attempts = 1;
  r.reason = stop_reason::reached;
  REQUIRE(r.summary() == "volume_fraction: reached 0.3502 of 0.35 after 1 attempt");
  // A miss never prints like a hit: more digits until the two differ.
  r.achieved = 0.34996;
  r.attempts = 9;
  r.reason = stop_reason::max_attempts;
  REQUIRE(r.summary() ==
          "volume_fraction: reached 0.34996 of 0.35 after 9 attempts (max_attempts)");
  r.target = 1.0;
  r.achieved = 0.9999999;
  REQUIRE(r.summary().find("reached 0.9999999 of 1 ") != std::string::npos);
  r.target = 0.35;
  r.termination = "number_of_inclusions";
  r.target = 2500000;
  r.achieved = 1234567;
  r.attempts = 0;   // not known (a generator that records no report)
  r.reason = stop_reason::max_attempts;
  REQUIRE(r.summary() ==
          "number_of_inclusions: reached 1234567 of 2500000 (max_attempts)");
}

// Strict runs, the config key, the deterministic generators and until_full
// through the pipeline.
void test_pipeline_strict() {
  namespace fs = std::filesystem;
  const fs::path out = "generation_report_strict.vox";
  fs::remove(out);
  auto missed = config("only_inside",
                       {{"type", "volume_fraction"}, {"target_fraction", 0.9}}, 400);
  missed["postprocess"] = json::array({json{
      {"type", "voxel"}, {"nx", 8}, {"ny", 8}, {"nz", 1}, {"output_path", out.string()}}});

  // strict_targets: an exception carrying the report, and no output file.
  {
    sinks s;
    auto o = s.options();
    o.strict_targets = true;
    bool thrown = false;
    try {
      (void)rvegen::run_pipeline(missed, o);
    } catch (rvegen::target_not_reached const& e) {
      thrown = true;
      REQUIRE(e.report.target == 0.9 && e.report.attempts == 400);
      REQUIRE(e.report.reason == stop_reason::max_attempts);
      REQUIRE(std::string{e.what()}.find("volume_fraction: reached 0.") == 0);
      REQUIRE(std::string{e.what()}.find("of 0.9 after 400 attempts (max_attempts)") !=
              std::string::npos);
    }
    REQUIRE(thrown);
    REQUIRE(!fs::exists(out));
    REQUIRE(s.warn.empty());
  }
  // "strict": true in the config does the same; false leaves the default.
  {
    auto cfg = missed;
    cfg["strict"] = true;
    bool thrown = false;
    try { (void)rvegen::run_pipeline(cfg); }
    catch (rvegen::target_not_reached const&) { thrown = true; }
    REQUIRE(thrown);
    REQUIRE(!fs::exists(out));
    cfg["strict"] = false;
    REQUIRE(rvegen::run_pipeline(cfg).report.target_missed());
    REQUIRE(fs::exists(out));
    fs::remove(out);
    cfg["strict"] = "yes";
    std::string msg;
    try { rvegen::check_config(cfg); } catch (std::invalid_argument const& e) { msg = e.what(); }
    REQUIRE(msg.find("\"strict\" must be true or false") != std::string::npos);
  }
  // A strict run that reaches its target is an ordinary run.
  {
    rvegen::pipeline_options o;
    o.strict_targets = true;
    const auto r = rvegen::generate(
        config("periodic", {{"type", "number_of_inclusions"}, {"target", 5}}, 100000), o);
    REQUIRE(primaries(r) == 5 && !r.report.target_missed());
  }
  // until_full: an info line, no warning, and no error when strict.
  {
    sinks s;
    auto o = s.options();
    o.strict_targets = true;
    const auto r = rvegen::generate(config("only_inside", {{"type", "until_full"}}, 300), o);
    REQUIRE(!r.shapes.empty());
    REQUIRE(s.warn.empty());
    REQUIRE(sinks::has(s.info, "rvegen: until_full: stopped after 300 attempts (max_attempts)"));
  }
  // Deterministic generators have no target: no line at all.
  {
    sinks s;
    auto o = s.options();
    o.strict_targets = true;
    const auto r = rvegen::generate(json::parse(R"({
      "domain_box": [1.0, 1.0, 0.0],
      "generator": {"type": "voronoi_2d", "n_seeds": 6}})"), o);
    REQUIRE(r.shapes.size() == 6);
    REQUIRE(r.report.termination.empty() && !r.report.target_missed());
    REQUIRE(r.report.reason == stop_reason::reached);
    REQUIRE(s.warn.empty());
    REQUIRE(s.info.size() == 1);   // "generated 6 shapes" only
  }
}

// The run_summary post-process writes seed, version, config and report.
void test_run_summary() {
  namespace fs = std::filesystem;
  const fs::path file = "generation_report_summary.json";
  fs::remove(file);
  auto cfg = config("periodic",
                    {{"type", "volume_fraction"}, {"target_fraction", 0.9}}, 400);
  cfg["postprocess"] = json::array(
      {json{{"type", "run_summary"}, {"output_path", file.string()}}});
  rvegen::pipeline_options o;
  o.seed = 23;   // overrides the config's 17
  const auto r = rvegen::run_pipeline(cfg, o);
  REQUIRE(r.post_processes_run == std::vector<std::string>{"run_summary"});
  std::ifstream in{file};
  REQUIRE(in.good());
  const json j = json::parse(in);
  REQUIRE(j.at("rvegen_version") == rvegen::version);
  REQUIRE(j.at("seed") == 23);
  REQUIRE(j.at("config") == cfg);           // as given, seed 17 included
  REQUIRE(j.at("domain_box") == json({1.0, 1.0, 0.0}));
  REQUIRE(j.at("n_shapes") == r.shapes.size());
  REQUIRE(j.at("n_inclusions") == primaries(r));
  REQUIRE(j.at("n_inclusions").get<std::size_t>() < r.shapes.size());   // copies exist
  REQUIRE(std::abs(j.at("volume_fraction").get<double>() -
                   rvegen::current_volume_fraction<double>(r.shapes, r.domain_box)) < 1e-15);
  auto const& rep = j.at("report");
  REQUIRE(rep.at("termination") == "volume_fraction");
  REQUIRE(rep.at("has_target") == true);
  REQUIRE(rep.at("target") == 0.9);
  REQUIRE(rep.at("achieved").get<double>() == r.report.achieved);
  REQUIRE(rep.at("attempts") == 400 && rep.at("max_attempts") == 400);
  REQUIRE(rep.at("stop_reason") == "max_attempts");
  REQUIRE(rep.at("target_reached") == false);
  fs::remove(file);

  // Outside the pipeline the writer still records what the shapes say.
  rvegen::run_summary_writer<double> direct{file.string()};
  direct.run(r.shapes, r.domain_box);
  std::ifstream in2{file};
  const json d = json::parse(in2);
  REQUIRE(d.at("n_shapes") == r.shapes.size());
  REQUIRE(!d.contains("seed") && !d.contains("report") && !d.contains("config"));
  fs::remove(file);

  // An empty path is rejected at validation, before any generation.
  cfg["postprocess"][0]["output_path"] = "";
  std::string msg;
  try { rvegen::check_config(cfg); } catch (std::exception const& e) { msg = e.what(); }
  REQUIRE(msg.find("postprocess[0] ('run_summary')") != std::string::npos);
  REQUIRE(msg.find("output_path is empty") != std::string::npos);
}

void test_version() {
#ifdef RVEGEN_EXPECTED_VERSION
  REQUIRE(std::string_view{rvegen::version} == RVEGEN_EXPECTED_VERSION);
#endif
  REQUIRE(!std::string_view{rvegen::version}.empty());
}
}  // namespace

int main() {
  test_report_matrix();
  test_generator_reports();
  test_unrecorded_and_stale_reports();
  test_summary_text();
  test_pipeline_strict();
  test_run_summary();
  test_version();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "generation_report_test OK\n";
  return EXIT_SUCCESS;
}
