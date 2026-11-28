#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../domain.h"
#include "../phase/phase.h"
#include "../shapes/shape_base.h"

namespace rvegen {

// Where a post-process sends what it has to say about its own output: a
// fact worth a line (`info`) or something the user should look at (`warn`).
// Either may be empty.
struct post_process_log {
  std::function<void(std::string const&)> info;
  std::function<void(std::string const&)> warn;
};

// Polymorphic post-processing step.
//
// Runs after generation against the shape vector + domain box, and is
// responsible for its own sink (file path, in-memory buffer, etc.). Use
// for derived data products that aren't a primary geometric serialization:
// voxelization for FFT, statistics dumps, periodicity checks, RDF, ...
//
// Distinct from output_base: writers stream the analytic shape representation
// to a single supplied stream; post-processes own their destination and may
// produce lossy / computed artefacts (e.g. a phase-id raster). A pipeline
// typically runs one writer plus zero-or-more post-processes.
template <typename T>
class post_process_base {
public:
  using value_type = T;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  post_process_base() = default;
  post_process_base(post_process_base const&) = default;
  post_process_base(post_process_base&&) noexcept = default;
  post_process_base& operator=(post_process_base const&) = default;
  post_process_base& operator=(post_process_base&&) noexcept = default;
  virtual ~post_process_base() = default;

  // Generic phase hook. The pipeline calls this on EVERY post-process
  // before run(); the default ignores the collection. Writers that
  // consume phases (voxel, gmsh_geo, vtk_legacy)
  // already expose a `set_phases(phase_collection<T> const*) noexcept`
  // with exactly this signature, which therefore overrides this virtual
  // implicitly. The pointer must outlive the next run().
  virtual void set_phases(phase_collection<T> const* /*phases*/) noexcept {}

  // Facts about the run that the shapes do not carry: an object with
  // "seed", "report" (the generation report) and "config" (as given). The
  // pipeline calls this on every post-process before run(); the default
  // ignores it (run_summary_writer writes it out). The pointer must
  // outlive the next run().
  virtual void set_run_info(nlohmann::json const* /*info*/) noexcept {}

  // Message sink for run() (#115). The pipeline attaches one that forwards
  // to its info and warning lines; without a sink a post-process is silent.
  // The pointer must outlive the next run(); nullptr detaches.
  void set_log(post_process_log const* log) noexcept { _log = log; }
  [[nodiscard]] bool has_log() const noexcept { return _log != nullptr; }

  // "min_gap = 0.02, wall_gap = 0.01": the minimum distances the placement
  // kept (#111), for a file header. Empty without run info and when no gap
  // was set, so a run without gaps writes the header it always wrote.
  [[nodiscard]] static std::string
  placement_gap_note(nlohmann::json const* info) {
    std::string note;
    if (info == nullptr || !info->is_object() || !info->contains("report") ||
        !info->at("report").is_object()) {
      return note;
    }
    auto const& report = info->at("report");
    for (char const* key : {"min_gap", "min_gap_relative", "wall_gap"}) {
      auto const it = report.find(key);
      if (it == report.end() || !it->is_number() || !(it->get<double>() > 0.0))
        continue;
      if (!note.empty()) note += ", ";
      note += std::string{key} + " = " + it->dump();
    }
    return note;
  }

  // Everything run() would reject that depends only on the configuration
  // and the domain box, not on the generated shapes: a missing output
  // path, a grid that is too large, an option that needs a 3D box. The
  // pipeline calls it for every writer BEFORE the first shape is placed
  // (and `rvegen --check` calls nothing else), so such a mistake does not
  // cost a generation run. Must not create or truncate any file. run()
  // implementations call it first, so the checks exist once.
  virtual void validate(domain<value_type> const& /*domain_box*/) const {}

  virtual void run(shape_vector const& shapes,
                   domain<value_type> const& domain_box) const = 0;

protected:
  // For derived classes, from run(): one line each, without a prefix (the
  // sink says which post-process is speaking).
  void log_info(std::string const& line) const {
    if (_log != nullptr && _log->info) _log->info(line);
  }
  void log_warn(std::string const& line) const {
    if (_log != nullptr && _log->warn) _log->warn(line);
  }

private:
  post_process_log const* _log{nullptr};
};

} // namespace rvegen
