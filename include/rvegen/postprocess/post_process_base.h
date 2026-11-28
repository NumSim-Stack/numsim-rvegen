#pragma once

#include <array>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

#include "../domain.h"
#include "../phase/phase.h"
#include "../shapes/shape_base.h"

namespace rvegen {

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
};

} // namespace rvegen
