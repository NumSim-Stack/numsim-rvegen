#pragma once

#include <array>
#include <memory>
#include <vector>

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

  virtual void run(shape_vector const& shapes,
                   std::array<value_type, 3> const& domain_box) const = 0;
};

} // namespace rvegen
