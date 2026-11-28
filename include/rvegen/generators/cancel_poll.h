#pragma once

// Cooperative cancellation for the deterministic generators' build loops.
//
// The sampling generators poll `progress_options::cancel` once per
// placement attempt. The deterministic ones (weave, tow_weave, voronoi)
// have no attempt loop, so their build() functions take an optional
// `cancel_fn const*` and call `poll_cancel` once per yarn / cell. When the
// callback reports true, `generation_cancelled` unwinds out of build();
// deterministic_generator::compute catches it and returns no shapes.

#include <functional>

namespace rvegen {

using cancel_fn = std::function<bool()>;

struct generation_cancelled {};

inline void poll_cancel(cancel_fn const* cancel) {
  if (cancel && *cancel && (*cancel)()) throw generation_cancelled{};
}

} // namespace rvegen
