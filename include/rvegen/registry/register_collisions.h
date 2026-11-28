#pragma once

// One-shot registration for the built-in shape collision pairs. Mirrors the
// register_all_*() pattern used for distributions / inputs / generators /
// terminations / outputs. Users call this once at startup if they want
// precise dispatch in the AABB-then-precise pipeline; otherwise unregistered
// pairs degrade to the AABB-overlap fallback.
//
// Adding a new shape pair: add one register_pair line here. New pairs
// instantly become available to every generator that uses
// collide_two_stage / collision_dispatcher::instance().

#include "../intersection/collision_details_extra.h"
#include "../intersection/collision_gap.h"
#include "../intersection/collision_dispatcher.h"
#include "../shapes/box.h"
#include "../shapes/circle.h"
#include "../shapes/ellipse.h"
#include "../shapes/mesh_inclusion.h"
#include "../shapes/polyline_tube.h"
#include "../shapes/rectangle.h"
#include "../shapes/sphere.h"

namespace rvegen {

template <typename T = double>
inline void register_all_collision_pairs() {
  auto& d = collision_dispatcher<T>::instance();

  // 2D pairs.
  d.template register_pair<circle<T>,    circle<T>>();
  d.template register_pair<rectangle<T>, rectangle<T>>();
  d.template register_pair<ellipse<T>,   ellipse<T>>();
  d.template register_pair<circle<T>,    rectangle<T>>();
  d.template register_pair<circle<T>,    ellipse<T>>();
  d.template register_pair<rectangle<T>, ellipse<T>>();

  // 3D pairs.
  d.template register_pair<sphere<T>, sphere<T>>();
  d.template register_pair<box<T>,    box<T>>();
  d.template register_pair<sphere<T>, box<T>>();

  // Fibres (#83): capsule-chain distance via GTE.
  d.template register_pair<polyline_tube<T>, polyline_tube<T>>();
  d.template register_pair<polyline_tube<T>, sphere<T>>();
  d.template register_pair<polyline_tube<T>, box<T>>();

  // Closed-mesh inclusions: triangle contact + containment (see
  // collision_details_extra.h for the open-mesh caveat).
  d.template register_pair<mesh_inclusion<T>, mesh_inclusion<T>>();
  d.template register_pair<mesh_inclusion<T>, sphere<T>>();
  d.template register_pair<mesh_inclusion<T>, box<T>>();
  d.template register_pair<mesh_inclusion<T>, polyline_tube<T>>();
}

} // namespace rvegen
