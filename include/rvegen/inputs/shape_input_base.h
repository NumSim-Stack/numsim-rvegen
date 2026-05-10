#pragma once

#include <memory>
#include <string>
#include <utility>

#include "../metadata/info.h"
#include "../shapes/shape_base.h"
#include "../types.h"

namespace rvegen {

// Polymorphic input descriptor. One per shape kind (circle_input, sphere_input,
// rectangle_input, ...). Owns or references the distributions for its
// parameters and produces fresh candidate shapes via new_shape().
//
// The generator drives the simulation by repeatedly calling new_shape() on
// each input until a termination condition fires.
//
// Metadata propagation:
//   Each input may carry an `info` blob that gets stamped onto every
//   shape it produces. Concrete inputs read metadata fields from the
//   schema (today: optional `phase_name`; future: orientation, source
//   paths, custom tags) and call `tag(*shape)` from their `new_shape()`
//   implementations. The shape carries the metadata through clone /
//   translate operations; output writers consume it (per-phase gmsh
//   Physical groups, voxel grid ids, DAMASK material.config).
template <typename T>
class shape_input_base {
public:
  using value_type = T;

  shape_input_base() = default;
  shape_input_base(shape_input_base const&) = default;
  shape_input_base(shape_input_base&&) noexcept = default;
  shape_input_base& operator=(shape_input_base const&) = default;
  shape_input_base& operator=(shape_input_base&&) noexcept = default;
  virtual ~shape_input_base() = default;

  // Sample a new candidate shape from this input's distributions. The shape
  // is returned without a bounding box computed; the caller is responsible
  // for calling make_bounding_box() before any geometric query.
  // [[nodiscard]] is mandatory: discarding the return value silently
  // destroys the freshly-allocated candidate.
  [[nodiscard]] virtual std::unique_ptr<shape_base<T>> new_shape() = 0;

  [[nodiscard]] rvegen::info const& info() const noexcept { return _info; }
  [[nodiscard]] rvegen::info& info() noexcept { return _info; }
  void set_info(rvegen::info i) noexcept { _info = std::move(i); }

  // Convenience pair for the phase-name special case.
  [[nodiscard]] std::string phase_name() const { return _info.phase_name(); }
  void set_phase_name(std::string name) {
    _info.set_phase_name(std::move(name));
  }

protected:
  // Concrete inputs call this from their schema-driven ctor body to
  // pull every recognised metadata field out of the handler in one
  // place. New metadata fields are added here once and every input
  // picks them up automatically — that's the point of the generic
  // info container.
  //
  // Two recognised fields:
  //   * `phase_name` (string) — convenience shortcut, stamped onto
  //     shape's info.phase_name.
  //   * `metadata` (JSON-encoded string) — arbitrary key/value pairs
  //     parsed via nlohmann::json and merged into the shape's info.
  //     The string-encoding is a stop-gap until the parameter visitor
  //     gains nested-object passthrough; until then, callers escape
  //     their JSON as a string in the config:
  //       "metadata": "{\"orientation_deg\": 42.5, \"source\": \"a.stl\"}"
  //
  // Order: `phase_name` is read first, then `metadata` is merged on
  // top — so a `phase_name` key inside the metadata object overrides
  // the explicit field.
  //
  // `merge_patch` follows RFC 7396 (JSON Merge Patch): a `null` value
  // in the patch ERASES the corresponding key. So
  // `"metadata": "{\"phase_name\": null}"` removes the phase tag set
  // by the explicit `phase_name` field. Useful for opt-out cases;
  // surprising if the caller expects "set to null" semantics — pass
  // an empty string instead if that's what you want.
  void read_metadata(parameter_handler_t const& handler) {
    if (handler.contains("phase_name")) {
      _info.set_phase_name(handler.template get<std::string>("phase_name"));
    }
    if (handler.contains("metadata")) {
      auto const& blob = handler.template get<std::string>("metadata");
      _info.json().merge_patch(nlohmann::json::parse(blob));
    }
  }

  // Concrete inputs call this from `new_shape()` to stamp the input's
  // metadata blob onto the shape before returning it.
  void tag(shape_base<T>& shape) const {
    shape.set_info(_info);
  }

  rvegen::info _info;
};

} // namespace rvegen
