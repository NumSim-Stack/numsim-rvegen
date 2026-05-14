#pragma once

// `info` — a generic string-keyed metadata container attached to shapes
// and shape inputs. Backed by `nlohmann::json` so values can be of any
// JSON type (string, number, bool, object, array) and serialise
// trivially when output writers want to dump the metadata. nlohmann is
// already a transitive rvegen dependency, so the storage choice incurs
// no new dep.
//
// Use cases (today and forward):
//   * Phase tagging — `info.set("phase_name", "fibre")` is the
//     primary use case driving this PR. Convenience accessors
//     `phase_name()` / `set_phase_name()` are provided for the common
//     case so callers don't reach into the json blob directly.
//   * Future metadata: shape ID for tracking, source-file path for
//     mesh inclusions, fibre orientation vector, custom tags emitted
//     by user post-processes — all flow through the same container
//     without further base-class changes.
//
// Why a class wrapper around `nlohmann::json`:
//   * The storage choice can evolve (flat_map, robin-hood) without
//     touching every caller.
//   * Convenience accessors live in one place.
//   * The "metadata" intent is explicit at the type level, distinct
//     from the JSON config blobs that flow through the registry.

#include <string>
#include <utility>

#include <nlohmann/json.hpp>

namespace rvegen {

class info {
public:
  info() = default;
  explicit info(nlohmann::json data) : _data{std::move(data)} {}

  // Generic set/get. `set` accepts any JSON-convertible value;
  // `get<T>` reads a value with type T (throws if absent or the
  // wrong type, matching nlohmann::json's get behaviour).
  template <typename T>
  void set(std::string const& key, T value) {
    _data[key] = std::move(value);
  }

  template <typename T>
  [[nodiscard]] T get(std::string const& key) const {
    return _data.at(key).template get<T>();
  }

  // Best-effort getter with a default — useful for optional fields
  // where the caller doesn't want to handle the missing-key throw.
  template <typename T>
  [[nodiscard]] T get_or(std::string const& key, T fallback) const {
    auto it = _data.find(key);
    if (it == _data.end()) return fallback;
    return it->template get<T>();
  }

  [[nodiscard]] bool contains(std::string const& key) const {
    return _data.contains(key);
  }

  [[nodiscard]] bool empty() const noexcept { return _data.empty(); }

  // Convenience pair for the common phase_name case. Routing through
  // the generic info means future writers can read `phase_name` the
  // same way as any other metadata key.
  [[nodiscard]] std::string phase_name() const {
    return get_or<std::string>("phase_name", {});
  }
  void set_phase_name(std::string name) {
    _data["phase_name"] = std::move(name);
  }

  // Direct json access — for writers that want to emit the whole
  // metadata blob (e.g. as a `metadata` block in DAMASK material.config).
  [[nodiscard]] nlohmann::json const& json() const noexcept { return _data; }
  [[nodiscard]] nlohmann::json& json() noexcept { return _data; }

private:
  nlohmann::json _data = nlohmann::json::object();
};

} // namespace rvegen
