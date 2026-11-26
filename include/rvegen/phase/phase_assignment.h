#pragma once

// phase_assignment — maps generated shapes to named phases (#7) when the
// shape inputs did not set `phase_name` themselves.
//
// JSON (top-level "phase_assignment", names refer to the "phases" array):
//
//   "phase_assignment": {
//     "all_inclusions": "fibre",
//     "by_shape_type": { "circle": "fibre1", "rectangle": "fibre2" },
//     "by_index":      { "0": "matrix", "1-3": "fibre", "4-": "coating" },
//     "matrix": "matrix"
//   }
//
// Semantics:
//   * Only shapes whose phase_name() is empty are touched; a phase_name
//     set by an input always wins.
//   * Precedence, most specific first: by_index > by_shape_type >
//     all_inclusions.
//   * by_index keys address positions in the generated shape vector:
//     "N", "N-M" (inclusive) or "N-" (to the end). With the periodic
//     generator the vector also holds wrap copies, so prefer
//     by_shape_type / all_inclusions there.
//   * Shape-type names: circle, ellipse, rectangle, sphere, box,
//     polyline_tube, convex_polygon, voronoi_cell, mesh_inclusion.
//   * "matrix" documents which declared phase is the matrix; it is
//     validated but assigns nothing (the matrix is whatever no shape
//     covers).
//   * Every referenced phase must exist in the phase_collection; unknown
//     keys / names / shape types throw at load time, not at write time.

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "../shapes/box.h"
#include "../shapes/circle.h"
#include "../shapes/convex_polygon.h"
#include "../shapes/ellipse.h"
#include "../shapes/mesh_inclusion.h"
#include "../shapes/polyline_tube.h"
#include "../shapes/rectangle.h"
#include "../shapes/shape_base.h"
#include "../shapes/sphere.h"
#include "../shapes/voronoi_cell.h"
#include "phase.h"

namespace rvegen {

// Name of the concrete shape type, or "" for an unknown subclass.
template <typename T>
[[nodiscard]] inline std::string shape_type_name(shape_base<T> const& s) {
  if (dynamic_cast<circle<T> const*>(&s)) return "circle";
  if (dynamic_cast<ellipse<T> const*>(&s)) return "ellipse";
  if (dynamic_cast<rectangle<T> const*>(&s)) return "rectangle";
  if (dynamic_cast<sphere<T> const*>(&s)) return "sphere";
  if (dynamic_cast<box<T> const*>(&s)) return "box";
  if (dynamic_cast<polyline_tube<T> const*>(&s)) return "polyline_tube";
  if (dynamic_cast<convex_polygon<T> const*>(&s)) return "convex_polygon";
  if (dynamic_cast<voronoi_cell<T> const*>(&s)) return "voronoi_cell";
  if (dynamic_cast<mesh_inclusion<T> const*>(&s)) return "mesh_inclusion";
  return {};
}

inline bool is_known_shape_type_name(std::string const& n) {
  for (auto const* k : {"circle", "ellipse", "rectangle", "sphere", "box",
                        "polyline_tube", "convex_polygon", "voronoi_cell",
                        "mesh_inclusion"})
    if (n == k) return true;
  return false;
}

template <typename T = double>
class phase_assignment {
public:
  struct index_range {
    std::size_t lo;
    std::size_t hi;   // inclusive
    std::string phase;
  };

  [[nodiscard]] static phase_assignment
  from_json(nlohmann::json const& j, phase_collection<T> const& phases) {
    if (!j.is_object())
      throw std::runtime_error{"phase_assignment: expected an object"};
    phase_assignment a;
    auto require_phase = [&](std::string const& where, nlohmann::json const& v) {
      if (!v.is_string())
        throw std::runtime_error{"phase_assignment: " + where +
                                 " must name a phase (string)"};
      auto name = v.get<std::string>();
      if (!phases.contains(name))
        throw std::runtime_error{"phase_assignment: " + where +
                                 " refers to unknown phase '" + name + "'"};
      return name;
    };
    for (auto const& [key, val] : j.items()) {
      if (key == "all_inclusions") {
        a._all = require_phase("all_inclusions", val);
      } else if (key == "matrix") {
        (void)require_phase("matrix", val);
      } else if (key == "by_shape_type") {
        if (!val.is_object())
          throw std::runtime_error{"phase_assignment: by_shape_type must be an object"};
        for (auto const& [type, ph] : val.items()) {
          if (!is_known_shape_type_name(type))
            throw std::runtime_error{"phase_assignment: unknown shape type '" +
                                     type + "' in by_shape_type"};
          a._by_type.emplace(type, require_phase("by_shape_type." + type, ph));
        }
      } else if (key == "by_index") {
        if (!val.is_object())
          throw std::runtime_error{"phase_assignment: by_index must be an object"};
        for (auto const& [range, ph] : val.items()) {
          auto [lo, hi] = parse_range(range);
          a._by_index.push_back({lo, hi, require_phase("by_index." + range, ph)});
        }
      } else {
        throw std::runtime_error{"phase_assignment: unknown key '" + key + "'"};
      }
    }
    return a;
  }

  // Phase for the shape at `index`, or nullopt if no rule applies.
  [[nodiscard]] std::optional<std::string>
  resolve(shape_base<T> const& s, std::size_t index) const {
    // Later by_index entries override earlier ones on overlap.
    for (auto it = _by_index.rbegin(); it != _by_index.rend(); ++it)
      if (index >= it->lo && index <= it->hi) return it->phase;
    if (!_by_type.empty()) {
      if (auto t = _by_type.find(shape_type_name(s)); t != _by_type.end())
        return t->second;
    }
    return _all;
  }

  // Stamp phase names onto shapes that have none. Returns the number of
  // shapes assigned.
  template <typename ShapeVector>
  std::size_t apply(ShapeVector& shapes) const {
    std::size_t n = 0;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
      auto& s = *shapes[i];
      if (!s.phase_name().empty()) continue;
      if (auto p = resolve(s, i)) {
        s.info().set_phase_name(*p);
        ++n;
      }
    }
    return n;
  }

private:
  static std::pair<std::size_t, std::size_t> parse_range(std::string const& k) {
    auto bad = [&] {
      return std::runtime_error{"phase_assignment: bad by_index key '" + k +
                                "' (expected N, N-M or N-)"};
    };
    auto to_num = [&](std::string const& t) {
      if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos)
        throw bad();
      return static_cast<std::size_t>(std::stoull(t));
    };
    auto dash = k.find('-');
    if (dash == std::string::npos) {
      auto n = to_num(k);
      return {n, n};
    }
    auto lo = to_num(k.substr(0, dash));
    auto rest = k.substr(dash + 1);
    auto hi = rest.empty() ? std::numeric_limits<std::size_t>::max()
                           : to_num(rest);
    if (hi < lo) throw bad();
    return {lo, hi};
  }

  std::optional<std::string> _all;
  std::unordered_map<std::string, std::string> _by_type;
  std::vector<index_range> _by_index;
};

} // namespace rvegen
