#pragma once

// Phase name <-> id table sidecar, shared by the voxel and VTK writers.
//
// Legacy VTK allows a single title line of at most 256 characters, so a
// name table cannot live in the VTK file itself. Both file-writing
// grid writers therefore write `<output_path>.phases.json` next to their
// output when a phase_collection is attached:
//
//   {
//     "format": "rvegen-phases v1",
//     "untagged_id": 0,
//     "matrix": "matrix",            // or null when none is declared
//     "phases": [ {"id": 1, "name": "matrix"}, {"id": 2, "name": "fibre"} ]
//   }
//
// Ids are exactly phase_collection::id_of, i.e. the ones in the grid.

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "../phase/phase.h"

namespace rvegen {

[[nodiscard]] inline std::string phase_table_path(std::string const& output_path) {
  return output_path + ".phases.json";
}

template <typename T>
[[nodiscard]] inline nlohmann::json
phase_table_json(phase_collection<T> const& phases) {
  nlohmann::json j;
  j["format"] = "rvegen-phases v1";
  j["untagged_id"] = 0;
  j["matrix"] = phases.has_matrix() ? nlohmann::json(phases.matrix_name())
                                    : nlohmann::json(nullptr);
  auto arr = nlohmann::json::array();
  for (auto const* p : phases.ordered())
    arr.push_back({{"id", p->id}, {"name", p->name}});
  j["phases"] = std::move(arr);
  return j;
}

template <typename T>
inline void write_phase_table_sidecar(std::string const& output_path,
                                      phase_collection<T> const& phases) {
  const auto path = phase_table_path(output_path);
  std::ofstream out{path};
  if (!out)
    throw std::runtime_error{"cannot open '" + path + "' for writing"};
  out << phase_table_json(phases).dump(2) << '\n';
}

} // namespace rvegen
