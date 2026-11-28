#pragma once

// Registry adapters for the Voronoi tessellations (#84).
//
// Registered names: "voronoi_2d" (convex_polygon cells, needs a 2D
// domain, Lz == 0) and "voronoi_3d" (voronoi_cell cells, needs Lz > 0).
//
// Seeds are either given explicitly (`seeds`: flat [x0,y0,(z0),x1,...])
// or generated: `n_seeds` points uniform in the domain from
// std::mt19937_64{seed}, mapping raw engine output to [0,1) directly
// (no std::uniform_real_distribution, whose sequence is
// implementation-defined), so the layout is identical across standard
// libraries. Same params + seed -> same tessellation.
//
// The generator's own `seed` key is an optional explicit override. When
// absent the engine is seeded from the pipeline's top-level `seed`
// (delivered via deterministic_generator::set_pipeline_seed; 1 when a
// library user never sets it), so configs differing only in the
// top-level seed give different tessellations.
//
// Termination and shape inputs are ignored (deterministic generators,
// see deterministic_generator.h).

#include <array>
#include <cstddef>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "deterministic_generator.h"
#include "voronoi_generator_2d.h"
#include "voronoi_generator_3d.h"

namespace rvegen {

namespace detail {

inline void add_voronoi_fields(parameter_controller_t& s) {
  s.template insert<std::size_t>("n_seeds")
      .template add<numsim_core::set_default>(std::size_t{0})
      .template add<min_only<std::size_t{0}>>()
      .template add<numsim_core::unit_label<"seeds">>()
      .template add<numsim_core::description_label<"number of random seeds (ignored when 'seeds' is given)">>();
  s.template insert<std::uint64_t>("seed")
      .template add<min_only<std::uint64_t{0}>>()
      .template add<numsim_core::description_label<"optional RNG seed override for generated seed points (default: the top-level 'seed')">>();
  s.template insert<std::vector<double>>("seeds")
      .template add<numsim_core::description_label<"optional explicit seed coordinates, flat [x0,y0,(z0),x1,...]">>();
  s.template insert<std::string>("phase_name")
      .template add<numsim_core::set_default>(std::string{})
      .template add<numsim_core::description_label<"optional phase name stamped on every cell">>();
  s.template insert<bool>("periodic")
      .template add<numsim_core::set_default>(false)
      .template add<numsim_core::description_label<"periodic tessellation: cells crossing a face are emitted with their wrapped copies (default false)">>();
}

// "auto" picks Voro++ when this build has it, otherwise the in-house
// tessellator (which refuses large seed counts, see voronoi_generator_3d.h).
[[nodiscard]] inline voronoi_backend voronoi_backend_from_name(std::string const& name) {
  if (name == "auto")
    return voro_pp_available() ? voronoi_backend::voro_pp
                               : voronoi_backend::in_house;
  if (name == "in_house") return voronoi_backend::in_house;
  if (name == "voro_pp") return voronoi_backend::voro_pp;
  throw std::runtime_error{"voronoi_3d: unknown backend '" + name +
                           "' (expected 'auto', 'in_house' or 'voro_pp')"};
}

// Flat coordinate list of seeds in `dim` dimensions inside the box.
template <typename T>
[[nodiscard]] std::vector<T>
voronoi_seed_coords(parameter_handler_t const& h, std::size_t dim,
                    std::array<T, 3> const& box,
                    std::uint64_t pipeline_seed) {
  std::vector<T> flat;
  if (h.contains("seeds")) {
    for (double v : h.template get<std::vector<double>>("seeds"))
      flat.push_back(static_cast<T>(v));
    if (!flat.empty()) {
      if (flat.size() % dim != 0)
        throw std::runtime_error{
            "voronoi: 'seeds' length must be a multiple of " +
            std::to_string(dim)};
      return flat;
    }
  }
  const auto n = h.template get<std::size_t>("n_seeds");
  if (n == 0)
    throw std::runtime_error{
        "voronoi: give 'n_seeds' >= 1 or an explicit 'seeds' list"};
  std::mt19937_64 eng{h.contains("seed")
                          ? h.template get<std::uint64_t>("seed")
                          : pipeline_seed};
  flat.reserve(n * dim);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t d = 0; d < dim; ++d)
      flat.push_back(box[d] * static_cast<T>(
          static_cast<double>(eng() >> 11) * 0x1.0p-53));
  return flat;
}

template <typename Vec>
void tag_phase(Vec& shapes, std::string const& phase) {
  if (phase.empty()) return;
  for (auto& s : shapes) s->set_phase_name(phase);
}

} // namespace detail

template <typename T = double>
class registered_voronoi_2d_generator final
    : public deterministic_generator<T> {
public:
  explicit registered_voronoi_2d_generator(parameter_handler_t const& h)
      : _handler{h} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    detail::add_voronoi_fields(s);
    return s;
  }

  void set_pipeline_seed(std::uint64_t seed) override {
    _pipeline_seed = seed;
  }

protected:
  [[nodiscard]] typename deterministic_generator<T>::shape_vector
  build(std::array<T, 3> const& box) const override {
    if (box[2] != T{0})
      throw std::runtime_error{
          "voronoi_2d: needs a 2D domain (domain_box z == 0)"};
    auto const flat = detail::voronoi_seed_coords<T>(_handler, 2, box, _pipeline_seed);
    std::vector<std::array<T, 2>> seeds;
    for (std::size_t i = 0; i + 1 < flat.size(); i += 2)
      seeds.push_back({flat[i], flat[i + 1]});
    auto shapes = voronoi_to_shapes<T>(voronoi_generator_2d<T>{
        box[0], box[1], std::move(seeds),
        _handler.template get<bool>("periodic")});
    detail::tag_phase(shapes, _handler.template get<std::string>("phase_name"));
    return shapes;
  }

private:
  parameter_handler_t _handler;
  std::uint64_t _pipeline_seed{1};
};

template <typename T = double>
class registered_voronoi_3d_generator final
    : public deterministic_generator<T> {
public:
  explicit registered_voronoi_3d_generator(parameter_handler_t const& h)
      : _handler{h} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    detail::add_voronoi_fields(s);
    s.template insert<std::string>("backend")
        .template add<numsim_core::set_default>(std::string{"auto"})
        .template add<numsim_core::description_label<"tessellation backend: auto, in_house or voro_pp (default auto: Voro++ when built with it)">>();
    return s;
  }

  void set_pipeline_seed(std::uint64_t seed) override {
    _pipeline_seed = seed;
  }

protected:
  [[nodiscard]] typename deterministic_generator<T>::shape_vector
  build(std::array<T, 3> const& box) const override {
    if (box[2] <= T{0})
      throw std::runtime_error{
          "voronoi_3d: needs a 3D domain (domain_box z > 0)"};
    auto const flat = detail::voronoi_seed_coords<T>(_handler, 3, box, _pipeline_seed);
    std::vector<std::array<T, 3>> seeds;
    for (std::size_t i = 0; i + 2 < flat.size(); i += 3)
      seeds.push_back({flat[i], flat[i + 1], flat[i + 2]});
    auto shapes = voronoi_to_shapes<T>(voronoi_generator_3d<T>{
        box[0], box[1], box[2], std::move(seeds),
        detail::voronoi_backend_from_name(
            _handler.template get<std::string>("backend")),
        _handler.template get<bool>("periodic")});
    detail::tag_phase(shapes, _handler.template get<std::string>("phase_name"));
    return shapes;
  }

private:
  parameter_handler_t _handler;
  std::uint64_t _pipeline_seed{1};
};

} // namespace rvegen
