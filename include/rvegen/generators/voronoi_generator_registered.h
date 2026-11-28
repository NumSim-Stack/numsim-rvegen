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
#include <utility>
#include <vector>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "deterministic_generator.h"
#include "grain_size_control.h"
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
    auto shapes = voronoi_to_shapes<T>(
        voronoi_generator_2d<T>{box[0], box[1], std::move(seeds),
                                _handler.template get<bool>("periodic")},
        this->cancel());
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
    // Grain-size control (grain_size_control.h): giving the mean diameter
    // replaces n_seeds / seeds. The three tuning keys carry no schema
    // default so that giving one without grain_diameter_mean is an error
    // instead of being ignored; their defaults are grain_defaults below.
    s.template insert<T>("grain_diameter_mean")
        .template add<min_only<T{0}>>()
        .template add<numsim_core::description_label<"optional mean equivalent grain diameter. When given, seeds and Laguerre weights are generated so the cell volumes follow a lognormal diameter distribution; the grain count follows from the box volume, and n_seeds / seeds must not be given. Use periodic: true for grains that are not cut by the box faces">>();
    s.template insert<T>("grain_diameter_sigma")
        .template add<min_only<T{0}>>()
        .template add<numsim_core::description_label<"standard deviation of ln(diameter) for grain-size control (default 0.3; convergence is expected up to about 0.8). Needs grain_diameter_mean">>();
    s.template insert<T>("grain_size_tolerance")
        .template add<min_only<T{0}>>()
        .template add<numsim_core::description_label<"largest allowed relative error of a cell volume against its target in grain-size control (> 0, default 0.01); the run fails if it is not reached. Needs grain_diameter_mean">>();
    s.template insert<std::size_t>("grain_size_max_iterations")
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"weight iterations per attempt in grain-size control (default 1000). Needs grain_diameter_mean">>();
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
    const auto backend = detail::voronoi_backend_from_name(
        _handler.template get<std::string>("backend"));
    const bool periodic = _handler.template get<bool>("periodic");
    std::vector<std::array<T, 3>> seeds;
    std::vector<T> weights;
    if (_handler.contains("grain_diameter_mean")) {
      grain_size_seeds(box, backend, periodic, seeds, weights);
    } else {
      for (char const* key : {"grain_diameter_sigma", "grain_size_tolerance",
                              "grain_size_max_iterations"}) {
        if (_handler.contains(key))
          throw std::runtime_error{"voronoi_3d: '" + std::string{key} +
                                   "' needs 'grain_diameter_mean'"};
      }
      auto const flat =
          detail::voronoi_seed_coords<T>(_handler, 3, box, _pipeline_seed);
      for (std::size_t i = 0; i + 2 < flat.size(); i += 3)
        seeds.push_back({flat[i], flat[i + 1], flat[i + 2]});
    }
    auto shapes = voronoi_to_shapes<T>(
        voronoi_generator_3d<T>{box[0], box[1], box[2], std::move(seeds),
                                backend, periodic, std::move(weights)},
        this->cancel());
    detail::tag_phase(shapes, _handler.template get<std::string>("phase_name"));
    return shapes;
  }

private:
  // Seeds + Laguerre weights for the requested lognormal grain diameters.
  void grain_size_seeds(std::array<T, 3> const& box, voronoi_backend backend,
                        bool periodic, std::vector<std::array<T, 3>>& seeds,
                        std::vector<T>& weights) const {
    const bool has_list =
        _handler.contains("seeds") &&
        !_handler.template get<std::vector<double>>("seeds").empty();
    if (has_list || _handler.template get<std::size_t>("n_seeds") != 0) {
      throw std::runtime_error{
          "voronoi_3d: 'grain_diameter_mean' sets the grain count from the "
          "box volume; remove 'n_seeds' / 'seeds'"};
    }
    // Same seeding rule as the plain mode; the engine is the 32-bit
    // mt19937 the grain-size code is written for, so reject wider seeds.
    const std::uint64_t seed = _handler.contains("seed")
                                   ? _handler.template get<std::uint64_t>("seed")
                                   : _pipeline_seed;
    if (!std::in_range<std::uint32_t>(seed)) {
      throw std::runtime_error{
          "voronoi_3d: 'seed' must be <= 4294967295 with grain-size control"};
    }
    std::mt19937 engine{static_cast<std::uint32_t>(seed)};
    progress_options opts;
    if (auto const* cancel = this->cancel(); cancel && *cancel) {
      opts.cancel = *cancel;
    }
    const T sigma = _handler.contains("grain_diameter_sigma")
                        ? _handler.template get<T>("grain_diameter_sigma")
                        : T{0.3};
    const T tolerance = _handler.contains("grain_size_tolerance")
                            ? _handler.template get<T>("grain_size_tolerance")
                            : T{0.01};
    const std::size_t max_iterations =
        _handler.contains("grain_size_max_iterations")
            ? _handler.template get<std::size_t>("grain_size_max_iterations")
            : std::size_t{1000};
    if (!(tolerance > T{0}))
      throw std::runtime_error{"voronoi_3d: 'grain_size_tolerance' must be > 0"};
    auto r = lognormal_grain_seeds_3d<T>(
        box[0], box[1], box[2],
        _handler.template get<T>("grain_diameter_mean"), sigma, engine,
        backend, periodic, tolerance, max_iterations, 2, opts);
    if (r.cancelled) throw generation_cancelled{};
    if (!r.converged) {
      throw std::runtime_error{
          "voronoi_3d: grain-size control did not reach the tolerance (" +
          std::to_string(r.seeds.size()) + " grains, largest relative volume "
          "error " + std::to_string(r.max_rel_volume_error) + " > " +
          std::to_string(tolerance) + ", " + std::to_string(r.empty_cells) +
          " empty cells). Use a smaller grain_diameter_sigma, a larger "
          "grain_size_tolerance or grain_size_max_iterations, or a box that "
          "holds more grains"};
    }
    seeds = std::move(r.seeds);
    weights = std::move(r.weights);
  }

  parameter_handler_t _handler;
  std::uint64_t _pipeline_seed{1};
};

} // namespace rvegen
