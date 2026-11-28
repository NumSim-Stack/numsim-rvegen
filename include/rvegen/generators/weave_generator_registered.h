#pragma once

// Registry adapters for the deterministic textile generators (#73).
//
// `weave_generator` / `tow_weave_generator` keep their typed C++ ctors;
// these thin adapters add the schema + `(parameter_handler_t const&)`
// ctor the generator registry expects and defer building to compute(),
// where the domain box is available.
//
// Registered names: "weave", "tow_weave". Termination and shape inputs
// are ignored (see deterministic_generator.h).

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "deterministic_generator.h"
#include "tow_weave_generator.h"
#include "weave_generator.h"

namespace rvegen {

namespace detail {

// Required keys shared by both adapters.
template <typename T>
[[nodiscard]] parameter_controller_t weave_required() {
  parameter_controller_t s;
  s.template required<std::size_t>("n_warp_yarns")
      .template add<min_only<std::size_t{1}>>()
      .template add<numsim_core::description_label<"number of yarns running along +x">>();
  s.template required<std::size_t>("n_weft_yarns")
      .template add<min_only<std::size_t{1}>>()
      .template add<numsim_core::description_label<"number of yarns running along +y">>();
  s.template required<T>("yarn_radius")
      .template add<min_only<T{0}>>()
      .template add<numsim_core::description_label<"yarn (tube) radius; must be > 0">>();
  s.template required<T>("amplitude")
      .template add<min_only<T{0}>>()
      .template add<numsim_core::description_label<"peak z-undulation of the yarn centerlines; must exceed yarn_radius in 3D (0 for a flat 2D weave)">>();
  return s;
}

// Optional knobs shared by both adapters (the handler always carries
// them: optional<> stores the default).
inline void add_weave_optionals(parameter_controller_t& s) {
  s.template optional<std::size_t>("n_segments_per_yarn", std::size_t{32})
      .template add<min_only<std::size_t{2}>>()
      .template add<numsim_core::description_label<"centerline samples per yarn (>= 2, default 32)">>();
  s.template optional<std::string>("pattern", std::string{"plain"})
      .template add<numsim_core::description_label<"interlacing pattern: plain, twill_2_2 or satin_5 (default plain)">>();
  s.template optional<std::string>("warp_phase_name", std::string{})
      .template add<numsim_core::description_label<"optional phase name for warp yarns">>();
  s.template optional<std::string>("weft_phase_name", std::string{})
      .template add<numsim_core::description_label<"optional phase name for weft yarns">>();
}

[[nodiscard]] inline weave_pattern weave_pattern_from_name(std::string const& name) {
  if (name == "plain") return weave_pattern::plain;
  if (name == "twill_2_2") return weave_pattern::twill_2_2;
  if (name == "satin_5") return weave_pattern::satin_5;
  throw std::runtime_error{"weave: unknown pattern '" + name +
                           "' (expected 'plain', 'twill_2_2' or 'satin_5')"};
}

template <typename T>
[[nodiscard]] weave_generator<T>
make_weave(parameter_handler_t const& h, rvegen::domain<T> const& domain) {
  weave_generator<T> w{domain,
                       h.template get<std::size_t>("n_warp_yarns"),
                       h.template get<std::size_t>("n_weft_yarns"),
                       h.template get<T>("yarn_radius"),
                       h.template get<T>("amplitude"),
                       h.template get<std::size_t>("n_segments_per_yarn"),
                       weave_pattern_from_name(
                           h.template get<std::string>("pattern"))};
  w.set_warp_phase_name(h.template get<std::string>("warp_phase_name"));
  w.set_weft_phase_name(h.template get<std::string>("weft_phase_name"));
  return w;
}

} // namespace detail

template <typename T = double>
class registered_weave_generator final : public deterministic_generator<T> {
public:
  explicit registered_weave_generator(parameter_handler_t const& handler)
      : _handler{handler} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    auto s = detail::weave_required<T>();
    detail::add_weave_optionals(s);
    return s;
  }

protected:
  [[nodiscard]] typename deterministic_generator<T>::shape_vector
  build(domain<T> const& domain_box) const override {
    return detail::make_weave<T>(_handler, domain_box).build(this->cancel());
  }

private:
  parameter_handler_t _handler;
};

template <typename T = double>
class registered_tow_weave_generator final : public deterministic_generator<T> {
public:
  explicit registered_tow_weave_generator(parameter_handler_t const& handler)
      : _handler{handler} {}

  [[nodiscard]] static parameter_controller_t parameters() {
    auto s = detail::weave_required<T>();
    detail::add_weave_optionals(s);
    s.template required<std::size_t>("n_fibres_per_tow")
        .template add<min_only<std::size_t{1}>>()
        .template add<numsim_core::description_label<"fibres hex-packed into each yarn cross-section">>();
    s.template optional<T>("fibre_radius_factor", T{0.2})
        .template add<numsim_core::description_label<"fibre radius as a fraction of the yarn radius, in (0, 1) (default 0.2)">>();
    return s;
  }

protected:
  [[nodiscard]] typename deterministic_generator<T>::shape_vector
  build(domain<T> const& domain_box) const override {
    tow_weave_generator<T> g{
        detail::make_weave<T>(_handler, domain_box),
        _handler.template get<std::size_t>("n_fibres_per_tow"),
        _handler.template get<T>("fibre_radius_factor")};
    return g.build(this->cancel());
  }

private:
  parameter_handler_t _handler;
};

} // namespace rvegen
