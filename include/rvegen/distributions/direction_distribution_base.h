#pragma once

// Polymorphic interface for distributions that draw unit 3-vectors —
// the orientation analogue of `distribution_base<T>` which draws
// scalars. Concrete implementations include `bingham_distribution`
// (axisymmetric Bingham/Watson) and any future von-Mises-Fisher,
// generic-3x3-Bingham, or fibre-orientation-tensor sampler.
//
// Why a separate base from `distribution_base<T>`:
//   * Return shape differs — scalar vs unit vector. Forcing a single
//     base would require collapsing the vector through three scalar
//     wrappers (which is exactly what `polyline_tube_directional_input`
//     does today, and the documented out-of-scope note in that
//     header explicitly asks for this dedicated base).
//   * Direction sampling is a separate concern at the input layer:
//     consumers like `polyline_tube_oriented_input` need ONE unit
//     vector per produced shape, not three scalar samples that
//     happen to be on a sphere.

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

namespace rvegen {

template <typename T = double>
class direction_distribution_base {
public:
  using value_type = T;

  virtual ~direction_distribution_base() = default;

  // One unit 3-vector per call. Concrete implementations are
  // responsible for ensuring the returned array has magnitude 1
  // (within floating-point tolerance); the consumer treats the
  // result as a direction and does not re-normalise.
  [[nodiscard]] virtual std::array<T, 3> operator()() = 0;
};

// Name → direction-distribution lookup table, parallel to
// `distribution_map_t<T>` for scalar distributions. Inputs that
// need orientation sampling take this map as a ctor argument and
// look up their direction-distribution dependency by name.
template <typename T = double>
using direction_distribution_map_t =
    std::unordered_map<std::string,
                       std::shared_ptr<direction_distribution_base<T>>>;

} // namespace rvegen
