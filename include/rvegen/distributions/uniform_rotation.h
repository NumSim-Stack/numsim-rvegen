#pragma once

// Uniformly random 3D rotations (Haar measure on SO(3)) via Shoemake's
// uniform-quaternion construction (K. Shoemake, "Uniform random
// rotations", Graphics Gems III, 1992). Given three independent
// U(0,1) samples u1, u2, u3,
//
//   q = ( sqrt(1−u1)·sin(2π·u2), sqrt(1−u1)·cos(2π·u2),
//         sqrt(u1)·sin(2π·u3),   sqrt(u1)·cos(2π·u3) )      (x, y, z, w)
//
// is uniformly distributed on the unit 3-sphere, so the rotation it
// represents is uniform on SO(3): rotated particles are isotropic
// (R·e has a uniform distribution on S² for any fixed unit e).
//
// Euler angles drawn uniformly are NOT isotropic (they cluster at the
// poles); use this instead whenever "random orientation" is meant.

#include <cmath>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

#include <Mathematics/Matrix3x3.h>
#include <Mathematics/Quaternion.h>
#include <Mathematics/Rotation.h>

#include "portable_random.h"

namespace rvegen {

template <typename T>
[[nodiscard]] gte::Quaternion<T> uniform_random_quaternion(T u1, T u2, T u3) {
  for (T u : {u1, u2, u3}) {
    if (!(u >= T{0} && u <= T{1})) {
      throw std::runtime_error{
          "uniform_random_quaternion: samples must lie in [0, 1], got " +
          std::to_string(u)};
    }
  }
  const T two_pi = T{2} * std::numbers::pi_v<T>;
  const T a = std::sqrt(T{1} - u1);
  const T b = std::sqrt(u1);
  return gte::Quaternion<T>(a * std::sin(two_pi * u2), a * std::cos(two_pi * u2),
                            b * std::sin(two_pi * u3), b * std::cos(two_pi * u3));
}

template <typename T>
[[nodiscard]] gte::Matrix3x3<T> uniform_random_rotation(T u1, T u2, T u3) {
  return static_cast<gte::Matrix3x3<T>>(
      gte::Rotation<3, T>(uniform_random_quaternion(u1, u2, u3)));
}

template <typename T = double, typename Engine>
[[nodiscard]] gte::Matrix3x3<T> uniform_random_rotation(Engine& engine) {
  portable_uniform_real<T> u{T{0}, T{1}};
  const T u1 = u(engine), u2 = u(engine), u3 = u(engine);
  return uniform_random_rotation(u1, u2, u3);
}

} // namespace rvegen
