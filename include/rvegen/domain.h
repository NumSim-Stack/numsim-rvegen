#pragma once

// The RVE box: its extents and, explicitly, its dimension (#122).
//
// A 2D RVE is a rectangle [0, Lx] x [0, Ly], a 3D RVE a cuboid
// [0, Lx] x [0, Ly] x [0, Lz]. Until this type existed the box was a bare
// std::array<T, 3> and "2D" was inferred all over the code from Lz == 0.
// That inference now lives here, in one constructor, together with the
// validation of the extents; everything else asks dimension() / is_3d().
//
// Transition: a domain converts implicitly from and to std::array<T, 3>
// (with Lz == 0 meaning 2D), so interfaces could move to the type without
// breaking every caller, and operator[] gives the extent along an axis.
// Do not use the extent to find the dimension (`box[2] > 0`): a test in
// the suite fails the build tree on such a line.

#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "util/format_number.h"

namespace rvegen {

template <typename T = double>
class domain {
public:
  using value_type = T;

  // The unit square.
  constexpr domain() noexcept = default;

  // 2D: domain{Lx, Ly}. 3D: domain{Lx, Ly, Lz}. Lz == 0 means 2D.
  // Throws std::invalid_argument unless the extents are finite, Lx and Ly
  // are > 0 and Lz is >= 0.
  constexpr domain(T lx, T ly, T lz = T{0})
      : domain(std::array<T, 3>{lx, ly, lz}) {}

  // Implicit on purpose (see "Transition" above).
  constexpr domain(std::array<T, 3> const& extents)   // NOLINT(google-explicit-constructor)
      : _extents{extents} {
    for (T const e : extents) {
      // Finite: neither NaN (fails both comparisons) nor +-inf.
      if (!(e >= std::numeric_limits<T>::lowest() &&
            e <= std::numeric_limits<T>::max()))
        throw std::invalid_argument{"domain_box entries must be finite, got " +
                                    describe(extents)};
    }
    if (!(extents[0] > T{0}) || !(extents[1] > T{0}))
      throw std::invalid_argument{
          "domain_box x and y extents must be > 0, got " + describe(extents)};
    if (extents[2] < T{0})
      throw std::invalid_argument{
          "domain_box z extent must be >= 0 (0 or absent means 2D), got " +
          describe(extents)};
    _dimension = extents[2] > T{0} ? 3 : 2;
  }

  // From the config's "domain_box": [Lx, Ly] (2D), [Lx, Ly, Lz] (3D), or
  // [Lx, Ly, 0] (2D, the older spelling).
  [[nodiscard]] static domain from_json(nlohmann::json const& j) {
    if (!j.is_array() || (j.size() != 2 && j.size() != 3))
      throw std::invalid_argument{
          "domain_box must be an array of 2 or 3 numbers, got " + j.dump()};
    std::array<T, 3> e{T{0}, T{0}, T{0}};
    for (std::size_t i = 0; i < j.size(); ++i) {
      if (!j[i].is_number())
        throw std::invalid_argument{
            "domain_box entries must be numbers, got " + j.dump()};
      e[i] = j[i].template get<T>();
    }
    return domain{e};
  }

  [[nodiscard]] constexpr int dimension() const noexcept { return _dimension; }
  [[nodiscard]] constexpr bool is_2d() const noexcept { return _dimension == 2; }
  [[nodiscard]] constexpr bool is_3d() const noexcept { return _dimension == 3; }

  // {Lx, Ly, Lz}; Lz is 0 for a 2D domain.
  [[nodiscard]] constexpr std::array<T, 3> const& extents() const noexcept {
    return _extents;
  }
  [[nodiscard]] constexpr T operator[](std::size_t axis) const noexcept {
    return _extents[axis];
  }
  constexpr operator std::array<T, 3> const&() const noexcept {   // NOLINT
    return _extents;
  }

  // Area (2D) or volume (3D) of the box.
  [[nodiscard]] constexpr T measure() const noexcept {
    return is_3d() ? _extents[0] * _extents[1] * _extents[2]
                   : _extents[0] * _extents[1];
  }

  [[nodiscard]] friend constexpr bool operator==(domain const& a,
                                                 domain const& b) noexcept {
    return a._extents == b._extents;
  }

private:
  [[nodiscard]] static std::string describe(std::array<T, 3> const& e) {
    return "[" + format_number(e[0]) + ", " + format_number(e[1]) + ", " +
           format_number(e[2]) + "]";
  }

  std::array<T, 3> _extents{T{1}, T{1}, T{0}};
  int _dimension{2};
};

} // namespace rvegen
