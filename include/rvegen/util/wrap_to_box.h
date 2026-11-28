#pragma once

// The one place that maps a coordinate into a periodic box (#135).

#include <cmath>

namespace rvegen {

// The periodic image of x in [0, L). L must be finite and positive.
//
//   * std::fmod is exact, so a coordinate that is already inside comes back
//     unchanged, bit for bit, and one that is at most one period outside is
//     moved by exactly +-L (rounded once).
//   * A negative x so small that x + L rounds to L (-1e-20 in a unit box)
//     lies just below the upper face; it becomes the largest value below L,
//     not 0, so it stays on its side of the face.
//   * x == L, x == -L and x == 0 all give 0 (a point on a face belongs to
//     the cell that starts there).
template <typename T>
[[nodiscard]] inline T wrap_to_box(T x, T L) noexcept {
  T r = std::fmod(x, L);
  if (r < T{0}) {
    r += L;
    if (!(r < L)) r = std::nextafter(L, T{0});
  }
  return r;
}

} // namespace rvegen
