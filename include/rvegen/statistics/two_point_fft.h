#pragma once

// Exact two-point statistics of a voxel grid by FFT (#127).
//
// For two sets of voxels A and B, the number of voxel pairs (p, p + l)
// with p in A and p + l in B is, for every lag l at once, the correlation
// of the two indicator arrays, which an FFT gives in O(N log N). The
// numbers are counts, so the result of the inverse transform is rounded to
// integers: what comes out does not depend on the FFT, the compiler or the
// machine, and no random sampling is involved.
//
//   periodic axis      the lag wraps around (circular correlation): every
//                      lag has the same number of pairs, the voxel count.
//   non-periodic axis  only pairs with both voxels inside the box count;
//                      the array is zero-padded so nothing wraps, and a
//                      lag l has (n - |l|) pairs along that axis.
//
// The two-point probability at a lag is count / pairs. `two_point_profile`
// is that ratio taken over all lags of a distance bin (the radial average)
// and over the lags along each axis (the directional profiles).

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../distributions/portable_random.h"
#include "../generators/cancel_poll.h"
#include "../postprocess/voxel_grid.h"
#include "../util/fft.h"

namespace rvegen {

// S2 against the distance. Bin b covers lags of length in
// [b, b + 1) * max_r / nbins, lag zero excepted: it has its own value,
// `zero`. A bin (or an axis within a bin) that holds no lag is NaN: with
// bins narrower than a voxel that happens.
struct two_point_profile {
  double zero{0.0};                  // lag 0: the voxel fraction for A = B
  std::vector<double> radial;        // all directions
  std::array<std::vector<double>, 3> axis;   // lags along x, y, z only
};

// Centre of bin b of `nbins` bins up to max_r.
[[nodiscard]] inline double two_point_bin_centre(std::size_t b, std::size_t nbins,
                                                 double max_r) noexcept {
  return max_r * (double(b) + 0.5) / double(nbins);
}

// How the box ends: `periodic` wraps every lag around (right for an RVE
// generated periodic), `bounded` counts only pairs inside the box (right
// for everything else, and unbiased for a periodic RVE too).
enum class two_point_boundary { periodic, bounded };

// Correlates indicator sets of one voxel grid. Build it once per grid and
// boundary treatment; transform() each set, correlate() any two.
class voxel_correlator {
public:
  using spectrum = std::vector<std::complex<double>>;

  // n: voxels per axis (nz = 1 for a 2D grid); extent: the box edges;
  // nbins bins up to max_r. Lags longer than max_r are never needed, which
  // bounds the padding of a non-periodic axis.
  voxel_correlator(std::array<std::size_t, 3> const& n,
                   std::array<double, 3> const& extent,
                   std::array<bool, 3> const& periodic, double max_r,
                   std::size_t nbins)
      : _n{n}, _periodic{periodic}, _max_r{max_r}, _nbins{nbins} {
    if (nbins == 0) throw std::invalid_argument{"two-point statistics: nbins must be >= 1"};
    if (!(max_r > 0.0) || !std::isfinite(max_r))
      throw std::invalid_argument{"two-point statistics: the largest distance must be > 0"};
    _voxels = checked_voxel_count(n[0], n[1], n[2]);
    std::uint64_t padded = 1;
    for (std::size_t a = 0; a < 3; ++a) {
      // A flat axis (one voxel) has the single lag 0.
      _d[a] = n[a] > 1 || extent[a] > 0.0 ? extent[a] / double(n[a]) : 0.0;
      std::size_t reach = 0;   // largest |lag| that can lie within max_r
      if (n[a] > 1 && _d[a] > 0.0) {
        const double lags = std::floor(max_r / _d[a]);
        reach = lags >= double(n[a] - 1) ? n[a] - 1 : std::size_t(lags);
      }
      if (periodic[a]) {
        _m[a] = n[a];
        // Each lag once: l and l - n are the same lag.
        _hi[a] = std::min(reach, n[a] / 2);
        _lo[a] = std::min(reach, (n[a] - 1) / 2);
      } else {
        _m[a] = n[a] == 1 ? 1
                          : rvegen_pocketfft::detail::util::good_size_real(
                                n[a] + reach);
        _hi[a] = _lo[a] = reach;
      }
      padded *= _m[a];
      if (padded > max_voxel_count) {
        throw std::invalid_argument{
            "two-point statistics: the padded grid of " +
            std::to_string(_m[0]) + " x " + std::to_string(_m[1]) + " x " +
            std::to_string(_m[2]) + " exceeds the 2^32 voxel limit"};
      }
    }
    _padded = static_cast<std::size_t>(padded);
    _half = _m[2] * _m[1] * (_m[0] / 2 + 1);
  }

  // Doubles this object holds at a time while correlating, per stored
  // spectrum and for the work arrays: what a caller can show as memory.
  [[nodiscard]] std::size_t padded_voxels() const noexcept { return _padded; }
  [[nodiscard]] std::array<std::size_t, 3> const& padded_shape() const noexcept {
    return _m;
  }

  // Forward transform of the indicator of {voxels where in(grid value)}.
  // `grid` is nx * ny * nz values in (k, j, i) order.
  template <typename Pred>
  [[nodiscard]] spectrum transform(std::vector<phase_id> const& grid,
                                   Pred&& in, cancel_fn const* cancel = nullptr) const {
    if (grid.size() != _voxels)
      throw std::invalid_argument{"two-point statistics: grid size does not match"};
    std::vector<double> real(_padded, 0.0);
    for (std::size_t k = 0; k < _n[2]; ++k) {
      poll_cancel(cancel);
      for (std::size_t j = 0; j < _n[1]; ++j) {
        phase_id const* const from = grid.data() + (k * _n[1] + j) * _n[0];
        double* const to = real.data() + (k * _m[1] + j) * _m[0];
        for (std::size_t i = 0; i < _n[0]; ++i) to[i] = in(from[i]) ? 1.0 : 0.0;
      }
    }
    spectrum out(_half);
    poll_cancel(cancel);
    rvegen_pocketfft::r2c(shape(), real_strides(), complex_strides(), axes(),
                          rvegen_pocketfft::FORWARD, real.data(), out.data(),
                          1.0);
    return out;
  }

  // The profile of the pair counts of A (first argument) against B.
  [[nodiscard]] two_point_profile correlate(spectrum const& a, spectrum const& b,
                                            cancel_fn const* cancel = nullptr) const {
    if (a.size() != _half || b.size() != _half)
      throw std::invalid_argument{"two-point statistics: spectrum size does not match"};
    spectrum product(_half);
    for (std::size_t q = 0; q < _half; ++q) {   // conj(a) * b
      const double ar = a[q].real(), ai = a[q].imag();
      const double br = b[q].real(), bi = b[q].imag();
      product[q] = {ar * br + ai * bi, ar * bi - ai * br};
    }
    std::vector<double> counts(_padded);
    poll_cancel(cancel);
    rvegen_pocketfft::c2r(shape(), complex_strides(), real_strides(), axes(),
                          rvegen_pocketfft::BACKWARD, product.data(),
                          counts.data(), 1.0 / double(_padded));
    poll_cancel(cancel);
    return reduce(counts);
  }

private:
  using shape_t = rvegen_pocketfft::shape_t;
  using stride_t = rvegen_pocketfft::stride_t;

  // Arrays are indexed (z, y, x) with x fastest; the real transform runs
  // along x, which is the axis the spectrum stores half of.
  [[nodiscard]] shape_t shape() const { return {_m[2], _m[1], _m[0]}; }
  [[nodiscard]] static shape_t axes() { return {0, 1, 2}; }
  [[nodiscard]] stride_t real_strides() const {
    const auto s = static_cast<std::ptrdiff_t>(sizeof(double));
    return {s * static_cast<std::ptrdiff_t>(_m[1] * _m[0]),
            s * static_cast<std::ptrdiff_t>(_m[0]), s};
  }
  [[nodiscard]] stride_t complex_strides() const {
    const auto s = static_cast<std::ptrdiff_t>(sizeof(std::complex<double>));
    const auto row = static_cast<std::ptrdiff_t>(_m[0] / 2 + 1);
    return {s * static_cast<std::ptrdiff_t>(_m[1]) * row, s * row, s};
  }

  // Pair counts (as the inverse transform left them) to the profile.
  [[nodiscard]] two_point_profile reduce(std::vector<double> const& counts) const {
    using portable_detail::rounded;
    const double dr = _max_r / double(_nbins);
    // A bin holds at most every lag once, each with at most N pairs, and
    // N <= 2^32: the sums stay below 2^64.
    struct tally {
      std::vector<std::uint64_t> count, pairs;
      explicit tally(std::size_t n) : count(n, 0), pairs(n, 0) {}
    };
    tally radial{_nbins};
    std::array<tally, 3> along{tally{_nbins}, tally{_nbins}, tally{_nbins}};
    two_point_profile out;
    double worst = 0.0;

    // Signed lag -> index in the (padded or circular) array.
    auto index = [](std::ptrdiff_t l, std::size_t m) {
      return l >= 0 ? static_cast<std::size_t>(l)
                    : m - static_cast<std::size_t>(-l);
    };
    auto pairs_along = [&](std::ptrdiff_t l, std::size_t a) {
      const auto len = static_cast<std::uint64_t>(_n[a]);
      return _periodic[a] ? len : len - static_cast<std::uint64_t>(l < 0 ? -l : l);
    };
    const std::array<std::ptrdiff_t, 3> lo{-static_cast<std::ptrdiff_t>(_lo[0]),
                                           -static_cast<std::ptrdiff_t>(_lo[1]),
                                           -static_cast<std::ptrdiff_t>(_lo[2])};
    const std::array<std::ptrdiff_t, 3> hi{static_cast<std::ptrdiff_t>(_hi[0]),
                                           static_cast<std::ptrdiff_t>(_hi[1]),
                                           static_cast<std::ptrdiff_t>(_hi[2])};
    for (std::ptrdiff_t lz = lo[2]; lz <= hi[2]; ++lz) {
      const double z = double(lz) * _d[2];
      const double z2 = rounded(z * z);
      const std::uint64_t pz = pairs_along(lz, 2);
      for (std::ptrdiff_t ly = lo[1]; ly <= hi[1]; ++ly) {
        const double y = double(ly) * _d[1];
        const double yz2 = rounded(y * y) + z2;
        const std::uint64_t pyz = pairs_along(ly, 1) * pz;
        double const* const row =
            counts.data() + (index(lz, _m[2]) * _m[1] + index(ly, _m[1])) * _m[0];
        for (std::ptrdiff_t lx = lo[0]; lx <= hi[0]; ++lx) {
          const double x = double(lx) * _d[0];
          const double r = std::sqrt(rounded(x * x) + yz2);
          if (!(r < _max_r)) continue;
          const double value = row[index(lx, _m[0])];
          const std::uint64_t pairs = pairs_along(lx, 0) * pyz;
          // A count lies in [0, pairs]; anything else (NaN included) is a
          // transform that has lost the integers, reported below.
          const double nearest = std::nearbyint(value);
          if (!(nearest >= 0.0) || !(nearest <= double(pairs))) {
            worst = 1.0;
            continue;
          }
          worst = std::max(worst, std::abs(value - nearest));
          const auto count = static_cast<std::uint64_t>(nearest);
          if (lx == 0 && ly == 0 && lz == 0) {
            out.zero = double(count) / double(pairs);
            continue;
          }
          const std::size_t bin =
              std::min(static_cast<std::size_t>(r / dr), _nbins - 1);
          radial.count[bin] += count;
          radial.pairs[bin] += pairs;
          const int zeros = (lx == 0) + (ly == 0) + (lz == 0);
          if (zeros == 2) {
            auto& t = along[lx != 0 ? 0 : (ly != 0 ? 1 : 2)];
            t.count[bin] += count;
            t.pairs[bin] += pairs;
          }
        }
      }
    }
    // The counts are integers; a transform that is this far off one has
    // lost them, and the numbers would no longer be exact.
    if (!(worst < 0.25)) {
      throw std::runtime_error{
          "two-point statistics: the FFT pair counts are not integers "
          "(off by " + std::to_string(worst) + "); the grid is too large"};
    }
    auto ratio = [&](tally const& t) {
      std::vector<double> v(_nbins, std::numeric_limits<double>::quiet_NaN());
      for (std::size_t b = 0; b < _nbins; ++b)
        if (t.pairs[b] > 0) v[b] = double(t.count[b]) / double(t.pairs[b]);
      return v;
    };
    out.radial = ratio(radial);
    for (std::size_t a = 0; a < 3; ++a)
      if (_n[a] > 1) out.axis[a] = ratio(along[a]);
    return out;
  }

  std::array<std::size_t, 3> _n;
  std::array<bool, 3> _periodic;
  double _max_r;
  std::size_t _nbins;
  std::size_t _voxels{0};
  std::array<double, 3> _d{};
  std::array<std::size_t, 3> _m{};        // array lengths (padded)
  std::array<std::size_t, 3> _lo{}, _hi{};   // lags run from -_lo to _hi
  std::size_t _padded{0}, _half{0};
};

} // namespace rvegen
