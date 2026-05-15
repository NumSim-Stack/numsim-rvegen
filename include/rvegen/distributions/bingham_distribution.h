#pragma once

// Axisymmetric Bingham (Watson) distribution on the 3-sphere — samples
// 3D unit vectors clustered around (κ > 0) or repelled from (κ < 0) a
// caller-specified mean axis.
//
// Density (up to a normalising constant) is f(x) ∝ exp(κ · (x · μ)²)
// for unit vector x and unit mean axis μ. Three regimes:
//   * κ = 0     → uniform on the sphere
//   * κ > 0     → prolate, samples cluster on the ±μ axis
//   * κ < 0     → oblate, samples lie near the equatorial girdle
//                  perpendicular to μ
//
// Why "axisymmetric Bingham" specifically:
//   The general Bingham distribution is parameterised by a 3×3
//   concentration matrix Z. The axisymmetric special case (Z =
//   κ·μμᵀ) covers the use cases relevant to fibre orientation in
//   composites — clustered around a fibre direction (prolate) or
//   in-plane random (oblate) — without exposing the user to the full
//   matrix parameterisation. The general 3×3 Bingham sampler
//   (Kume-Walker) is a follow-up.
//
// Why not inherit `distribution_base<T>`:
//   `distribution_base<T>` returns a scalar T per call. Bingham's
//   natural output is a 3-vector; squashing it through a scalar
//   interface would lose information. Exposing a `sample()` method
//   returning `gte::Vector<3, T>` is the honest API. A parallel
//   `direction_distribution_base` hierarchy + JSON-registry hookup is
//   tracked as a follow-up — until then, 3D fibre inputs that want
//   Bingham orientations construct this directly in C++.
//
// What this header lands today (3D companion to PR #14's 2D von-Mises):
//   * `bingham_distribution<T, Engine>` — ctor taking (mean_axis,
//     kappa, engine).
//   * `sample()` returning a unit `gte::Vector<3, T>`.
//   * Naive rejection sampling on a uniform `cos θ` proposal for the
//     axisymmetric case in both κ > 0 and κ < 0 regimes.
//
// On the sampling algorithm and its useful range:
//   This is not Wood (1994) — Wood uses a tailored proposal density
//   that gives near-100 % acceptance even at very high |κ|. The naive
//   uniform proposal here has acceptance ≈ 1 / √(π |κ| / 2) for large
//   |κ|, so it's efficient up to roughly |κ| ≲ 50 (≥ 10 % accept). At
//   |κ| ≳ 200 each accept costs tens of candidates; at |κ| ≳ 1000 the
//   sampler is unusable. If callers need that regime, replace with a
//   real Wood proposal — tracked as part of the Kume-Walker follow-up.
//
// Out of scope here, ships in follow-up PRs against #6:
//   * Generic 3×3 Bingham (Kume-Walker sampler) — also fixes the
//     high-|κ| performance cliff above.
//   * `direction_distribution_base` hierarchy + JSON registry entry.
//   * 3D fibre input that consumes this for tube orientation.

#include <array>
#include <cmath>
#include <random>

#include <Mathematics/Vector3.h>

#include "direction_distribution_base.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
class bingham_distribution : public direction_distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;
  using vector_type = gte::Vector<3, T>;

  bingham_distribution(vector_type mean_axis, T kappa, engine_type& engine)
      : _kappa{kappa}, _engine{engine}, _uniform{T{0}, T{1}} {
    set_mean_axis(mean_axis);
  }

  // Convenience overload accepting std::array — most callers don't
  // already have a gte::Vector handy.
  bingham_distribution(std::array<T, 3> const& mean_axis, T kappa,
                       engine_type& engine)
      : bingham_distribution{vector_to_gte(mean_axis), kappa, engine} {}

  // Draws one unit 3-vector. Uses rejection sampling: for κ > 0 (prolate)
  // and κ < 0 (oblate) the marginal density of x = cos(θ) is exp(κ·x²)
  // up to normalisation, with θ measured from the mean axis. Sample
  // a candidate x uniform on [-1, 1] and accept with probability
  // exp(κ(x² − 1)) for κ > 0, or exp(κ·x²) for κ < 0 (each scaled to
  // a max of 1). Once x is accepted, φ is uniform on [0, 2π) and the
  // 3-vector is built in the mean-axis-aligned frame and rotated.
  // Polymorphic entry from `direction_distribution_base<T>` — same
  // sample as `sample()` but returned as `std::array<T, 3>` to match
  // the abstract base's signature. Callers that need the native
  // `gte::Vector<3, T>` (for GTE intersection queries etc.) should
  // call `sample()` directly.
  [[nodiscard]] std::array<T, 3> operator()() override {
    const auto v = sample();
    return {v[0], v[1], v[2]};
  }

  [[nodiscard]] vector_type sample() {
    if (std::abs(_kappa) <
        std::numeric_limits<T>::epsilon() * T{1024}) {
      return sample_uniform_sphere();
    }
    T x = T{0};
    while (true) {
      const T u1 = _uniform(_engine.get());
      x = T{2} * u1 - T{1};   // candidate cos θ in [-1, 1]
      const T u2 = _uniform(_engine.get());
      // Acceptance probability normalised so its max over [-1,1] is 1.
      // For κ > 0 the density peaks at x² = 1; bound by exp(0) = 1.
      // For κ < 0 the density peaks at x² = 0; bound by exp(0) = 1.
      const T accept = (_kappa > T{0})
                           ? std::exp(_kappa * (x * x - T{1}))
                           : std::exp(_kappa * (x * x));
      if (u2 <= accept) break;
    }
    const T phi = T{2} * std::numbers::pi_v<T> * _uniform(_engine.get());
    const T sin_theta = std::sqrt(std::max(T{0}, T{1} - x * x));
    vector_type v_local;
    v_local[0] = sin_theta * std::cos(phi);
    v_local[1] = sin_theta * std::sin(phi);
    v_local[2] = x;
    // Rotate so the local z-axis points along the mean axis.
    return _frame_x * v_local[0] + _frame_y * v_local[1] +
           _mean_axis * v_local[2];
  }

  [[nodiscard]] T kappa() const noexcept { return _kappa; }
  [[nodiscard]] vector_type const& mean_axis() const noexcept {
    return _mean_axis;
  }

private:
  static vector_type vector_to_gte(std::array<T, 3> const& a) {
    vector_type v;
    v[0] = a[0]; v[1] = a[1]; v[2] = a[2];
    return v;
  }

  // Builds an orthonormal frame (frame_x, frame_y, mean_axis) so the
  // local-frame "up" matches the user-specified mean axis. Picks an
  // auxiliary vector not parallel to mean_axis to seed the cross
  // products — guards against the degenerate case where mean_axis is
  // close to ẑ.
  void set_mean_axis(vector_type axis) {
    const T n2 = gte::Dot(axis, axis);
    if (n2 > T{0}) axis /= std::sqrt(n2);
    _mean_axis = axis;
    // Pick the auxiliary vector farthest from `_mean_axis` to maximise
    // the cross-product magnitude. Choosing the world axis with the
    // smallest |component| of `_mean_axis` guarantees |aux × axis| ≥
    // √(2/3), comfortably away from the degenerate-parallel case.
    vector_type aux;
    aux[0] = T{0}; aux[1] = T{0}; aux[2] = T{0};
    const T ax = std::abs(_mean_axis[0]);
    const T ay = std::abs(_mean_axis[1]);
    const T az = std::abs(_mean_axis[2]);
    if (ax <= ay && ax <= az) aux[0] = T{1};
    else if (ay <= az) aux[1] = T{1};
    else aux[2] = T{1};
    _frame_x = gte::Cross(_mean_axis, aux);
    const T fx = std::sqrt(gte::Dot(_frame_x, _frame_x));
    // fx ≥ √(2/3) ≈ 0.816 by construction; the tolerance below is a
    // belt-and-braces guard for non-finite inputs.
    if (fx > std::numeric_limits<T>::epsilon() * T{1024}) {
      _frame_x /= fx;
    } else {
      _frame_x[0] = T{1}; _frame_x[1] = T{0}; _frame_x[2] = T{0};
    }
    _frame_y = gte::Cross(_mean_axis, _frame_x);
  }

  vector_type sample_uniform_sphere() {
    // Marsaglia 1972 — uniform on S². Two uniform variates on the
    // unit disc, then map to the sphere.
    while (true) {
      const T u = T{2} * _uniform(_engine.get()) - T{1};
      const T v = T{2} * _uniform(_engine.get()) - T{1};
      const T s = u * u + v * v;
      if (s < T{1} && s > T{0}) {
        const T factor = T{2} * std::sqrt(T{1} - s);
        vector_type out;
        out[0] = u * factor;
        out[1] = v * factor;
        out[2] = T{1} - T{2} * s;
        return out;
      }
    }
  }

  vector_type _mean_axis;
  vector_type _frame_x;
  vector_type _frame_y;
  T _kappa;
  std::reference_wrapper<engine_type> _engine;
  std::uniform_real_distribution<T> _uniform;
};

} // namespace rvegen
