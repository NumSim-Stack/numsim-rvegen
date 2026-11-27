#pragma once

// Axisymmetric Bingham (Watson) distribution on the 2-sphere S² — samples
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
// What this header provides (3D companion to the 2D von-Mises):
//   * `bingham_distribution<T, Engine>` — ctor taking (mean_axis,
//     kappa, engine), plus a schema-driven ctor for the JSON registry.
//   * `sample()` returning a unit `gte::Vector<3, T>`.
//   * Exact rejection samplers with O(1) expected cost per draw for every
//     supported kappa (one branch per regime, below).
//
// Prolate branch (κ > 0):
//   Let t = 1 − |cos θ|. The target exp(−κ t (2 − t)) is bounded by the
//   truncated exponential exp(−κ t) on [0, 1], which is sampled exactly by
//   inversion (log1p/expm1, so there is no loss of precision for tiny or
//   huge κ); the sign of cos θ is a fair coin. Acceptance probability is
//   exp(−κ t (1 − t)), overall rate ≥ (1 + e^−κ)/2 ≥ 1/2. sin²θ is formed
//   as t (2 − t) rather than 1 − cos²θ, so the transverse spread ~ 1/κ is
//   not lost to cancellation when κ is large.
//
// Oblate branch (κ ≤ −1): cos θ is exactly a normal with variance
//   1/(2|κ|) truncated to [−1, 1]; rejecting |x| > 1 has acceptance
//   erf(√|κ|) ≥ 0.84.
//
// Weakly oblate branch (−1 < κ < 0): uniform cos θ envelope with
//   acceptance ∫₀¹ exp(κx²) dx ≥ 0.37.
//
// Supported κ range and iteration cap:
//   * κ ∈ [−1e10, 1e10] (`min_kappa` / `max_kappa`); anything outside
//     (or non-finite) throws std::invalid_argument at construction. The
//     bounds are not sampler limits (both branches are O(1) and stable
//     beyond them) but the range that was verified against quadrature
//     (second moment, and the transverse spread sin²θ ≈ 1/κ at the prolate
//     end, with no overflow or loss of 1/κ precision) and that is
//     physically meaningful: at |κ| = 1e10 the angular spread is already
//     ~1e-5 rad, far finer than any orientation input needs.
//   * Each sample() is capped at a number of candidates derived from the
//     branch's acceptance lower bound p, so that (1 − p)^cap ≤ e⁻¹⁰⁰: a
//     spurious std::runtime_error cannot happen in practice and the cap
//     only fires on a genuine numerical breakdown. `set_max_iterations`
//     overrides it (tests use a tiny cap to exercise the throw path).
//
// Not covered: the general 3×3 Bingham (Kume-Walker sampler).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <tuple>

#include <Mathematics/Vector3.h>
#include <numsim-core/input_parameter_controller.h>

#include "direction_distribution_base.h"
#include "../schema/field_list.h"
#include "../types.h"

namespace rvegen {

template <typename T = double, typename Engine = std::mt19937>
class bingham_distribution : public direction_distribution_base<T> {
public:
  using value_type = T;
  using engine_type = Engine;
  using vector_type = gte::Vector<3, T>;

  bingham_distribution(vector_type mean_axis, T kappa, engine_type& engine)
      : _kappa{kappa}, _engine{engine}, _uniform{T{0}, T{1}},
        _normal{T{0}, T{1}} {
    if (!std::isfinite(_kappa)) {
      throw std::invalid_argument{
          "bingham_distribution: kappa must be finite"};
    }
    if (_kappa <= T{-1}) {
      _normal = std::normal_distribution<T>{
          T{0}, std::sqrt(T{1} / (T{2} * -_kappa))};
    }
    if (_kappa > max_kappa) {
      throw std::invalid_argument{
          "bingham_distribution: kappa above 1e10 is outside the supported "
          "range [-1e10, 1e10] (angular spread already ~1e-5 rad at the "
          "bound)"};
    }
    if (_kappa < min_kappa) {
      throw std::invalid_argument{
          "bingham_distribution: kappa below -1e10 is outside the supported "
          "range [-1e10, 1e10] (girdle thickness already ~1e-5 rad at the "
          "bound)"};
    }
    set_mean_axis(mean_axis);
    // Acceptance lower bound p of the branch used (header comment); the
    // cap is 100 / -ln(1 - p) candidates, i.e. (1 - p)^cap <= e^-100.
    const T p_min = _kappa > T{0}    ? T{0.5}
                    : _kappa <= T{-1} ? T{0.84}
                                      : T{0.37};
    _max_iterations = static_cast<std::size_t>(
        std::ceil(T{100} / -std::log1p(-p_min)));
  }

  // Supported concentration range; see the header comment.
  static constexpr T max_kappa = T{1e10};
  static constexpr T min_kappa = T{-1e10};

  // Convenience overload accepting std::array — most callers don't
  // already have a gte::Vector handy.
  bingham_distribution(std::array<T, 3> const& mean_axis, T kappa,
                       engine_type& engine)
      : bingham_distribution{vector_to_gte(mean_axis), kappa, engine} {}

  using fields = field_list<
      field<"mean_axis_x", T, true,
            numsim_core::description_label<"x-component of the Bingham mean axis (renormalised internally)">>,
      field<"mean_axis_y", T, true,
            numsim_core::description_label<"y-component of the Bingham mean axis (renormalised internally)">>,
      field<"mean_axis_z", T, true,
            numsim_core::description_label<"z-component of the Bingham mean axis (renormalised internally)">>,
      field<"kappa", T, true,
            numsim_core::description_label<"Bingham concentration: κ > 0 = prolate (samples cluster on ±mean_axis), κ < 0 = oblate (samples lie near the great circle perpendicular to mean_axis), κ ≈ 0 = uniform on the sphere. Supported range [-1e10, 1e10]; values outside are rejected">>>;

  // Schema-driven ctor for the JSON registry path. Mean axis is
  // declared as three scalar fields rather than a vector field
  // because `parameter_handler<string, any>` doesn't carry a typed
  // array slot — three scalars is the rvegen idiom (matches
  // polyline_tube_directional_input). The axis is renormalised
  // inside the helper ctor, so the user can drop unit-vector
  // hygiene from their config.
  bingham_distribution(parameter_handler_t const& handler,
                       engine_type& engine)
      : bingham_distribution(fields::extract(handler), engine) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    return fields::schema();
  }

  // Draws one unit 3-vector: x = cos(θ) (θ measured from the mean axis)
  // has density ∝ exp(κ·x²) and is drawn by the branch for the regime of κ
  // (header comment); φ is uniform on [0, 2π) and the 3-vector is built in
  // the mean-axis-aligned frame and rotated.
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
    T sin2 = T{0};   // sin²θ, formed without cancellation where possible
    std::size_t it = 0;
    while (true) {
      if (++it > _max_iterations) {
        throw std::runtime_error{
            "bingham_distribution: rejection sampler exceeded its "
            "iteration cap (numerical breakdown or an overridden cap)"};
      }
      T accept;
      if (_kappa > T{0}) {
        // Prolate: exponential envelope in t = 1 - |x| (see header
        // comment); acceptance >= 1/2 for every kappa > 0.
        const T u1 = _uniform(_engine.get());
        const T t = -std::log1p(-u1 * (-std::expm1(-_kappa))) / _kappa;
        const T ax = T{1} - std::min(t, T{1});
        const T u0 = _uniform(_engine.get());
        x = (u0 < T{0.5}) ? -ax : ax;
        // target exp(-k t (2 - t)) / envelope exp(-k t)
        accept = std::exp(-_kappa * t * (T{1} - t));
        sin2 = std::min(t, T{1}) * (T{2} - std::min(t, T{1}));
      } else {
        if (_kappa <= T{-1}) {
          // Oblate, |κ| ≥ 1: x ~ N(0, 1/(2|κ|)) truncated to [-1, 1] is
          // exactly the target; accept iff |x| ≤ 1 (rate erf(√|κ|) ≥ 0.84).
          x = _normal(_engine.get());
          accept = (std::abs(x) <= T{1}) ? T{1} : T{0};
        } else {
          // Weakly oblate: uniform envelope, acceptance ≥ ~0.37.
          const T u1 = _uniform(_engine.get());
          x = T{2} * u1 - T{1};   // candidate cos θ in [-1, 1]
          accept = std::exp(_kappa * (x * x));
        }
      }
      const T u2 = (accept >= T{1}) ? T{0} : _uniform(_engine.get());
      if (u2 <= accept) break;
    }
    if (_kappa <= T{0}) sin2 = std::max(T{0}, T{1} - x * x);
    const T phi = T{2} * std::numbers::pi_v<T> * _uniform(_engine.get());
    const T sin_theta = std::sqrt(sin2);
    vector_type v_local;
    v_local[0] = sin_theta * std::cos(phi);
    v_local[1] = sin_theta * std::sin(phi);
    v_local[2] = x;
    // Rotate so the local z-axis points along the mean axis.
    return _frame_x * v_local[0] + _frame_y * v_local[1] +
           _mean_axis * v_local[2];
  }

  [[nodiscard]] T kappa() const noexcept { return _kappa; }

  // Rejection-candidate cap per sample(); see the header comment. The
  // setter exists so tests can exercise the throw path cheaply.
  [[nodiscard]] std::size_t max_iterations() const noexcept {
    return _max_iterations;
  }
  void set_max_iterations(std::size_t n) noexcept { _max_iterations = n; }
  [[nodiscard]] vector_type const& mean_axis() const noexcept {
    return _mean_axis;
  }

private:
  // Tuple-taking helper used by the schema-driven ctor. Lets us delegate
  // through `fields::extract(handler)` without re-extracting per element.
  bingham_distribution(std::tuple<T, T, T, T> mxk, engine_type& engine)
      : bingham_distribution(
            std::array<T, 3>{std::get<0>(mxk), std::get<1>(mxk),
                             std::get<2>(mxk)},
            std::get<3>(mxk), engine) {}

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
    if (!std::isfinite(axis[0]) || !std::isfinite(axis[1]) ||
        !std::isfinite(axis[2])) {
      throw std::invalid_argument{
          "bingham_distribution: mean_axis must be finite"};
    }
    // Robust normalise scales by the max |component| first, so very
    // large or very small (but non-zero) axes still normalise cleanly.
    if (!(gte::Normalize(axis, true) > T{0})) {
      throw std::invalid_argument{
          "bingham_distribution: mean_axis must be non-zero"};
    }
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
  std::size_t _max_iterations{0};
  std::reference_wrapper<engine_type> _engine;
  std::uniform_real_distribution<T> _uniform;
  std::normal_distribution<T> _normal;
};

} // namespace rvegen
