#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

#include <Eigen/Geometry>
#include <Mathematics/Hyperellipsoid.h>
#include <numsim-core/input_parameter_controller.h>

#include "../types.h"
#include "rectangle_bounding.h"
#include "shape_base.h"

namespace rvegen {

// rvegen ellipse: IS-A gte::Ellipse2<T> (Hyperellipsoid<2, T>) + IS-A
// shape_base. GTE storage is (center, axis[2], extent[2]).
//
// Eigen is used in the user-friendly ctor to derive the orthonormal axis
// pair from a single rotation angle. is_inside reverse-rotates by dot
// product against the stored axes (no Eigen needed at query time).
template <typename T = double>
class ellipse
    : public numsim_core::static_indexing<ellipse<T>, shape_base<T>>,
      public gte::Ellipse2<T> {
public:
  using value_type = T;
  using size_type = std::size_t;

  ellipse() = default;

  // Construct from centre, semi-axes, and rotation in radians (CCW from +x).
  ellipse(T x, T y, T radius_a, T radius_b, T rotation_rad) {
    this->center = {x, y};
    this->extent = {radius_a, radius_b};
    Eigen::Rotation2D<T> R{rotation_rad};
    Eigen::Matrix<T, 2, 1> a = R * Eigen::Matrix<T, 2, 1>{T{1}, T{0}};
    Eigen::Matrix<T, 2, 1> b = R * Eigen::Matrix<T, 2, 1>{T{0}, T{1}};
    this->axis[0] = {a[0], a[1]};
    this->axis[1] = {b[0], b[1]};
  }

  explicit ellipse(parameter_handler_t const& handler)
      : ellipse(handler.template get<T>("x"),
                handler.template get<T>("y"),
                handler.template get<T>("radius_a"),
                handler.template get<T>("radius_b"),
                handler.template get<T>("rotation")) {}

  [[nodiscard]] static parameter_controller_t parameters() {
    parameter_controller_t s;
    s.template insert<T>("x").template add<numsim_core::is_required>();
    s.template insert<T>("y").template add<numsim_core::is_required>();
    s.template insert<T>("radius_a").template add<numsim_core::is_required>();
    s.template insert<T>("radius_b").template add<numsim_core::is_required>();
    s.template insert<T>("rotation").template add<numsim_core::is_required>();
    return s;
  }

  [[nodiscard]] T operator()(size_type idx) const noexcept {
    return this->center[idx];
  }
  [[nodiscard]] T& operator()(size_type idx) noexcept {
    return this->center[idx];
  }

  [[nodiscard]] T radius_a() const noexcept { return this->extent[0]; }
  [[nodiscard]] T radius_b() const noexcept { return this->extent[1]; }
  [[nodiscard]] T rotation() const noexcept {
    return std::atan2(this->axis[0][1], this->axis[0][0]);
  }

  // shape_base contract.
  [[nodiscard]] T area() const override {
    return this->extent[0] * this->extent[1] * std::numbers::pi_v<T>;
  }
  [[nodiscard]] T volume() const override { return T{0}; }

  [[nodiscard]] std::array<T, 3> max_expansion() const override {
    const auto m = std::max(this->extent[0], this->extent[1]);
    return {m, m, T{0}};
  }

  [[nodiscard]] std::array<T, 3> get_middle_point() const override {
    return {this->center[0], this->center[1], T{0}};
  }
  void set_middle_point(std::array<T, 3> middle_point) override {
    this->center[0] = middle_point[0];
    this->center[1] = middle_point[1];
  }

  // Reverse-rotate the query point into the ellipse-local frame via dot
  // product against the stored axes — direct on the GTE representation,
  // no Eigen at query time.
  [[nodiscard]] bool is_inside(std::array<T, 3> const& point) const override {
    const T dx = point[0] - this->center[0];
    const T dy = point[1] - this->center[1];
    const T u = dx * this->axis[0][0] + dy * this->axis[0][1];
    const T v = dx * this->axis[1][0] + dy * this->axis[1][1];
    const T un = u / this->extent[0];
    const T vn = v / this->extent[1];
    return un * un + vn * vn <= T{1};
  }

  // Tight rotated AABB:
  //   half-extent on x = sqrt((ra · a0[0])² + (rb · a1[0])²)
  //   half-extent on y = sqrt((ra · a0[1])² + (rb · a1[1])²)
  void make_bounding_box() override {
    const T ra = this->extent[0], rb = this->extent[1];
    const T ax = ra * this->axis[0][0], ay = ra * this->axis[0][1];
    const T bx = rb * this->axis[1][0], by = rb * this->axis[1][1];
    const T half_w = std::sqrt(ax * ax + bx * bx);
    const T half_h = std::sqrt(ay * ay + by * by);

    auto box = std::make_unique<rectangle_bounding<T>>();
    box->top_point()    = {this->center[0] + half_w, this->center[1] + half_h, T{0}};
    box->bottom_point() = {this->center[0] - half_w, this->center[1] - half_h, T{0}};
    this->_bounding_box = std::move(box);
  }

  [[nodiscard]] std::unique_ptr<shape_base<T>> clone() const override {
    return std::make_unique<ellipse<T>>(*this);
  }

  [[nodiscard]] numsim_core::type_id shape_id() const noexcept override {
    return ellipse::m_id;
  }
};

} // namespace rvegen
