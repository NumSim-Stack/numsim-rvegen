#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>

#include <numsim-core/static_indexing.h>

#include "../metadata/info.h"
#include "bounding_box_base.h"

namespace rvegen {

// Polymorphic interface for any shape that participates in RVE generation.
// Concrete shapes (circle, sphere, rectangle, ellipse, ...) inherit and
// implement this contract. After generation, the algorithm's accepted-shape
// container is std::vector<std::unique_ptr<shape_base<T>>>; output writers
// iterate that container and dispatch through these virtuals.
//
// During the inner generation/collision loop, however, dispatch is
// compile-time via the collision_details(A, B) overload set on concrete shape
// types — virtuals are not on the hot path.
//
// Construction-order invariant for derived shapes:
//   Every concrete shape inherits BOTH from shape_base<T> (this class) AND
//   from a GTE primitive (gte::Circle2<T>, gte::AlignedBox<2,T>, ...). The
//   GTE base supplies the geometric data fields (center, radius, min/max,
//   axis, extent). At construction time the bases run in declaration order
//   of the derived class, so shape_base may run BEFORE the GTE base. This
//   class's ctors therefore MUST NOT read any geometric fields — they
//   belong to a base that may not yet exist. Default-init only.
template <typename T>
class shape_base {
public:
  using value_type = T;

  shape_base() = default;

  // Deep-copy of the bounding box via its polymorphic clone(). This is what
  // allows derived shapes to be implicitly copy-constructible: the derived
  // class's compiler-generated copy ctor calls this one + the GTE base's
  // copy ctor, producing a fully-copied shape with a freshly-cloned bb.
  shape_base(shape_base const& other)
      : _info{other._info},
        _bounding_box{other._bounding_box ? other._bounding_box->clone()
                                          : nullptr} {}

  shape_base(shape_base&&) noexcept = default;

  virtual ~shape_base() = default;

protected:
  // Copy / move assignment are protected to prevent base-only slicing of
  // the shape_base subobject (e.g. `shape_base& dst = circle_a; dst = sphere_b;`
  // would otherwise compile and silently update only the bb). Derived
  // classes invoke these implicitly through their own (public) op=.
  shape_base& operator=(shape_base const& other) {
    if (this != &other) {
      _info = other._info;
      _bounding_box = other._bounding_box ? other._bounding_box->clone()
                                          : nullptr;
    }
    return *this;
  }
  shape_base& operator=(shape_base&&) noexcept = default;

public:

  // 3-component middle point (z = 0 for 2D shapes).
  [[nodiscard]] virtual std::array<value_type, 3> get_middle_point() const = 0;
  virtual void set_middle_point(std::array<value_type, 3> middle_point) = 0;

  // Surface area (2D) — meaningful for 2D shapes; 3D shapes return 0.
  [[nodiscard]] virtual value_type area() const = 0;

  // Volume (3D) — meaningful for 3D shapes; 2D shapes return 0.
  [[nodiscard]] virtual value_type volume() const = 0;

  // Maximum half-extent on each axis (used to test if a shape can fit
  // entirely inside the RVE box for OnlyInside placement).
  [[nodiscard]] virtual std::array<value_type, 3> max_expansion() const = 0;

  // (Re)compute the bounding box from the shape's current geometry.
  virtual void make_bounding_box() = 0;

  // Point-in-shape predicate. Used by voxelization (regular-grid sampling for
  // FFT homogenization) and any other tool that needs to ask "is this voxel
  // inside this inclusion?". 2D shapes ignore the z component.
  [[nodiscard]] virtual bool is_inside(std::array<value_type, 3> const& point) const = 0;

  // Polymorphic copy. shape_base itself is non-copyable (the _bounding_box
  // unique_ptr makes the implicit copy ctor deleted), so periodic wrap
  // generation and other "duplicate this shape and translate it" needs an
  // explicit virtual. Each concrete shape returns a fresh unique_ptr with
  // the same geometric state and a freshly-computed bounding box.
  [[nodiscard]] virtual std::unique_ptr<shape_base<value_type>> clone() const = 0;

  // Per-derived static type id (assigned by numsim_core::static_indexing).
  // Concrete shapes pick up the id by inheriting via
  //   static_indexing<concrete_shape<T>, shape_base<T>>
  // and forwarding through `return m_id`. Used as the integer key in the
  // collision dispatcher's 2D table — replaces RTTI-based type_index
  // keying so the inner-loop lookup is O(1) array indexing.
  [[nodiscard]] virtual numsim_core::type_id shape_id() const noexcept = 0;

  [[nodiscard]] bounding_box_base<value_type>* bounding_box() const noexcept {
    return _bounding_box.get();
  }

  // Generic metadata blob attached to this shape — see `rvegen::info`
  // for the storage. Defaults to an empty json object. Set by
  // `shape_input_base` when the input's schema includes metadata fields
  // (today: `phase_name`; later: orientation, source paths, custom
  // tags), then propagated through clone/translate operations and
  // consumed by output writers.
  //
  // For the common phase-name case the convenience pair
  // `phase_name() / set_phase_name()` routes through `_info` so callers
  // never have to reach into the json blob directly.
  [[nodiscard]] rvegen::info const& info() const noexcept { return _info; }
  [[nodiscard]] rvegen::info& info() noexcept { return _info; }
  void set_info(rvegen::info i) noexcept { _info = std::move(i); }

  [[nodiscard]] std::string phase_name() const { return _info.phase_name(); }
  void set_phase_name(std::string name) {
    _info.set_phase_name(std::move(name));
  }

protected:
  rvegen::info _info;
  std::unique_ptr<bounding_box_base<value_type>> _bounding_box{};
};

} // namespace rvegen
