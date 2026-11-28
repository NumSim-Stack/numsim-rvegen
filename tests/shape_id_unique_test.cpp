// Shape-id uniqueness (#85).
//
// collision_dispatcher and mesh_dispatcher key flat tables by the
// numsim_core::static_indexing id of each shape type. Under clang the
// pinned numsim-core handed every type id 0, so both dispatchers kept a
// single slot and circles were collided as box x sphere — silently.
//
// 1. Every built-in shape type, for double and float, has its own id, and
//    an instance's shape_id() reports it.
// 2. Registering a second type under an id that is already taken throws
//    std::logic_error in both dispatchers (instead of overwriting the
//    slot), and leaves the existing registration working.
// 3. Re-registering the same type stays a no-op.

#include <cstddef>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

#include "rvegen/rvegen.h"
#include "rvegen/shapes/shape_id_claims.h"

namespace {

int failures = 0;

#define REQUIRE(cond)                                                          \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::cerr << "REQUIRE failed: " #cond " at " __FILE__ ":" << __LINE__    \
                << "\n";                                                       \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

template <typename T> void test_builtin_ids_unique() {
  using namespace rvegen;
  std::set<numsim_core::type_id> const ids{
      circle<T>::m_id,         rectangle<T>::m_id,      ellipse<T>::m_id,
      sphere<T>::m_id,         box<T>::m_id,            polyline_tube<T>::m_id,
      convex_polygon<T>::m_id, voronoi_cell<T>::m_id,   mesh_inclusion<T>::m_id};
  if (ids.size() != 9) {
    std::cerr << "shape ids:"
              << " circle " << circle<T>::m_id << " rectangle " << rectangle<T>::m_id
              << " ellipse " << ellipse<T>::m_id << " sphere " << sphere<T>::m_id
              << " box " << box<T>::m_id << " polyline_tube " << polyline_tube<T>::m_id
              << " convex_polygon " << convex_polygon<T>::m_id
              << " voronoi_cell " << voronoi_cell<T>::m_id
              << " mesh_inclusion " << mesh_inclusion<T>::m_id << "\n";
  }
  REQUIRE(ids.size() == 9);

  REQUIRE(circle<T>(T(0), T(0), T(1)).shape_id() == circle<T>::m_id);
  REQUIRE(sphere<T>(T(0), T(0), T(0), T(1)).shape_id() == sphere<T>::m_id);
  REQUIRE(box<T>(T(0), T(0), T(0), T(1), T(1), T(1)).shape_id() == box<T>::m_id);
}

// The id of a sphere<double>, but a different C++ type: what a duplicate
// static_indexing id looks like to the dispatchers. Deriving from sphere
// keeps to_mesh / collision_details callable on it.
struct impostor_sphere : rvegen::sphere<double> {
  using rvegen::sphere<double>::sphere;
  static inline numsim_core::type_id m_id = 0;
};

void test_claims_direct() {
  rvegen::shape_id_claims claims;
  claims.claim(3, typeid(int), "test");
  claims.claim(3, typeid(int), "test");  // same type again: fine
  claims.claim(4, typeid(long), "test");
  bool threw = false;
  try {
    claims.claim(3, typeid(long), "test");
  } catch (std::logic_error const& e) {
    threw = std::string(e.what()).find("share static_indexing id 3") != std::string::npos;
  }
  REQUIRE(threw);
  claims.clear();
  claims.claim(3, typeid(long), "test");  // cleared: id free again
}

void test_mesh_dispatcher_rejects_duplicate_id() {
  auto& d = rvegen::mesh_dispatcher<double>::instance();
  d.clear();
  rvegen::register_all_meshes<double>();
  rvegen::register_all_meshes<double>();  // idempotent
  std::size_t const before = d.size();

  impostor_sphere::m_id = rvegen::sphere<double>::m_id;
  bool threw = false;
  try {
    d.register_shape<impostor_sphere>();
  } catch (std::logic_error const&) {
    threw = true;
  }
  REQUIRE(threw);
  REQUIRE(d.size() == before);

  // The real sphere still owns its slot.
  rvegen::sphere<double> s{0.5, 0.5, 0.5, 0.25};
  auto const direct = rvegen::to_mesh(s);
  auto const via = d(static_cast<rvegen::shape_base<double> const&>(s));
  REQUIRE(!via.empty());
  REQUIRE(via.verts.size() == direct.verts.size());
  d.clear();
}

void test_collision_dispatcher_rejects_duplicate_id() {
  auto& d = rvegen::collision_dispatcher<double>::instance();
  d.clear();
  rvegen::register_all_collision_pairs<double>();
  rvegen::register_all_collision_pairs<double>();  // idempotent
  std::size_t const before = d.size();

  impostor_sphere::m_id = rvegen::sphere<double>::m_id;
  bool threw = false;
  try {
    d.register_pair<impostor_sphere, rvegen::box<double>>();
  } catch (std::logic_error const&) {
    threw = true;
  }
  REQUIRE(threw);
  REQUIRE(d.size() == before);

  // Two circles that overlap only in their AABB corners must not collide
  // through the circle x circle narrow phase.
  rvegen::circle<double> a{0.0, 0.0, 1.0}, b{1.6, 1.6, 1.0};
  REQUIRE(!d(a, b));
  rvegen::circle<double> c{1.5, 0.0, 1.0};
  REQUIRE(d(a, c));
  d.clear();
}

} // namespace

int main() {
  test_builtin_ids_unique<double>();
  test_builtin_ids_unique<float>();
  test_claims_direct();
  test_mesh_dispatcher_rejects_duplicate_id();
  test_collision_dispatcher_rejects_duplicate_id();
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "shape_id_unique_test: all passed\n";
  return 0;
}
