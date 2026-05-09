#pragma once

// Material model interface.
//
// A `material<T>` carries the constitutive description for one phase
// of the RVE: matrix, inclusion-type-A, void, etc. Concrete materials
// implement `stiffness()` (returning the 6×6 Voigt-stiffness matrix)
// and a kind-tag string used for downstream output writers (DAMASK
// material.config, gmsh Physical Surface tags).
//
// Phase IDs from the voxel grid (0 = matrix, 1..N = inclusion index)
// map to materials via the JSON `phase_assignment` block — kept
// outside this header so the material layer stays decoupled from the
// shape layer.

#include <array>
#include <string_view>

namespace rvegen {

// Voigt-form 6×6 stiffness matrix, T-typed. Symmetric in practice;
// the matrix carries the full 36 entries for ease of indexing
// (downstream FFT / FE codes accept either redundant or upper-
// triangular forms).
template <typename T>
struct stiffness_tensor {
  std::array<std::array<T, 6>, 6> c{};
};

template <typename T>
class material_base {
public:
  using value_type = T;

  material_base() = default;
  material_base(material_base const&) = delete;
  material_base(material_base&&) = delete;
  material_base& operator=(material_base const&) = delete;
  material_base& operator=(material_base&&) = delete;
  virtual ~material_base() = default;

  // Effective small-strain stiffness in Voigt form.
  [[nodiscard]] virtual stiffness_tensor<T> stiffness() const = 0;

  // Short kind tag used by output writers — "linear_elastic", "void",
  // future "plastic_J2" etc. Returned as string_view so concrete
  // materials can return a literal without heap allocation.
  [[nodiscard]] virtual std::string_view kind() const noexcept = 0;
};

} // namespace rvegen
