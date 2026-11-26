#pragma once

// Mean-field (analytical) homogenization of rvegen microstructures.
//
// This is an ADAPTER. The schemes themselves (Voigt, Reuss, dilute,
// Mori-Tanaka, self-consistent) and the Eshelby tensors of ellipsoids
// live in numsim-materials (`mean_field/kernels.h`, `core/eshelby.h`),
// on top of tmech's rank-4 tensor algebra. rvegen's job is to
//   * extract per-phase volume fractions and inclusion shape descriptors
//     (semi-axes + orientation) from the generated shape vector,
//   * turn them into "pseudo-grains" (one phase per distinct orientation,
//     the standard way to orientation-average a mean-field scheme),
//   * call the numsim-materials kernel and assemble
//     C_eff = sum_r c_r C_r : A_r.
// Only the Hashin-Shtrikman bounds are missing upstream; they are
// implemented here (closed form, isotropic phases) and would belong in
// numsim-materials next to the other kernels.
//
// Only compiled when RVEGEN_WITH_HOMOGENIZATION=ON (needs numsim-materials
// and tmech); the header-only core never includes this file.
//
// Conventions
//   * 3D stiffness tensors (tmech rank-4, minor symmetries), K/G isotropic
//     moduli of the matrix and of every inclusion phase.
//   * A 2D RVE (domain_box[2] == 0) is read as a generalised plane-strain
//     section: circles are infinite cylinders along z, ellipses elliptic
//     cylinders along z, area fractions are volume fractions. The full 3D
//     stiffness is returned; in-plane components are the plane-strain ones.
//   * Hashin-Shtrikman bounds are the bounds of a statistically isotropic
//     composite (isotropic comparison medium).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <tmech/tmech.h>

#include <numsim-materials/core/elastic_constants.h>
#include <numsim-materials/core/eshelby.h>
#include <numsim-materials/core/isotropize.h>
#include <numsim-materials/core/object_spec.h>
#include <numsim-materials/core/parameter_helpers.h>
#include <numsim-materials/core/traits.h>
#include <numsim-materials/mean_field/kernel_registry.h>

#include "../shapes/circle.h"
#include "../shapes/ellipse.h"
#include "../shapes/shape_base.h"
#include "../shapes/sphere.h"

namespace rvegen::homogenization {

namespace nm = numsim::materials;
namespace mf = numsim::materials::mean_field;

using policy = nm::material_policy_default;
using tensor2 = tmech::tensor<double, 3, 2>;
using tensor4 = tmech::tensor<double, 3, 4>;
using frame = std::array<std::array<double, 3>, 3>; // rows; columns are the local axes

inline constexpr double infinite_axis = std::numeric_limits<double>::infinity();

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------
struct isotropic_material {
  double K{0};  // bulk modulus
  double G{0};  // shear modulus

  [[nodiscard]] static isotropic_material from_K_G(double K, double G) {
    return validated({K, G});
  }
  [[nodiscard]] static isotropic_material from_E_nu(double E, double nu) {
    return validated({nm::bulk_modulus(E, nu), nm::shear_modulus(E, nu)});
  }
  // Minimal schema: {"K":..,"G":..} or {"E":..,"nu":..}.
  [[nodiscard]] static isotropic_material from_json(nlohmann::json const& j) {
    if (j.contains("K") && j.contains("G"))
      return from_K_G(j.at("K").get<double>(), j.at("G").get<double>());
    if (j.contains("E") && j.contains("nu"))
      return from_E_nu(j.at("E").get<double>(), j.at("nu").get<double>());
    throw std::invalid_argument{
        "isotropic_material: expected {\"K\",\"G\"} or {\"E\",\"nu\"}"};
  }
  [[nodiscard]] tensor4 stiffness() const {
    return nm::isotropic_stiffness<double, 3>(K, G);
  }
  [[nodiscard]] double poisson() const { return nm::poisson_ratio(K, G); }

private:
  static isotropic_material validated(isotropic_material m) {
    if (!(m.K > 0) || !(m.G > 0))
      throw std::invalid_argument{"isotropic_material: K and G must be > 0"};
    return m;
  }
};

// ---------------------------------------------------------------------------
// Inclusion shapes
// ---------------------------------------------------------------------------

// An ellipsoid with semi-axes (a1,a2,a3) along the local axes and the
// orientation `axes`: column j is local axis j in global coordinates.
// At most one semi-axis may be infinite (elliptic cylinder).
struct oriented_ellipsoid {
  std::array<double, 3> semi_axes{1, 1, 1};
  frame axes{{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}};
};

namespace detail {

inline tensor2 to_tensor(frame const& f) {
  tensor2 R;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) R(i, j) = f[i][j];
  return R;
}

// Right-handed frame whose first column is `d` (need not be normalised).
inline frame frame_with_axis(std::array<double, 3> d) {
  double const n = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
  if (!(n > 0)) throw std::invalid_argument{"inclusion direction must be non-zero"};
  for (auto& x : d) x /= n;
  std::array<double, 3> t = std::abs(d[0]) < 0.9 ? std::array<double, 3>{1, 0, 0}
                                                 : std::array<double, 3>{0, 1, 0};
  std::array<double, 3> e2{d[1] * t[2] - d[2] * t[1], d[2] * t[0] - d[0] * t[2],
                           d[0] * t[1] - d[1] * t[0]};
  double const m = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
  for (auto& x : e2) x /= m;
  std::array<double, 3> const e3{d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2],
                                 d[0] * e2[1] - d[1] * e2[0]};
  frame f{};
  for (std::size_t i = 0; i < 3; ++i) { f[i][0] = d[i]; f[i][1] = e2[i]; f[i][2] = e3[i]; }
  return f;
}

inline frame rotation_about_z(double angle) {
  double const c = std::cos(angle), s = std::sin(angle);
  return {{{c, -s, 0}, {s, c, 0}, {0, 0, 1}}};
}

inline bool is_sphere(std::array<double, 3> const& a) {
  return std::isfinite(a[0]) && std::abs(a[0] - a[1]) <= 1e-12 * a[0] &&
         std::abs(a[0] - a[2]) <= 1e-12 * a[0];
}

} // namespace detail

// Eshelby tensor of the oriented ellipsoid in an isotropic matrix of Poisson's ratio nu.
[[nodiscard]] inline tensor4 eshelby_tensor(oriented_ellipsoid const& e, double nu) {
  if (detail::is_sphere(e.semi_axes)) return nm::eshelby::sphere(nu);
  tensor4 const S = nm::eshelby::ellipsoid(nu, e.semi_axes[0], e.semi_axes[1], e.semi_axes[2]);
  return nm::rotate(S, detail::to_tensor(e.axes));
}

// --- descriptors -------------------------------------------------------------

[[nodiscard]] inline oriented_ellipsoid sphere_inclusion() { return {}; }

// Infinite circular cylinder (fibre) along `direction`.
[[nodiscard]] inline oriented_ellipsoid cylinder_inclusion(std::array<double, 3> direction = {0, 0, 1}) {
  return {{infinite_axis, 1, 1}, detail::frame_with_axis(direction)};
}

// Spheroid of aspect ratio `aspect_ratio` = (axis length)/(diameter) along
// `direction`; > 1 prolate (short fibre), < 1 oblate (platelet).
[[nodiscard]] inline oriented_ellipsoid spheroid_inclusion(double aspect_ratio,
                                                           std::array<double, 3> direction = {0, 0, 1}) {
  if (!(aspect_ratio > 0)) throw std::invalid_argument{"spheroid aspect ratio must be > 0"};
  return {{aspect_ratio, 1, 1}, detail::frame_with_axis(direction)};
}

// Elliptic cylinder along z with in-plane semi-axes a (at angle `rotation` from x) and b.
[[nodiscard]] inline oriented_ellipsoid elliptic_cylinder_inclusion(double a, double b, double rotation = 0) {
  if (!(a > 0) || !(b > 0)) throw std::invalid_argument{"semi-axes must be > 0"};
  return {{a, b, infinite_axis}, detail::rotation_about_z(rotation)};
}

[[nodiscard]] inline oriented_ellipsoid ellipsoid_inclusion(std::array<double, 3> semi_axes,
                                                            frame axes = {{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}}) {
  return {semi_axes, axes};
}

// Uniform random orientation of an axisymmetric inclusion (sphere, spheroid
// with symmetry axis local-1, cylinder along local-1): the six diagonals of
// an icosahedron, a spherical 5-design, are exact for the rank-4
// orientation average.
[[nodiscard]] inline std::vector<oriented_ellipsoid> randomize_axisymmetric(oriented_ellipsoid const& e) {
  if (detail::is_sphere(e.semi_axes)) return {e};
  if (e.semi_axes[1] != e.semi_axes[2])
    throw std::invalid_argument{"randomize_axisymmetric: needs semi_axes[1] == semi_axes[2]"};
  double const g = std::numbers::phi;
  std::array<std::array<double, 3>, 6> const dirs{{{0, 1, g}, {0, 1, -g}, {1, g, 0}, {1, -g, 0}, {g, 0, 1}, {-g, 0, 1}}};
  std::vector<oriented_ellipsoid> out;
  for (auto const& d : dirs) out.push_back({e.semi_axes, detail::frame_with_axis(d)});
  return out;
}

// Uniform in-plane orientation of an elliptic cylinder along z (n angles over [0, pi)).
[[nodiscard]] inline std::vector<oriented_ellipsoid> randomize_in_plane(oriented_ellipsoid const& e,
                                                                        std::size_t n = 8) {
  std::vector<oriented_ellipsoid> out;
  for (std::size_t k = 0; k < n; ++k)
    out.push_back({e.semi_axes, detail::rotation_about_z(std::numbers::pi * double(k) / double(n))});
  return out;
}

// ---------------------------------------------------------------------------
// Microstructure description
// ---------------------------------------------------------------------------

// One pseudo-grain: an inclusion of one orientation with its share of the volume.
struct grain {
  std::string phase;      // rvegen phase name
  double fraction{0};     // volume fraction of the whole RVE
  oriented_ellipsoid shape;
  isotropic_material material;
};

struct microstructure {
  isotropic_material matrix;
  std::vector<grain> grains;

  [[nodiscard]] double inclusion_fraction() const {
    double s = 0;
    for (auto const& g : grains) s += g.fraction;
    return s;
  }
};

enum class orientation_mode {
  from_shapes,  // every shape keeps its own orientation (ellipse rotation)
  aligned,      // orientation discarded: ellipse axes along x/y
  random        // each shape replaced by a uniform in-plane orientation average
};

// Extract the grains (volume fraction + shape descriptor) of a generated RVE.
// `domain_box` is {Lx, Ly, Lz} with Lz == 0 for 2D. Materials are looked up by
// the shape's phase name in `inclusion_materials` ("" is the fallback key).
// Supported shapes: circle, ellipse (2D), sphere (3D).
template <typename T>
[[nodiscard]] microstructure extract_microstructure(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes,
    std::array<T, 3> const& domain_box, isotropic_material matrix,
    std::map<std::string, isotropic_material> const& inclusion_materials,
    orientation_mode mode = orientation_mode::from_shapes) {
  bool const planar = !(domain_box[2] > 0);
  double const domain = planar ? double(domain_box[0]) * double(domain_box[1])
                               : double(domain_box[0]) * double(domain_box[1]) * double(domain_box[2]);
  if (!(domain > 0)) throw std::invalid_argument{"extract_microstructure: empty domain"};

  microstructure ms{matrix, {}};
  for (auto const& sp : shapes) {
    std::string const name = sp->phase_name();
    auto it = inclusion_materials.find(name);
    if (it == inclusion_materials.end()) it = inclusion_materials.find("");
    if (it == inclusion_materials.end())
      throw std::invalid_argument{"extract_microstructure: no material for phase '" + name + "'"};
    double const frac = double(planar ? sp->area() : sp->volume()) / domain;

    if (dynamic_cast<circle<T> const*>(sp.get())) {
      ms.grains.push_back({name, frac, cylinder_inclusion(), it->second});
    } else if (auto const* e = dynamic_cast<ellipse<T> const*>(sp.get())) {
      auto base = elliptic_cylinder_inclusion(
          double(e->radius_a()), double(e->radius_b()),
          mode == orientation_mode::from_shapes ? double(e->rotation()) : 0.0);
      if (mode == orientation_mode::random) {
        auto const pool = randomize_in_plane(base);
        for (auto const& p : pool) ms.grains.push_back({name, frac / double(pool.size()), p, it->second});
      } else {
        ms.grains.push_back({name, frac, base, it->second});
      }
    } else if (dynamic_cast<sphere<T> const*>(sp.get())) {
      ms.grains.push_back({name, frac, sphere_inclusion(), it->second});
    } else {
      throw std::invalid_argument{
          "extract_microstructure: unsupported shape type (circle, ellipse, sphere only)"};
    }
  }
  return ms;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------
enum class model {
  voigt,
  reuss,
  hashin_shtrikman_lower,
  hashin_shtrikman_upper,
  dilute,
  mori_tanaka,
  self_consistent
};

[[nodiscard]] inline std::string to_string(model m) {
  switch (m) {
    case model::voigt: return "voigt";
    case model::reuss: return "reuss";
    case model::hashin_shtrikman_lower: return "hs_lower";
    case model::hashin_shtrikman_upper: return "hs_upper";
    case model::dilute: return "dilute";
    case model::mori_tanaka: return "mori_tanaka";
    case model::self_consistent: return "self_consistent";
  }
  return {};
}

struct effective_properties {
  tensor4 C;      // effective stiffness (3D, rank-4)
  double K{0};    // isotropic projection of C
  double G{0};
  double E{0};
  double nu{0};
};

struct options {
  double sc_relaxation{0.5};
  double sc_tolerance{1e-10};
  int sc_max_iterations{200};
};

namespace detail {

// numsim-materials shape for the self-consistent kernel: any oriented ellipsoid.
class oriented_ellipsoid_shape final
    : public mf::shape_base<oriented_ellipsoid_shape, policy> {
public:
  using base = mf::shape_base<oriented_ellipsoid_shape, policy>;
  explicit oriented_ellipsoid_shape(typename base::parameter_handler p) : base(std::move(p)) {
    auto const& a = base::template get<std::vector<double>>("semi_axes");
    auto const& r = base::template get<std::vector<double>>("axes");
    if (a.size() != 3 || r.size() != 9)
      throw std::invalid_argument{"rvegen_ellipsoid: semi_axes needs 3, axes needs 9 values"};
    for (std::size_t i = 0; i < 3; ++i) {
      m_e.semi_axes[i] = a[i];
      for (std::size_t j = 0; j < 3; ++j) m_e.axes[i][j] = r[3 * i + j];
    }
  }
  static typename base::input_parameter_controller parameters() {
    typename base::input_parameter_controller para;
    nm::required<std::vector<double>>(para, "semi_axes");
    nm::required<std::vector<double>>(para, "axes");
    return para;
  }
  [[nodiscard]] tensor4 eshelby(double nu) const override { return eshelby_tensor(m_e, nu); }

private:
  oriented_ellipsoid m_e;
};

inline void register_shapes_once() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    mf::shape_registry<policy>().template add<oriented_ellipsoid_shape>("rvegen_ellipsoid");
  });
}

inline nm::object_spec<policy::ParameterHandler> shape_spec(oriented_ellipsoid const& e) {
  policy::ParameterHandler p;
  p.insert<std::vector<double>>("semi_axes", {e.semi_axes[0], e.semi_axes[1], e.semi_axes[2]});
  std::vector<double> r(9);
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 3; ++j) r[3 * i + j] = e.axes[i][j];
  p.insert<std::vector<double>>("axes", std::move(r));
  return {"rvegen_ellipsoid", std::move(p)};
}

[[nodiscard]] inline effective_properties finish(tensor4 const& C) {
  auto const [K, G] = nm::isotropize(C);
  return {C, K, G, nm::youngs_modulus(K, G), nm::poisson_ratio(K, G)};
}

// Multi-phase Hashin-Shtrikman bound of isotropic phases; reference = stiffest
// (upper) or softest (lower) K and G of all phases.
[[nodiscard]] inline effective_properties hashin_shtrikman(microstructure const& ms, bool upper) {
  struct ph { double c, K, G; };
  std::vector<ph> p{{1.0 - ms.inclusion_fraction(), ms.matrix.K, ms.matrix.G}};
  for (auto const& g : ms.grains) p.push_back({g.fraction, g.material.K, g.material.G});
  double K0 = p[0].K, G0 = p[0].G;
  for (auto const& x : p) {
    if (upper) { K0 = std::max(K0, x.K); G0 = std::max(G0, x.G); }
    else       { K0 = std::min(K0, x.K); G0 = std::min(G0, x.G); }
  }
  double const zeta = G0 * (9 * K0 + 8 * G0) / (6 * (K0 + 2 * G0));
  double sk = 0, sg = 0;
  for (auto const& x : p) {
    if (x.c <= 0) continue;
    sk += x.c / (x.K + 4.0 / 3.0 * G0);
    sg += x.c / (x.G + zeta);
  }
  double const K = 1.0 / sk - 4.0 / 3.0 * G0;
  double const G = 1.0 / sg - zeta;
  return finish(nm::isotropic_stiffness<double, 3>(K, G));
}

} // namespace detail

// Effective stiffness of the microstructure under the chosen mean-field model.
[[nodiscard]] inline effective_properties homogenize(microstructure const& ms, model m,
                                                     options const& opt = {}) {
  double const phi = ms.inclusion_fraction();
  if (phi < -1e-12 || phi > 1.0 + 1e-9)
    throw std::invalid_argument{"homogenize: inclusion volume fraction outside [0,1] (" + std::to_string(phi) + ")"};
  if (ms.grains.empty() || phi <= 0) return detail::finish(ms.matrix.stiffness());

  if (m == model::hashin_shtrikman_lower || m == model::hashin_shtrikman_upper)
    return detail::hashin_shtrikman(ms, m == model::hashin_shtrikman_upper);

  std::size_t const n = ms.grains.size() + 1;
  double const nu_m = ms.matrix.poisson();

  std::vector<tensor4> C(n), S(n);
  std::vector<double> frac(n);
  std::vector<std::string> names(n);
  C[0] = ms.matrix.stiffness();
  S[0] = nm::eshelby::sphere(nu_m);
  frac[0] = std::max(0.0, 1.0 - phi);
  names[0] = "matrix";
  for (std::size_t i = 0; i < ms.grains.size(); ++i) {
    auto const& g = ms.grains[i];
    C[i + 1] = g.material.stiffness();
    S[i + 1] = eshelby_tensor(g.shape, nu_m);
    frac[i + 1] = g.fraction;
    names[i + 1] = "g" + std::to_string(i);
  }

  policy::ParameterHandler params;
  std::string type;
  switch (m) {
    case model::voigt: type = "voigt"; break;
    case model::reuss: type = "reuss"; break;
    case model::dilute:
      type = "dilute";
      if (frac[0] < 1e-12) throw std::invalid_argument{"dilute: matrix fraction must be > 0"};
      break;
    case model::mori_tanaka: type = "mori_tanaka"; break;
    case model::self_consistent: {
      type = "self_consistent";
      detail::register_shapes_once();
      using spec = nm::object_spec<policy::ParameterHandler>;
      std::map<std::string, spec> shapes;
      shapes.emplace("matrix", spec{"sphere", {}});
      for (std::size_t i = 0; i < ms.grains.size(); ++i)
        shapes.emplace(names[i + 1], detail::shape_spec(ms.grains[i].shape));
      params.insert<std::map<std::string, spec>>("shapes", std::move(shapes));
      params.insert<double>("relaxation", opt.sc_relaxation);
      params.insert<double>("tolerance", opt.sc_tolerance);
      params.insert<int>("max_iterations", opt.sc_max_iterations);
      break;
    }
    default: break;
  }

  auto kernel = mf::kernel_registry<policy>::instance().make(
      nm::object_spec<policy::ParameterHandler>{type, std::move(params)});
  kernel->bind(names);

  std::vector<mf::phase_view<double>> views;
  views.reserve(n);
  for (std::size_t i = 0; i < n; ++i) views.push_back({frac[i], &C[i], &S[i]});
  std::vector<tensor4> A(n);
  kernel->concentrations(views, A);

  tensor4 Ceff;
  for (std::size_t i = 0; i < n; ++i)
    Ceff += frac[i] * tmech::dcontract(C[i], A[i]);
  return detail::finish(Ceff);
}

// Convenience: shapes + domain -> effective properties.
template <typename T>
[[nodiscard]] effective_properties homogenize(
    std::vector<std::unique_ptr<shape_base<T>>> const& shapes, std::array<T, 3> const& domain_box,
    isotropic_material matrix, std::map<std::string, isotropic_material> const& inclusion_materials,
    model m, orientation_mode mode = orientation_mode::from_shapes, options const& opt = {}) {
  return homogenize(extract_microstructure(shapes, domain_box, matrix, inclusion_materials, mode), m, opt);
}

} // namespace rvegen::homogenization
