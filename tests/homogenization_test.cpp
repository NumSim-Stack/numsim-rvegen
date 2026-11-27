// Validation of the mean-field homogenization adapter (RVEGEN_WITH_HOMOGENIZATION).
//
// Checks against closed forms: Eshelby tensors of the sphere and the circular
// cylinder, the Voigt >= HS+ >= MT >= HS- >= Reuss chain, MT == HS lower for
// stiff spheres in a matrix that is softest in both K and G (NOT in general:
// mixed K/G ordering is tested separately), the dilute limit, phi -> 0 / 1, Hill's self-consistent
// equations, Hashin's plane-strain bulk modulus of aligned fibres, rotation
// and orientation-average consistency, and extraction from rvegen shapes.

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numbers>
#include <vector>

#include "rvegen/homogenization/mean_field.h"

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

#define REQUIRE_NEAR(a, b, tol)                                                \
  do {                                                                         \
    double const va_ = (a), vb_ = (b);                                         \
    if (!(std::abs(va_ - vb_) <= (tol) * std::max(1.0, std::abs(vb_)))) {      \
      std::cerr << "REQUIRE_NEAR failed: " #a " = " << va_ << " vs " #b " = "  \
                << vb_ << " at " __FILE__ ":" << __LINE__ << "\n";             \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

namespace h = rvegen::homogenization;
using h::model;

constexpr double Km = 3.0, Gm = 1.0, Ki = 60.0, Gi = 30.0;

h::microstructure spheres(double c, h::isotropic_material mat = {Ki, Gi},
                          h::isotropic_material matrix = {Km, Gm}) {
  h::microstructure ms{matrix, {}};
  ms.grains.push_back({"inc", c, h::sphere_inclusion(), mat});
  return ms;
}

double relnorm(h::tensor4 const& a, h::tensor4 const& b) {
  return tmech::norm(h::tensor4{a - b}) / tmech::norm(b);
}

h::tensor4 nm_iso(double K, double G) { return h::nm::isotropic_stiffness<double, 3>(K, G); }
h::tensor4 nm_ellipsoid_check(double nu) { return h::nm::eshelby::ellipsoid(nu, 1.0, 1.0 + 1e-9, 1.0 - 1e-9); }

double zeta(double K, double G) { return G * (9 * K + 8 * G) / (6 * (K + 2 * G)); }

// closed-form MT (= HS with the matrix as reference) for spheres
std::pair<double, double> mt_closed(double c, double Kmx, double Gmx, double Kin, double Gin) {
  double const K = Kmx + c * (Kin - Kmx) / (1 + (1 - c) * (Kin - Kmx) / (Kmx + 4 * Gmx / 3));
  double const G = Gmx + c * (Gin - Gmx) / (1 + (1 - c) * (Gin - Gmx) / (Gmx + zeta(Kmx, Gmx)));
  return {K, G};
}

void test_eshelby_analytic() {
  double const nu = 0.3, d = 1 - nu;
  auto const S = h::eshelby_tensor(h::sphere_inclusion(), nu);
  REQUIRE_NEAR(S(0, 0, 0, 0), (7 - 5 * nu) / (15 * d), 1e-13);
  REQUIRE_NEAR(S(0, 0, 1, 1), (5 * nu - 1) / (15 * d), 1e-13);
  REQUIRE_NEAR(S(0, 1, 0, 1), (4 - 5 * nu) / (15 * d), 1e-13);
  // general ellipsoid code path collapses to the sphere for equal axes
  auto const Se = nm_ellipsoid_check(nu);
  REQUIRE(relnorm(Se, S) < 1e-6);

  // circular cylinder along z
  auto const C = h::eshelby_tensor(h::cylinder_inclusion({0, 0, 1}), nu);
  REQUIRE_NEAR(C(0, 0, 0, 0), (5 - 4 * nu) / (8 * d), 1e-12);
  REQUIRE_NEAR(C(0, 0, 1, 1), (4 * nu - 1) / (8 * d), 1e-12);
  REQUIRE_NEAR(C(0, 0, 2, 2), nu / (2 * d), 1e-12);
  REQUIRE_NEAR(C(0, 1, 0, 1), (3 - 4 * nu) / (8 * d), 1e-12);
  REQUIRE_NEAR(C(0, 2, 0, 2), 0.25, 1e-12);
  REQUIRE_NEAR(C(2, 2, 2, 2), 0.0, 1e-12);
  // the same cylinder as a=b elliptic cylinder at an arbitrary in-plane angle
  auto const Ce = h::eshelby_tensor(h::elliptic_cylinder_inclusion(1, 1, 0.7), nu);
  REQUIRE(relnorm(Ce, C) < 1e-12);
  // cylinder along x equals the axis-permuted one: S_1111(x-cyl) = S_3333(z-cyl) = 0
  auto const Cx = h::eshelby_tensor(h::cylinder_inclusion({1, 0, 0}), nu);
  REQUIRE_NEAR(Cx(0, 0, 0, 0), 0.0, 1e-12);
  REQUIRE_NEAR(Cx(1, 1, 1, 1), (5 - 4 * nu) / (8 * d), 1e-12);
  std::cout << "eshelby sphere S1111=" << S(0, 0, 0, 0) << " analytic " << (7 - 5 * nu) / (15 * d) << "\n";
}

void test_bounds_chain_and_mt() {
  double const c = 0.3;
  auto const ms = spheres(c);
  auto const v = h::homogenize(ms, model::voigt);
  auto const r = h::homogenize(ms, model::reuss);
  auto const hu = h::homogenize(ms, model::hashin_shtrikman_upper);
  auto const hl = h::homogenize(ms, model::hashin_shtrikman_lower);
  auto const mt = h::homogenize(ms, model::mori_tanaka);
  auto const sc = h::homogenize(ms, model::self_consistent);
  auto const di = h::homogenize(ms, model::dilute);

  std::cout << std::setprecision(8) << "K: voigt " << v.K << " hs+ " << hu.K << " sc " << sc.K << " mt " << mt.K
            << " hs- " << hl.K << " reuss " << r.K << "\nG: voigt " << v.G << " hs+ " << hu.G << " sc " << sc.G
            << " mt " << mt.G << " hs- " << hl.G << " reuss " << r.G << "\n";
  REQUIRE(v.K >= hu.K && hu.K >= mt.K && mt.K >= hl.K - 1e-12 && hl.K >= r.K);
  REQUIRE(v.G >= hu.G && hu.G >= mt.G && mt.G >= hl.G - 1e-12 && hl.G >= r.G);
  REQUIRE(hu.K >= sc.K && sc.K >= hl.K - 1e-12 && hu.G >= sc.G && sc.G >= hl.G - 1e-12);

  // Voigt / Reuss closed forms
  REQUIRE_NEAR(v.K, (1 - c) * Km + c * Ki, 1e-12);
  REQUIRE_NEAR(v.G, (1 - c) * Gm + c * Gi, 1e-12);
  REQUIRE_NEAR(r.K, 1 / ((1 - c) / Km + c / Ki), 1e-12);
  REQUIRE_NEAR(r.G, 1 / ((1 - c) / Gm + c / Gi), 1e-12);

  // MT equals closed form and HS lower (stiff spheres in a matrix softest in
  // BOTH K and G; see test_mixed_ordering for the general case)
  auto const [Kc, Gc] = mt_closed(c, Km, Gm, Ki, Gi);
  REQUIRE_NEAR(mt.K, Kc, 1e-10);
  REQUIRE_NEAR(mt.G, Gc, 1e-10);
  REQUIRE_NEAR(mt.K, hl.K, 1e-10);
  REQUIRE_NEAR(mt.G, hl.G, 1e-10);
  REQUIRE(relnorm(mt.C, hl.C) < 1e-10);
  std::cout << "MT-HS- relative rank-4 difference " << relnorm(mt.C, hl.C) << "\n";

  // HS upper = MT with inclusion as reference (matrix is the 'inclusion')
  auto const [Ku, Gu] = mt_closed(1 - c, Ki, Gi, Km, Gm);
  REQUIRE_NEAR(hu.K, Ku, 1e-10);
  REQUIRE_NEAR(hu.G, Gu, 1e-10);

  // Hill's self-consistent equations for spheres
  double const res_k = (1 - c) * (Km - sc.K) / (Km + 4 * sc.G / 3) + c * (Ki - sc.K) / (Ki + 4 * sc.G / 3);
  double const res_g = (1 - c) * (Gm - sc.G) / (Gm + zeta(sc.K, sc.G)) + c * (Gi - sc.G) / (Gi + zeta(sc.K, sc.G));
  REQUIRE(std::abs(res_k) < 1e-8 && std::abs(res_g) < 1e-8);
  std::cout << "SC residuals " << res_k << " " << res_g << "\n";

  // isotropy of the result
  REQUIRE(relnorm(mt.C, nm_iso(mt.K, mt.G)) < 1e-10);
  (void)di;
}

void test_dilute_limit() {
  double const c = 1e-6;
  auto const ms = spheres(c);
  double const dK = (Ki - Km) / (1 + (Ki - Km) / (Km + 4 * Gm / 3));
  double const dG = (Gi - Gm) / (1 + (Gi - Gm) / (Gm + zeta(Km, Gm)));
  for (auto m : {model::dilute, model::mori_tanaka, model::self_consistent}) {
    auto const e = h::homogenize(ms, m);
    double const sK = (e.K - Km) / c, sG = (e.G - Gm) / c;
    std::cout << h::to_string(m) << " slope K " << sK << " (" << dK << ") G " << sG << " (" << dG << ")\n";
    REQUIRE_NEAR(sK, dK, 1e-4);
    REQUIRE_NEAR(sG, dG, 1e-4);
  }
}

void test_limits() {
  auto const Cm = h::isotropic_material{Km, Gm}.stiffness();
  auto const Ci = h::isotropic_material{Ki, Gi}.stiffness();
  for (auto m : {model::voigt, model::reuss, model::hashin_shtrikman_lower, model::hashin_shtrikman_upper,
                 model::dilute, model::mori_tanaka, model::self_consistent}) {
    REQUIRE(relnorm(h::homogenize(spheres(0.0), m).C, Cm) < 1e-12);
    // phi -> 1 (exact for the bounds, limit for the schemes; dilute is not meant for it)
    if (m == model::dilute) continue;
    double const tol = 1e-5;  // matrix fraction 1e-7 perturbs by O(1e-7 * contrast)
    REQUIRE(relnorm(h::homogenize(spheres(1.0 - 1e-7), m).C, Ci) < tol);
    REQUIRE(relnorm(h::homogenize(spheres(1e-8), m).C, Cm) < 1e-6);
  }
  // exactly phi = 1: the bounds and MT/SC (matrix fraction 0) give the inclusion
  for (auto m : {model::voigt, model::reuss, model::hashin_shtrikman_lower, model::hashin_shtrikman_upper,
                 model::mori_tanaka})
    REQUIRE(relnorm(h::homogenize(spheres(1.0), m).C, Ci) < 1e-9);
  std::cout << "limits phi->0, phi->1 ok\n";
}

void test_aligned_fibres() {
  // aligned infinite fibres along z: MT plane-strain bulk modulus k = (C11+C12)/2 (Hashin)
  double const c = 0.4;
  h::microstructure ms{{Km, Gm}, {}};
  ms.grains.push_back({"f", c, h::cylinder_inclusion({0, 0, 1}), {Ki, Gi}});
  auto const mt = h::homogenize(ms, model::mori_tanaka);
  double const km = Km + Gm / 3, ki = Ki + Gi / 3;
  double const k_expected = km + c * (ki - km) / (1 + (1 - c) * (ki - km) / (km + Gm));
  double const k_got = (mt.C(0, 0, 0, 0) + mt.C(0, 0, 1, 1)) / 2;
  std::cout << "fibre MT plane-strain k " << k_got << " vs Hashin " << k_expected << "\n";
  REQUIRE_NEAR(k_got, k_expected, 1e-10);

  // axial modulus along the fibres: rule of mixtures with Poisson correction, close to Voigt
  auto const v = h::homogenize(ms, model::voigt);
  REQUIRE(mt.C(2, 2, 2, 2) <= v.C(2, 2, 2, 2) + 1e-9);
  REQUIRE(mt.C(2, 2, 2, 2) > mt.C(0, 0, 0, 0));  // stiff along the fibres

  // rotation consistency: fibres along x vs z swap the roles of the axes
  h::microstructure mx = ms;
  mx.grains[0].shape = h::cylinder_inclusion({1, 0, 0});
  auto const mtx = h::homogenize(mx, model::mori_tanaka);
  REQUIRE_NEAR(mtx.C(0, 0, 0, 0), mt.C(2, 2, 2, 2), 1e-10);
  REQUIRE_NEAR(mtx.C(1, 1, 1, 1), mt.C(0, 0, 0, 0), 1e-10);
  REQUIRE_NEAR(mtx.C(1, 1, 2, 2), mt.C(0, 0, 1, 1), 1e-10);

  // SC with fibres converges and lies within Reuss/Voigt
  auto const sc = h::homogenize(ms, model::self_consistent);
  auto const r = h::homogenize(ms, model::reuss);
  REQUIRE(sc.K <= v.K && sc.K >= r.K);
  std::cout << "fibre SC K=" << sc.K << " G=" << sc.G << "\n";

  // ellipse rotated by 90 deg == ellipse with swapped semi-axes
  h::microstructure e1{{Km, Gm}, {}}, e2{{Km, Gm}, {}};
  e1.grains.push_back({"e", 0.3, h::elliptic_cylinder_inclusion(2, 1, std::numbers::pi / 2), {Ki, Gi}});
  e2.grains.push_back({"e", 0.3, h::elliptic_cylinder_inclusion(1, 2, 0), {Ki, Gi}});
  for (auto m : {model::mori_tanaka, model::self_consistent})
    REQUIRE(relnorm(h::homogenize(e1, m).C, h::homogenize(e2, m).C) < 1e-8);
}

void test_spheroids() {
  double const c = 0.25;
  // aspect ratio 1 spheroid == sphere, for MT and (through the custom shape) SC
  h::microstructure a{{Km, Gm}, {}}, b{{Km, Gm}, {}};
  a.grains.push_back({"s", c, h::spheroid_inclusion(1.0 + 1e-9, {0, 0, 1}), {Ki, Gi}});
  b.grains.push_back({"s", c, h::sphere_inclusion(), {Ki, Gi}});
  for (auto m : {model::mori_tanaka, model::self_consistent})
    REQUIRE(relnorm(h::homogenize(a, m).C, h::homogenize(b, m).C) < 1e-6);

  // random orientation of prolate spheroids / cylinders -> isotropic, within HS bounds
  for (double ar : {5.0, 20.0, h::infinite_axis}) {
    h::microstructure ms{{Km, Gm}, {}};
    auto const base = std::isinf(ar) ? h::cylinder_inclusion({1, 0, 0}) : h::spheroid_inclusion(ar, {1, 0, 0});
    auto const pool = h::randomize_axisymmetric(base);
    for (auto const& p : pool) ms.grains.push_back({"s", c / double(pool.size()), p, {Ki, Gi}});
    auto const mt = h::homogenize(ms, model::mori_tanaka);
    auto const hu = h::homogenize(ms, model::hashin_shtrikman_upper);
    auto const hl = h::homogenize(ms, model::hashin_shtrikman_lower);
    double const aniso = relnorm(mt.C, nm_iso(mt.K, mt.G));
    std::cout << "random AR " << ar << ": MT K=" << mt.K << " G=" << mt.G << " (HS- " << hl.G << ", HS+ " << hu.G
              << ") anisotropy " << aniso << "\n";
    REQUIRE(aniso < 1e-10);
    REQUIRE(mt.K <= hu.K + 1e-9 && mt.K >= hl.K - 1e-9);
    REQUIRE(mt.G <= hu.G + 1e-9 && mt.G >= hl.G - 1e-9);
  }
  // aligned prolate: stiffer along the axis than transverse
  h::microstructure al{{Km, Gm}, {}};
  al.grains.push_back({"s", c, h::spheroid_inclusion(10, {0, 0, 1}), {Ki, Gi}});
  auto const e = h::homogenize(al, model::mori_tanaka);
  REQUIRE(e.C(2, 2, 2, 2) > e.C(0, 0, 0, 0));
}

void test_extraction() {
  using T = double;
  std::vector<std::unique_ptr<rvegen::shape_base<T>>> shapes;
  shapes.push_back(std::make_unique<rvegen::circle<T>>(0.3, 0.3, 0.1));
  shapes.push_back(std::make_unique<rvegen::circle<T>>(0.7, 0.7, 0.15));
  for (auto& s : shapes) s->set_phase_name("fibre");
  std::array<T, 3> const box{1, 1, 0};
  std::map<std::string, h::isotropic_material> mats{{"fibre", {Ki, Gi}}};
  auto const ms = h::extract_microstructure(shapes, box, {Km, Gm}, mats);
  double const phi = std::numbers::pi * (0.01 + 0.0225);
  REQUIRE_NEAR(ms.inclusion_fraction(), phi, 1e-12);
  REQUIRE(ms.grains.size() == 2);
  auto const e = h::homogenize(shapes, box, h::isotropic_material{Km, Gm}, mats, model::mori_tanaka);
  h::microstructure ref{{Km, Gm}, {}};
  ref.grains.push_back({"f", phi, h::cylinder_inclusion(), {Ki, Gi}});
  REQUIRE(relnorm(e.C, h::homogenize(ref, model::mori_tanaka).C) < 1e-10);

  // ellipses: shape rotation and area flow through; random mode is in-plane isotropic
  std::vector<std::unique_ptr<rvegen::shape_base<T>>> el;
  el.push_back(std::make_unique<rvegen::ellipse<T>>(0.5, 0.5, 0.3, 0.1, 0.4));
  el.back()->set_phase_name("fibre");
  auto const msr = h::extract_microstructure(el, box, {Km, Gm}, mats, h::orientation_mode::random);
  REQUIRE(msr.grains.size() == 8);
  REQUIRE_NEAR(msr.inclusion_fraction(), std::numbers::pi * 0.03, 1e-12);
  auto const er = h::homogenize(msr, model::mori_tanaka);
  REQUIRE_NEAR(er.C(0, 0, 0, 0), er.C(1, 1, 1, 1), 1e-9);

  // missing material for a phase is reported
  bool threw = false;
  try { h::extract_microstructure(el, box, {Km, Gm}, {}); } catch (std::invalid_argument const&) { threw = true; }
  REQUIRE(threw);

  // 3D spheres
  std::vector<std::unique_ptr<rvegen::shape_base<T>>> sp;
  double const r = std::cbrt(0.2 * 3 / (4 * std::numbers::pi));
  sp.push_back(std::make_unique<rvegen::sphere<T>>(0.5, 0.5, 0.5, r));
  std::map<std::string, h::isotropic_material> any{{"", {Ki, Gi}}};
  auto const e3 = h::homogenize(sp, std::array<T, 3>{1, 1, 1}, h::isotropic_material{Km, Gm}, any, model::mori_tanaka);
  auto const [Kc, Gc] = mt_closed(0.2, Km, Gm, Ki, Gi);
  REQUIRE_NEAR(e3.K, Kc, 1e-10);
  REQUIRE_NEAR(e3.G, Gc, 1e-10);
}

// Independent Walpole/Berryman multi-phase HS form for isotropic phases.
std::pair<double, double> hs_ref(std::vector<std::array<double, 3>> const& ph, double K0, double G0) {
  double const z = zeta(K0, G0);
  double sk = 0, sg = 0;
  for (auto const& p : ph) { sk += p[0] / (p[1] + 4 * G0 / 3); sg += p[0] / (p[2] + z); }
  return {1 / sk - 4 * G0 / 3, 1 / sg - z};
}

void test_mixed_ordering() {
  // (a) two inclusion phases with mixed K/G ordering: A stiff in K, soft in G;
  // B soft in K, stiff in G; matrix 3/1.
  h::microstructure ms{{3, 1}, {}};
  ms.grains.push_back({"A", 0.2, h::sphere_inclusion(), {100, 2}});
  ms.grains.push_back({"B", 0.2, h::sphere_inclusion(), {5, 40}});
  std::vector<std::array<double, 3>> const ph{{0.6, 3, 1}, {0.2, 100, 2}, {0.2, 5, 40}};
  auto const [Kl, Gl] = hs_ref(ph, 3, 1);       // softest K = 3, softest G = 1
  auto const [Ku, Gu] = hs_ref(ph, 100, 40);    // stiffest K = 100, stiffest G = 40
  auto const hl = h::homogenize(ms, model::hashin_shtrikman_lower);
  auto const hu = h::homogenize(ms, model::hashin_shtrikman_upper);
  REQUIRE_NEAR(hl.K, Kl, 1e-10); REQUIRE_NEAR(hl.G, Gl, 1e-10);
  REQUIRE_NEAR(hu.K, Ku, 1e-10); REQUIRE_NEAR(hu.G, Gu, 1e-10);
  auto const v = h::homogenize(ms, model::voigt);
  auto const r = h::homogenize(ms, model::reuss);
  REQUIRE(v.K >= hu.K && hu.K >= hl.K && hl.K >= r.K);
  REQUIRE(v.G >= hu.G && hu.G >= hl.G && hl.G >= r.G);

  // (b) matrix soft in K but stiff in G, spheres the other way round
  // (K 1/G 20 matrix, K 50/G 2 spheres): MT is NOT the HS lower bound; its K
  // coincides with the HS upper K (the matrix G is the largest shear modulus).
  h::microstructure m2{{1, 20}, {}};
  m2.grains.push_back({"S", 0.3, h::sphere_inclusion(), {50, 2}});
  auto const mt = h::homogenize(m2, model::mori_tanaka);
  auto const l2 = h::homogenize(m2, model::hashin_shtrikman_lower);
  auto const u2 = h::homogenize(m2, model::hashin_shtrikman_upper);
  auto const [Kc, Gc] = mt_closed(0.3, 1, 20, 50, 2);
  REQUIRE_NEAR(mt.K, Kc, 1e-10);
  REQUIRE_NEAR(mt.G, Gc, 1e-10);
  REQUIRE(std::abs(mt.K - l2.K) > 1e-3 * mt.K);   // not the lower bound
  REQUIRE_NEAR(mt.K, u2.K, 1e-10);
  // still inside Reuss/Voigt
  auto const v2 = h::homogenize(m2, model::voigt);
  auto const r2 = h::homogenize(m2, model::reuss);
  REQUIRE(mt.K <= v2.K && mt.K >= r2.K && mt.G <= v2.G && mt.G >= r2.G);
  // HS bounds themselves are ordered
  REQUIRE(l2.K <= u2.K && l2.G <= u2.G);

  // (c) matrix softest in both (documented equality) and stiffest in both
  auto const a = h::homogenize(spheres(0.3), model::mori_tanaka);
  REQUIRE_NEAR(a.K, h::homogenize(spheres(0.3), model::hashin_shtrikman_lower).K, 1e-10);
  auto const stiff = h::homogenize(spheres(0.3, {3, 1}, {60, 30}), model::mori_tanaka);
  REQUIRE_NEAR(stiff.K, h::homogenize(spheres(0.3, {3, 1}, {60, 30}), model::hashin_shtrikman_upper).K, 1e-10);
  REQUIRE_NEAR(stiff.G, h::homogenize(spheres(0.3, {3, 1}, {60, 30}), model::hashin_shtrikman_upper).G, 1e-10);
}

void test_self_consistent_error_is_typed() {
  // Near-void spheres at c = 0.5 are past the percolation threshold of the
  // SC scheme: the iteration fails and must surface as self_consistent_error.
  h::microstructure s{{3, 1}, {}};
  s.grains.push_back({"A", 0.5, h::sphere_inclusion(), {3e-6, 1e-6}});
  bool typed = false, other = false;
  try { (void)h::homogenize(s, model::self_consistent); }
  catch (h::self_consistent_error const&) { typed = true; }
  catch (...) { other = true; }
  REQUIRE(typed && !other);
}

void test_extraction_dimension_mismatch_throws() {
  using T = double;
  std::map<std::string, h::isotropic_material> mats{{"", {Ki, Gi}}};
  std::vector<std::unique_ptr<rvegen::shape_base<T>>> c;
  c.push_back(std::make_unique<rvegen::circle<T>>(0.5, 0.5, 0.1));
  bool threw = false;
  try { (void)h::extract_microstructure(c, std::array<T, 3>{1, 1, 1}, {Km, Gm}, mats); }
  catch (std::invalid_argument const&) { threw = true; }
  REQUIRE(threw);
  std::vector<std::unique_ptr<rvegen::shape_base<T>>> s;
  s.push_back(std::make_unique<rvegen::sphere<T>>(0.5, 0.5, 0.5, 0.1));
  threw = false;
  try { (void)h::extract_microstructure(s, std::array<T, 3>{1, 1, 0}, {Km, Gm}, mats); }
  catch (std::invalid_argument const&) { threw = true; }
  REQUIRE(threw);
}

} // namespace

int main() {
  test_mixed_ordering();
  test_self_consistent_error_is_typed();
  test_extraction_dimension_mismatch_throws();
  test_eshelby_analytic();
  test_bounds_chain_and_mt();
  test_dilute_limit();
  test_limits();
  test_aligned_fibres();
  test_spheroids();
  test_extraction();
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "homogenization_test: all checks passed\n";
  return 0;
}
