// Partial-volume voxels and resolution guidance (#115).
//
//  1. sample_voxel_fractions against a reference that tests every sample
//     point of every voxel against the shapes in order (first match), for
//     overlapping shapes in 2D and 3D, both id schemes, several grids and
//     subsample counts: dominant ids, the list of composite voxels, the
//     shares and the totals all agree exactly.
//  2. With one sample per voxel the result is sample_voxel_grid's.
//  3. The volume fraction of the grid converges on the exact one with the
//     subsamples, and reaches 0.5 % at 64 x 64 with 4 x 4 points on the
//     dense circle packing of examples/dense_circles_svg.json.
//  4. shape_base::min_width of every built-in shape.
//  5. check_voxel_resolution: counts, threshold, suggested grid; fires for
//     examples/bingham_fibres.json at 32^3 and not at the suggested grid.
//  6. The writers: the file of subsamples = 1 is unchanged (v3), the v4
//     voxel file and the VTK volume_fraction array read back to the
//     sampled shares, and the run reports fractions and the warning.

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "rvegen/rvegen.h"
#include "rvegen/pipeline.h"
#include "rvegen/postprocess/voxel_fractions.h"
#include "rvegen/postprocess/voxel_grid.h"
#include "rvegen/postprocess/voxel_writer.h"
#include "rvegen/postprocess/vtk_legacy_writer.h"

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

using json = nlohmann::json;
using shape_ptr = std::unique_ptr<rvegen::shape_base<double>>;
using shape_vector = std::vector<shape_ptr>;
using rvegen::phase_id;

template <typename S, typename... A>
shape_ptr make(A&&... a) {
  auto s = std::make_unique<S>(std::forward<A>(a)...);
  s->make_bounding_box();
  return s;
}

// Overlapping circles, rectangles and ellipses in a box that is not the
// unit square; some stick out of it.
shape_vector shapes_2d(unsigned seed, std::size_t n, double lx, double ly) {
  std::mt19937 engine{seed};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  shape_vector shapes;
  for (std::size_t s = 0; s < n; ++s) {
    const double x = (u(engine) * 1.1 - 0.05) * lx;
    const double y = (u(engine) * 1.1 - 0.05) * ly;
    switch (s % 3) {
      case 0:
        shapes.push_back(make<rvegen::circle<double>>(
            x, y, (0.02 + 0.1 * u(engine)) * lx));
        break;
      case 1:
        shapes.push_back(make<rvegen::rectangle<double>>(
            x, y, (0.01 + 0.2 * u(engine)) * lx, (0.01 + 0.2 * u(engine)) * ly));
        break;
      default:
        shapes.push_back(make<rvegen::ellipse<double>>(
            x, y, (0.02 + 0.1 * u(engine)) * lx, (0.01 + 0.04 * u(engine)) * lx,
            3.0 * u(engine)));
    }
  }
  return shapes;
}

shape_vector shapes_3d(unsigned seed, std::size_t n,
                       std::array<double, 3> const& l) {
  std::mt19937 engine{seed};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  shape_vector shapes;
  for (std::size_t s = 0; s < n; ++s) {
    const double x = (u(engine) * 1.1 - 0.05) * l[0];
    const double y = (u(engine) * 1.1 - 0.05) * l[1];
    const double z = (u(engine) * 1.1 - 0.05) * l[2];
    if (s % 2 == 0) {
      shapes.push_back(make<rvegen::sphere<double>>(
          x, y, z, (0.05 + 0.15 * u(engine)) * l[0]));
    } else {
      shapes.push_back(make<rvegen::box<double>>(
          x, y, z, (0.02 + 0.3 * u(engine)) * l[0],
          (0.02 + 0.3 * u(engine)) * l[1], (0.02 + 0.3 * u(engine)) * l[2]));
    }
  }
  return shapes;
}

// The reference: every sample point of every voxel, shapes in order.
struct reference {
  std::vector<std::map<phase_id, std::uint32_t>> voxels;
  std::map<phase_id, std::uint64_t> totals;
};

reference brute_force(shape_vector const& shapes,
                      rvegen::domain<double> const& box, std::size_t nx,
                      std::size_t ny, std::size_t nz, std::size_t k,
                      std::vector<phase_id> const& ids, phase_id fill) {
  const bool is_3d = box.is_3d();
  const std::size_t nz_eff = is_3d ? nz : 1, kz = is_3d ? k : 1;
  reference r;
  r.voxels.resize(nx * ny * nz_eff);
  for (std::size_t c = 0; c < nz_eff * kz; ++c)
    for (std::size_t b = 0; b < ny * k; ++b)
      for (std::size_t a = 0; a < nx * k; ++a) {
        const std::array<double, 3> p{
            (double(a) + 0.5) * (box[0] / double(nx * k)),
            (double(b) + 0.5) * (box[1] / double(ny * k)),
            is_3d ? (double(c) + 0.5) * (box[2] / double(nz_eff * kz)) : 0.0};
        phase_id id = fill;
        for (std::size_t s = 0; s < shapes.size(); ++s)
          if (shapes[s]->is_inside(p)) {
            id = ids[s];
            break;
          }
        ++r.voxels[((c / kz) * ny + b / k) * nx + a / k][id];
        ++r.totals[id];
      }
  return r;
}

void compare(rvegen::voxel_fractions const& f, reference const& r,
             std::size_t k, bool is_3d) {
  const std::uint32_t per_voxel =
      static_cast<std::uint32_t>(k * k * (is_3d ? k : 1));
  REQUIRE(f.samples_per_voxel == per_voxel);
  REQUIRE(f.dominant.size() == r.voxels.size());
  REQUIRE(f.first.size() == f.mixed.size() + 1);
  REQUIRE(f.first.back() == f.shares.size());
  std::size_t m = 0, wrong = 0;
  for (std::size_t v = 0; v < r.voxels.size(); ++v) {
    auto const& expect = r.voxels[v];
    // Dominant: the largest count, the smallest id on a tie (map order).
    phase_id best = 0;
    std::uint32_t most = 0;
    for (auto const& [id, count] : expect)
      if (count > most) { most = count; best = id; }
    if (f.dominant[v] != best) ++wrong;
    if (expect.size() == 1) {
      if (m < f.mixed.size() && f.mixed[m] == v) ++wrong;
      continue;
    }
    if (m >= f.mixed.size() || f.mixed[m] != v) { ++wrong; continue; }
    std::vector<rvegen::voxel_share> listed(
        f.shares.begin() + static_cast<std::ptrdiff_t>(f.first[m]),
        f.shares.begin() + static_cast<std::ptrdiff_t>(f.first[m + 1]));
    std::vector<rvegen::voxel_share> wanted;
    for (auto const& [id, count] : expect) wanted.push_back({id, count});
    if (listed != wanted) ++wrong;
    ++m;
  }
  REQUIRE(wrong == 0);
  REQUIRE(m == f.mixed.size());
  std::uint64_t sum = 0;
  for (std::size_t id = 0; id < f.samples_of_id.size(); ++id) {
    auto const it = r.totals.find(static_cast<phase_id>(id));
    REQUIRE(f.samples_of_id[id] == (it == r.totals.end() ? 0 : it->second));
    sum += f.samples_of_id[id];
  }
  REQUIRE(sum == f.total_samples());
}

void test_against_reference_2d() {
  const rvegen::domain<double> box{{1.3, 0.7, 0.0}};
  for (unsigned seed : {1u, 2u, 3u}) {
    const auto shapes = shapes_2d(seed, 25, 1.3, 0.7);
    const auto ids = rvegen::detail::inclusion_index_of_shapes(shapes);
    for (std::size_t k : {1u, 2u, 3u, 4u, 5u}) {
      for (auto const& n : {std::array<std::size_t, 2>{17, 9},
                            std::array<std::size_t, 2>{32, 32}}) {
        const auto f =
            rvegen::sample_voxel_fractions(shapes, box, n[0], n[1], 7, k);
        REQUIRE(f.nz == 1);   // a 2D domain has one layer whatever nz says
        compare(f, brute_force(shapes, box, n[0], n[1], 1, k, ids, 0), k,
                false);
      }
    }
  }
}

void test_against_reference_3d() {
  const std::array<double, 3> l{1.0, 0.8, 1.6};
  const rvegen::domain<double> box{l};
  for (unsigned seed : {4u, 5u}) {
    const auto shapes = shapes_3d(seed, 14, l);
    const auto ids = rvegen::detail::inclusion_index_of_shapes(shapes);
    for (std::size_t k : {1u, 2u, 3u}) {
      for (auto const& n : {std::array<std::size_t, 3>{9, 7, 11},
                            std::array<std::size_t, 3>{12, 12, 1}}) {
        const auto f =
            rvegen::sample_voxel_fractions(shapes, box, n[0], n[1], n[2], k);
        compare(f, brute_force(shapes, box, n[0], n[1], n[2], k, ids, 0), k,
                true);
      }
    }
  }
}

// Phase ids: three phases, one of them the matrix, some shapes untagged.
void test_against_reference_with_phases() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  auto shapes = shapes_2d(9, 30, 1.0, 1.0);
  rvegen::phase_collection<double> phases;
  phases.add("matrix");
  phases.add("fibre");
  phases.add("filler");
  phases.set_matrix("matrix");
  for (std::size_t s = 0; s < shapes.size(); ++s) {
    if (s % 5 == 0) continue;   // untagged: id 0
    shapes[s]->set_phase_name(s % 2 == 0 ? "fibre" : "filler");
  }
  const auto ids = rvegen::detail::phase_id_of_shapes(shapes, phases, false);
  const phase_id fill = rvegen::detail::matrix_phase_id(phases);
  REQUIRE(fill == phases.id_of("matrix"));
  for (std::size_t k : {1u, 4u}) {
    const auto f =
        rvegen::sample_voxel_fractions(shapes, box, 40, 24, 1, k, phases);
    compare(f, brute_force(shapes, box, 40, 24, 1, k, ids, fill), k, false);
    // The fractions of all ids add up to one.
    double sum = 0.0;
    for (std::size_t id = 0; id < f.samples_of_id.size(); ++id)
      sum += f.fraction_of(static_cast<phase_id>(id));
    REQUIRE(std::abs(sum - 1.0) < 1e-12);
  }
  // strict: an untagged shape is an error, as for sample_voxel_grid.
  bool threw = false;
  try {
    (void)rvegen::sample_voxel_fractions(shapes, box, 8, 8, 1, 2, phases, true);
  } catch (std::runtime_error const&) { threw = true; }
  REQUIRE(threw);
}

void test_one_sample_is_the_plain_grid() {
  {
    const rvegen::domain<double> box{{1.3, 0.7, 0.0}};
    const auto shapes = shapes_2d(6, 30, 1.3, 0.7);
    const auto f = rvegen::sample_voxel_fractions(shapes, box, 33, 21, 1, 1);
    REQUIRE(f.dominant == rvegen::sample_voxel_grid(shapes, box, 33, 21, 1));
    REQUIRE(f.mixed.empty() && f.shares.empty());
    REQUIRE(f.samples_of_id == rvegen::count_voxel_ids(f.dominant) ||
            f.samples_of_id.size() > rvegen::count_voxel_ids(f.dominant).size());
  }
  {
    const std::array<double, 3> l{1.0, 0.8, 1.6};
    const rvegen::domain<double> box{l};
    const auto shapes = shapes_3d(7, 12, l);
    for (std::size_t nz : {1u, 6u}) {
      const auto f = rvegen::sample_voxel_fractions(shapes, box, 10, 9, nz, 1);
      REQUIRE(f.dominant == rvegen::sample_voxel_grid(shapes, box, 10, 9, nz));
      REQUIRE(f.mixed.empty());
    }
  }
}

void test_bad_arguments() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  const shape_vector none;
  for (std::size_t k : {std::size_t{0}, rvegen::max_voxel_subsamples + 1}) {
    bool threw = false;
    try {
      (void)rvegen::sample_voxel_fractions(none, box, 4, 4, 1, k);
    } catch (std::invalid_argument const& e) {
      threw = std::string{e.what()}.find("subsamples") != std::string::npos;
    }
    REQUIRE(threw);
  }
  // No shape at all: everything is id 0, nothing is mixed.
  const auto f = rvegen::sample_voxel_fractions(none, box, 4, 3, 1, 4);
  REQUIRE(f.dominant == std::vector<phase_id>(12, 0));
  REQUIRE(f.mixed.empty() && f.fraction_of(0) == 1.0);
  REQUIRE(f.fraction_of(7) == 0.0);
  // The sample points of one row of voxels are bounded: that row is the
  // least the sampler holds, whatever the rest of the grid is.
  bool threw = false;
  try {
    rvegen::validate_voxel_sampling(box, std::size_t{1} << 24, 1, 1, 16);
  } catch (std::invalid_argument const& e) {
    threw = std::string{e.what()}.find("sample points in a row") !=
            std::string::npos;
  }
  REQUIRE(threw);
  rvegen::validate_voxel_sampling(box, 64, 64, 1, 16);   // fine
  // A grid whose layer is far more than the buffer is fine too: the layer
  // is worked through in bands of rows.
  rvegen::validate_voxel_sampling(box, std::size_t{1} << 15,
                                  std::size_t{1} << 15, 1, 16);
}

// The sampler holds a band of voxel rows at a time. Whatever the band
// (one row, a few, the whole layer), the result is the same.
void test_bands_of_rows() {
  auto same = [](rvegen::voxel_fractions const& a,
                 rvegen::voxel_fractions const& b) {
    return a.dominant == b.dominant && a.mixed == b.mixed &&
           a.first == b.first && a.shares == b.shares &&
           a.samples_of_id == b.samples_of_id;
  };
  {
    const rvegen::domain<double> box{{1.3, 0.7, 0.0}};
    auto shapes = shapes_2d(11, 25, 1.3, 0.7);
    // One shape confined to a few rows, so that some bands hold nothing.
    shape_vector few;
    few.push_back(make<rvegen::circle<double>>(0.4, 0.2, 0.05));
    for (auto const* set : {&shapes, &few}) {
      const auto ids = rvegen::detail::inclusion_index_of_shapes(*set);
      const auto whole = rvegen::detail::sample_voxel_fractions_with_ids(
          *set, box, 19, 23, 1, 3, ids, 0, nullptr);
      compare(whole, brute_force(*set, box, 19, 23, 1, 3, ids, 0), 3, false);
      // A row of voxels is 19 * 9 sample points.
      for (std::size_t buffer : {std::size_t{1}, std::size_t{19 * 9},
                                 std::size_t{19 * 9 * 2 + 5},
                                 std::size_t{19 * 9 * 22}}) {
        REQUIRE(same(whole, rvegen::detail::sample_voxel_fractions_with_ids(
                                *set, box, 19, 23, 1, 3, ids, 0, nullptr,
                                buffer)));
      }
    }
  }
  {
    const std::array<double, 3> l{1.0, 0.8, 1.6};
    const rvegen::domain<double> box{l};
    const auto shapes = shapes_3d(12, 14, l);
    const auto ids = rvegen::detail::inclusion_index_of_shapes(shapes);
    const auto whole = rvegen::detail::sample_voxel_fractions_with_ids(
        shapes, box, 9, 7, 5, 2, ids, 0, nullptr);
    for (std::size_t buffer : {std::size_t{1}, std::size_t{9 * 8 * 3}}) {
      REQUIRE(same(whole, rvegen::detail::sample_voxel_fractions_with_ids(
                              shapes, box, 9, 7, 5, 2, ids, 0, nullptr,
                              buffer)));
    }
  }
}

// Shapes the sampler cannot cull: without a bounding box they are tested
// at every sample point. Periodic copies, which share their inclusion's id.
// And single precision, where the sample points are float values.
template <typename T>
void reference_more_shapes() {
  using ptr = std::unique_ptr<rvegen::shape_base<T>>;
  using vec = std::vector<ptr>;
  auto check = [](vec const& shapes, rvegen::domain<T> const& box,
                  std::size_t nx, std::size_t ny, std::size_t nz,
                  std::size_t k) {
    const auto ids = rvegen::detail::inclusion_index_of_shapes(shapes);
    const bool is_3d = box.is_3d();
    const std::size_t nz_eff = is_3d ? nz : 1, kz = is_3d ? k : 1;
    reference r;
    r.voxels.resize(nx * ny * nz_eff);
    const T dx = box[0] / static_cast<T>(nx * k);
    const T dy = box[1] / static_cast<T>(ny * k);
    const T dz = is_3d ? box[2] / static_cast<T>(nz_eff * kz) : T{0};
    for (std::size_t c = 0; c < nz_eff * kz; ++c)
      for (std::size_t b = 0; b < ny * k; ++b)
        for (std::size_t a = 0; a < nx * k; ++a) {
          const std::array<T, 3> p{
              (static_cast<T>(a) + T{0.5}) * dx,
              (static_cast<T>(b) + T{0.5}) * dy,
              is_3d ? (static_cast<T>(c) + T{0.5}) * dz : T{0}};
          phase_id id = 0;
          for (std::size_t s = 0; s < shapes.size(); ++s)
            if (shapes[s]->is_inside(p)) {
              id = ids[s];
              break;
            }
          ++r.voxels[((c / kz) * ny + b / k) * nx + a / k][id];
          ++r.totals[id];
        }
    compare(rvegen::sample_voxel_fractions(shapes, box, nx, ny, nz, k), r, k,
            is_3d);
    if (k == 1) {
      REQUIRE(rvegen::sample_voxel_fractions(shapes, box, nx, ny, nz, 1)
                  .dominant ==
              rvegen::sample_voxel_grid(shapes, box, nx, ny, nz));
    }
  };
  std::mt19937 engine{21};
  std::uniform_real_distribution<double> u{0.0, 1.0};
  auto next = [&] { return static_cast<T>(u(engine)); };
  {   // 2D: every other shape without a bounding box, one covering all
    const rvegen::domain<T> box{{T(1.3), T(0.7), T(0)}};
    vec shapes;
    for (int s = 0; s < 14; ++s) {
      auto c = std::make_unique<rvegen::circle<T>>(
          next() * T(1.3), next() * T(0.7), T(0.03) + T(0.12) * next());
      if (s % 2 == 0) c->make_bounding_box();
      shapes.push_back(std::move(c));
    }
    shapes.push_back(std::make_unique<rvegen::circle<T>>(T(0.6), T(0.3), T(5)));
    for (std::size_t k : {1u, 2u, 3u, 5u})
      for (auto const& n : {std::array<std::size_t, 2>{23, 11},
                            std::array<std::size_t, 2>{1, 9}})
        check(shapes, box, n[0], n[1], 1, k);
  }
  {   // 3D, spheres and boxes, some without a bounding box
    const rvegen::domain<T> box{{T(1), T(0.8), T(1.6)}};
    vec shapes;
    for (int s = 0; s < 10; ++s) {
      ptr shape;
      if (s % 2 == 0) {
        shape = std::make_unique<rvegen::sphere<T>>(
            next(), next() * T(0.8), next() * T(1.6), T(0.05) + T(0.2) * next());
      } else {
        shape = std::make_unique<rvegen::box<T>>(
            next(), next() * T(0.8), next() * T(1.6), T(0.05) + T(0.3) * next(),
            T(0.05) + T(0.3) * next(), T(0.05) + T(0.3) * next());
      }
      if (s % 3 != 0) shape->make_bounding_box();
      shapes.push_back(std::move(shape));
    }
    for (std::size_t k : {1u, 2u, 3u})
      for (auto const& n : {std::array<std::size_t, 3>{9, 7, 5},
                            std::array<std::size_t, 3>{5, 3, 1}})
        check(shapes, box, n[0], n[1], n[2], k);
  }
  {   // periodic copies
    const rvegen::domain<T> box{{T(1), T(1), T(0)}};
    vec shapes;
    for (std::size_t s = 0; s < 12; ++s) {
      auto c = std::make_unique<rvegen::circle<T>>(next(), next(),
                                                   T(0.05) + T(0.1) * next());
      c->make_bounding_box();
      c->set_inclusion_id(s);
      auto wraps = rvegen::generate_periodic_wraps(*c, box);
      shapes.push_back(std::move(c));
      for (auto& w : wraps) shapes.push_back(std::move(w));
    }
    REQUIRE(shapes.size() > 12);
    for (std::size_t k : {1u, 2u, 4u}) check(shapes, box, 17, 13, 1, k);
  }
}

void test_reference_more_shapes() {
  reference_more_shapes<double>();
  reference_more_shapes<float>();
}

// One circle at many positions: the error of the grid's area shrinks with
// the subsamples, and 4 x 4 points at 64 x 64 are within half a percent.
void test_fraction_converges() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  std::mt19937 engine{21};
  std::uniform_real_distribution<double> u{0.3, 0.7};
  const double r = 0.11, exact = std::numbers::pi * r * r;
  double worst1 = 0.0, worst4 = 0.0, mean1 = 0.0, mean4 = 0.0;
  const int trials = 40;
  for (int t = 0; t < trials; ++t) {
    shape_vector shapes;
    shapes.push_back(make<rvegen::circle<double>>(u(engine), u(engine), r));
    const double e1 = std::abs(
        rvegen::sample_voxel_fractions(shapes, box, 64, 64, 1, 1).fraction_of(1) -
        exact) / exact;
    const double e4 = std::abs(
        rvegen::sample_voxel_fractions(shapes, box, 64, 64, 1, 4).fraction_of(1) -
        exact) / exact;
    worst1 = std::max(worst1, e1);
    worst4 = std::max(worst4, e4);
    mean1 += e1 / trials;
    mean4 += e4 / trials;
  }
  REQUIRE(worst4 < 0.005);
  REQUIRE(mean4 < 0.3 * mean1);
  REQUIRE(worst4 < worst1);
}

json read_example(std::string const& name) {
  std::ifstream in{std::string{RVEGEN_EXAMPLES_DIR} + "/" + name};
  REQUIRE(in.good());
  return json::parse(in, nullptr, true, true);
}

struct lines {
  std::vector<std::string> info, warn;
  [[nodiscard]] rvegen::pipeline_options options() {
    rvegen::pipeline_options o;
    o.info = [this](std::string const& s) { info.push_back(s); };
    o.warn = [this](std::string const& s) { warn.push_back(s); };
    return o;
  }
  [[nodiscard]] static std::string find(std::vector<std::string> const& v,
                                        std::string const& part) {
    for (auto const& l : v)
      if (l.find(part) != std::string::npos) return l;
    return {};
  }
};

// Acceptance: the dense circle packing at 64 x 64 with 4 x 4 points.
void test_dense_circles_within_half_a_percent() {
  auto cfg = read_example("dense_circles_svg.json");
  cfg["postprocess"] = json::array();
  auto result = rvegen::generate(cfg);
  const double geometric =
      rvegen::volume_fraction_in_domain(result.shapes, result.domain_box);
  REQUIRE(geometric > 0.29);
  const auto f4 = rvegen::sample_voxel_fractions(result.shapes,
                                                 result.domain_box, 64, 64, 1, 4);
  const double grid4 = 1.0 - f4.fraction_of(0);
  REQUIRE(std::abs(grid4 - geometric) / geometric < 0.005);
  // The shape sizes give the same number as the statistics do.
  const auto ids = rvegen::detail::inclusion_index_of_shapes(result.shapes);
  const auto sizes =
      rvegen::shape_size_fractions(result.shapes, result.domain_box, ids, 0);
  REQUIRE(std::abs((1.0 - sizes[0]) - geometric) < 1e-12);
  // The circles are 3.8 to 7.7 voxels across at 64: under the default 8.
  const auto res = rvegen::check_voxel_resolution(result.shapes,
                                                  result.domain_box, 64, 64, 1, 8.0);
  REQUIRE(!res.resolved());
  REQUIRE(res.shapes == result.shapes.size());
}

// ---- min_width --------------------------------------------------------------

void test_min_width() {
  auto near = [](double a, double b) { return std::abs(a - b) < 1e-12; };
  REQUIRE(near(rvegen::circle<double>(0.3, 0.4, 0.07).min_width(), 0.14));
  REQUIRE(near(rvegen::sphere<double>(0.3, 0.4, 0.5, 0.07).min_width(), 0.14));
  REQUIRE(near(rvegen::rectangle<double>(0.5, 0.5, 0.3, 0.08).min_width(), 0.08));
  REQUIRE(near(rvegen::box<double>(0.5, 0.5, 0.5, 0.3, 0.2, 0.05).min_width(),
               0.05));
  // An ellipse at any angle: twice the minor semi-axis.
  for (double angle : {0.0, 0.7, 1.5707963267948966, 2.9})
    REQUIRE(near(rvegen::ellipse<double>(0.5, 0.5, 0.2, 0.03, angle).min_width(),
                 0.06));
  REQUIRE(near(rvegen::ellipse<double>(0.5, 0.5, 0.03, 0.2, 0.4).min_width(),
               0.06));
  // A tube along a space diagonal: its diameter, not its bounding box.
  {
    const rvegen::polyline_tube<double> tube{
        std::vector<std::array<double, 3>>{{0.1, 0.1, 0.1}, {0.9, 0.9, 0.9}},
        0.02};
    REQUIRE(near(tube.min_width(), 0.04));
  }
  // Convex polygons: the smallest altitude of a triangle, the short side
  // of a rotated rectangle.
  {
    const rvegen::convex_polygon<double> triangle{
        std::vector<std::array<double, 2>>{{0, 0}, {4, 0}, {0, 3}}};
    REQUIRE(near(triangle.min_width(), 2.4));   // altitude on the hypotenuse
    const double c = std::cos(0.6), s = std::sin(0.6);
    auto at = [&](double x, double y) {
      return std::array<double, 2>{0.5 + c * x - s * y, 0.5 + s * x + c * y};
    };
    const rvegen::convex_polygon<double> slab{
        std::vector<std::array<double, 2>>{at(-0.3, -0.02), at(0.3, -0.02),
                                           at(0.3, 0.02), at(-0.3, 0.02)}};
    REQUIRE(near(slab.min_width(), 0.04));
  }
  // A Voronoi cell: a thin rotated brick is as wide as it is thick.
  {
    using V = gte::Vector<3, double>;
    const double c = std::cos(0.5), s = std::sin(0.5);
    auto mk = [&](double x, double y, double z) {
      V v;
      v[0] = c * x - s * z;   // rotate about y
      v[1] = y;
      v[2] = s * x + c * z;
      return v;
    };
    const double a = 0.4, b = 0.3, t = 0.03;
    std::vector<V> verts{mk(0, 0, 0), mk(a, 0, 0), mk(a, b, 0), mk(0, b, 0),
                         mk(0, 0, t), mk(a, 0, t), mk(a, b, t), mk(0, b, t)};
    std::vector<std::vector<std::size_t>> faces{
        {0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4},
        {2, 3, 7, 6}, {1, 2, 6, 5}, {0, 4, 7, 3}};
    const rvegen::voronoi_cell<double> brick{std::move(verts), std::move(faces)};
    REQUIRE(std::abs(brick.min_width() - t) < 1e-12);
    // The default (axis-aligned extent) would have said far more.
    const auto half = brick.max_expansion();
    REQUIRE(2.0 * std::min({half[0], half[1], half[2]}) > 5.0 * t);
  }
}

// ---- resolution -------------------------------------------------------------

void test_resolution_check() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  shape_vector shapes;
  shapes.push_back(make<rvegen::circle<double>>(0.3, 0.3, 0.05));    // 0.1
  shapes.push_back(make<rvegen::circle<double>>(0.7, 0.7, 0.1));     // 0.2
  shapes.push_back(make<rvegen::rectangle<double>>(0.5, 0.2, 0.3, 0.04));
  {
    const auto r = rvegen::check_voxel_resolution(shapes, box, 64, 64, 1, 8.0);
    REQUIRE(r.shapes == 3 && r.under == 2);   // 6.4 and 2.56 voxels
    REQUIRE(std::abs(r.smallest_width - 0.04) < 1e-12);
    REQUIRE(std::abs(r.voxels_across() - 2.56) < 1e-9);
    REQUIRE((r.suggested == std::array<std::size_t, 3>{200, 200, 1}));
    REQUIRE(!r.sampled_z);
    const auto text = r.message();
    REQUIRE(text.find("2 of 3 shapes are less than 8 voxels across") !=
            std::string::npos);
    REQUIRE(text.find("nx >= 200, ny >= 200 gives it 8 voxels") !=
            std::string::npos);
    REQUIRE(text.find("nz") == std::string::npos);
  }
  // Exactly at the threshold counts as resolved, and so does the suggestion.
  {
    const auto r = rvegen::check_voxel_resolution(shapes, box, 200, 200, 1, 8.0);
    REQUIRE(r.resolved() && r.message().empty());
    REQUIRE(r.shapes == 3);
  }
  // The coarsest axis decides.
  {
    const auto r = rvegen::check_voxel_resolution(shapes, box, 400, 100, 1, 8.0);
    REQUIRE(r.under == 1 && std::abs(r.voxel - 0.01) < 1e-15);
    REQUIRE((r.suggested == std::array<std::size_t, 3>{400, 200, 1}));
  }
  // Threshold 0: no check. No shape: nothing to resolve.
  REQUIRE(rvegen::check_voxel_resolution(shapes, box, 4, 4, 1, 0.0).resolved());
  REQUIRE(rvegen::check_voxel_resolution(shape_vector{}, box, 4, 4, 1, 8.0)
              .resolved());
  // Periodic copies are the same inclusion: counted once.
  {
    rvegen::domain<double> pbox{{1.0, 1.0, 0.0}};
    shape_vector periodic;
    auto primary = make<rvegen::circle<double>>(0.02, 0.5, 0.05);
    auto wraps = rvegen::generate_periodic_wraps(*primary, pbox);
    REQUIRE(!wraps.empty());
    periodic.push_back(std::move(primary));
    for (auto& w : wraps) periodic.push_back(std::move(w));
    const auto r = rvegen::check_voxel_resolution(periodic, pbox, 32, 32, 1, 8.0);
    REQUIRE(r.shapes == 1 && r.under == 1);
    // ... and in the fraction from the shape sizes.
    const auto ids = rvegen::detail::inclusion_index_of_shapes(periodic);
    const auto sizes = rvegen::shape_size_fractions(periodic, pbox, ids, 0);
    REQUIRE(std::abs((1.0 - sizes[0]) - std::numbers::pi * 0.0025) < 1e-12);
  }
  // A shape without any width is the thinnest there can be: under the
  // threshold, and no grid to suggest.
  {
    shape_vector flat;
    flat.push_back(make<rvegen::circle<double>>(0.5, 0.5, 0.2));
    flat.push_back(make<rvegen::circle<double>>(0.2, 0.2, 0.0));
    const auto r = rvegen::check_voxel_resolution(flat, box, 64, 64, 1, 8.0);
    REQUIRE(r.shapes == 2 && r.under == 1 && r.smallest_width == 0.0);
    REQUIRE((r.suggested == std::array<std::size_t, 3>{64, 64, 1}));
    REQUIRE(r.message().find("the thinnest has no width") != std::string::npos);
    REQUIRE(r.message().find("nx >=") == std::string::npos);
  }
  // A grid nobody can have is named as such.
  {
    const rvegen::domain<double> cube{{1.0, 1.0, 1.0}};
    shape_vector thin;
    thin.push_back(make<rvegen::sphere<double>>(0.5, 0.5, 0.5, 1e-4));
    const auto r = rvegen::check_voxel_resolution(thin, cube, 32, 32, 32, 8.0);
    REQUIRE(r.under == 1 && !r.suggestion_fits);
    REQUIRE((r.suggested == std::array<std::size_t, 3>{40000, 40000, 40000}));
    REQUIRE(r.message().find("more than the 2^32 voxels") != std::string::npos);
    shape_vector fine;
    fine.push_back(make<rvegen::sphere<double>>(0.5, 0.5, 0.5, 0.02));
    const auto ok = rvegen::check_voxel_resolution(fine, cube, 32, 32, 32, 8.0);
    REQUIRE(ok.under == 1 && ok.suggestion_fits);
    REQUIRE(ok.message().find("2^32") == std::string::npos);
  }
  // Shapes that overlap add up to more than the box: nothing negative is
  // left for the rest, and the line of the writers says what happened.
  {
    shape_vector big;
    big.push_back(make<rvegen::circle<double>>(0.5, 0.5, 0.45));
    big.push_back(make<rvegen::circle<double>>(0.5, 0.5, 0.45));
    const auto ids = rvegen::detail::inclusion_index_of_shapes(big);
    const auto sizes = rvegen::shape_size_fractions(big, box, ids, 0);
    REQUIRE(sizes[0] == 0.0);
    REQUIRE(sizes[1] + sizes[2] > 1.2);
    std::vector<std::string> lines;
    const auto counts = rvegen::count_voxel_ids(
        rvegen::sample_voxel_grid(big, box, 32, 32, 1));
    rvegen::report_voxel_sampling<double>(
        big, box, 32, 32, 1, 1, 0.0, nullptr, false, counts,
        [&](std::string const& l) { lines.push_back(l); }, {});
    REQUIRE(lines.size() == 1);
    REQUIRE(lines[0].find("their sizes add up to more than the box") !=
            std::string::npos);
    REQUIRE(lines[0].find("-") == std::string::npos);   // no negative number
    // Shapes that fit say nothing of the kind.
    lines.clear();
    shape_vector one;
    one.push_back(make<rvegen::circle<double>>(0.5, 0.5, 0.2));
    rvegen::report_voxel_sampling<double>(
        one, box, 32, 32, 1, 1, 0.0, nullptr, false,
        rvegen::count_voxel_ids(rvegen::sample_voxel_grid(one, box, 32, 32, 1)),
        [&](std::string const& l) { lines.push_back(l); }, {});
    REQUIRE(lines.size() == 1 && lines[0].find("add up") == std::string::npos);
  }
  // 3D: z counts when it is sampled, not for a single slice.
  {
    const rvegen::domain<double> cube{{1.0, 1.0, 2.0}};
    shape_vector spheres;
    spheres.push_back(make<rvegen::sphere<double>>(0.5, 0.5, 1.0, 0.1));
    const auto r = rvegen::check_voxel_resolution(spheres, cube, 100, 100, 20, 8.0);
    REQUIRE(r.sampled_z && r.under == 1 && std::abs(r.voxel - 0.1) < 1e-15);
    REQUIRE((r.suggested == std::array<std::size_t, 3>{100, 100, 80}));
    REQUIRE(r.message().find("nz >= 80") != std::string::npos);
    const auto slice =
        rvegen::check_voxel_resolution(spheres, cube, 100, 100, 1, 8.0);
    REQUIRE(slice.resolved() && !slice.sampled_z);
  }
}

// Acceptance: bingham_fibres at 32^3 warns, the suggested grid does not.
void test_bingham_fibres_warning() {
  auto cfg = read_example("bingham_fibres.json");
  cfg["postprocess"] = json::array(
      {json{{"type", "voxel"}, {"nx", 32}, {"ny", 32}, {"nz", 32},
            {"output_path", "voxel_fractions_fibres.vox"}}});
  lines l;
  auto result = rvegen::run_pipeline(cfg, l.options());
  const auto warning = lines::find(l.warn, "voxels across");
  REQUIRE(warning.find("rvegen: warning - post-process 'voxel': ") == 0);
  REQUIRE(warning.find("shapes are less than 8 voxels across") !=
          std::string::npos);
  REQUIRE(warning.find("the thinnest is 0.04 wide, 1.28 voxels of 0.03125") !=
          std::string::npos);
  REQUIRE(warning.find("nx >= 200, ny >= 200, nz >= 200") != std::string::npos);
  REQUIRE(warning.find("min_voxels_per_feature") != std::string::npos);
  // The fraction line is there too.
  REQUIRE(!lines::find(l.info, "post-process 'voxel': volume fraction of the "
                               "inclusions").empty());
  // At the suggested grid nothing is under-resolved (checked on the
  // shapes: writing 200^3 voxels is not a unit test).
  REQUIRE(rvegen::check_voxel_resolution(result.shapes, result.domain_box, 200,
                                         200, 200, 8.0).resolved());
  // A lower threshold, or none, and the same grid passes silently.
  for (double threshold : {1.0, 0.0}) {
    cfg["postprocess"][0]["min_voxels_per_feature"] = threshold;
    lines quiet;
    (void)rvegen::run_pipeline(cfg, quiet.options());
    REQUIRE(lines::find(quiet.warn, "voxels across").empty());
  }
  std::filesystem::remove("voxel_fractions_fibres.vox");
}

// ---- writers ----------------------------------------------------------------

std::vector<std::string> split_lines(std::string const& text) {
  std::vector<std::string> out;
  std::istringstream in{text};
  for (std::string line; std::getline(in, line);) out.push_back(line);
  return out;
}

void test_voxel_writer_formats() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  const auto shapes = shapes_2d(12, 12, 1.0, 1.0);
  // subsamples = 1: the v3 file, with or without the new members touched.
  std::ostringstream plain, touched;
  rvegen::voxel_writer<double> w{24, 16, 1};
  w.write(plain, shapes, box);
  REQUIRE(w.subsamples() == 1 && w.min_voxels_per_feature() == 8.0);
  w.set_min_voxels_per_feature(3.0);
  w.set_subsamples(1);
  w.write(touched, shapes, box);
  REQUIRE(plain.str() == touched.str());
  REQUIRE(plain.str().find("[rvegen-voxel-format: v3]") != std::string::npos);
  REQUIRE(plain.str().find("subsamples") == std::string::npos);
  REQUIRE(plain.str().find("composite") == std::string::npos);

  // subsamples = 4: v4, read back.
  w.set_subsamples(4);
  std::ostringstream out;
  w.write(out, shapes, box);
  const auto f = w.sample_fractions(shapes, box);
  REQUIRE(!f.mixed.empty());
  const auto all = split_lines(out.str());
  std::vector<std::string> header, body;
  for (auto const& line : all) (line.starts_with('#') ? header : body).push_back(line);
  REQUIRE(header.at(0) == "# rvegen voxel grid [rvegen-voxel-format: v4]");
  REQUIRE(header.at(1) == "# nx ny nz: 24 16 1");
  REQUIRE(header.at(3) == "# subsamples: 4 (16 sample points per voxel)");
  REQUIRE(lines::find(header, "# composite voxels: " +
                                  std::to_string(f.mixed.size()) + " ") != "");
  // No comment line after the first value.
  bool seen_value = false, comment_after = false;
  for (auto const& line : all) {
    if (!line.starts_with('#')) seen_value = true;
    else if (seen_value) comment_after = true;
  }
  REQUIRE(!comment_after);
  REQUIRE(body.size() == 24 * 16 + f.mixed.size());
  for (std::size_t v = 0; v < 24 * 16; ++v)
    REQUIRE(body[v] == std::to_string(f.dominant[v]));
  for (std::size_t m = 0; m < f.mixed.size(); ++m) {
    std::istringstream in{body[24 * 16 + m]};
    std::size_t voxel = 0;
    in >> voxel;
    REQUIRE(voxel == f.mixed[m]);
    std::uint32_t sum = 0;
    std::size_t e = f.first[m];
    for (phase_id id = 0, n = 0; in >> id >> n; ++e) {
      REQUIRE(e < f.first[m + 1]);
      REQUIRE((f.shares[e] == rvegen::voxel_share{id, n}));
      sum += n;
    }
    REQUIRE(e == f.first[m + 1] && sum == 16);
  }
  // sample() stays the centre sample whatever the subsamples.
  REQUIRE(w.sample(shapes, box) == rvegen::sample_voxel_grid(shapes, box, 24, 16, 1));
  // validate() refuses a subsample count out of range.
  w.set_subsamples(17);
  bool threw = false;
  try { w.write(out, shapes, box); } catch (std::invalid_argument const&) { threw = true; }
  REQUIRE(threw);
}

// The shares of the VTK volume_fraction array, one row per cell.
std::vector<std::vector<double>> read_fractions(std::string const& text,
                                                std::size_t& components,
                                                std::vector<phase_id>& scalar) {
  const auto all = split_lines(text);
  std::vector<std::vector<double>> rows;
  std::size_t at = 0;
  while (at < all.size() && !all[at].starts_with("LOOKUP_TABLE")) ++at;
  ++at;
  while (at < all.size() && !all[at].starts_with("FIELD")) {
    scalar.push_back(static_cast<phase_id>(std::stoul(all[at])));
    ++at;
  }
  if (at >= all.size()) return rows;
  REQUIRE(all[at] == "FIELD FieldData 1");
  std::istringstream head{all[at + 1]};
  std::string name, type;
  std::size_t tuples = 0;
  head >> name >> components >> tuples >> type;
  REQUIRE(name == "volume_fraction" && type == "float");
  for (std::size_t r = at + 2; r < all.size(); ++r) {
    std::istringstream in{all[r]};
    std::vector<double> row;
    for (double v = 0; in >> v;) row.push_back(v);
    rows.push_back(std::move(row));
  }
  REQUIRE(rows.size() == tuples);
  return rows;
}

void test_vtk_writer_formats() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  auto shapes = shapes_2d(13, 12, 1.0, 1.0);
  {
    rvegen::vtk_legacy_writer<double> w{20, 12, 1};
    std::ostringstream plain;
    w.write(plain, shapes, box);
    REQUIRE(plain.str().find("[rvegen-vtk-format: v3]") != std::string::npos);
    REQUIRE(plain.str().find("FIELD") == std::string::npos);
    REQUIRE(plain.str().find("volume_fraction") == std::string::npos);

    // Without phases: two components, no shape and any shape.
    w.set_subsamples(3);
    std::ostringstream out;
    w.write(out, shapes, box);
    const auto text = out.str();
    REQUIRE(text.find("[rvegen-vtk-format: v4]") != std::string::npos);
    REQUIRE(split_lines(text).at(1).size() < 256);   // VTK's title limit
    std::size_t components = 0;
    std::vector<phase_id> scalar;
    const auto rows = read_fractions(text, components, scalar);
    const auto f = rvegen::sample_voxel_fractions(shapes, box, 20, 12, 1, 3);
    REQUIRE(components == 2 && rows.size() == 240);
    REQUIRE(scalar == f.dominant);
    std::vector<double> covered(240, -1.0);
    for (std::size_t v = 0; v < 240; ++v) covered[v] = f.dominant[v] == 0 ? 0 : 1;
    for (std::size_t m = 0; m < f.mixed.size(); ++m) {
      std::uint32_t in_shapes = 0;
      for (std::size_t e = f.first[m]; e < f.first[m + 1]; ++e)
        if (f.shares[e].id != 0) in_shapes += f.shares[e].samples;
      covered[f.mixed[m]] = in_shapes / 9.0;
    }
    double total = 0.0;
    for (std::size_t v = 0; v < 240; ++v) {
      REQUIRE(rows[v].size() == 2);
      REQUIRE(std::abs(rows[v][1] - covered[v]) < 1e-5);
      REQUIRE(std::abs(rows[v][0] + rows[v][1] - 1.0) < 1e-5);
      total += rows[v][1] / 240.0;
    }
    REQUIRE(std::abs(total - (1.0 - f.fraction_of(0))) < 1e-5);
  }
  // With phases: one component per phase id, 0 included.
  {
    rvegen::phase_collection<double> phases;
    phases.add("matrix");
    phases.add("fibre");
    phases.add("filler");
    phases.set_matrix("matrix");
    for (std::size_t s = 0; s < shapes.size(); ++s)
      shapes[s]->set_phase_name(s % 2 == 0 ? "fibre" : "filler");
    rvegen::vtk_legacy_writer<double> w{20, 12, 1};
    w.set_phases(&phases);
    w.set_subsamples(4);
    REQUIRE(w.fraction_components() == 4);
    std::ostringstream out;
    w.write(out, shapes, box);
    REQUIRE(split_lines(out.str()).at(1).size() < 256);
    std::size_t components = 0;
    std::vector<phase_id> scalar;
    const auto rows = read_fractions(out.str(), components, scalar);
    const auto f = rvegen::sample_voxel_fractions(shapes, box, 20, 12, 1, 4, phases);
    REQUIRE(components == 4 && rows.size() == 240 && scalar == f.dominant);
    std::vector<double> of_id(4, 0.0);
    for (auto const& row : rows) {
      REQUIRE(row.size() == 4);
      double sum = 0.0;
      for (std::size_t c = 0; c < 4; ++c) { sum += row[c]; of_id[c] += row[c] / 240.0; }
      REQUIRE(std::abs(sum - 1.0) < 1e-5);
    }
    for (phase_id id = 0; id < 4; ++id)
      REQUIRE(std::abs(of_id[id] - f.fraction_of(id)) < 1e-5);
    REQUIRE(of_id[0] == 0.0);   // every shape has a phase, the rest is matrix

    // Too many phases for a dense array: refused, with the way out.
    rvegen::phase_collection<double> many;
    for (int p = 0; p < 300; ++p) many.add("p" + std::to_string(p));
    w.set_phases(&many);
    bool threw = false;
    try { (void)w.fraction_components(); }
    catch (std::runtime_error const& e) {
      threw = std::string{e.what()}.find("voxel writer") != std::string::npos;
    }
    REQUIRE(threw);
    // ... by write() too, and before it has written anything.
    std::ostringstream untouched;
    threw = false;
    try { w.write(untouched, shapes, box); }
    catch (std::runtime_error const&) { threw = true; }
    REQUIRE(threw && untouched.str().empty());
  }
}

// The lines of a run: fractions per phase, the warning, and the config keys.
void test_pipeline_lines_and_keys() {
  auto cfg = read_example("phases_voxel.json");
  cfg["postprocess"] = json::array(
      {json{{"type", "voxel"}, {"nx", 64}, {"ny", 64}, {"nz", 1},
            {"subsamples", 4}, {"output_path", "voxel_fractions_phases.vox"}},
       json{{"type", "vtk_legacy"}, {"nx", 64}, {"ny", 64}, {"nz", 1},
            {"subsamples", 2}, {"min_voxels_per_feature", 0},
            {"output_path", "voxel_fractions_phases.vtk"}}});
  lines l;
  (void)rvegen::run_pipeline(cfg, l.options());
  const auto voxel = lines::find(l.info, "post-process 'voxel': volume fractions");
  REQUIRE(voxel.find("volume fractions on the grid (from the shape sizes): "
                     "matrix 0.9") != std::string::npos);
  REQUIRE(voxel.find(", fibre 0.0") != std::string::npos);
  REQUIRE(voxel.find(", filler 0.0") != std::string::npos);
  REQUIRE(voxel.find("(64 x 64 voxels, 4 x 4 sample points each)") !=
          std::string::npos);
  REQUIRE(voxel.find("id 0") == std::string::npos);   // nothing untagged
  const auto vtk =
      lines::find(l.info, "post-process 'vtk_legacy': volume fractions");
  REQUIRE(vtk.find("2 x 2 sample points each") != std::string::npos);
  // The rectangles are 2.6 to 4.5 voxels thick: the voxel writer warns, the
  // VTK writer was told not to check.
  REQUIRE(!lines::find(l.warn, "post-process 'voxel': ").empty());
  REQUIRE(lines::find(l.warn, "post-process 'vtk_legacy': ").empty());
  {
    std::ifstream in{"voxel_fractions_phases.vox"};
    std::string first;
    std::getline(in, first);
    REQUIRE(first == "# rvegen voxel grid [rvegen-voxel-format: v4]");
  }
  // The keys are checked like every other: a value out of range is an
  // error before anything is generated.
  for (auto const& [key, value] :
       std::vector<std::pair<std::string, json>>{
           {"subsamples", 0}, {"subsamples", 17},
           {"min_voxels_per_feature", -1.0}}) {
    for (std::size_t writer : {0u, 1u}) {
      auto bad = cfg;
      bad["postprocess"][writer][key] = value;
      std::string message;
      try { rvegen::check_config(bad); }
      catch (std::exception const& e) { message = e.what(); }
      REQUIRE(message.find(key) != std::string::npos);
    }
  }
  std::filesystem::remove("voxel_fractions_phases.vox");
  std::filesystem::remove("voxel_fractions_phases.vox.phases.json");
  std::filesystem::remove("voxel_fractions_phases.vtk");
  std::filesystem::remove("voxel_fractions_phases.vtk.phases.json");
}

// A writer used on its own says nothing, and one with a sink says it there.
void test_log_sink() {
  const rvegen::domain<double> box{{1.0, 1.0, 0.0}};
  shape_vector shapes;
  shapes.push_back(make<rvegen::circle<double>>(0.5, 0.5, 0.2));
  rvegen::voxel_writer<double> w{10, 10, 1, "voxel_fractions_sink.vox"};
  REQUIRE(!w.has_log());
  w.run(shapes, box);   // no sink: nothing to call, nothing thrown
  std::vector<std::string> info, warn;
  const rvegen::post_process_log log{
      [&](std::string const& s) { info.push_back(s); },
      [&](std::string const& s) { warn.push_back(s); }};
  w.set_log(&log);
  w.run(shapes, box);
  REQUIRE(info.size() == 1 && warn.size() == 1);
  REQUIRE(info[0].find("volume fraction of the inclusions 0.12 on the grid, "
                       "0.1257 from the shape sizes (10 x 10 voxels)") == 0);
  REQUIRE(warn[0].find("1 of 1 shape is less than 8 voxels across") == 0);
  // A sink with one side only.
  const rvegen::post_process_log only_warn{{}, [&](std::string const& s) {
                                             warn.push_back(s);
                                           }};
  w.set_log(&only_warn);
  w.run(shapes, box);
  REQUIRE(info.size() == 1 && warn.size() == 2);
  w.set_log(nullptr);
  w.run(shapes, box);
  REQUIRE(info.size() == 1 && warn.size() == 2);
  std::filesystem::remove("voxel_fractions_sink.vox");
}

} // namespace

int main() {
  test_against_reference_2d();
  test_against_reference_3d();
  test_against_reference_with_phases();
  test_bands_of_rows();
  test_reference_more_shapes();
  test_one_sample_is_the_plain_grid();
  test_bad_arguments();
  test_fraction_converges();
  test_dense_circles_within_half_a_percent();
  test_min_width();
  test_resolution_check();
  test_bingham_fibres_warning();
  test_voxel_writer_formats();
  test_vtk_writer_formats();
  test_pipeline_lines_and_keys();
  test_log_sink();
  if (failures) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "voxel_fractions_test OK\n";
  return 0;
}
