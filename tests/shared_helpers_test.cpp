// Shared helpers (#135): wrap_to_box, format_number, extent_of. Each of
// them replaced several local versions that did not agree in corner cases.

#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "rvegen/rvegen.h"

namespace {
int failures = 0;
#define REQUIRE(cond)                                                    \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << "REQUIRE failed: " #cond " at " << __FILE__ << ":"    \
                << __LINE__ << "\n";                                     \
      ++failures;                                                        \
    }                                                                    \
  } while (0)

void test_wrap_to_box() {
  using rvegen::wrap_to_box;
  const double L = 1.5;
  // Inside: returned unchanged, bit for bit.
  for (double x : {0.0, 1e-300, 0.1, 0.75, 1.4999999999999998}) {
    REQUIRE(wrap_to_box(x, L) == x);
  }
  // Exactly on a face: the cell that starts there.
  REQUIRE(wrap_to_box(L, L) == 0.0);
  REQUIRE(wrap_to_box(-L, L) == 0.0);
  REQUIRE(wrap_to_box(3.0 * L, L) == 0.0);
  REQUIRE(wrap_to_box(-0.0, L) == 0.0);
  // One period outside, either side.
  REQUIRE(wrap_to_box(L + 0.25, L) == 0.25);
  REQUIRE(wrap_to_box(-0.25, L) == 1.25);
  REQUIRE(wrap_to_box(-L - 0.25, L) == 1.25);
  // A negative value too small to survive "+ L" stays just below the upper
  // face instead of jumping to L (outside) or to 0 (the other side).
  for (double tiny : {-1e-20, -1e-300, -std::numeric_limits<double>::denorm_min()}) {
    const double r = wrap_to_box(tiny, L);
    REQUIRE(r < L);
    REQUIRE(r == std::nextafter(L, 0.0));
  }
  // Always in [0, L), also far away and for an awkward period.
  for (double period : {1.0, 1.5, 0.1, 3.3e-7, 7.7e9}) {
    for (double f : {-1e9, -123.456, -2.0, -1.0, -0.5, -1e-17, 0.0, 0.5, 1.0,
                     2.0, 99.99, 1e9}) {
      const double r = wrap_to_box(f * period, period);
      REQUIRE(r >= 0.0);
      REQUIRE(r < period);
      // r differs from x by a whole number of periods.
      const double n = (f * period - r) / period;
      REQUIRE(std::abs(n - std::round(n)) < 1e-6);
    }
  }
  // float works the same way.
  REQUIRE(wrap_to_box(-1e-20f, 1.0f) == std::nextafter(1.0f, 0.0f));
  REQUIRE(wrap_to_box(1.25f, 1.0f) == 0.25f);
}

void test_format_number() {
  using rvegen::format_number;
  REQUIRE(format_number(0.05) == "0.05");
  REQUIRE(format_number(1.0) == "1");
  REQUIRE(format_number(-2.5) == "-2.5");
  REQUIRE(format_number(0.0) == "0");
  REQUIRE(format_number(1.0 / 3.0) == "0.3333333333333333");
  REQUIRE(format_number(0.1f) == "0.1");
  // Reads back to the same value, for values that need all 17 digits too.
  for (double v : {1.0 / 48.0, 0.1 + 0.2, 6.02214076e23, 1e-310,
                   std::numeric_limits<double>::max(),
                   std::numeric_limits<double>::min(), 123456789.123456789}) {
    const std::string s = format_number(v);
    double back = 0.0;
    const auto res = std::from_chars(s.data(), s.data() + s.size(), back);
    REQUIRE(res.ec == std::errc{});
    REQUIRE(back == v);
  }
  // Independent of the stream it ends up in: precision and flags of the
  // stream do not touch a string.
  std::ostringstream os;
  os.precision(3);
  os << std::fixed << format_number(0.123456789);
  REQUIRE(os.str() == "0.123456789");
}

void test_extent_of() {
  // Without a bounding box: centre +- max_expansion().
  rvegen::sphere<double> s{0.5, 0.25, 0.75, 0.2};
  REQUIRE(s.bounding_box() == nullptr);
  auto e = rvegen::extent_of<double>(s);
  for (int k = 0; k < 3; ++k) {
    REQUIRE(std::abs((e.top[k] - e.bottom[k]) - 0.4) < 1e-15);
  }
  REQUIRE(std::abs(e.bottom[0] - 0.3) < 1e-15);
  REQUIRE(std::abs(e.top[2] - 0.95) < 1e-15);
  // With one: the box itself.
  s.make_bounding_box();
  auto const* bb = s.bounding_box();
  REQUIRE(bb != nullptr);
  e = rvegen::extent_of<double>(s);
  REQUIRE(e.top == bb->top_point());
  REQUIRE(e.bottom == bb->bottom_point());
  // It is a bounding box: the AABB predicates take it.
  REQUIRE(rvegen::aabb_inside<double>(e, 1.0, 1.0, 1.0));
  REQUIRE(!rvegen::aabb_inside<double>(e, 1.0, 1.0, 0.9));

  // A tube's extents() are valid without a bounding box and equal the box
  // once it is made; the collision code reads the same numbers.
  std::vector<std::array<double, 3>> line{{0.1, 0.2, 0.3}, {0.6, 0.1, 0.5},
                                          {0.4, 0.7, 0.2}};
  rvegen::polyline_tube<double> tube{line, 0.05};
  const auto ext = tube.extents();
  REQUIRE(std::abs(ext[0][0] - 0.05) < 1e-15);
  REQUIRE(std::abs(ext[1][1] - 0.75) < 1e-15);
  REQUIRE(std::abs(ext[0][2] - 0.15) < 1e-15);
  const auto cb = rvegen::detail::tube_bounds(tube);
  REQUIRE(cb.lo == ext[0]);
  REQUIRE(cb.hi == ext[1]);
  tube.make_bounding_box();
  REQUIRE(tube.bounding_box()->bottom_point() == ext[0]);
  REQUIRE(tube.bounding_box()->top_point() == ext[1]);
}
}  // namespace

int main() {
  test_wrap_to_box();
  test_format_number();
  test_extent_of();
  if (failures) { std::cerr << failures << " failure(s)\n"; return EXIT_FAILURE; }
  std::cout << "shared_helpers_test OK\n";
  return EXIT_SUCCESS;
}
