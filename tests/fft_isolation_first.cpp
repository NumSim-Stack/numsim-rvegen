// A program's own pocketfft, included BEFORE rvegen's (#127): both must
// exist side by side, each in its namespace, and the program's macros must
// be what they were.
#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_CACHE_SIZE 3   // a setting of the program
#include "rvegen/external/pocketfft/pocketfft_hdronly.h"

#include "rvegen/util/fft.h"

#ifndef POCKETFFT_HDRONLY_H
#error "the program's include guard is gone"
#endif
#ifndef POCKETFFT_NAMESPACE
#error "the namespace macro the program's copy left defined is gone"
#endif
#if POCKETFFT_CACHE_SIZE != 3
#error "the program's setting was changed"
#endif

// Still the program's namespace, not rvegen's.
namespace program_fft = POCKETFFT_NAMESPACE;

#include <complex>
#include <vector>

// Sum of a forward transform's first bin, by both copies.
double fft_isolation_first() {
  const std::vector<double> in{1.0, 2.0, 3.0, 4.0};
  std::vector<std::complex<double>> a(3), b(3);
  const std::ptrdiff_t s = sizeof(double), c = sizeof(std::complex<double>);
  program_fft::r2c({4}, {s}, {c}, 0, pocketfft::FORWARD, in.data(), a.data(), 1.0);
  rvegen_pocketfft::r2c({4}, {s}, {c}, 0, rvegen_pocketfft::FORWARD, in.data(),
                        b.data(), 1.0);
  return a[0].real() + b[0].real();
}
