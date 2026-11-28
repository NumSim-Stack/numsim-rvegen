// A program's own pocketfft, included AFTER rvegen's (#127): rvegen must
// not have consumed the include guard or left its settings behind.
#include "rvegen/util/fft.h"

#ifdef POCKETFFT_HDRONLY_H
#error "rvegen's include of pocketfft blocks the program's own"
#endif
#if defined(POCKETFFT_NAMESPACE) || defined(POCKETFFT_THROW)
#error "rvegen's pocketfft settings leaked"
#endif

#define POCKETFFT_NO_MULTITHREADING
#define POCKETFFT_CACHE_SIZE 3   // as in fft_isolation_first.cpp: one program
#include "rvegen/external/pocketfft/pocketfft_hdronly.h"

#include <complex>
#include <vector>

double fft_isolation_second() {
  const std::vector<double> in{1.0, 2.0, 3.0, 4.0};
  std::vector<std::complex<double>> a(3), b(3);
  const std::ptrdiff_t s = sizeof(double), c = sizeof(std::complex<double>);
  pocketfft::r2c({4}, {s}, {c}, 0, pocketfft::FORWARD, in.data(), a.data(), 1.0);
  rvegen_pocketfft::r2c({4}, {s}, {c}, 0, rvegen_pocketfft::FORWARD, in.data(),
                        b.data(), 1.0);
  return a[0].real() + b[0].real();
}
