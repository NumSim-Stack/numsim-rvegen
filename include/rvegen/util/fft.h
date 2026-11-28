#pragma once

// The FFT rvegen uses (statistics/two_point_fft.h): the bundled pocketfft
// header, placed in the namespace rvegen_pocketfft and built without its
// thread pool and plan cache. A program that includes its own pocketfft,
// before or after this header and with whatever settings, keeps it in
// `pocketfft` untouched: every macro that configures the header, its
// include guard and the one macro it leaves defined are saved here and
// restored afterwards.

#pragma push_macro("POCKETFFT_HDRONLY_H")
#pragma push_macro("POCKETFFT_NAMESPACE")
#pragma push_macro("POCKETFFT_NO_MULTITHREADING")
#pragma push_macro("POCKETFFT_CACHE_SIZE")
#pragma push_macro("POCKETFFT_NO_VECTORS")
#pragma push_macro("POCKETFFT_NO_EXCEPTIONS")
#pragma push_macro("POCKETFFT_PTHREADS")
#pragma push_macro("POCKETFFT_USE_POSIX_MEMALIGN")
#pragma push_macro("POCKETFFT_THROW")
#undef POCKETFFT_HDRONLY_H
#undef POCKETFFT_NAMESPACE
#undef POCKETFFT_NO_MULTITHREADING
#undef POCKETFFT_CACHE_SIZE
#undef POCKETFFT_NO_VECTORS
#undef POCKETFFT_NO_EXCEPTIONS
#undef POCKETFFT_PTHREADS
#undef POCKETFFT_USE_POSIX_MEMALIGN
#undef POCKETFFT_THROW
#define POCKETFFT_NAMESPACE rvegen_pocketfft
#define POCKETFFT_NO_MULTITHREADING

#include "../external/pocketfft/pocketfft_hdronly.h"

#pragma pop_macro("POCKETFFT_THROW")
#pragma pop_macro("POCKETFFT_USE_POSIX_MEMALIGN")
#pragma pop_macro("POCKETFFT_PTHREADS")
#pragma pop_macro("POCKETFFT_NO_EXCEPTIONS")
#pragma pop_macro("POCKETFFT_NO_VECTORS")
#pragma pop_macro("POCKETFFT_CACHE_SIZE")
#pragma pop_macro("POCKETFFT_NO_MULTITHREADING")
#pragma pop_macro("POCKETFFT_NAMESPACE")
#pragma pop_macro("POCKETFFT_HDRONLY_H")
