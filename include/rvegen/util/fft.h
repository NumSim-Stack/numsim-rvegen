#pragma once

// The FFT rvegen uses (two_point_fft.h): the bundled pocketfft header,
// placed in the namespace rvegen_pocketfft and built without its thread
// pool. A program that includes its own pocketfft, before or after this
// header and with whatever settings, keeps it in `pocketfft` untouched:
// the three macros that select namespace, threading and the include guard
// are saved here and restored afterwards.

#pragma push_macro("POCKETFFT_HDRONLY_H")
#pragma push_macro("POCKETFFT_NAMESPACE")
#pragma push_macro("POCKETFFT_NO_MULTITHREADING")
#pragma push_macro("POCKETFFT_CACHE_SIZE")
#pragma push_macro("POCKETFFT_NO_VECTORS")
#undef POCKETFFT_HDRONLY_H
#undef POCKETFFT_NAMESPACE
#undef POCKETFFT_NO_MULTITHREADING
#undef POCKETFFT_CACHE_SIZE
#undef POCKETFFT_NO_VECTORS
#define POCKETFFT_NAMESPACE rvegen_pocketfft
#define POCKETFFT_NO_MULTITHREADING

#include "../external/pocketfft/pocketfft_hdronly.h"

#pragma pop_macro("POCKETFFT_NO_VECTORS")
#pragma pop_macro("POCKETFFT_CACHE_SIZE")
#pragma pop_macro("POCKETFFT_NO_MULTITHREADING")
#pragma pop_macro("POCKETFFT_NAMESPACE")
#pragma pop_macro("POCKETFFT_HDRONLY_H")
