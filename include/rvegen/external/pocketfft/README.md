pocketfft (C++ header-only version), https://github.com/mreineck/pocketfft,
branch `cpp`, commit 4d865cba24359b05b37da4e75aee24498c3b78e7, unmodified.
BSD 3-Clause, see LICENSE.md and the notice at the top of the header.

rvegen includes it only through `rvegen/util/fft.h`, which places it in the
namespace `rvegen_pocketfft`, so a program that uses its own copy of
pocketfft is not affected.
