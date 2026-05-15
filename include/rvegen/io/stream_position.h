#pragma once

// Tiny helper shared between the STL and PLY readers — formats a
// stream's current byte offset as a parenthesised " (at byte N)" tail
// for parse-error messages. Returns an empty string on non-seekable
// streams (where `tellg()` returns -1), so the resulting error
// message doesn't carry a confusing "-1" position marker for pipe /
// socket inputs.
//
// Note on stream state: many parse-error sites reach this helper
// AFTER a `>>` extraction has failed and put the stream into
// failbit / eofbit, at which point `tellg()` itself returns -1 even
// on a seekable underlying source. We clear() the state before
// asking — the caller is about to throw anyway, so any side effect
// on the stream is benign and the position is dramatically more
// useful in the error message.

#include <istream>
#include <string>

namespace rvegen::detail {

inline std::string position_marker(std::istream& in) {
  in.clear();
  if (auto pos = in.tellg(); pos >= 0) {
    return " (at byte " + std::to_string(static_cast<long long>(pos)) + ")";
  }
  return "";
}

} // namespace rvegen::detail
