#pragma once

// Tiny helpers shared between the STL and PLY readers.
//
// `position_marker(in)` formats a stream's current byte offset as a
// parenthesised " (at byte N)" tail for parse-error messages. Returns
// an empty string on non-seekable streams (where `tellg()` returns
// -1), so the resulting error message doesn't carry a confusing "-1"
// position marker for pipe / socket inputs.
//
// Note on stream state: many parse-error sites reach this helper
// AFTER a `>>` extraction has failed and put the stream into
// failbit / eofbit, at which point `tellg()` itself returns -1 even
// on a seekable underlying source. We clear() the state before
// asking — the caller is about to throw anyway, so any side effect
// on the stream is benign and the position is dramatically more
// useful in the error message.
//
// `remaining_bytes(in)` reports how many bytes are left between the
// current read position and the end of a seekable stream (nullopt on
// non-seekable streams). The readers use it to bound header-declared
// counts BEFORE reserving memory, so a hostile 84-byte file cannot
// request gigabytes.
//
// `parse_real(token, out)` parses a whole whitespace-delimited token
// as a floating-point number via `std::from_chars`. Unlike `>>`, a
// token such as "1,5" or "1.0abc" is rejected outright instead of
// being split into a number and a leftover that confuses the next
// extraction. A leading '+' is accepted (`from_chars` alone refuses
// it). "nan" / "inf" parse successfully; callers that need finite
// values check `std::isfinite` themselves.

#include <charconv>
#include <cstdint>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace rvegen::detail {

inline std::string position_marker(std::istream& in) {
  in.clear();
  if (auto pos = in.tellg(); pos >= 0) {
    return " (at byte " + std::to_string(static_cast<long long>(pos)) + ")";
  }
  return "";
}

inline std::optional<std::uint64_t> remaining_bytes(std::istream& in) {
  if (!in) return std::nullopt;
  const auto cur = in.tellg();
  if (cur < 0) return std::nullopt;
  if (!in.seekg(0, std::ios::end)) {
    in.clear();
    in.seekg(cur);
    return std::nullopt;
  }
  const auto end = in.tellg();
  in.seekg(cur);
  if (end < cur) return std::nullopt;
  return static_cast<std::uint64_t>(end - cur);
}

inline bool parse_real(std::string_view token, double& out) {
  if (!token.empty() && token.front() == '+') token.remove_prefix(1);
  if (token.empty()) return false;
  const char* first = token.data();
  const char* last = token.data() + token.size();
  const auto [ptr, ec] = std::from_chars(first, last, out);
  return ec == std::errc{} && ptr == last;
}

} // namespace rvegen::detail
