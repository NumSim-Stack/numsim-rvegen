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

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

namespace rvegen::detail {

// Locale-independent ASCII helpers. File formats are ASCII by spec;
// std::tolower / std::isprint depend on the global C locale and on
// the sign of `char`, so the readers use these instead.
constexpr char ascii_lower(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

constexpr bool ascii_is_text(unsigned char c) noexcept {
  return (c >= 0x20 && c <= 0x7e) || c == '\t' || c == '\n' ||
         c == '\r' || c == '\v' || c == '\f';
}

inline std::string ascii_to_lower(std::string s) {
  for (auto& c : s) c = ascii_lower(c);
  return s;
}

// Consume a leading UTF-8 byte-order mark (EF BB BF) if present.
// Works on non-seekable streams via peek(); a stream that starts with
// a partial BOM is not valid ASCII anyway and fails in the parser.
inline void skip_utf8_bom(std::istream& in) {
  constexpr unsigned char bom[3] = {0xEF, 0xBB, 0xBF};
  for (unsigned char b : bom) {
    if (in.peek() != static_cast<int>(b)) return;
    in.get();
  }
}

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

// parse_real for standard libraries without floating-point from_chars
// (libc++ before 20): strtod on a copy, restricted to what from_chars
// accepts (no leading white space, no hexadecimal form) and to the whole
// token. strtod follows the C locale's decimal point; rvegen never changes
// the locale. Always compiled, so the tests compare it with from_chars
// where both exist. `token` is non-empty and carries no leading '+'.
inline bool parse_real_strtod(std::string_view token, double& out) {
  if (token.empty()) return false;
  if (std::isspace(static_cast<unsigned char>(token.front()))) return false;
  const std::size_t digits = token.front() == '-' ? 1 : 0;
  if (token.size() > digits + 1 && token[digits] == '0' &&
      (token[digits + 1] == 'x' || token[digits + 1] == 'X')) {
    return false;
  }
  if (token.size() > digits && token[digits] == '+') return false;   // "-+1"
  const std::string copy{token};
  char* end = nullptr;
  errno = 0;
  const double value = std::strtod(copy.c_str(), &end);
  if (end != copy.c_str() + copy.size()) return false;
  // from_chars reports result_out_of_range for overflow and for underflow
  // to zero, but accepts a subnormal result (for which strtod also sets
  // ERANGE).
  if (errno == ERANGE && (value == 0.0 || value == HUGE_VAL || value == -HUGE_VAL))
    return false;
  out = value;
  return true;
}

inline bool parse_real(std::string_view token, double& out) {
  if (!token.empty() && token.front() == '+') {
    token.remove_prefix(1);
    // "+-1" / "++1": from_chars would accept the '-' after our strip.
    if (!token.empty() && (token.front() == '-' || token.front() == '+')) {
      return false;
    }
  }
  if (token.empty()) return false;
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
  const char* first = token.data();
  const char* last = token.data() + token.size();
  const auto [ptr, ec] = std::from_chars(first, last, out);
  return ec == std::errc{} && ptr == last;
#else
  return parse_real_strtod(token, out);
#endif
}

} // namespace rvegen::detail
