#pragma once

// What a generator achieved against what was asked for (#119). A placement
// generator also returns when it runs out of attempts; the report says so
// instead of leaving the caller to measure the result.

#include <cmath>
#include <cstddef>
#include <format>
#include <string>

namespace rvegen {

enum class stop_reason {
  reached,       // the termination predicate was met
  max_attempts,  // the generator ran out of attempts first
  cancelled,     // the cancel callback asked it to stop
  no_inputs,     // there was no shape input to draw from
};

[[nodiscard]] constexpr char const* to_string(stop_reason r) noexcept {
  switch (r) {
    case stop_reason::reached:      return "reached";
    case stop_reason::max_attempts: return "max_attempts";
    case stop_reason::cancelled:    return "cancelled";
    case stop_reason::no_inputs:    return "no_inputs";
  }
  return "?";
}

struct generation_report {
  // termination_base::kind() of the predicate ("volume_fraction",
  // "number_of_inclusions", "until_full"). Empty: the generator does not
  // place towards a target (the deterministic generators).
  std::string termination;
  // false: the predicate never stops by itself (until_full), so running
  // out of attempts is the expected end, not a missed target.
  bool has_target{false};
  double target{0};     // requested value, in the predicate's own unit
  double achieved{0};   // value the returned shapes have
  std::size_t attempts{0};       // placement attempts made
  std::size_t max_attempts{0};   // the generator's limit; 0 = not known
  // Minimum distances the placement kept (#111, placement_gap.h); 0 where
  // the generator has no such setting or it was not used.
  double min_gap{0};
  double min_gap_relative{0};
  double wall_gap{0};
  stop_reason reason{stop_reason::reached};

  [[nodiscard]] bool target_missed() const noexcept {
    return has_target && reason != stop_reason::reached;
  }

  // One line: "volume_fraction: reached 0.223 of 0.35 after 50 attempts
  // (max_attempts)". Empty when there is no termination to report on.
  [[nodiscard]] std::string summary() const {
    if (termination.empty()) return {};
    auto number = [](double v, int digits) {
      // Counts print as integers, fractions with `digits` significant digits.
      if (std::isfinite(v) && v == std::floor(v) && std::abs(v) < 1e15)
        return std::format("{:.0f}", v);
      return std::format("{:.{}g}", v, digits);
    };
    std::string line = termination + ": ";
    if (has_target) {
      // Four significant digits, more when a miss would otherwise print
      // like a hit ("reached 0.35 of 0.35" for 0.34996).
      int digits = 4;
      std::string got = number(achieved, digits), want = number(target, digits);
      while (achieved != target && got == want && digits < 17) {
        ++digits;
        got = number(achieved, digits);
        want = number(target, digits);
      }
      line += "reached " + got + " of " + want;
    } else {
      line += "stopped";
    }
    if (attempts > 0 || reason == stop_reason::no_inputs)
      line += " after " + std::to_string(attempts) + " attempt" +
              (attempts == 1 ? "" : "s");
    if (reason != stop_reason::reached)
      line += std::string{" ("} + to_string(reason) + ")";
    return line;
  }
};

} // namespace rvegen
