#pragma once

namespace rvegen {

// Polymorphic random scalar source. Concrete implementations
// (uniform_real_distribution, normal_distribution, constant_distribution, ...)
// hold their parameters and an engine reference; calling operator() draws one
// value.
//
// Engine injection: every concrete distribution takes its engine by reference
// at construction. This is the architecturally correct alternative to a global
// singleton — distributions are reproducible (same seed → same sequence) and
// safe to use across threads when each thread has its own engine.
//
// Threading: a distribution and its engine form a unit. Sharing either across
// threads is undefined behaviour (std::mt19937 is not thread-safe). Each
// thread constructs its own engine and its own distributions referencing
// that engine. See tests/concurrency_smoke.cpp.
template <typename T>
class distribution_base {
public:
  using value_type = T;

  distribution_base() = default;
  distribution_base(distribution_base const&) = default;
  distribution_base(distribution_base&&) noexcept = default;
  distribution_base& operator=(distribution_base const&) = default;
  distribution_base& operator=(distribution_base&&) noexcept = default;
  virtual ~distribution_base() = default;

  [[nodiscard]] virtual value_type operator()() = 0;
};

} // namespace rvegen
