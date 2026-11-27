#pragma once

// field_list — compile-time schema declaration that collapses three-way
// duplication into one source of truth.
//
// Today, every registered rvegen type declares its members in three
// different places:
//   1. C++ class fields (e.g. `T _radius;`)
//   2. The `parameters()` schema (e.g. `s.insert<T>("radius").add<is_required>();`)
//   3. The schema-driven ctor's name-by-name extraction (e.g.
//      `_radius{handler.get<T>("radius")}`)
//
// Drift between the three is a recurring source of bugs. This header
// provides the scaffolding to express the schema once via a
// `field_list<field<"name", T>...>` type alias, and have both
// `parameters()` and the ctor's tuple-of-values derive from that single
// declaration.
//
// What this header lands today (phase 1 of #1):
//   * `field<Name, T, Required = true>` — single-field declaration.
//   * `field_list<Fields...>::schema()` — produces a
//     `parameter_controller_t` matching the field declarations.
//   * `field_list<Fields...>::extract(handler)` — pulls the named
//     values out of a `parameter_handler_t` into a `std::tuple<>`.
//   * Compile-time uniqueness check on field names.
//   * Standalone unit tests that exercise both, with no migration of
//     existing types.
//
// Phase 2 (this PR): per-field annotations. `field` now accepts a
// trailing pack of annotation types that are applied to the schema
// slot via `slot.add<A>()` — exactly the same set the hand-written
// `parameters()` declarations use today. Typical examples:
//   * `numsim_core::description_label<"...">`
//   * `numsim_core::unit_label<"...">`
//   * `min_only<X>` / `max_only<X>` / `range<X, Y>`
//
// `is_required` is still expressed via the boolean `Required` template
// parameter (the canonical way to opt in to required-ness in rvegen);
// the annotation pack carries everything else.
//
// Phase 3: optional fields.
//   * `field<Name, T, false, ...>` — optional, no default. Emitted
//     without `is_required`; `extract()` yields `std::optional<T>`
//     (empty when the key is absent).
//   * `field_with_default<Name, T, Default, ...>` — optional with a
//     compile-time default. Emitted without `is_required` and with
//     numsim-core's `set_default` check (added before the annotations,
//     so range checks also see the default), which makes
//     `input_parameter_controller::accept()` insert the default into
//     the handler when the key is missing. `extract()` also falls back
//     to the default, so a hand-built handler that never went through
//     `accept()` behaves the same. `Default` is an NTTP: an arithmetic
//     or bool value convertible to T, or a string literal (via
//     numsim_core::fixed_string) for std::string fields.
//     Limitation: numsim-core has no hint side-base for defaults
//     (`set_default` keeps its value private), so a GUI walking the
//     schema can tell that a default exists (dynamic_cast to
//     `set_default<T, K, H>`) but cannot read its value. Put the
//     default in the field's description_label until numsim-core grows
//     a `default_hint_base`.
//   * `field_list::value<"name">(handler)` — extracts one field by
//     name (same semantics as `extract()`), so ctors that initialise
//     members one at a time don't index the tuple by position.

#include <array>
#include <cstddef>
#include <string>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>

#include <numsim-core/input_parameter_controller.h>

#include "../types.h"

namespace rvegen {

namespace detail {

// True if converting `d` to T and back reproduces `d` exactly and keeps
// its sign, i.e. a `field_with_default` default is not silently narrowed
// (e.g. -1 into size_t, 0.1 into float, 1.5 into int).
template <typename T, typename D>
[[nodiscard]] constexpr bool default_is_lossless(D d) {
  if constexpr (std::is_arithmetic_v<T> && std::is_arithmetic_v<D>) {
    const T v = static_cast<T>(d);
    return static_cast<D>(v) == d && ((v < T{}) == (d < D{}));
  } else {
    return true;
  }
}

} // namespace detail

// One field declaration. `Name` is a compile-time string (NTTP via
// numsim_core::fixed_string), `T` is the value type, `Required`
// determines whether the field is added with `is_required` in the
// emitted schema, and `Annotations...` is any pack of annotation tag
// types that the parameter slot understands (description_label,
// unit_label, range, min_only, etc.).
//
// Default `Required = true` mirrors the rvegen convention that nearly
// every schema field is required (matches the explicit `is_required`
// add()s in every existing `parameters()` declaration). Pass
// `Required=false` for an optional field without a default (extracted
// as std::optional<T>), or use `field_with_default` below.
//
// Annotation types must be valid arguments to `slot.add<A>()` — they
// are folded into the slot at schema-build time. Example:
//
//   using fields = field_list<
//     field<"radius", double, true,
//           numsim_core::description_label<"sphere radius">,
//           numsim_core::unit_label<"m">,
//           min_only<0.0>>,
//     ...>;
template <numsim_core::fixed_string Name, typename T, bool Required = true,
          typename... Annotations>
struct field {
  static constexpr auto name = Name;
  using value_type = T;
  static constexpr bool required = Required;
  static constexpr bool has_default = false;
  using annotations = std::tuple<Annotations...>;
};

// Optional field with a compile-time default; see the header comment.
//
//   field_with_default<"periodic", bool, false,
//                      numsim_core::description_label<"... (default false)">>
//   field_with_default<"label", std::string, numsim_core::fixed_string{"abc"},
//                      numsim_core::description_label<"...">>
template <numsim_core::fixed_string Name, typename T, auto Default,
          typename... Annotations>
struct field_with_default {
  static constexpr auto name = Name;
  using value_type = T;
  static constexpr bool required = false;
  static constexpr bool has_default = true;
  using annotations = std::tuple<Annotations...>;

  [[nodiscard]] static T default_value() {
    if constexpr (std::is_same_v<T, std::string>) {
      return std::string{Default.view()};
    } else {
      static_assert(std::is_convertible_v<decltype(Default), T>,
                    "field_with_default: Default must convert to T");
      static_assert(detail::default_is_lossless<T>(Default),
                    "field_with_default: Default does not survive conversion "
                    "to T unchanged (narrowing, sign change or truncation)");
      return static_cast<T>(Default);
    }
  }
};

namespace detail {

// Compile-time check that a list of field names is unique. Catches the
// `field_list<field<"x", T>, field<"y", T>, field<"x", T>>` typo at
// type definition rather than at runtime. O(N²) — fine for typical
// schemas (3–7 fields, never more than ~20).
template <typename... Fields>
constexpr bool field_names_unique() {
  std::array<std::string_view, sizeof...(Fields)> names{
      Fields::name.view()...};
  for (std::size_t i = 0; i < names.size(); ++i) {
    for (std::size_t j = i + 1; j < names.size(); ++j) {
      if (names[i] == names[j]) return false;
    }
  }
  return true;
}

} // namespace detail

// A pack of fields. `schema()` builds the `parameter_controller_t`;
// `extract()` pulls a tuple of values out of a `parameter_handler_t`.
//
// Note: `extract()` returns the values by value via `std::make_tuple`,
// which copies. `parameter_handler::get<T>` returns `T const&`, so a
// string-typed field is copied once per `extract()`. Negligible at
// construction time (handler.get is one-shot per shape ctor) and
// in line with how the existing inputs read from the handler today.
template <typename... Fields>
struct field_list {
  static_assert(detail::field_names_unique<Fields...>(),
                "field_list: duplicate field names");

  [[nodiscard]] static parameter_controller_t schema() {
    parameter_controller_t s;
    (insert_one<Fields>(s), ...);
    return s;
  }

  // Returns a tuple with one element per field, in declaration order,
  // populated by name-keyed lookups against the supplied handler:
  //   * required field            -> T (throws if the key is missing)
  //   * field<..., false>         -> std::optional<T>
  //   * field_with_default<...>   -> T (the default if the key is missing)
  [[nodiscard]] static auto extract(parameter_handler_t const& handler) {
    return std::make_tuple(extract_one<Fields>(handler)...);
  }

  // Index of the field called `Name` (compile error if there is none).
  template <numsim_core::fixed_string Name>
  [[nodiscard]] static consteval std::size_t index_of() {
    constexpr std::array<std::string_view, sizeof...(Fields)> names{
        Fields::name.view()...};
    for (std::size_t i = 0; i < names.size(); ++i) {
      if (names[i] == Name.view()) return i;
    }
    throw "field_list::index_of: no field with this name";
  }

  // Extracts the single field called `Name`, with the same semantics as
  // the corresponding element of `extract()`. Lets ctors that
  // initialise members one by one read by name:
  //   _nx{fields::template value<"nx">(handler)}
  template <numsim_core::fixed_string Name>
  [[nodiscard]] static auto value(parameter_handler_t const& handler) {
    using F = std::tuple_element_t<index_of<Name>(), std::tuple<Fields...>>;
    return extract_one<F>(handler);
  }

private:
  template <typename F>
  [[nodiscard]] static auto extract_one(parameter_handler_t const& handler) {
    using T = typename F::value_type;
    const std::string key{F::name.view()};
    if constexpr (F::has_default) {
      return handler.contains(key) ? T{handler.template get<T>(key)}
                                   : F::default_value();
    } else if constexpr (!F::required) {
      return handler.contains(key)
                 ? std::optional<T>{handler.template get<T>(key)}
                 : std::optional<T>{};
    } else {
      return T{handler.template get<T>(key)};
    }
  }

  template <typename F>
  static void insert_one(parameter_controller_t& s) {
    auto& slot = s.template insert<typename F::value_type>(
        std::string{F::name.view()});
    if constexpr (F::required) {
      slot.template add<numsim_core::is_required>();
    }
    if constexpr (F::has_default) {
      slot.template add<numsim_core::set_default>(F::default_value());
    }
    apply_annotations<F>(slot, typename F::annotations{});
  }

  // Fold the annotation pack — held as `std::tuple<A...>` inside the
  // field — into individual `slot.add<A>()` calls.
  template <typename F, typename Slot, typename... A>
  static void apply_annotations(Slot& slot, std::tuple<A...>) {
    (slot.template add<A>(), ...);
  }
};

} // namespace rvegen
