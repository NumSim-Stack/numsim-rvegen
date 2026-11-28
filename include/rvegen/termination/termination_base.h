#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../domain.h"
#include "../shapes/shape_base.h"

namespace rvegen {

// What a criterion has added up over the first `seen` accepted shapes. A
// generator keeps one per run and passes it to is_met() after every
// acceptance, so a criterion that adds something up per shape looks at the
// new shapes only instead of at all of them: with a pass over all accepted
// shapes per acceptance a run costs the square of the number of inclusions
// (#153). A tally belongs to one criterion and one list: between two calls
// with the same tally the accepted list may only grow at its end.
template <typename T>
struct termination_tally {
  std::size_t seen{0};    // shapes of `accepted` already included
  T sum{};                // for criteria that add a measure
  std::size_t count{0};   // for criteria that count
};

// Polymorphic stop predicate for the generator's accept loop.
// operator() returns true to halt generation. Common implementations:
//   * number_of_inclusions: stop after N accepted shapes
//   * volume_fraction:      stop after sum-of-volumes >= target ratio of box
//   * max_iterations:       stop after N attempts (success or failure)
template <typename T>
class termination_base {
public:
  using value_type = T;
  using shape_vector = std::vector<std::unique_ptr<shape_base<T>>>;

  termination_base() = default;
  termination_base(termination_base const&) = default;
  termination_base(termination_base&&) noexcept = default;
  termination_base& operator=(termination_base const&) = default;
  termination_base& operator=(termination_base&&) noexcept = default;
  virtual ~termination_base() = default;

  [[nodiscard]] virtual bool operator()(shape_vector const& accepted) const = 0;

  // Same predicate with a hint. `count_all == false` is the operator()
  // behaviour: periodic copies (shape_base::is_primary() == false) are not
  // counted again. `count_all == true` is the caller's promise that the
  // vector holds no periodic copies (only_inside_generator,
  // random_generator), which lets a counting termination skip the pass
  // over the shapes. Terminations that do not distinguish ignore it.
  [[nodiscard]] virtual bool is_met(shape_vector const& accepted,
                                    bool /*count_all*/) const {
    return (*this)(accepted);
  }

  // is_met() with a running tally. The answer is the one is_met(accepted,
  // count_all) gives, bit for bit: an overrider adds the new shapes in the
  // order of the list, which is the order of the full pass. The default is
  // the full pass, so a criterion that keeps no tally needs no override.
  [[nodiscard]] virtual bool is_met(shape_vector const& accepted,
                                    bool count_all,
                                    termination_tally<T>& /*tally*/) const {
    return is_met(accepted, count_all);
  }

  // Optional progress hooks consumed by the generator base when filling
  // `progress_info` for the on_progress callback. Concrete terminations
  // override whichever applies; the defaults of 0 mean "unknown" and
  // surface in Tessera's progress bar as a count-only display rather
  // than a percentage.
  [[nodiscard]] virtual std::size_t target_count() const noexcept { return 0; }
  [[nodiscard]] virtual double target_volume_fraction() const noexcept { return 0.0; }

  // What the predicate measures, for the generation report (#119):
  //   kind()            its name ("volume_fraction", ...);
  //   has_target()      false for a predicate that never stops by itself
  //                     (until_full): running out of attempts is then the
  //                     expected end;
  //   target_value()    the requested value;
  //   achieved_value()  the value `accepted` has, by the same accounting
  //                     (and the same `count_all` hint) as is_met().
  // The defaults describe a predicate without a number to report.
  [[nodiscard]] virtual std::string kind() const { return "termination"; }
  [[nodiscard]] virtual bool has_target() const noexcept { return true; }
  [[nodiscard]] virtual double target_value() const noexcept { return 0.0; }
  [[nodiscard]] virtual double achieved_value(shape_vector const& /*accepted*/,
                                              bool /*count_all*/) const {
    return 0.0;
  }
};

} // namespace rvegen
