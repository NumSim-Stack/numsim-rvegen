# #119 — Wire `shapes_target` and `volume_fraction` into `progress_info`

> Drafted issue + PR text for when rvegen lands on GitHub. Branch:
> `feature/wire-progress-targets`.

## Issue body

The progress callback (`progress_options::on_progress`) receives a
`progress_info{shapes_placed, shapes_target, volume_fraction}`, but
all three generators (`only_inside`, `random`, `periodic`) hardcode
`shapes_target = 0` and `volume_fraction = 0.0`. Only
`shapes_placed` is meaningfully populated.

Means the GUI's progress bar can show "X shapes placed" but cannot
display "X / N" or a percentage of the target volume fraction without
side-channel polling. The infrastructure exists; the wiring doesn't.

### Proposed fix

Surface termination targets through optional accessors on
`termination_base<T>` (default returns "unknown"), populate them in
each generator before invoking `on_progress`:

```cpp
// On termination_base<T>:
[[nodiscard]] virtual std::size_t target_count() const noexcept { return 0; }
[[nodiscard]] virtual double target_volume_fraction() const noexcept { return 0.0; }

// In number_of_inclusions: return _target;
// In volume_fraction:      return static_cast<double>(_target);
```

Generators read these via the termination they hold and stuff them
into `progress_info` before the callback. A free helper
`current_volume_fraction(accepted, domain_box)` (in
`rve_generator_base.h`) computes the live fraction so it can be
shared across all generators.

### Acceptance

- [ ] All three generators populate `target_count` and
      `volume_fraction` (live measured) on each callback.
- [ ] `progress_cancel_test` extended to assert `shapes_target` is
      non-zero when `number_of_inclusions` termination is in use.
- [ ] All existing ctest still green.

## PR body

Closes #119.

### Summary

- Adds two virtual accessors on `termination_base<T>` —
  `target_count()` and `target_volume_fraction()` — defaulted to 0
  (meaning "unknown") and overridden by the relevant terminations.
- Adds a free helper `current_volume_fraction(accepted, domain_box)`
  in `rve_generator_base.h` that filters wrap copies via
  centre-in-domain and sums area (2D) or volume (3D) over the unit
  cell.
- Updates the three generators to populate `progress_info` from
  these sources on every `on_progress` call.

The defaults of 0 keep terminations like `until_full` working
without modification — the GUI displays "no target" rather than a
mis-computed percentage when the field reads zero.

### Test plan

- 15/15 ctest green; new test in `progress_cancel_test` covers the
  target-count surfacing.
- Manual: progress bar in Tessera now shows "X / N" and
  "P%" without the GUI needing to interrogate the termination
  itself.
