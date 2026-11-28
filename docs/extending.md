# Extending rvegen

Every selectable part of a config is a type registered by name in one of the
registries under `include/rvegen/registry/`. Each registry is a
`numsim_core::object_registry` that maps the JSON `"type"` string to a class.
A registered class provides

- a static `parameters()` returning the `parameter_controller_t` schema (field
  names, `is_required`, ranges, `description_label`), and
- a constructor taking the arguments the registry passes (see the table).

The CLI (`apps/rvegen_cli.cpp`, `register_everything()`) calls
`register_all_*()` for each registry once at start-up, then builds each config
section with `build_from_json` (`registry/build.h`): it reads `"type"`, asks the
registry for the schema, fills a `parameter_handler_t` from the JSON and calls
`registry.create(type, handler, extra_args...)`. Nothing else in the CLI or the
pipeline has to change for a new type.

| Registry (header)                              | Base class            | Constructor arguments                      |
|------------------------------------------------|-----------------------|--------------------------------------------|
| `input_registry_t` (`register_inputs.h`)       | `shape_input_base<T>` | `(parameter_handler_t const&, distribution_map_t<T> const&)` |
| `generator_registry_t` (`register_generators.h`) | `rve_generator_base<T>` | `(parameter_handler_t const&)`           |
| `post_process_registry_t` (`register_post_processes.h`) | `post_process_base<T>` | `(parameter_handler_t const&)`    |
| `termination_registry_t` (`register_terminations.h`) | `termination_base<T>` | `(parameter_handler_t const&, std::array<T,3> const& domain_box)` |

Distributions live in `register_distributions.h` and follow the same pattern.

## Adding a shape

1. **Shape class** under `include/rvegen/shapes/`. Derive from
   `numsim_core::static_indexing<my_shape<T>, shape_base<T>>` (see `circle.h`)
   and implement the pure virtuals of `shape_base<T>` (`shapes/shape_base.h`):
   `get_middle_point`, `set_middle_point`, `area`, `volume`, `max_expansion`,
   `make_bounding_box`, `is_inside`, `clone` and `shape_id`.
2. **Input class** under `include/rvegen/inputs/`, derived from
   `shape_input_base<T>`: a schema-driven constructor
   `(parameter_handler_t const&, distribution_map_t<T> const&)` that looks the
   named distributions up in the map, calls `this->read_metadata(handler)`, a
   static `parameters()`, and `new_shape()` which builds the shape and calls
   `this->tag(*shape)` (see `inputs/circle_input.h`).
3. **Register the input** in `registry/register_inputs.h`: add a function like
   `register_circle_input()`
   (`input_registry_t<T>::instance().template register_type<my_input<T>>("my_input")`)
   and call it from `register_all_inputs()`.
4. **Collisions** (optional): pairs registered in
   `registry/register_collisions.h` (`register_pair<A, B>()`) get the precise
   narrow-phase test; unregistered pairs fall back to the bounding-box overlap
   test, as that header states.
5. **Visualisation** (optional): add a free function
   `to_mesh(my_shape<T> const&)` returning a `triangle_mesh<T>` under
   `include/rvegen/visualization/` (see `circle_mesh.h`) and register it in
   `registry/register_meshes.h` with
   `d.template register_shape<my_shape<T>>()`. Shapes without a registered mesh
   give an empty mesh, which renderers skip.
6. **Tests.** Add the shape to `tests/extra_types_smoke.cpp` and its id to the
   uniqueness set in `tests/shape_id_unique_test.cpp`. The dispatchers claim
   each `static_indexing` id at registration (`shapes/shape_id_claims.h`) and
   throw `std::logic_error` if two shape types share one.

The JSON can use `"type": "my_input"` as soon as step 3 is in place.

## Adding a generator

Derive from `rve_generator_base<T>` (`generators/rve_generator_base.h`) and
implement `compute(inputs, termination, domain_box, progress_options const&)`,
returning the accepted shapes. Provide a constructor from
`parameter_handler_t const&` and a static `parameters()` (see
`generators/random_generator.h`). Register it in `registry/register_generators.h`
with `generator_registry_t<T>::instance().template register_type<...>("name")`
and add the call to `register_all_generators()`.

## Adding a post-process

Derive from `post_process_base<T>` (`postprocess/post_process_base.h`) and
implement `run(shapes, domain_box) const`. A post-process owns its output
file, so its schema normally has a required `output_path` (see
`postprocess/voxel_writer.h`). Provide the constructor from
`parameter_handler_t const&` and `parameters()`, and register it in
`registry/register_post_processes.h` inside a `register_*()` function that
`register_all_post_processes()` calls. A config may list any number of
post-processes in its `postprocess` array; they run in order.

Put every check that needs only the options and the domain box (a missing
output path, a grid that is too large, an option that needs a 3D box) into
`validate(domain_box) const` and call it at the top of `run()`. The pipeline
calls `validate` for all post-processes before the first shape is placed, and
`rvegen --check` runs it without generating anything, so such a mistake does
not cost a generation run. `validate` must not create or truncate a file.

## Checking the result

- `rvegen --list-types` should list the new name. Update the catalogue in
  `README.md` with `python3 tests/readme_types_check.py --rvegen build/rvegen
  --readme README.md --update`; the `readme_type_catalogue` test fails when it
  is out of date.
- `ctest -R schema_audit_test` checks that every schema field carries a
  description and that integral fields have a range or unit.
- Add an example config under `examples/`; every file there gets a
  `cli_check_<name>` test that runs the CLI and inspects the files written.
