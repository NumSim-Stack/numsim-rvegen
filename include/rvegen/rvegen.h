#pragma once

// rvegen — RVE (Representative Volume Element) generation for FFT/FEM
// homogenization. Header-only library; consumers include this umbrella
// and link no compiled artifact.
//
// Layered architecture:
//   - Library types (this tree) are JSON-naive plain C++ classes.
//   - The adapter layer (rvegen/json/, rvegen/registry/) translates JSON
//     config + numsim-core registries into library types.
//   - Output writers (rvegen/output/) consume the library's shape vector
//     and emit gmsh .geo, voxel grids, VTK, etc.
//
// Threading model:
//   * Engine injection through distributions means rvegen has no global
//     mutable state on the hot path. Each thread builds its own engine
//     (e.g. std::mt19937), its own distributions referencing that engine,
//     and its own pipeline; threads never share these.
//   * Registry singletons (object_registry::instance()) are populated
//     once via register_all_*() at startup, then read-only. Concurrent
//     create() / schema() calls are safe; concurrent register_type calls
//     during the parallel section are not.
//   * Same seed → same shape sequence, regardless of thread count or
//     interleaving. This is verified by tests/concurrency_smoke.cpp.
//
// See REFACTORING_PLAN.md for the phase-by-phase migration roadmap.

// shapes
#include "shapes/shape_base.h"
#include "shapes/bounding_box_base.h"
#include "shapes/rectangle_bounding.h"
#include "shapes/box_bounding.h"
#include "shapes/circle.h"
#include "shapes/sphere.h"
#include "shapes/rectangle.h"
#include "shapes/box.h"
#include "shapes/ellipse.h"
#include "shapes/polyline_tube.h"
#include "shapes/voronoi_cell.h"
#include "shapes/mesh_inclusion.h"
#include "shapes/shape_variant.h"
#include "shapes/shape_pool.h"

// intersection
#include "intersection/collision_details.h"
#include "intersection/aabb_overlap.h"
#include "intersection/collision_dispatcher.h"

// types
#include "types.h"

// distributions
#include "distributions/distribution_base.h"
#include "distributions/direction_distribution_base.h"
#include "distributions/uniform_real_distribution.h"
#include "distributions/normal_distribution.h"
#include "distributions/constant_distribution.h"
#include "distributions/bingham_distribution.h"

// registry (opt-in: pulls in object_registry)
#include "registry/register_distributions.h"
#include "registry/register_terminations.h"
#include "registry/register_generators.h"
#include "registry/register_inputs.h"
#include "registry/register_collisions.h"
#include "registry/register_post_processes.h"
#include "registry/register_meshes.h"
#include "registry/build.h"

// visualization (boundary types between rvegen shapes and out-of-tree
// renderers like Tessera's VTK viewport)
#include "visualization/triangle_mesh.h"
#include "visualization/mesh_dispatcher.h"
#include "visualization/box_mesh.h"
#include "visualization/circle_mesh.h"
#include "visualization/ellipse_mesh.h"
#include "visualization/polyline_tube_mesh.h"
#include "visualization/rectangle_mesh.h"
#include "visualization/sphere_mesh.h"

// json (opt-in: pulls in nlohmann)
#include "json/parameter_visitor_nlohmann.h"

// metadata — generic string-to-value container attached to shapes and inputs
#include "metadata/info.h"

// io — STL / PLY mesh readers
#include "io/stl_reader.h"
#include "io/ply_reader.h"
#include "io/mesh_reader.h"
// schema scaffolding — collapse name/type duplication into one declaration
#include "schema/field_list.h"

// homogenization — analytical bounds (Voigt / Reuss / Hill / MT / HS / SC)
#include "homogenization/mean_field.h"
#include "homogenization/phase_bridge.h"
#include "homogenization/homogenize.h"

// phase model — named regions with opaque material_config blobs that
// downstream solvers (numsim-materials) interpret.
#include "phase/phase.h"

// inputs
#include "inputs/shape_input_base.h"
#include "inputs/circle_input.h"
#include "inputs/sphere_input.h"
#include "inputs/rectangle_input.h"
#include "inputs/box_input.h"
#include "inputs/ellipse_input.h"
#include "inputs/polyline_tube_directional_input.h"
#include "inputs/polyline_tube_oriented_input.h"
#include "inputs/polyline_tube_input.h"
#include "inputs/mesh_inclusion_input.h"

// termination
#include "termination/termination_base.h"
#include "termination/number_of_inclusions.h"
#include "termination/volume_fraction.h"
#include "termination/until_full.h"

// generators
#include "generators/rve_generator_base.h"
#include "generators/only_inside_generator.h"
#include "generators/periodic_generator.h"
#include "generators/random_generator.h"

// postprocess (writers + the base they inherit from)
#include "postprocess/post_process_base.h"
#include "postprocess/voxel_grid.h"
#include "postprocess/gmsh_geo_writer.h"
#include "postprocess/svg_writer.h"
#include "postprocess/svg_3d_writer.h"
#include "postprocess/three_js_writer.h"
#include "postprocess/voxel_writer.h"
#include "postprocess/vtk_legacy_writer.h"
