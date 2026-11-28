cmake_minimum_required(VERSION 3.20)

project(rvegen
    VERSION 0.1.0
    LANGUAGES CXX
)

include(FetchContent)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Compiler capability gate.
#
# numsim-core uses <std::print> from C++23, which lands in libstdc++ 14+
# and libc++ 17+. Without this check, an older compiler silently produces a
# wall of template-instantiation noise on the first numsim-core include.
# Catch it early with a clear, actionable error.
# ---------------------------------------------------------------------------
include(CheckCXXSourceCompiles)
set(_rvegen_check_flags_save "${CMAKE_REQUIRED_FLAGS}")
set(CMAKE_REQUIRED_FLAGS "-std=c++23")
check_cxx_source_compiles(
  "#include <print>
   int main() { std::print(\"x\"); }"
  RVEGEN_HAS_STD_PRINT)
set(CMAKE_REQUIRED_FLAGS "${_rvegen_check_flags_save}")
unset(_rvegen_check_flags_save)
if(NOT RVEGEN_HAS_STD_PRINT)
  message(FATAL_ERROR
    "rvegen requires <print> from libstdc++ 14+ / libc++ 17+; "
    "your ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} doesn't have it. "
    "Try: cmake -DCMAKE_CXX_COMPILER=g++-14 ..")
endif()

# Default to Release if the user didn't pick a build type. Without this
# CMake leaves CXX_FLAGS empty (effectively -O0) — silently catastrophic
# for the benchmarks and for any production use. Override with
# `cmake -DCMAKE_BUILD_TYPE=Debug` explicitly if needed.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No CMAKE_BUILD_TYPE set — defaulting to Release.")
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---------------------------------------------------------------------------
# Dependencies.
#
# Header-only Boost (boost::container::flat_map in the collision_dispatcher),
# nlohmann_json for config + writers, Eigen3 for shape-side rotation math,
# and the vendored GeometricTools headers under external/.
# ---------------------------------------------------------------------------
find_package(Boost REQUIRED)

find_package(Eigen3 QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found via find_package, fetching with FetchContent...")
  FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
  )
  FetchContent_MakeAvailable(Eigen3)
endif()

find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  message(STATUS "nlohmann/json not found, fetching with FetchContent...")
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.12.0
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

# ---------------------------------------------------------------------------
# rvegen library (header-only).
#
# In-tree consumers (apps/, tests/, examples/) get the BUILD_INTERFACE
# include paths; downstream consumers that pull rvegen via the install
# tree (vcpkg / conda-forge / `find_package(rvegen)`) get
# INSTALL_INTERFACE paths so the CMake config relocates correctly.
#
# numsim-core: rvegen depends on numsim-core (>= 1.3), found below through
# its installed CMake package or, as a fallback, a sibling checkout. The
# rvegen install does NOT bundle numsim-core headers - that would be a
# relicensing concern and goes against single-source-of-truth packaging;
# the installed rvegen config re-finds the package via find_dependency.
#
# The vendored GTE headers ARE installed alongside rvegen's own
# headers (separately licenced under Boost; bundling is fine).
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

add_library(rvegen_lib INTERFACE)
add_library(rvegen::rvegen ALIAS rvegen_lib)
# Make the install-tree target name match the in-tree alias: callers
# always say `rvegen::rvegen` regardless of which CMake context they're
# in (build tree via add_subdirectory, or install tree via find_package).
set_target_properties(rvegen_lib PROPERTIES EXPORT_NAME rvegen)

target_include_directories(rvegen_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/GeometricTools/GTE>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rvegen/external>
)
target_link_libraries(rvegen_lib INTERFACE
    Eigen3::Eigen
    Boost::headers
    nlohmann_json::nlohmann_json
)
target_compile_features(rvegen_lib INTERFACE cxx_std_23)

# numsim-core: prefer its installed CMake package (target
# numsim-core::numsim-core, shipped since numsim-core ef6d2ac). Point
# CMAKE_PREFIX_PATH / numsim-core_DIR at the install prefix. Version 1.3 is
# the minimum: it has the lazy static_indexing ids the shape-id guard
# needs; an older package is treated as not found. If no package is found,
# fall back to a sibling checkout (../../numsim-core/include) so
# the historical layout keeps working. When both exist the package wins;
# if it is stale (e.g. an old install in /usr/local), configure with
# -DCMAKE_DISABLE_FIND_PACKAGE_numsim-core=ON to use the sibling instead.
find_package(numsim-core 1.3 CONFIG QUIET)
set(RVEGEN_NUMSIM_CORE_FROM_PACKAGE OFF)
set(_rvegen_core_sibling "${CMAKE_CURRENT_SOURCE_DIR}/../../numsim-core/include")
if(numsim-core_FOUND)
  message(STATUS "numsim-core: using CMake package (${numsim-core_DIR})")
  if(EXISTS "${_rvegen_core_sibling}/numsim-core")
    message(STATUS
      "numsim-core: sibling checkout ${_rvegen_core_sibling} is NOT used "
      "(the package wins); pass -DCMAKE_DISABLE_FIND_PACKAGE_numsim-core=ON "
      "to build against it instead")
  endif()
  target_link_libraries(rvegen_lib INTERFACE numsim-core::numsim-core)
  set(RVEGEN_NUMSIM_CORE_FROM_PACKAGE ON)
else()
  if(EXISTS "${_rvegen_core_sibling}/numsim-core")
    message(STATUS "numsim-core: package not found, using sibling ${_rvegen_core_sibling}")
  else()
    message(WARNING
      "numsim-core >= 1.3 not found: no CMake package (or one that is too "
      "old) and no sibling checkout. "
      "Install numsim-core and set CMAKE_PREFIX_PATH, or add its include "
      "directory via CMAKE_CXX_FLAGS=-I<path>.")
  endif()
  target_include_directories(rvegen_lib INTERFACE
      $<BUILD_INTERFACE:${_rvegen_core_sibling}>)
endif()

# ---------------------------------------------------------------------------
# Optional Voro++ backend for voronoi_generator_3d (#70). OFF by default so
# the library stays header-only with no extra dependency; the in-house
# tessellator remains the default backend either way. When ON, Voro++
# (BSD-3, pinned tag) is fetched, built as a small static library, and wired
# in through the RVEGEN_WITH_VORO_PP compile definition. Voro++ v0.4.6 ships
# no CMake build, hence the SOURCE_SUBDIR trick (populate only) plus our own
# target. Only the in-tree build links it; the installed package stays
# header-only (see docs in voronoi_backend_voropp.h).
# ---------------------------------------------------------------------------
option(RVEGEN_WITH_VORO_PP "Enable the Voro++ backend for voronoi_generator_3d" OFF)
if(RVEGEN_WITH_VORO_PP)
  FetchContent_Declare(
    voropp
    GIT_REPOSITORY https://github.com/chr1shr/voro.git
    GIT_TAG 43d460b83d5f33392bb599a401f05c37af3d7441   # v0.4.6 (tags are mutable; pin the commit)
    SOURCE_SUBDIR no_cmake_here
  )
  FetchContent_MakeAvailable(voropp)
  add_library(rvegen_voropp STATIC
    ${voropp_SOURCE_DIR}/src/cell.cc
    ${voropp_SOURCE_DIR}/src/common.cc
    ${voropp_SOURCE_DIR}/src/container.cc
    ${voropp_SOURCE_DIR}/src/unitcell.cc
    ${voropp_SOURCE_DIR}/src/v_compute.cc
    ${voropp_SOURCE_DIR}/src/c_loops.cc
    ${voropp_SOURCE_DIR}/src/v_base.cc
    ${voropp_SOURCE_DIR}/src/wall.cc
    ${voropp_SOURCE_DIR}/src/pre_container.cc
    ${voropp_SOURCE_DIR}/src/container_prd.cc
  )
  target_include_directories(rvegen_voropp SYSTEM PUBLIC
    ${voropp_SOURCE_DIR}/src)
  set_target_properties(rvegen_voropp PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_compile_options(rvegen_voropp PRIVATE -w)   # third-party code
  # No FMA contraction in Voro++: its periodic weighted (container_poly)
  # tessellation returns overlapping cells when the compiler fuses a*b+c
  # (8 of 20 random cases bad with contraction, 0 of 20 without). GCC and
  # Clang contract by default for -march with FMA (and on aarch64); MSVC
  # uses /fp:precise by default, which does not contract.
  if(NOT MSVC)
    target_compile_options(rvegen_voropp PRIVATE -ffp-contract=off)
  endif()
  # Voro++ calls exit() on fatal errors; force-include the shim that turns
  # that into a C++ exception (same shim the consumer-side include applies).
  if(MSVC)
    target_compile_options(rvegen_voropp PRIVATE
      /FI${CMAKE_CURRENT_SOURCE_DIR}/include/rvegen/generators/voronoi_voropp_exit_shim.h)
  else()
    target_compile_options(rvegen_voropp PRIVATE
      -include ${CMAKE_CURRENT_SOURCE_DIR}/include/rvegen/generators/voronoi_voropp_exit_shim.h)
  endif()
  target_link_libraries(rvegen_lib INTERFACE $<BUILD_INTERFACE:rvegen_voropp>)
  target_compile_definitions(rvegen_lib INTERFACE
    $<BUILD_INTERFACE:RVEGEN_WITH_VORO_PP=1>)
endif()

# ---------------------------------------------------------------------------
# Tests. Gated on BUILD_TESTING (CTest convention) so consumers that pull
# rvegen via add_subdirectory (e.g. Tessera) can disable rvegen's tests
# from registering into the parent CTest database with `set(BUILD_TESTING
# OFF CACHE BOOL "" FORCE)` before add_subdirectory().
# ---------------------------------------------------------------------------
if(NOT DEFINED BUILD_TESTING OR BUILD_TESTING)
enable_testing()

add_executable(rvegen_new_tree_smoke tests/new_tree_smoke.cpp)
target_link_libraries(rvegen_new_tree_smoke PRIVATE rvegen_lib)
add_test(NAME new_tree_smoke COMMAND rvegen_new_tree_smoke)

add_executable(rvegen_collision_pairs_test tests/collision_pairs_test.cpp)
target_link_libraries(rvegen_collision_pairs_test PRIVATE rvegen_lib)
add_test(NAME collision_pairs_test COMMAND rvegen_collision_pairs_test)

add_executable(rvegen_placement_test tests/placement_test.cpp)
target_link_libraries(rvegen_placement_test PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME placement_test COMMAND rvegen_placement_test)

add_executable(rvegen_end_to_end_smoke tests/end_to_end_smoke.cpp)
target_link_libraries(rvegen_end_to_end_smoke PRIVATE rvegen_lib)
add_test(NAME end_to_end_smoke COMMAND rvegen_end_to_end_smoke)

add_executable(rvegen_weave_geometry_test tests/weave_geometry_test.cpp)
target_link_libraries(rvegen_weave_geometry_test PRIVATE rvegen_lib)
add_test(NAME weave_geometry_test COMMAND rvegen_weave_geometry_test)

add_executable(rvegen_json_visitor_smoke tests/json_visitor_smoke.cpp)
target_link_libraries(rvegen_json_visitor_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME json_visitor_smoke COMMAND rvegen_json_visitor_smoke)

add_executable(rvegen_orientation_sampler_test tests/orientation_sampler_test.cpp)
target_link_libraries(rvegen_orientation_sampler_test PRIVATE rvegen_lib)
add_test(NAME orientation_sampler_test COMMAND rvegen_orientation_sampler_test)

add_executable(rvegen_orientation_a2_test tests/orientation_a2_test.cpp)
target_link_libraries(rvegen_orientation_a2_test PRIVATE rvegen_lib)
add_test(NAME orientation_a2_test COMMAND rvegen_orientation_a2_test)

add_executable(rvegen_registry_smoke tests/registry_smoke.cpp)
target_link_libraries(rvegen_registry_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME registry_smoke COMMAND rvegen_registry_smoke)

add_executable(rvegen_input_validation_test tests/input_validation_test.cpp)
target_link_libraries(rvegen_input_validation_test PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME input_validation_test COMMAND rvegen_input_validation_test)

add_executable(rvegen_golden_random_test tests/golden_random_test.cpp)
target_link_libraries(rvegen_golden_random_test PRIVATE rvegen_lib)
add_test(NAME golden_random_test COMMAND rvegen_golden_random_test)

add_executable(rvegen_full_registry_smoke tests/full_registry_smoke.cpp)
target_link_libraries(rvegen_full_registry_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME full_registry_smoke COMMAND rvegen_full_registry_smoke)

add_executable(rvegen_deterministic_generators_test tests/deterministic_generators_test.cpp)
target_link_libraries(rvegen_deterministic_generators_test PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME deterministic_generators_test COMMAND rvegen_deterministic_generators_test)

add_executable(rvegen_phases_test tests/phases_test.cpp)
target_link_libraries(rvegen_phases_test PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME phases_test COMMAND rvegen_phases_test
    $<TARGET_FILE:rvegen>
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/phases_voxel.json)

add_executable(rvegen_pipeline_test tests/pipeline_test.cpp)
target_link_libraries(rvegen_pipeline_test PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME pipeline_test COMMAND rvegen_pipeline_test
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(rvegen_extra_types_smoke tests/extra_types_smoke.cpp)
target_link_libraries(rvegen_extra_types_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME extra_types_smoke COMMAND rvegen_extra_types_smoke)

find_package(Threads REQUIRED)
add_executable(rvegen_concurrency_smoke tests/concurrency_smoke.cpp)
target_link_libraries(rvegen_concurrency_smoke PRIVATE
    rvegen_lib
    Threads::Threads
)
add_test(NAME concurrency_smoke COMMAND rvegen_concurrency_smoke)

add_executable(rvegen_mesh_io_test tests/mesh_io_test.cpp)
target_link_libraries(rvegen_mesh_io_test PRIVATE rvegen_lib)
add_test(NAME mesh_io_test COMMAND rvegen_mesh_io_test)

add_executable(rvegen_mesh_transform_test tests/mesh_transform_test.cpp)
target_link_libraries(rvegen_mesh_transform_test PRIVATE rvegen_lib)
add_test(NAME mesh_transform_test COMMAND rvegen_mesh_transform_test)

add_executable(rvegen_post_process_smoke tests/post_process_smoke.cpp)
target_link_libraries(rvegen_post_process_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME post_process_smoke COMMAND rvegen_post_process_smoke)

add_executable(rvegen_voxel_stats_3d_test tests/voxel_stats_3d_test.cpp)
target_link_libraries(rvegen_voxel_stats_3d_test PRIVATE rvegen_lib)
add_test(NAME voxel_stats_3d_test COMMAND rvegen_voxel_stats_3d_test)

add_executable(rvegen_mesh_dispatch_test tests/mesh_dispatch_test.cpp)
target_link_libraries(rvegen_mesh_dispatch_test PRIVATE rvegen_lib)
add_test(NAME mesh_dispatch_test COMMAND rvegen_mesh_dispatch_test)

# Shape-id uniqueness (#85): every built-in shape type has its own
# static_indexing id, and the dispatchers refuse a duplicate id.
add_executable(rvegen_shape_id_unique_test tests/shape_id_unique_test.cpp)
target_link_libraries(rvegen_shape_id_unique_test PRIVATE rvegen_lib)
add_test(NAME shape_id_unique_test COMMAND rvegen_shape_id_unique_test)

add_executable(rvegen_visualization_writers_test tests/visualization_writers_test.cpp)
target_link_libraries(rvegen_visualization_writers_test PRIVATE rvegen_lib Threads::Threads)
add_test(NAME visualization_writers_test COMMAND rvegen_visualization_writers_test)

add_executable(rvegen_schema_audit_test tests/schema_audit_test.cpp)
target_link_libraries(rvegen_schema_audit_test PRIVATE rvegen_lib)
add_test(NAME schema_audit_test COMMAND rvegen_schema_audit_test)

add_executable(rvegen_schema_construct_test tests/schema_construct_test.cpp)
target_link_libraries(rvegen_schema_construct_test PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME schema_construct_test COMMAND rvegen_schema_construct_test)

add_executable(rvegen_voronoi_robustness_test tests/voronoi_robustness_test.cpp)
target_link_libraries(rvegen_voronoi_robustness_test PRIVATE rvegen_lib)
add_test(NAME voronoi_robustness_test COMMAND rvegen_voronoi_robustness_test)

add_executable(rvegen_voronoi_backend_test tests/voronoi_backend_test.cpp)
target_link_libraries(rvegen_voronoi_backend_test PRIVATE rvegen_lib)
add_test(NAME voronoi_backend_test COMMAND rvegen_voronoi_backend_test)

add_executable(rvegen_voronoi_polycrystal_test tests/voronoi_polycrystal_test.cpp)
target_link_libraries(rvegen_voronoi_polycrystal_test PRIVATE rvegen_lib)
add_test(NAME voronoi_polycrystal_test COMMAND rvegen_voronoi_polycrystal_test)

add_executable(rvegen_progress_cancel_test tests/progress_cancel_test.cpp)
target_link_libraries(rvegen_progress_cancel_test PRIVATE rvegen_lib)
add_test(NAME progress_cancel_test COMMAND rvegen_progress_cancel_test)

# gmsh round-trip (#74): meshes gmsh_geo_writer output with a real gmsh
# and checks validity, conformity, clipping and periodic node matching.
# Only registered when gmsh is on the PATH (or RVEGEN_GMSH_EXECUTABLE is
# set explicitly).
find_program(RVEGEN_GMSH_EXECUTABLE gmsh)
# CI sets this so a missing gmsh fails the configure instead of quietly
# dropping the .geo meshing checks.
option(RVEGEN_REQUIRE_GMSH "Fail the configure if gmsh is not found" OFF)
if(RVEGEN_REQUIRE_GMSH AND NOT RVEGEN_GMSH_EXECUTABLE)
  message(FATAL_ERROR "RVEGEN_REQUIRE_GMSH is ON but gmsh was not found")
endif()
if(RVEGEN_GMSH_EXECUTABLE)
  add_executable(rvegen_gmsh_geo_roundtrip_test tests/gmsh_geo_roundtrip_test.cpp)
  target_link_libraries(rvegen_gmsh_geo_roundtrip_test PRIVATE rvegen_lib)
  target_compile_definitions(rvegen_gmsh_geo_roundtrip_test PRIVATE
      RVEGEN_GMSH_EXECUTABLE="${RVEGEN_GMSH_EXECUTABLE}")
  add_test(NAME gmsh_geo_roundtrip_test COMMAND rvegen_gmsh_geo_roundtrip_test
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
  message(STATUS "gmsh not found — skipping gmsh_geo_roundtrip_test")
endif()

# Collision-dispatch comparison benchmark. Not added to ctest —
# benchmarks produce numbers, not pass/fail. Run manually:
#   ./rvegen_dispatch_bench --benchmark_format=console
find_package(benchmark QUIET)
if(benchmark_FOUND)
  add_executable(rvegen_dispatch_bench bench/dispatch_compare.cpp)
  target_link_libraries(rvegen_dispatch_bench PRIVATE
      rvegen_lib
      benchmark::benchmark
  )
endif()

endif() # NOT DEFINED BUILD_TESTING OR BUILD_TESTING

# ---------------------------------------------------------------------------
# rvegen CLI.
#
# Production executable: argv → JSON config → registries → pipeline → file.
# ---------------------------------------------------------------------------
add_executable(rvegen apps/rvegen_cli.cpp)
target_link_libraries(rvegen PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)

# Install rules only for a top-level build. When rvegen is pulled in with
# add_subdirectory (e.g. by the Python wheel build, python/CMakeLists.txt),
# its dependencies may be FetchContent'd targets that are in no export set,
# so install(EXPORT) would fail at generate time, and the wheel must not
# ship the CLI / headers anyway.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
install(TARGETS rvegen
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# `cmake --build build --target gallery` renders every config in examples/
# into docs/gallery/ (PNGs plus a README.md index). Not part of `all`:
# it needs Python 3 with numpy + matplotlib, and optionally Inkscape,
# Chrome and gmsh for the SVG, three.js and .geo outputs.
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_custom_target(gallery
        COMMAND ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_gallery.py
                --rvegen $<TARGET_FILE:rvegen>
                --out ${CMAKE_CURRENT_SOURCE_DIR}/docs/gallery --index
                ${CMAKE_CURRENT_SOURCE_DIR}/examples
        DEPENDS rvegen
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Rendering the example gallery into docs/gallery"
        USES_TERMINAL
        VERBATIM)
endif()

# ---------------------------------------------------------------------------
# Install / export so downstream `find_package(rvegen)` works.
#
# Installs:
#   * rvegen public headers under <prefix>/include/rvegen/
#   * vendored GTE headers under <prefix>/include/rvegen/external/Mathematics/
#   * `rvegenTargets.cmake` exporting the `rvegen::rvegen` interface target
#   * `rvegenConfig.cmake` so `find_package(rvegen CONFIG)` resolves
#   * `rvegenConfigVersion.cmake` for SemVer compatibility checks
# ---------------------------------------------------------------------------
install(DIRECTORY include/rvegen/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rvegen
    FILES_MATCHING PATTERN "*.h"
)

# Vendored GTE — install under rvegen/external/ so the install-tree
# include path matches the source-tree layout used in our headers
# (`#include <Mathematics/...>`). Consumers who already have GTE
# elsewhere can ignore this; rvegen's CMake config doesn't propagate
# system-GTE detection.
install(DIRECTORY external/GeometricTools/GTE/Mathematics
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rvegen/external
    FILES_MATCHING PATTERN "*.h"
)

install(TARGETS rvegen_lib
    EXPORT rvegenTargets
)

# Licence texts: rvegen's own and those of the bundled GTE headers.
install(FILES LICENSE THIRD_PARTY_NOTICES.md
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/rvegen
)

install(EXPORT rvegenTargets
    FILE rvegenTargets.cmake
    NAMESPACE rvegen::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rvegen
)

include(CMakePackageConfigHelpers)

# Version comes from `project(rvegen VERSION ...)` at the top of this
# file — single source of truth shared with packaging metadata.
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rvegenConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rvegen
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/rvegen
)
endif() # top-level install rules

# Smoke tests: drive the CLI against the bundled example configs. Each
# config self-describes its output destination via output_path entries in
# its postprocess array — the CLI takes only the config path. Set
# WORKING_DIRECTORY so those relative paths resolve into the build tree
# rather than the source tree.
if(NOT DEFINED BUILD_TESTING OR BUILD_TESTING)
# The wheel is built from python/ alone, so it carries copies of the licence
# files; fail when a copy differs from the file at the repository root.
foreach(_license_file LICENSE THIRD_PARTY_NOTICES.md)
  add_test(
      NAME license_files_in_sync_${_license_file}
      COMMAND ${CMAKE_COMMAND} -E compare_files
          ${CMAKE_CURRENT_SOURCE_DIR}/${_license_file}
          ${CMAKE_CURRENT_SOURCE_DIR}/python/${_license_file}
  )
endforeach()

add_test(
    NAME cli_circles_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/circles_in_unit_box.json
)
set_tests_properties(cli_circles_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "generated 10 shapes"
)

# Unreachable target: the generator gives up at max_attempts and the CLI
# warns on stderr (PASS_REGULAR_EXPRESSION matches stdout and stderr).
add_test(
    NAME cli_max_attempts_warning
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_max_attempts.json
)
set_tests_properties(cli_max_attempts_warning PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "warning .* stopped at max_attempts"
)

add_test(
    NAME cli_spheres_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/spheres_in_cube.json
)
set_tests_properties(cli_spheres_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "generated [0-9]+ shapes"
)

# Top-level "seed" / "domain_box" validation. Each case writes a minimal
# config with the given fragment and expects the CLI to reject it (message
# naming the key) or, for the *_ok cases, to run.
function(rvegen_add_cli_toplevel_test name fragment regex)
  set(cfg ${CMAKE_CURRENT_BINARY_DIR}/cli_${name}.json)
  file(WRITE ${cfg} "{ ${fragment},
  \"distributions\": { \"u\": { \"type\": \"uniform_real\", \"a\": 0.2, \"b\": 0.8 },
                      \"r\": { \"type\": \"uniform_real\", \"a\": 0.03, \"b\": 0.05 } },
  \"shapes\": [ { \"type\": \"circle_input\", \"pos_x_dist\": \"u\",
                 \"pos_y_dist\": \"u\", \"radius_dist\": \"r\" } ],
  \"generator\": { \"type\": \"only_inside\", \"max_attempts\": 1000 },
  \"termination\": { \"type\": \"number_of_inclusions\", \"target\": 2 } }
")
  add_test(NAME cli_${name} COMMAND rvegen ${cfg})
  set_tests_properties(cli_${name} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      PASS_REGULAR_EXPRESSION "${regex}")
endfunction()

rvegen_add_cli_toplevel_test(seed_negative   "\"seed\": -1"
    "key 'seed' expects a non-negative integer")
rvegen_add_cli_toplevel_test(seed_fractional "\"seed\": 1.5"
    "key 'seed' expects a non-negative integer")
rvegen_add_cli_toplevel_test(seed_string     "\"seed\": \"7\""
    "key 'seed' expects a non-negative integer")
rvegen_add_cli_toplevel_test(seed_too_large  "\"seed\": 4294967296"
    "seed must be <= 4294967295")
rvegen_add_cli_toplevel_test(seed_max_ok     "\"seed\": 4294967295"
    "generated 2 shapes")
rvegen_add_cli_toplevel_test(domain_box_zero "\"domain_box\": [0, 0, 0]"
    "domain_box x and y extents must be > 0")
rvegen_add_cli_toplevel_test(domain_box_negative "\"domain_box\": [-1, 1, 0]"
    "domain_box x and y extents must be > 0")
rvegen_add_cli_toplevel_test(domain_box_negative_z "\"domain_box\": [1, 1, -1]"
    "domain_box z extent must be >= 0")
rvegen_add_cli_toplevel_test(domain_box_length "\"domain_box\": [1]"
    "domain_box must be an array of 2 or 3 numbers")
rvegen_add_cli_toplevel_test(domain_box_type "\"domain_box\": [1, \"a\"]"
    "domain_box entries must be numbers")
rvegen_add_cli_toplevel_test(domain_box_two_ok "\"domain_box\": [1, 1]"
    "generated 2 shapes")

add_test(
    NAME cli_periodic_circles_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/periodic_circles.json
)
set_tests_properties(cli_periodic_circles_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "generated [0-9]+ shapes"
)

# Deterministic generators (#73/#84) and Bingham-oriented tubes through
# the CLI. Each example's output must be written, not just reported.
foreach(_ex IN ITEMS voronoi_2d voronoi_3d plain_weave tow_weave bingham_fibres)
  add_test(
      NAME cli_${_ex}_example
      COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/${_ex}.json
  )
  set_tests_properties(cli_${_ex}_example PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      PASS_REGULAR_EXPRESSION "generated [0-9]+ shapes.*post-process '[a-z_]+' done"
  )
endforeach()

# JSON "phases" + "phase_assignment" written through the voxel and VTK writers (#7).
add_test(
    NAME cli_phases_voxel_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/phases_voxel.json
)
set_tests_properties(cli_phases_voxel_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "phase_assignment tagged 12 shapes.*post-process 'voxel' done.*post-process 'vtk_legacy' done"
)

# CLI validation (regression tests for the pipeline's load-time checks).
# PASS_REGULAR_EXPRESSION overrides the exit code, so these match the message.
function(rvegen_cli_msg_test name cfg regex)
  add_test(NAME cli_${name}
           COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/${cfg})
  set_tests_properties(cli_${name} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      PASS_REGULAR_EXPRESSION "${regex}")
endfunction()
rvegen_cli_msg_test(direction_in_scalar_slot direction_in_scalar_slot.json
    "refers to direction distribution 'd'")
rvegen_cli_msg_test(deterministic_ignored_keys deterministic_ignored_keys.json
    "\"shapes\" is ignored.*\"termination\" is ignored")
rvegen_cli_msg_test(generator_unknown_key generator_unknown_key.json
    "unknown key 'n_seedz' for generator 'voronoi_2d' is ignored")

# Combined run: gmsh_geo + voxel as two post-processes from one JSON.
add_test(
    NAME cli_circles_with_voxelize_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/circles_with_voxelize.json
)
set_tests_properties(cli_circles_with_voxelize_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "post-process 'voxel' done"
)

# End-to-end FE-RVE workflow: periodic generator + every gmsh knob
# (size, order, algorithm, recombine). Smoke-tests that the JSON-driven
# path through the registry consumes the new gmsh-writer parameters.
add_test(
    NAME cli_fe_rve_workflow_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/fe_rve_workflow.json
)
set_tests_properties(cli_fe_rve_workflow_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "generated [0-9]+ shapes"
)

# `rvegen --check` (#120): validation only. Every shipped example must pass
# it, quickly (the check builds every block but places no shape), and a
# misspelt section must be rejected with the section it meant. Run from
# examples/ because mesh inputs are read at validation time and their paths
# are relative to the working directory.
file(GLOB _rvegen_check_configs CONFIGURE_DEPENDS
     ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.json)
foreach(_cfg IN LISTS _rvegen_check_configs)
  get_filename_component(_name ${_cfg} NAME_WE)
  add_test(NAME cli_validate_${_name} COMMAND rvegen --check ${_cfg})
  set_tests_properties(cli_validate_${_name} PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/examples
    PASS_REGULAR_EXPRESSION "rvegen: config OK"
    TIMEOUT 20 LABELS "cli")
endforeach()
add_test(NAME cli_validate_rejects_misspelt_section
  COMMAND rvegen --check
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/misspelt_postprocess.json)
set_tests_properties(cli_validate_rejects_misspelt_section PROPERTIES
  PASS_REGULAR_EXPRESSION
    "unknown top-level key 'post_process' \\(did you mean 'postprocess'\\?\\)"
  LABELS "cli")
add_test(NAME cli_rejects_unknown_option
  COMMAND rvegen --chekc ${CMAKE_CURRENT_SOURCE_DIR}/examples/circles_in_unit_box.json)
set_tests_properties(cli_rejects_unknown_option PROPERTIES
  PASS_REGULAR_EXPRESSION "unknown option '--chekc'" LABELS "cli")

# Output-checking CLI tests (#85): one test per config in examples/. Exit
# status and stdout are not enough, so tests/cli_example_check.py runs the
# CLI and then inspects every file the config's postprocess entries wrote
# (voxel/VTK header vs value count, SVG as XML, statistics CSV, .geo through
# gmsh when found). Needs Python 3.
if(Python3_Interpreter_FOUND)
  file(GLOB _rvegen_example_configs CONFIGURE_DEPENDS
       ${CMAKE_CURRENT_SOURCE_DIR}/examples/*.json)
  foreach(_cfg IN LISTS _rvegen_example_configs)
    get_filename_component(_name ${_cfg} NAME_WE)
    add_test(NAME cli_check_${_name}
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_example_check.py
              --rvegen $<TARGET_FILE:rvegen>
              --examples ${CMAKE_CURRENT_SOURCE_DIR}/examples
              --gmsh "${RVEGEN_GMSH_EXECUTABLE}"
              ${_cfg})
    # Pass/fail is the script's exit status; 77 = a .geo was not meshed
    # because gmsh is missing, reported as Skipped rather than Passed.
    set_tests_properties(cli_check_${_name} PROPERTIES
      TIMEOUT 900 LABELS "cli"
      SKIP_RETURN_CODE 77)
  endforeach()

  # The checker must reject an under-filled RVE: this config stops at
  # max_attempts 50 far below its volume-fraction target. The test passes
  # only if the checker reports the shortfall (the regex decides, not the
  # exit status). cli_check_selftest covers the checker's own parsing
  # (statistics CSV with an empty g(r), the accepted fill window).
  add_test(NAME cli_check_rejects_underfill
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_example_check.py
            --rvegen $<TARGET_FILE:rvegen>
            --examples ${CMAKE_CURRENT_SOURCE_DIR}/examples
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/underfilled_polydisperse.json)
  set_tests_properties(cli_check_rejects_underfill PROPERTIES
    TIMEOUT 300 LABELS "cli"
    PASS_REGULAR_EXPRESSION "FAIL: volume fraction [0-9.]+ not within")
  add_test(NAME cli_check_selftest
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_example_check.py --self-test)
  set_tests_properties(cli_check_selftest PROPERTIES LABELS "cli")

  # The README type catalogue is generated from `rvegen --list-types`.
  add_test(NAME readme_type_catalogue
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/readme_types_check.py
            --rvegen $<TARGET_FILE:rvegen>
            --readme ${CMAKE_CURRENT_SOURCE_DIR}/README.md)

  # Gallery smoke: render every example into a scratch directory. Needs
  # numpy + matplotlib; Inkscape / Chrome / gmsh are optional and skipped
  # by the script when absent (--browser none: no Chrome in the test).
  # gallery_script_check covers the script itself: a failing renderer
  # must give a non-zero exit, and same-stem configs must not collide.
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy, matplotlib"
    RESULT_VARIABLE _rvegen_py_plot_rc
    OUTPUT_QUIET ERROR_QUIET)
  if(_rvegen_py_plot_rc EQUAL 0)
    add_test(NAME gallery_smoke
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_gallery.py
              --rvegen $<TARGET_FILE:rvegen>
              --out ${CMAKE_CURRENT_BINARY_DIR}/gallery_smoke
              --no-index --browser none
              ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    set_tests_properties(gallery_smoke PROPERTIES TIMEOUT 1800 LABELS "cli")
    add_test(NAME gallery_script_check
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/tests/gallery_script_check.py
              --rvegen $<TARGET_FILE:rvegen>
              --examples ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    set_tests_properties(gallery_script_check PROPERTIES TIMEOUT 900 LABELS "cli")
    # The committed docs/gallery index and PNGs must list exactly what the
    # script generates from examples/ (renders nothing).
    add_test(NAME gallery_index_check
      COMMAND ${Python3_EXECUTABLE}
              ${CMAKE_CURRENT_SOURCE_DIR}/scripts/make_gallery.py --check
              --out ${CMAKE_CURRENT_SOURCE_DIR}/docs/gallery
              ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    set_tests_properties(gallery_index_check PROPERTIES LABELS "cli")
  else()
    message(STATUS "numpy/matplotlib not found - skipping gallery_smoke and gallery_script_check")
  endif()
else()
  message(STATUS "Python 3 not found - skipping CLI output checks")
endif()

endif() # NOT DEFINED BUILD_TESTING OR BUILD_TESTING (CLI smoke tests)

# ---------------------------------------------------------------------------
# Doxygen API reference (opt-in).
#
# `cmake -DRVEGEN_BUILD_DOCS=ON` enables a `docs` target that builds
# HTML API documentation from the public headers + README + the
# extending.md contributor doc. Output: build/docs/html/index.html.
#
# Default OFF — most builds don't need it, and pulling in doxygen is a
# non-trivial dependency on minimal CI images.
# ---------------------------------------------------------------------------
option(RVEGEN_BUILD_DOCS "Build Doxygen API reference" OFF)
if(RVEGEN_BUILD_DOCS)
  find_package(Doxygen REQUIRED)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in
    ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    @ONLY
  )
  add_custom_target(docs
    COMMAND Doxygen::doxygen ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating API reference with Doxygen → build/docs/html/"
    VERBATIM
  )
endif()

# ---------------------------------------------------------------------------
# Python bindings (opt-in)
#
# `cmake -DRVEGEN_BUILD_PYTHON=ON` enables the pybind11 module under
# python/. Default OFF because pybind11 is a non-trivial fetch on minimal
# CI images and most C++-only builds don't need it. The python/ directory
# also functions standalone via scikit-build-core (`pip install ./python`).
# ---------------------------------------------------------------------------
option(RVEGEN_BUILD_PYTHON "Build the pybind11 Python bindings" OFF)
if(RVEGEN_BUILD_PYTHON)
  add_subdirectory(python)
endif()
