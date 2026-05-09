cmake_minimum_required(VERSION 3.20)

project(rvegen
    VERSION   0.1.0
    LANGUAGES CXX
    DESCRIPTION "RVE generator for FFT/FEM homogenization"
    HOMEPAGE_URL "https://github.com/numsim-stack/rvegen"
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
# BUILD_INTERFACE: paths used while building rvegen and any consumer that
# pulls rvegen via add_subdirectory().
# INSTALL_INTERFACE: paths used by consumers that find rvegen via
# find_package(rvegen). The Mathematics subset of GTE is installed under
# rvegen/_thirdparty/ so the consumer's global include namespace stays
# clean; numsim-core is found by the consumer's own find_dependency()
# call (chained from rvegenConfig.cmake).
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

add_library(rvegen_lib INTERFACE)
target_include_directories(rvegen_lib INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/external/GeometricTools/GTE>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../numsim-core/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/rvegen/_thirdparty>
)
target_link_libraries(rvegen_lib INTERFACE
    Eigen3::Eigen
    Boost::headers
)
target_compile_features(rvegen_lib INTERFACE cxx_std_23)

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

add_executable(rvegen_end_to_end_smoke tests/end_to_end_smoke.cpp)
target_link_libraries(rvegen_end_to_end_smoke PRIVATE rvegen_lib)
add_test(NAME end_to_end_smoke COMMAND rvegen_end_to_end_smoke)

add_executable(rvegen_json_visitor_smoke tests/json_visitor_smoke.cpp)
target_link_libraries(rvegen_json_visitor_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME json_visitor_smoke COMMAND rvegen_json_visitor_smoke)

add_executable(rvegen_registry_smoke tests/registry_smoke.cpp)
target_link_libraries(rvegen_registry_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME registry_smoke COMMAND rvegen_registry_smoke)

add_executable(rvegen_full_registry_smoke tests/full_registry_smoke.cpp)
target_link_libraries(rvegen_full_registry_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME full_registry_smoke COMMAND rvegen_full_registry_smoke)

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

add_executable(rvegen_post_process_smoke tests/post_process_smoke.cpp)
target_link_libraries(rvegen_post_process_smoke PRIVATE
    rvegen_lib
    nlohmann_json::nlohmann_json
)
add_test(NAME post_process_smoke COMMAND rvegen_post_process_smoke)

add_executable(rvegen_mesh_dispatch_test tests/mesh_dispatch_test.cpp)
target_link_libraries(rvegen_mesh_dispatch_test PRIVATE rvegen_lib)
add_test(NAME mesh_dispatch_test COMMAND rvegen_mesh_dispatch_test)

add_executable(rvegen_schema_audit_test tests/schema_audit_test.cpp)
target_link_libraries(rvegen_schema_audit_test PRIVATE rvegen_lib)
add_test(NAME schema_audit_test COMMAND rvegen_schema_audit_test)

add_executable(rvegen_progress_cancel_test tests/progress_cancel_test.cpp)
target_link_libraries(rvegen_progress_cancel_test PRIVATE rvegen_lib)
add_test(NAME progress_cancel_test COMMAND rvegen_progress_cancel_test)

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

install(TARGETS rvegen
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# ---------------------------------------------------------------------------
# Install + export for find_package(rvegen) consumers.
#
# Downstream usage:
#     find_package(rvegen 0.1.0 CONFIG REQUIRED)
#     target_link_libraries(my_target PRIVATE rvegen::rvegen_lib)
#
# The four upstream packages — Boost, Eigen3, nlohmann_json, numsim-core —
# are pulled in by find_dependency() in rvegenConfig.cmake and must be
# discoverable by the consumer's CMake.
# ---------------------------------------------------------------------------
include(CMakePackageConfigHelpers)

# rvegen's own headers.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/rvegen
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Vendored GTE Mathematics — what rvegen's headers actually #include.
# Installed under rvegen/_thirdparty/ to keep the consumer's global
# include namespace clean; rvegen_lib's INSTALL_INTERFACE adds the dir.
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/external/GeometricTools/GTE/Mathematics
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rvegen/_thirdparty
    FILES_MATCHING PATTERN "*.h"
)

# Header-only — exporting the target records include-dir + link-dependency
# metadata for consumers, no binary artefacts.
install(TARGETS rvegen_lib EXPORT rvegen_Targets)

install(EXPORT rvegen_Targets
    FILE rvegenTargets.cmake
    NAMESPACE rvegen::
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rvegen/cmake
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/rvegenConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rvegen/cmake
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/rvegenConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rvegen/cmake
)

# Smoke tests: drive the CLI against the bundled example configs. Each
# config self-describes its output destination via output_path entries in
# its postprocess array — the CLI takes only the config path. Set
# WORKING_DIRECTORY so those relative paths resolve into the build tree
# rather than the source tree.
if(NOT DEFINED BUILD_TESTING OR BUILD_TESTING)
add_test(
    NAME cli_circles_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/circles_in_unit_box.json
)
set_tests_properties(cli_circles_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "generated 10 shapes"
)

add_test(
    NAME cli_spheres_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/spheres_in_cube.json
)
set_tests_properties(cli_spheres_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "generated [0-9]+ shapes"
)

add_test(
    NAME cli_periodic_circles_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/periodic_circles.json
)
set_tests_properties(cli_periodic_circles_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "generated [0-9]+ shapes"
)

# Combined run: gmsh_geo + voxel as two post-processes from one JSON.
add_test(
    NAME cli_circles_with_voxelize_example
    COMMAND rvegen ${CMAKE_CURRENT_SOURCE_DIR}/examples/circles_with_voxelize.json
)
set_tests_properties(cli_circles_with_voxelize_example PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PASS_REGULAR_EXPRESSION "post-process 'voxel' done"
)

endif() # NOT DEFINED BUILD_TESTING OR BUILD_TESTING (CLI smoke tests)
