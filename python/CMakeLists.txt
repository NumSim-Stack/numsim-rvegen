# Python bindings for rvegen — built only when SKBUILD_PROJECT_NAME is set
# (i.e. when invoked through scikit-build-core via `pip install .` from
# the python/ directory) OR when -DRVEGEN_BUILD_PYTHON=ON is passed
# explicitly to the top-level rvegen CMake build.
#
# Phase 1 scope: bind shape primitives only. Generators / distributions /
# the JSON config path come in follow-up PRs once shape lifetime + GIL
# concerns are sorted.

cmake_minimum_required(VERSION 3.20)

if(NOT TARGET rvegen_lib)
  # We're being invoked standalone (via scikit-build-core), so configure
  # the parent project as a subdirectory.
  project(rvegen_python LANGUAGES CXX)
  add_subdirectory(.. rvegen_parent)
endif()

# pybind11 — fetched if not found.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

pybind11_add_module(_core src/rvegen_module.cpp)

target_link_libraries(_core PRIVATE rvegen_lib)
target_compile_features(_core PRIVATE cxx_std_23)

if(SKBUILD_PROJECT_NAME)
  install(TARGETS _core DESTINATION rvegen)
endif()
