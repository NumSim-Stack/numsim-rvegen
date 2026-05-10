# Standalone consumer that exercises the rvegen install/export.
# Driven by ../../.github/workflows/ci.yml: after installing rvegen to a
# temp prefix, this project does `find_package(rvegen CONFIG REQUIRED)`
# and links via the public `rvegen::rvegen` target. If the install tree
# is misconfigured (missing headers, wrong include paths, broken CMake
# config), the configure or compile step fails — much closer to the
# real downstream-consumer experience than `ctest` against the build
# tree.
#
# numsim-core's headers must be on the include path; until numsim-core
# itself ships a CMake config the workflow injects them via
# CMAKE_CXX_FLAGS=-I... rather than relying on find_dependency.

cmake_minimum_required(VERSION 3.20)
project(rvegen_install_smoke LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rvegen CONFIG REQUIRED)

add_executable(install_smoke main.cpp)
target_link_libraries(install_smoke PRIVATE rvegen::rvegen)
