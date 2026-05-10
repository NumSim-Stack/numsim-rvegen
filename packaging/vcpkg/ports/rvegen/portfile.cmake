# vcpkg port for rvegen — header-only library; install rule comes from the
# upstream CMake install/export configured by issue #103.
#
# The SHA512 below is a placeholder; replace with the real hash once the
# v0.1.0 tag is published on GitHub. Until then, vcpkg will fail with a
# hash-mismatch error at install time — that's expected during the
# "ports file is committed but no release tag yet" interim.
#
# numsim-core is listed as a dependency in vcpkg.json. It needs its own
# vcpkg port (separate work, tracked on the numsim-core repo) before this
# port can be PR'd to microsoft/vcpkg. Until then, this overlay-port lets
# downstream consumers vendor numsim-core themselves and wire it through
# `vcpkg install rvegen --overlay-ports ./packaging/vcpkg/ports`.

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO NumSim-Stack/numsim-rvegen
    REF v0.1.0
    SHA512 0
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DBUILD_TESTING=OFF
)

vcpkg_cmake_install()

vcpkg_cmake_config_fixup(PACKAGE_NAME rvegen)

# Header-only — strip debug includes; the library has no .so / .a artifacts.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")

# License must be installed under share/rvegen/copyright per vcpkg policy.
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
