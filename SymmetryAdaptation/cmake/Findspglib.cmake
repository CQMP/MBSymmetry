#
# Find the spglib library.
# Sets:
# SPGLIB_FOUND        - system has spglib
# SPGLIB_INCLUDE_DIR  - spglib include directories
# SPGLIB_LIBRARY      - spglib library
#

if (DEFINED ENV{SPGLIB_DIR})
    set(SPGLIB_DIR "$ENV{SPGLIB_DIR}")
endif()

find_library(SPGLIB_LIBRARY
        NAMES spglib symspg
        PATH_SUFFIXES project build bin lib
        HINTS ${SPGLIB_DIR}
        )
find_path(SPGLIB_INCLUDE_DIR
        NAMES spglib.h
        HINTS ${SPGLIB_DIR}
        PATH_SUFFIXES "include"
        )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(spglib
        REQUIRED_VARS SPGLIB_LIBRARY SPGLIB_INCLUDE_DIR
        )

mark_as_advanced(SPGLIB_LIBRARY SPGLIB_INCLUDE_DIR)

