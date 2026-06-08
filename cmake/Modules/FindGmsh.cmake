# FindGmsh.cmake - Locate the Gmsh library and headers
#
# This module defines:
#   GMSH_FOUND          - True if Gmsh was found
#   GMSH_INCLUDE_DIR    - Directory containing gmsh.h
#   GMSH_LIBRARY        - Path to libgmsh.so / libgmsh.a
#
# It also creates an imported target:
#   Gmsh::Gmsh

find_path(GMSH_INCLUDE_DIR
        NAMES gmsh.h
        PATH_SUFFIXES gmsh
)

find_library(GMSH_LIBRARY
        NAMES gmsh libgmsh
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Gmsh
        REQUIRED_VARS GMSH_INCLUDE_DIR GMSH_LIBRARY
)

if(GMSH_FOUND AND NOT TARGET Gmsh::Gmsh)
    add_library(Gmsh::Gmsh UNKNOWN IMPORTED)
    set_target_properties(Gmsh::Gmsh PROPERTIES
            IMPORTED_LOCATION "${GMSH_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${GMSH_INCLUDE_DIR}"
    )
endif()
