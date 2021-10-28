## SPDX-FileCopyrightText: 2021 Benjamin Brock
##
## SPDX-License-Identifier: BSD-3-Clause

if (TARGET GASNET_EX::GASNET_EX OR GASNET_EX_FOUND)
  return()
endif()

find_path(GASNET_EX_INCLUDE_DIRS NAMES gasnetex.h)

#find_library(GASNET_EX_LIBRARIES NAMES gasnetex)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GASNET_EX
  DEFAULT_MSG
  GASNET_EX_INCLUDE_DIRS
  #GASNET_EX_LIBRARIES
)

mark_as_advanced(GASNET_EX_INCLUDE_DIRS GASNET_EX_LIBRARIES)

add_library(GASNET_EX::GASNET_EX UNKNOWN IMPORTED)
set_target_properties(GASNET_EX::GASNET_EX PROPERTIES
  #IMPORTED_LOCATION "${GASNET_EX_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${GASNET_EX_INCLUDE_DIRS}"
)
