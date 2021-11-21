## SPDX-FileCopyrightText: 2021 Benjamin Brock
##
## SPDX-License-Identifier: BSD-3-Clause

if (TARGET UPCXX::UPCXX OR UPCXX_FOUND)
  return()
endif()

find_path(UPCXX_INCLUDE_DIRS NAMES upcxx.h)

#find_library(UPCXX_LIBRARIES NAMES gasnetex)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(UPCXX
  DEFAULT_MSG
  UPCXX_INCLUDE_DIRS
  #UPCXX_LIBRARIES
)

mark_as_advanced(UPCXX_INCLUDE_DIRS UPCXX_LIBRARIES)

add_library(UPCXX::UPCXX UNKNOWN IMPORTED)
set_target_properties(UPCXX::UPCXX PROPERTIES
  #IMPORTED_LOCATION "${UPCXX_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${UPCXX_INCLUDE_DIRS}"
)
