## SPDX-FileCopyrightText: 2021 Benjamin Brock
##
## SPDX-License-Identifier: BSD-3-Clause

if (TARGET SHMEM::SHMEM OR SHMEM_FOUND)
  return()
endif()

find_path(SHMEM_INCLUDE_DIRS NAMES shmem.h PATHS mpp)

#find_library(SHMEM_LIBRARIES NAMES gasnetex)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SHMEM
  DEFAULT_MSG
  GASNETEX_INCLUDE_DIRS
  #GASNETEX_LIBRARIES
)

mark_as_advanced(SHMEM_INCLUDE_DIRS SHMEM_LIBRARIES)

add_library(SHMEM::SHMEM UNKNOWN IMPORTED)
set_target_properties(SHMEM::SHMEM PROPERTIES
  #IMPORTED_LOCATION "${SHMEM_LIBRARIES}"
  INTERFACE_INCLUDE_DIRECTORIES "${SHMEM_INCLUDE_DIRS}/.."
)

