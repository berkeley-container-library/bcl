## SPDX-FileCopyrightText: 2021 Benjamin Brock
##
## SPDX-License-Identifier: BSD-3-Clause

find_package(MPI REQUIRED)

function(add_bcl_test TEST_NAME)
  add_executable(${TEST_NAME} ${ARGN})
  target_link_libraries(${TEST_NAME} PRIVATE bcl::mpi)
  add_test(
  NAME
    ${TEST_NAME}
  COMMAND
    ${MPIEXEC_EXECUTABLE} ${CMAKE_BINARY_DIR}/${TEST_NAME} -n 4
  )
endfunction()

add_subdirectory(hashmap)
add_subdirectory(simple)
