## SPDX-FileCopyrightText: 2021 Benjamin Brock
##
## SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.10)

# Guard against multiple 'find_package(bcl)' calls
if (TARGET bcl OR bcl_FOUND)
  return()
endif()

set(bcl_LOC ${CMAKE_CURRENT_LIST_DIR}/..)
list(APPEND CMAKE_MODULE_PATH ${bcl_LOC}/cmake)

## Base target ##

add_library(bcl::core INTERFACE IMPORTED)
target_include_directories(bcl::core INTERFACE ${bcl_LOC})

## MPI ##

find_package(MPI QUIET)
if (TARGET MPI::MPI_CXX)
  add_library(bcl::mpi INTERFACE IMPORTED)
  target_link_libraries(bcl::mpi INTERFACE bcl::core MPI::MPI_CXX)
  target_compile_definitions(bcl::mpi INTERFACE BCL_MPI)
endif()

## SHMEM ##

find_package(SHMEM QUIET MODULE)
if (TARGET SHMEM::SHMEM)
  add_library(bcl::shmem INTERFACE IMPORTED)
  target_link_libraries(bcl::shmem INTERFACE bcl::core SHMEM::SHMEM)
  target_compile_definitions(bcl::shmem INTERFACE SHMEM)
endif()

## GASNET_EX ##

find_package(GASNET_EX QUIET MODULE)
if (TARGET GASNET_EX::GASNET_EX)
  add_library(bcl::gasnet_ex INTERFACE IMPORTED)
  target_link_libraries(bcl::gasnet_ex INTERFACE bcl::core GASNET_EX::GASNET_EX)
  target_compile_definitions(bcl::gasnet_ex INTERFACE GASNET_EX)
endif()

set(bcl_FOUND ON)
