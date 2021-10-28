## SPDX-FileCopyrightText: 2021 Benjamin Brock
##
## SPDX-License-Identifier: BSD-3-Clause

cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(bcl LANGUAGES CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

enable_testing()

set(bcl_DIR ${CMAKE_CURRENT_LIST_DIR}/cmake)
find_package(bcl REQUIRED)

add_subdirectory(examples)
