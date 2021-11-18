// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <bcl/core/detail/detail.hpp>
#include <bcl/core/except.hpp>
#include <bcl/core/GlobalPtr.hpp>
#include <bcl/core/malloc.hpp>
#include <bcl/core/alloc.hpp>

#ifdef BCL_BACKEND_SHMEM
  #include <bcl/backends/shmem/backend.hpp>
#elif BCL_BACKEND_GASNET_EX
  #include <bcl/backends/gasnet-ex/backend.hpp>
#elif BCL_BACKEND_UPCXX
  #include <bcl/backends/upcxx/backend.hpp>
#else
  #ifndef BCL_BACKEND_MPI
    #define BCL_BACKEND_MPI
  #endif
  #include <bcl/backends/mpi/backend.hpp>
#endif

#include <bcl/core/comm.hpp>
#include <bcl/core/teams.hpp>
#include <bcl/core/util.hpp>

namespace BCL {
  // TODO: put these in a compilation unit.
  uint64_t shared_segment_size;
  void *smem_base_ptr;
}
