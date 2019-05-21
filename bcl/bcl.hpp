#pragma once

#include <bcl/core/except.hpp>
#include <bcl/core/GlobalPtr.hpp>
#include <bcl/core/malloc.hpp>
#include <bcl/core/alloc.hpp>

#ifdef SHMEM
  #include <bcl/backends/shmem/backend.hpp>
#elif GASNET_EX
  #include <bcl/backends/gasnet-ex/backend.hpp>
#elif UPCXX
  #include <bcl/backends/upcxx/backend.hpp>
#else
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
