// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <mpi.h>

#include <bcl/core/global_state.hpp>
#include <bcl/core/malloc.hpp>

#include "alloc.hpp"
#include "comm.hpp"
#include "ops.hpp"
#include "request.hpp"

#include "async_allocator.hpp"

namespace BCL {

namespace backend {

inline MPI_Comm comm;
inline MPI_Win win;
inline MPI_Info info;

inline bool we_initialized;
inline bool bcl_finalized;

inline std::size_t rank_;
inline std::size_t nprocs_;

inline std::size_t rank() {
  return rank_;
}

inline std::size_t nprocs() {
  return nprocs_;
}

inline bool mpi_finalized() {
  int flag;
  MPI_Finalized(&flag);
  return flag;
}

inline bool mpi_initialized() {
  int flag;
  MPI_Initialized(&flag);
  return flag;
}

inline void barrier() {
  int error_code = MPI_Win_flush_all(win);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL barrier(): MPI_Win_lock_all returned error code " + std::to_string(error_code));
          }
  )
  error_code = MPI_Barrier(comm);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL barrier(): MPI_Barrier returned error code " + std::to_string(error_code));
          }
  )
}

inline void flush() {
  MPI_Win_flush_all(win);
}

// MPI communicator, shared_segment_size in MB,
// and whether to start the progress thread.
inline void init(std::size_t shared_segment_size = 256, bool thread_safe = false) {
  BCL::comm = MPI_COMM_WORLD;
  BCL::global_state::shared_segment_size = 1024*1024*shared_segment_size;

  if (!mpi_initialized()) {
    if (!thread_safe) {
      MPI_Init(NULL, NULL);
    } else {
      int provided;
      MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
      if (provided < MPI_THREAD_MULTIPLE) {
        throw BCL::error("BCL asked for MPI_THREAD_MULTIPLE, but was denied. "
                         "You need to use a thread-safe MPI implementation.");
      }
    }
    we_initialized = true;
  }

  int rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  rank_ = rank;
  nprocs_ = nprocs;

  MPI_Info_create(&info);
  MPI_Info_set(info, "accumulate_ordering", "none");
  MPI_Info_set(info, "accumulate_ops", "same_op_no_op");
  MPI_Info_set(info, "same_size", "true");
  MPI_Info_set(info, "same_disp_unit", "true");

  MPI_Win_allocate(BCL::shared_segment_size, 1, info, BCL::comm,
    &smem_base_ptr, &win);

  bcl_finalized = false;

  init_malloc();

  MPI_Barrier(BCL::comm);
  MPI_Win_lock_all(0, win);
  BCL::barrier();
}

void finalize() {
  barrier();
  MPI_Win_unlock_all(win);
  MPI_Info_free(&info);
  MPI_Win_free(&win);
  if (we_initialized && !mpi_finalized()) {
    MPI_Finalize();
  }
  bcl_finalized = true;
}

} // end backend
} // end BCL
