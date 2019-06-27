#pragma once

#include <stdexcept>
#include <cstring>
#include <vector>

#include <type_traits>

#include "backend.hpp"
#include "ops.hpp"
#include "request.hpp"
#include "async_allocator.hpp"
#include "atomics.hpp"
#include "team_conv.hpp"

namespace BCL {

extern MPI_Comm comm;
extern MPI_Win win;

extern void barrier();

// Read size T's from src -> dst
// Blocks until dst is ready.
template <typename T>
inline void read(const GlobalPtr <T> &src, T *dst, const size_t size) {
  BCL_DEBUG(
    if (src.rank > BCL::backend::nprocs()) {
      throw debug_error("BCL read(): request to read from rank " +
                        std::to_string(src.rank) + ", which does not exist");
    }
  )
  /*
  if (src.rank == BCL::rank()) {
    std::memcpy(dst, src.local(), size*sizeof(T));
  } else {
    MPI_Request request;
    MPI_Rget(dst, size*sizeof(T), MPI_CHAR, src.rank, src.ptr, size*sizeof(T),
      MPI_CHAR, BCL::win, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
  */
  MPI_Request request;

  int error_code = MPI_Rget(dst, size*sizeof(T), MPI_CHAR,
                            src.rank, src.ptr, size*sizeof(T), MPI_CHAR,
                            BCL::win, &request);
  BCL_DEBUG(
    if (error_code != MPI_SUCCESS) {
      throw debug_error("BCL read(): MPI_Rget returned error code " + std::to_string(error_code));
    }
  )

  error_code = MPI_Wait(&request, MPI_STATUS_IGNORE);
  BCL_DEBUG(
    if (error_code != MPI_SUCCESS) {
      throw debug_error("BCL read(): MPI_Wait returned error code " + std::to_string(error_code));
    }
  )
}

// Read size T's from src -> dst
// Blocks until dst is ready.
template <typename T>
inline void atomic_read(const GlobalPtr <T> &src, T *dst, size_t size) {
  BCL_DEBUG(
    if (src.rank > BCL::backend::nprocs()) {
      throw debug_error("atomic_read(): request to read from rank "
                        + std::to_string(src.rank) + ", which does not exist");
    }
  )
  MPI_Request request;

  int error_code = MPI_Rget(dst, size*sizeof(T), MPI_CHAR,
                            src.rank, src.ptr, size*sizeof(T), MPI_CHAR,
                            BCL::win, &request);
  BCL_DEBUG(
    if (error_code != MPI_SUCCESS) {
      throw debug_error("BCL atomic_read(): MPI_Rget return error code " + std::to_string(error_code));
    }
  )

  error_code = MPI_Wait(&request, MPI_STATUS_IGNORE);
  BCL_DEBUG(
    if (error_code != MPI_SUCCESS) {
      throw debug_error("BCL atomic_read(): MPI_Wait return error code " + std::to_string(error_code));
    }
  )
}

// Write size T's from src -> dst
// Returns after src buffer is sent
// and okay to be modified; no
// guarantee that memop is complete
// until BCL::barrier()
template <typename T>
inline void write(const T *src, const GlobalPtr <T> &dst, size_t size) {
  BCL_DEBUG(
    if (dst.rank > BCL::backend::nprocs()) {
      throw debug_error("BCL write(): request to write to rank " +
                        std::to_string(dst.rank) + ", which does not exist");
    }
  )
  if (dst.rank == BCL::rank()) {
    std::memcpy(dst.local(), src, size*sizeof(T));
  } else {
    MPI_Request request;

    int error_code = MPI_Rput(src, size*sizeof(T), MPI_CHAR,
                              dst.rank, dst.ptr, size*sizeof(T), MPI_CHAR,
                              BCL::win, &request);
    BCL_DEBUG(
            if (error_code != MPI_SUCCESS) {
              throw debug_error("BCL write(): MPI_Rput return error code." + std::to_string(error_code));
            }
    )

    error_code = MPI_Wait(&request, MPI_STATUS_IGNORE);

    BCL_DEBUG(
            if (error_code != MPI_SUCCESS) {
              throw debug_error("BCL write(): MPI_Wait return error code " + std::to_string(error_code));
            }
    )
  }
}

template <typename T>
inline BCL::request async_read(const GlobalPtr<T>& src, T* dst, size_t size) {
  MPI_Request request;

  int error_code = MPI_Rget(dst, size*sizeof(T), MPI_CHAR,
                            src.rank, src.ptr, size*sizeof(T), MPI_CHAR,
                            BCL::win, &request);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL async_read(): MPI_Rget return error code " + std::to_string(error_code));
          }
  )
  return BCL::request(request);
}

template <typename T>
inline BCL::request async_write(const T* src, const GlobalPtr<T>& dst, size_t size) {
  MPI_Request request;

  int error_code = MPI_Rput(src, size*sizeof(T), MPI_CHAR,
                            dst.rank, dst.ptr, size*sizeof(T), MPI_CHAR,
                            BCL::win, &request);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL async_write(): MPI_Rput return error code " + std::to_string(error_code));
          }
  )
  return BCL::request(request);
}

template <typename T>
inline T broadcast(T &val, uint64_t root) {
  int error_code = MPI_Bcast(&val, sizeof(T), MPI_CHAR, root, BCL::comm);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL broadcast(): MPI_Bcast return error code " + std::to_string(error_code));
          }
  )
  return val;
}

template <typename T>
inline void broadcast(T* val, uint64_t root, size_t size, const BCL::Team& team) {
  BCL::backend::MPICommWrapper comm(team);
  if (team.in_team()) {
    int error_code = MPI_Bcast(val, size*sizeof(T), MPI_CHAR, root, comm.comm());
    BCL_DEBUG(
            if (error_code != MPI_SUCCESS) {
              throw debug_error("BCL broadcast(): MPI_Bcast return error code " + std::to_string(error_code));
            }
    )
  }
}

template <typename T>
inline auto arbroadcast(T* val, uint64_t root, size_t size, const BCL::Team& team) {
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::backend::MPICommWrapper comm(team);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  if (BCL::rank() == 0) {
    printf("%lf making comm\n", duration);
  }
  MPI_Request request;
  if (team.in_team()) {
    int error_code = MPI_Ibcast(val, size*sizeof(T), MPI_CHAR, root, comm.comm(), &request);
    BCL_DEBUG(
            if (error_code != MPI_SUCCESS) {
              throw debug_error("BCL arbroadcast(): MPI_Ibcast return error code " + std::to_string(error_code));
            }
    )
  }
  return BCL::request(request);
}

template <typename T, typename Allocator>
inline void broadcast(std::vector<T, Allocator>& val, uint64_t root, const BCL::Team& team) {
  BCL::backend::MPICommWrapper comm(team);
  if (team.in_team()) {
    int error_code = MPI_Bcast(val.data(), val.size()*sizeof(T), MPI_CHAR, root, comm.comm());
    BCL_DEBUG(
            if (error_code != MPI_SUCCESS) {
              throw debug_error("BCL broadcast(): MPI_Bcast return error code " + std::to_string(error_code));
            }
    )
  }
}

template <typename T>
inline T broadcast(T& val, uint64_t root, const BCL::Team& team) {
  BCL::backend::MPICommWrapper comm(team);
  if (team.in_team()) {
    int error_code = MPI_Bcast(&val, sizeof(T), MPI_CHAR, root, comm.comm());
    BCL_DEBUG(
            if (error_code != MPI_SUCCESS) {
              throw debug_error("BCL broadcast(): MPI_Bcast return error code " + std::to_string(error_code));
            }
    )
  }
  return val;
}

template <typename T, typename Op>
inline T reduce(const T& val, Op fn, size_t dst_rank) {
  BCL::GlobalPtr<T> recv_vals = nullptr;
  if (BCL::rank() == 0) {
    recv_vals = BCL::alloc<T>(BCL::backend::nprocs());
  }
  recv_vals = BCL::broadcast(recv_vals, 0);

  recv_vals[BCL::rank()] = val;
  BCL::barrier();

  if (BCL::rank() == 0) {
    T acc_val = val;

    for (size_t i = 1; i < BCL::backend::nprocs(); i++) {
      acc_val = fn(acc_val, recv_vals.local()[i]);
    }
    BCL::dealloc<T>(recv_vals);
    return acc_val;
  } else {
    return val;
  }
}

template <typename T, typename Op>
inline T allreduce(const T& val, Op fn) {
  T rv;

  rv = reduce(val, fn, 0);
  rv = broadcast(rv, 0);

  return rv;
}

template <typename T>
inline T allreduce(const T &val, const abstract_op <T> &op) {
  T rv;
  MPI_Allreduce(&val, &rv, 1, op.type(), op.op(), BCL::comm);
  return rv;
}

template <typename T>
inline T fetch_and_op(const GlobalPtr <T> ptr, const T &val, const atomic_op <T> &op) {
  T rv;
  MPI_Request request;

  int error_code = MPI_Rget_accumulate(&val, 1, op.type(),
                                       &rv, 1, op.type(),
                                       ptr.rank, ptr.ptr, 1, op.type(),
                                       op.op(), BCL::win, &request);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL fetch_and_op(): MPI_Rget_accumulate return error code " + std::to_string(error_code));
          }
  )

  error_code = MPI_Wait(&request, MPI_STATUS_IGNORE);

  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL fetch_and_op(): MPI_Rget_accumulate (MPI_Wait) return error code " + std::to_string(error_code));
          }
  )
  return rv;
}

template <typename T>
inline BCL::future<T> arfetch_and_op(const GlobalPtr<T> ptr, const T& val, const atomic_op<T>& op) {
  future<T> future;
  MPI_Request request;

  int error_code = MPI_Rget_accumulate(&val, 1, op.type(),
                                       future.value_.get(), 1, op.type(),
                                       ptr.rank, ptr.ptr, 1, op.type(),
                                       op.op(), BCL::win, &request);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL arfetch_and_op(): MPI_Rget_accumulate return error code " + std::to_string(error_code));
          }
  )
  future.update(request);
  return std::move(future);
}

// MPI one-sided is basically terrible :/
// TODO: beter CAS syntax
inline int int_compare_and_swap(const GlobalPtr <int> ptr, const int old_val,
  const int new_val) {
  int result;

  int error_code = MPI_Compare_and_swap(&new_val, &old_val, &result, MPI_INT, ptr.rank, ptr.ptr, BCL::win);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL int_compare_and_swap(): MPI_Compare_and_swap return error code " + std::to_string(error_code));
          }
  )

  error_code = MPI_Win_flush_local(ptr.rank, BCL::win);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL int_compare_and_swap(): MPI_Win_flush_local return error code " + std::to_string(error_code));
          }
  )
  return result;
}

inline uint16_t uint16_compare_and_swap(const GlobalPtr <uint16_t> ptr, const uint16_t old_val, const uint16_t new_val) {
  uint16_t result;
  int error_code = MPI_Compare_and_swap(&new_val, &old_val, &result, MPI_UNSIGNED_SHORT, ptr.rank, ptr.ptr, BCL::win);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL uint16_compare_and_swap(): MPI_Wait return error code " + std::to_string(error_code));
          }
  )

  error_code = MPI_Win_flush_local(ptr.rank, BCL::win);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL uint16_compare_and_swap(): MPI_Wait return error code " + std::to_string(error_code));
          }
  )
  return result;
}

inline uint64_t uint64_compare_and_swap(const GlobalPtr <uint64_t> ptr, const uint64_t old_val, const uint64_t new_val) {
  uint64_t result;
  int error_code = MPI_Compare_and_swap(&new_val, &old_val, &result, MPI_UNSIGNED_LONG_LONG, ptr.rank, ptr.ptr, BCL::win);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL uint64_compare_and_swap(): MPI_Compare_and_swap return error code " + std::to_string(error_code));
          }
  )

  error_code = MPI_Win_flush_local(ptr.rank, BCL::win);
  BCL_DEBUG(
          if (error_code != MPI_SUCCESS) {
            throw debug_error("BCL uint64_compare_and_swap(): MPI_Win_flush_local return error code " + std::to_string(error_code));
          }
  )
  return result;
}

} // end BCL
