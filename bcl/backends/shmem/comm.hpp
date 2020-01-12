#pragma once

#include <functional>
#include <cstring>
#include <vector>

#include "backend.hpp"
#include "ops.hpp"
#include "atomics.hpp"
#include "request.hpp"

namespace BCL {

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
  if (src.rank == BCL::rank()) {
    std::memcpy(dst, src.rptr(), size*sizeof(T));
  } else {
    shmem_getmem(dst, src.rptr(), sizeof(T)*size, src.rank);
  }
}

// Read size T's from src -> dst
// Blocks until dst is ready.
template <typename T>
inline void atomic_read(const GlobalPtr <T> &src, T *dst, const size_t size) {
  BCL_DEBUG(
    if (src.rank > BCL::backend::nprocs()) {
      throw debug_error("atomic_read(): request to read from rank "
                        + std::to_string(src.rank) + ", which does not exist");
    }
  )
  shmem_getmem(dst, src.rptr(), sizeof(T)*size, src.rank);
}

// Write size T's from src -> dst
// Returns after src buffer is sent
// and okay to be modified; no
// guarantee that memop is complete
// until BCL::barrier()
template <typename T>
inline void write(const T *src, const GlobalPtr <T> &dst, const size_t size) {
  BCL_DEBUG(
    if (dst.rank > BCL::backend::nprocs()) {
        throw debug_error("BCL write(): request to write to rank " +
                          std::to_string(dst.rank) + ", which does not exist");
    }
  )
  if (dst.rank == BCL::rank()) {
    std::memcpy(dst.rptr(), src, size*sizeof(T));
  } else {
    shmem_putmem(dst.rptr(), src, sizeof(T)*size, dst.rank);
  }
}

template <typename T>
inline BCL::request async_read(const GlobalPtr<T>& src, T* dst, size_t size) {
  shmem_getmem_nbi(dst, src.rptr(), sizeof(T)*size, src.rank);
  return BCL::request();
}

template <typename T>
inline BCL::request async_write(const T* src, const GlobalPtr<T>& dst, size_t size) {
  shmem_putmem_nbi(dst.rptr(), src, size*sizeof(T), dst.rank);
  return BCL::request();
}

template <typename T>
inline T broadcast(T &val, uint64_t root) {
  static long pSync[SHMEM_BCAST_SYNC_SIZE];
  for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }
  shmem_barrier_all();
  constexpr size_t memsize = (sizeof(T) + sizeof(long) - 1) / sizeof(long);
  static long rv[memsize];
  static long startval[memsize];
  std::memcpy(startval, &val, sizeof(T));
  shmem_broadcast64(rv, startval, memsize, root, 0, 0, BCL::nprocs(), pSync);
  if (BCL::rank() != root) {
    std::memcpy(&val, rv, sizeof(T));
  }
  return val;
}

template <typename T>
inline T broadcast(const T &val, uint64_t root) {
  static long pSync[SHMEM_BCAST_SYNC_SIZE];
  for (int i = 0; i < SHMEM_BCAST_SYNC_SIZE; i++) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }
  shmem_barrier_all();
  constexpr size_t memsize = (sizeof(T) + sizeof(long) - 1) / sizeof(long);
  static long rv[memsize];
  static long startval[memsize];
  std::memcpy(startval, &val, sizeof(T));
  shmem_broadcast64(rv, startval, memsize, root, 0, 0, BCL::nprocs(), pSync);
  if (BCL::rank() != root) {
    return *(reinterpret_cast <T *> (&rv));
  } else {
    return val;
  }
}

template <typename T, typename BinaryOp>
inline T allreduce(const T &val, BinaryOp op) {
  T rv = val;
  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    T current = broadcast(val, rank);
    if (rank != BCL::rank()) {
      rv = op(rv, current);
    }
  }
  return rv;
}

inline int allreduce(const int &val, std::plus <int> op) {
  static int ival = val;
  static int rv;
  static long pSync[SHMEM_REDUCE_SYNC_SIZE];
  for (int i = 0; i < SHMEM_REDUCE_SYNC_SIZE; i++) {
    pSync[i] = SHMEM_SYNC_VALUE;
  }
  static int pWrk[SHMEM_REDUCE_MIN_WRKDATA_SIZE];
  shmem_barrier_all();
  shmem_int_sum_to_all(&rv, &ival, 1, 0, 0, BCL::nprocs(), pWrk, pSync);
  return rv;
}

template <typename T>
T fetch_and_op(const GlobalPtr <T> ptr, const T &val, const atomic_op <T> &op) {
  return op.shmem_atomic_op(ptr, val);
}

template <typename T>
future<T> arfetch_and_op(const GlobalPtr <T> ptr, const T &val, const atomic_op <T> &op) {
  static_assert(std::is_same<T, int32_t>::value);
  future<T> future;
  int rv =  op.shmem_atomic_op(ptr, val);
  *future.value_ = rv;
  return std::move(future);
}

int int_compare_and_swap(const GlobalPtr <int> ptr, int old_val,
  const int new_val) {
  old_val = shmem_int_cswap(ptr.rptr(), old_val, new_val, ptr.rank);
  return old_val;
}

/*
short short_compare_and_swap(const GlobalPtr <short> ptr, short old_val,
  const short new_val) {
  old_val = shmem_short_cswap(ptr.rptr(), old_val, new_val, ptr.rank);
  return old_val;
}
*/

uint64_t uint64_compare_and_swap(const GlobalPtr <uint64_t> ptr, uint64_t old_val,
  const uint64_t new_val) {
  long long ov = *((long long *) &old_val);
  long long nv = *((long long *) &old_val);
  long long rv = shmem_longlong_cswap((long long *) ptr.rptr(), ov, nv, ptr.rank);
  return *((uint64_t *) &rv);
}

/*
uint64_t uint64_atomic_fetch_or(const GlobalPtr <uint64_t> ptr, uint64_t val) {
  return shmem_uint64_atomic_fetch_or(ptr.rptr(), val, ptr.rank);
}
*/

} // end BCL
