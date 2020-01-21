#pragma once

#include "backend.hpp"
#include "ops.hpp"
#include "request.hpp"

#include <vector>
#include <functional>
#include <gasnet_coll.h>

#include "atomics.hpp"

namespace BCL {
extern gasnet_seginfo_t *gasnet_seginfo;
extern gex_TM_t tm;

extern void barrier();

template <typename T>
extern void* gasnet_resolve_address(const GlobalPtr<T> ptr);

template <typename T>
inline void read(const GlobalPtr <T> &src, T* dst, const size_t size) {
  void* src_ptr = gasnet_resolve_address(src);
  gex_RMA_GetBlocking(tm, dst, src.rank, src_ptr, size*sizeof(T), 0);
}

template <typename T>
inline void atomic_read(const GlobalPtr<T>& src, T* dst, size_t size) {
  assert(size == 1);
  static_assert(std::is_same<T, int32_t>::value);
  void* src_ptr = gasnet_resolve_address(src);

  gex_Event_t event = shim_gex_AD_OpNB<T>(ad_i32, dst, src.rank, src_ptr,
                                          GEX_OP_GET, T(), T(), 0);
  gex_Event_Wait(event);
}

template <typename T>
inline void write(const T *src, const GlobalPtr <T> &dst, const size_t size) {
  void* dst_ptr = gasnet_resolve_address(dst);
  gex_RMA_PutBlocking(tm, dst.rank, dst_ptr, (T *) src, size*sizeof(T), 0);
}

template <typename T>
inline BCL::request async_read(const GlobalPtr<T>& src, T* dst, size_t size) {
  void* src_ptr = gasnet_resolve_address(src);
  gex_Event_t request = gex_RMA_GetNB(tm, dst, src.rank, src_ptr, size*sizeof(T), 0);
  return BCL::request(request);
}

template <typename T>
inline BCL::request async_write(const T* src, const GlobalPtr<T>& dst, size_t size) {
  void* dst_ptr = gasnet_resolve_address(dst);
  // TODO: remove const_cast (after GASNet-EX fixes)
  gex_Event_t request = gex_RMA_PutNB(tm, dst.rank, dst_ptr, const_cast<T*>(src),
                                      size*sizeof(T), GEX_EVENT_DEFER, 0);
  return BCL::request(request);
}

template <typename T>
inline T broadcast(T& val, uint64_t root) {
  T rv;
  gasnet_coll_broadcast(GASNET_TEAM_ALL, &rv, root, &val, sizeof(T),
                        GASNET_COLL_IN_ALLSYNC | GASNET_COLL_OUT_ALLSYNC | GASNET_COLL_LOCAL);

  return rv;
}


template <typename T, typename Op>
inline T reduce(const T& val, Op fn, size_t dst_rank) {
  BCL::GlobalPtr<T> recv_vals = nullptr;
  if (BCL::rank() == 0) {
    recv_vals = BCL::alloc<T>(BCL::nprocs());
  }
  recv_vals = BCL::broadcast(recv_vals, 0);

  recv_vals[BCL::rank()] = val;
  BCL::barrier();

  if (BCL::rank() == 0) {
    T acc_val = val;

    for (size_t i = 1; i < BCL::nprocs(); i++) {
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

#include <type_traits>

template <typename T>
T fetch_and_op(const GlobalPtr <T> ptr, const T &val, const atomic_op <T> &op) {
  static_assert(std::is_same<T, int32_t>::value || std::is_same<T, float>::value);
  T rv;
  void* dst_ptr = gasnet_resolve_address(ptr);
  // TODO: select the correct AD
  gex_Event_t event = shim_gex_AD_OpNB<T>(get_gex_ad<T>(), &rv, ptr.rank, dst_ptr,
                                          op.op(), val, val, 0);
  gex_Event_Wait(event);
  return rv;
}

template <typename T>
future<T> arfetch_and_op(const GlobalPtr <T> ptr, const T &val, const atomic_op <T> &op) {
  static_assert(std::is_same<T, int32_t>::value);
  future<T> future;
  void* dst_ptr = gasnet_resolve_address(ptr);
  // TODO: select the correct AD
  gex_Event_t event = shim_gex_AD_OpNB<T>(get_gex_ad<T>(), future.value_.get(), ptr.rank, dst_ptr,
                                          op.op(), val, val, 0);
  future.update(event);
  return std::move(future);
}

int32_t int_compare_and_swap(GlobalPtr<int32_t> ptr, int32_t old_val,
                             int32_t new_val) {
  void* dst_ptr = gasnet_resolve_address(ptr);
  int32_t rv;
  gex_Event_t event = gex_AD_OpNB_I32(get_gex_ad<int32_t>(), &rv, ptr.rank, dst_ptr,
                                      GEX_OP_FCAS, old_val, new_val, 0);

  gex_Event_Wait(event);
  return rv;
}

}
