#pragma once

#include <mpp/shmem.h>
#include <bcl/core/GlobalPtr.hpp>

namespace BCL {

template <typename T>
struct compare_and_swap_impl_;

template <>
struct compare_and_swap_impl_<int32_t> {
  static int32_t op(BCL::GlobalPtr<int32_t> ptr, int32_t old_val, int32_t new_val) {
    return shmem_int32_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

template <>
struct compare_and_swap_impl_<int64_t> {
  static int64_t op(BCL::GlobalPtr<int64_t> ptr, int64_t old_val, int64_t new_val) {
    return shmem_int64_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

template <>
struct compare_and_swap_impl_<uint32_t> {
  static uint32_t op(BCL::GlobalPtr<uint32_t> ptr, uint32_t old_val, uint32_t new_val) {
    return shmem_uint32_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

template <>
struct compare_and_swap_impl_<uint64_t> {
  static uint64_t op(BCL::GlobalPtr<uint64_t> ptr, uint64_t old_val, uint64_t new_val) {
    return shmem_uint64_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

}
