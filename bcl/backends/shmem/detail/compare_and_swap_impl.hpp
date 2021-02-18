#pragma once

#include <mpp/shmem.h>
#include <bcl/core/GlobalPtr.hpp>

namespace BCL {

template <typename T>
struct compare_and_swap_impl_;

template <>
struct compare_and_swap_impl_<int> {
  static int op(BCL::GlobalPtr<int> ptr, int old_val, int new_val) {
    return shmem_int_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

template <>
struct compare_and_swap_impl_<long long> {
  static long long op(BCL::GlobalPtr<long long> ptr, long long old_val, long long new_val) {
    return shmem_longlong_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

template <>
struct compare_and_swap_impl_<unsigned int> {
  static unsigned int op(BCL::GlobalPtr<unsigned int> ptr, unsigned int old_val, unsigned int new_val) {
    return shmem_uint_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

template <>
struct compare_and_swap_impl_<unsigned long long> {
  static unsigned long long op(BCL::GlobalPtr<unsigned long long> ptr, unsigned long long old_val, unsigned long long new_val) {
    return shmem_ulonglong_atomic_compare_swap(ptr.rptr(), old_val, new_val, ptr.rank);
  }
};

}
