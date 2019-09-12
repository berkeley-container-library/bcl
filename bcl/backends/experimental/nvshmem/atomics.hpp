#pragma once

namespace BCL {
namespace cuda {

__device__ int compare_and_swap(BCL::cuda::ptr<int> ptr, int old_value, int new_value) {
  return nvshmem_int_cswap(ptr.rptr(), old_value, new_value, ptr.rank_);
}

__device__ int fetch_and_add(BCL::cuda::ptr<int> ptr, int value) {
  return nvshmem_int_fadd(ptr.rptr(), value, ptr.rank_);
}

} // end cuda
} // end BCL
