#pragma once

#include <bcl/bcl.hpp>
#include <cuda.h>

// This is just a convenience function, *launch*, to launch a kernel.

namespace BCL {

namespace cuda {

template <typename Fn, typename... Args>
__global__ void bcl_cuda_kernel_launch_impl_(size_t extent, Fn fn, Args... args) {
  size_t tid = threadIdx.x + blockIdx.x *blockDim.x;
  if (tid < extent) {
    fn(tid, args...);
  }
}

template <typename Fn, typename... Args>
__global__ void bcl_cuda_global_kernel_launch_impl_(size_t host, size_t local_extent, size_t global_extent, Fn fn, Args... args) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_t gtid = host * local_extent + tid;

  if (gtid < global_extent) {
    fn(gtid, args...);
  }
}

template <typename Fn, typename... Args>
void launch(size_t extent, Fn fn, Args&& ... args) {
  size_t block_size = std::min(std::size_t(1024), extent);
  size_t num_blocks = (extent + block_size - 1) / block_size;
  bcl_cuda_kernel_launch_impl_<<<num_blocks, block_size>>>(extent, fn, args...);
}

template <typename Fn, typename... Args>
void global_launch(size_t extent, Fn fn, Args&& ... args) {
  size_t local_extent = (extent + BCL::nprocs() - 1) / BCL::nprocs();

  size_t block_size = std::min(std::size_t(1024), local_extent);
  size_t num_blocks = (local_extent + block_size - 1) / block_size;

  bcl_cuda_global_kernel_launch_impl_<<<num_blocks, local_extent>>>(BCL::rank(), local_extent, extent, fn, args...);
}

} // end cuda
} // end BCL
