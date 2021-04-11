// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <bcl/bcl.hpp>
#include <cuda.h>

// This is just a convenience function, *launch*, to launch a kernel.

namespace BCL {

namespace cuda {

struct LaunchInfo {
  size_t tid = 0;
  size_t ltid = 0;
  size_t gtid = 0;

  size_t extent = 0;
  size_t local_extent = 0;
  size_t global_extent = 0;

  __host__ __device__ operator size_t() const noexcept {
    return tid;
  }
};

template <typename Fn, typename... Args>
__global__ void bcl_cuda_kernel_launch_impl_(size_t extent, Fn fn, Args... args) {
  size_t tid = threadIdx.x + blockIdx.x *blockDim.x;
  LaunchInfo info;
  info.tid = tid;
  info.ltid  = tid;
  info.extent = extent;
  info.local_extent = extent;
  if (tid < extent) {
    fn(info, args...);
  }
}

template <typename Fn, typename... Args>
__global__ void bcl_cuda_global_kernel_launch_impl_(size_t host, size_t local_extent, size_t global_extent, Fn fn, Args... args) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t gtid = host * local_extent + tid;

  LaunchInfo info;
  info.tid  = gtid;
  info.ltid = tid;
  info.gtid = gtid;
  info.extent = global_extent;
  info.local_extent = local_extent;
  info.global_extent = global_extent;

  if (gtid < global_extent && tid < local_extent) {
    fn(info, args...);
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

  bcl_cuda_global_kernel_launch_impl_<<<num_blocks, block_size>>>(BCL::rank(), local_extent, extent, fn, args...);
}

} // end cuda
} // end BCL
