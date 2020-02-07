#pragma once

#include "nvshmem.h"
#include "nvshmemx.h"

#include <bcl/bcl.hpp>
#include "malloc.hpp"
#include "comm.hpp"
#include "allocator.hpp"
#include "gpu_side_allocator.hpp"

namespace BCL {

namespace cuda {

extern size_t shared_segment;
extern char* smem_base_ptr;
__device__ char* device_smem_base_ptr;

__device__ size_t rank_;
__device__ size_t nprocs_;

__global__ void set_device_ptr(char* smem_base_ptr, size_t rank, size_t nprocs) {
  device_smem_base_ptr = smem_base_ptr;
  rank_ = rank;
  nprocs_ = nprocs;
}

inline void barrier() {
  nvshmem_barrier_all();
  BCL::barrier();
}

inline __device__ __host__ size_t rank() {
  #ifdef __CUDA_ARCH__
    return rank_;
  #else
    return BCL::rank();
  #endif
}

inline __device__ __host__ size_t nprocs() {
  #ifdef __CUDA_ARCH__
    return nprocs_;
  #else
    return BCL::rank();
  #endif
}

inline void init(size_t shared_segment_size = 256) {
  BCL::cuda::shared_segment_size = 1024*1024*shared_segment_size;

  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &BCL::comm;

  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

  int device_count;
  cudaGetDeviceCount(&device_count);

  // TODO: check that we are not oversubscribing GPUs
  //       (only because NVSHMEM does not support this)

  cudaSetDevice(BCL::rank() % device_count);
  BCL::barrier();

  BCL::cuda::smem_base_ptr = (char *) nvshmem_malloc(BCL::cuda::shared_segment_size);

  if (smem_base_ptr == nullptr) {
    throw std::runtime_error("BCL: nvshmem backend could not allocate shared memory segment.");
  }

  if (nvshmem_my_pe() != BCL::rank() || nvshmem_n_pes() != BCL::nprocs()) {
    throw std::runtime_error("BCL: MPI and nvshmem ranks do not match up.");
  }

  set_device_ptr<<<1, 1>>>(smem_base_ptr, BCL::rank(), BCL::nprocs());
  cudaDeviceSynchronize();

  double total = BCL::cuda::shared_segment_size*0.1;
  init_gpu_side_allocator(total);

  BCL::cuda::barrier();
}

inline __host__ __device__ void flush() {
  nvshmem_quiet();
}

inline void finalize() {
  BCL::cuda::barrier();
  finalize_gpu_side_allocator();
  nvshmem_free(smem_base_ptr);
  nvshmem_finalize();
}

} //end cuda

} // end BCL
