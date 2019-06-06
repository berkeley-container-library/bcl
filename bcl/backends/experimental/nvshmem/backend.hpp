#pragma once

#include "nvshmem.h"
#include "nvshmemx.h"

#include <bcl/bcl.hpp>
#include "malloc.hpp"
#include "comm.hpp"

namespace BCL {

namespace cuda {

extern size_t shared_segment;
extern char* smem_base_ptr;
__device__ char* device_smem_base_ptr;

__global__ void set_device_ptr(char* smem_base_ptr) {
  device_smem_base_ptr = smem_base_ptr;
}

inline void barrier() {
  nvshmem_barrier_all();
  BCL::barrier();
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

  BCL::cuda::smem_base_ptr = (char *) nvshmem_malloc(BCL::shared_segment_size);

  if (smem_base_ptr == nullptr) {
    throw std::runtime_error("BCL: nvshmem backend could not allocate shared memory segment.");
  }

  if (nvshmem_my_pe() != BCL::rank() || nvshmem_n_pes() != BCL::nprocs()) {
    throw std::runtime_error("BCL: MPI and nvshmem ranks do not match up.");
  }

  set_device_ptr<<<1, 1>>>(smem_base_ptr);
  cudaDeviceSynchronize();

  BCL::cuda::barrier();
}

inline __host__ __device__ void flush() {
  nvshmem_quiet();
}

inline void finalize() {
  BCL::cuda::barrier();
  nvshmem_free(smem_base_ptr);
  nvshmem_finalize();
}

} //end cuda

} // end BCL
