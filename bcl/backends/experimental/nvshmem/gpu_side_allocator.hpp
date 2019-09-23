#pragma once

#include <cuda.h>

namespace emalloc {

char* segment_ptr;
size_t segment_size;

__device__ char* d_segment_ptr;
__device__ size_t d_segment_size;
__device__ unsigned long long* d_heap_ptr;

__global__ void set_device_vars(char* segptr, size_t size, unsigned long long* hptr) {
  d_segment_ptr = segptr;
  d_segment_size = size;
  d_heap_ptr = hptr;
  *d_heap_ptr = 0;
}

void init_emalloc(void* ptr, size_t n_bytes) {
  segment_ptr = (char *) ptr;
  segment_size = n_bytes;

  unsigned long long* hptr;
  cudaMalloc((void **) &hptr, sizeof(unsigned long long));

  set_device_vars<<<1, 1>>>(segment_ptr, segment_size, hptr);
  cudaDeviceSynchronize();
}

__device__ void* emalloc(size_t size) {
  unsigned long long value = atomicAdd(d_heap_ptr, (unsigned long long) size);
  if (value + size <= d_segment_size) {
    return d_segment_ptr + value;
  } else {
    return nullptr;
  }
}

// Do nothing...
__device__ void efree(void* ptr) {
}

} // end emalloc
namespace BCL {

namespace cuda {

BCL::cuda::ptr<char> gpu_side_segment;

void init_gpu_side_allocator(size_t size) {
  gpu_side_segment = BCL::cuda::alloc<char>(size);
  emalloc::init_emalloc(gpu_side_segment.local(), size);
}

void finalize_gpu_side_allocator() {
  BCL::cuda::dealloc(gpu_side_segment);
}

} // end BCL

} // end cuda
