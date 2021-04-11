// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda.h>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

__global__ void kernel(BCL::cuda::device_vector<int> vec) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  int value = vec.data()[tid];
  printf("%lu: %d\n", tid, value);
}

int main(int argc, char** argv) {
  cudaSetDevice(0);

  constexpr size_t n = 16;

  std::vector<int> vec(n);
  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = i;
  }

  BCL::cuda::device_vector<int> v(vec.begin(), vec.end());

  printf("First Launch (should be in order 0 -> n)\n");
  fflush(stdout);
  BCL::cuda::launch(v.size(),
                    [] __device__ (size_t tid, BCL::cuda::device_vector<int> v) {
                      int val = v[tid];
                      printf("Element %lu == %d\n", tid, val);
                    }, v);
  cudaDeviceSynchronize();
  fflush(stdout);

  printf("Second Launch (modifying values)\n");
  fflush(stdout);
  BCL::cuda::launch(v.size(),
                    [] __device__ (size_t tid, BCL::cuda::device_vector<int> v) {
                      v[tid] = v.size() - tid;
                    }, v);

  cudaDeviceSynchronize();
  fflush(stdout);

  printf("Third Launch (should be in order n -> 1)\n");
  fflush(stdout);

  BCL::cuda::launch(v.size(),
                    [] __device__ (size_t tid, BCL::cuda::device_vector<int> v) {
                      int val = v[tid];
                      printf("Element %lu == %d\n", tid, val);
                    }, v);

  cudaDeviceSynchronize();
  fflush(stdout);

  v.destroy();

  return 0;
}
