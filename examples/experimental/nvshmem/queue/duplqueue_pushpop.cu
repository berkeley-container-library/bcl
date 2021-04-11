// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/DuplQueue.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

#include <chrono>

int main(int argc, char** argv) {
  BCL::init(16);

  printf("Hello, world! I am rank %lu/%lu\n",
         BCL::rank(), BCL::nprocs());

  BCL::cuda::init(8*1024);

  size_t num_inserts = 2*1024;
  size_t insert_size = 1024;

  size_t total_inserts = num_inserts*insert_size;

  BCL::cuda::DuplQueue<int> queue(0, total_inserts);

  BCL::cuda::device_vector<int, BCL::cuda::bcl_allocator<int>> values(insert_size);
  // BCL::cuda::device_vector<int> values(insert_size);
  std::vector<int> values_local(insert_size, BCL::rank());
  values.assign(values_local.begin(), values_local.end());

  BCL::cuda::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  BCL::cuda::global_launch(num_inserts,
                     [] __device__ (size_t idx, BCL::cuda::DuplQueue<int>& queue,
                                    BCL::cuda::device_vector<int, BCL::cuda::bcl_allocator<int>>& values) {
                                    // BCL::cuda::device_vector<int>& values) {
                       bool success = queue.push(values.data(), values.size());
                       if (!success) {
                         printf("AGH! I have failed!\n");
                       }
                     }, queue, values);

  cudaDeviceSynchronize();
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  double data_moved = total_inserts*sizeof(int);

  double bw = data_moved / duration;
  double bw_gb = bw*1e-9;

  BCL::print("Total %lf s (%lf GB/s)\n", duration, bw_gb);

  if (BCL::rank() == 0) {
    std::vector<int> histogram_local(BCL::nprocs(), 0);
    BCL::cuda::device_vector<int> histogram(BCL::nprocs());
    histogram.assign(histogram_local.begin(), histogram_local.end());
    BCL::cuda::launch(total_inserts,
                      [] __device__ (size_t idx, BCL::cuda::DuplQueue<int>& queue,
                                     BCL::cuda::device_vector<int>& histogram) {
                        int value = 12;
                        bool success = queue.local_pop(value);
                        if (success && value >= 0 && value < BCL::cuda::nprocs()) {
                          atomicAdd(&histogram.data()[value], 1);
                        } else {
                          printf("Error! Missing values in the queue (%lu)\n", idx);
                        }
                      }, queue, histogram);
    cudaDeviceSynchronize();

    size_t total_counted = 0;
    for (size_t i = 0; i < histogram.size(); i++) {
      int hval = histogram[i];
      printf("%lu: %d\n", i, hval);
      total_counted += hval;
    }
    assert(total_counted == total_inserts);
  }
  BCL::cuda::barrier();

  BCL::print("Finished...\n");

  BCL::finalize();
  return 0;
}
