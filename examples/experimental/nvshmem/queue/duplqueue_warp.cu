// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/DuplQueue.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

#include <chrono>

#define NUM_INSERTS 2*8*1024

int main(int argc, char** argv) {
  BCL::init(16);

  printf("Hello, world! I am rank %lu/%lu\n",
         BCL::rank(), BCL::nprocs());

  BCL::cuda::init(14*1024);

  size_t num_inserts = NUM_INSERTS;
  size_t insert_size = 8*1024;

  // Round up so each rank has an equal number of inserts.
  size_t inserts_per_rank = (num_inserts + BCL::nprocs() - 1) / BCL::nprocs();
  inserts_per_rank *= BCL::nprocs();
  num_inserts = inserts_per_rank * BCL::nprocs();

  BCL::cuda::DuplQueue<int> queue(0, num_inserts*insert_size);

  BCL::cuda::device_vector<int, BCL::cuda::bcl_allocator<int>> values(insert_size);
  // BCL::cuda::device_vector<int> values(insert_size);
  std::vector<int> values_local(insert_size, BCL::rank());
  values.assign(values_local.begin(), values_local.end());

  BCL::cuda::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  BCL::cuda::launch(inserts_per_rank*32,
                    [] __device__ (size_t idx, BCL::cuda::DuplQueue<int>& queue,
                                   BCL::cuda::device_vector<int, BCL::cuda::bcl_allocator<int>>& values) {
                      // BCL::cuda::device_vector<int>& values) {
                      bool success = queue.push_warp(values.data(), values.size());
                      if (!success) {
                        printf("AGH! I have failed!\n");
                      }
                    }, queue, values);

  cudaDeviceSynchronize();
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  double data_moved = num_inserts*insert_size*sizeof(int);
  double data_moved_gb = data_moved*1e-9;

  double bw = data_moved / duration;
  double bw_gb = bw*1e-9;

  BCL::print("Total %lf s (%lf GB) (%lf GB/s)\n", duration, data_moved_gb, bw_gb);

  if (BCL::rank() == 0) {
    BCL::cuda::launch(num_inserts,
                      [] __device__ (size_t idx, BCL::cuda::DuplQueue<int>& queue) {
                        int value = 12;
                        bool success = queue.local_pop(value);
                        // printf("%lu: %d (%s)\n", idx, value, (success) ? "success" : "failure");
                      }, queue);
    cudaDeviceSynchronize();
  }
  BCL::cuda::barrier();

  BCL::print("Here...\n");

  BCL::cuda::barrier();
  BCL::print("After barrier...\n");

  BCL::finalize();
  return 0;
}
