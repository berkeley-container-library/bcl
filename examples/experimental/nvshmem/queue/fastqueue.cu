#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/FastQueue.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

#include <chrono>

#define NUM_INSERTS 64*1024*1024

int main(int argc, char** argv) {
  BCL::init(16);

  printf("Hello, world! I am rank %lu/%lu\n",
         BCL::rank(), BCL::nprocs());

  BCL::cuda::init(16*1024);

  size_t num_inserts = NUM_INSERTS;

  BCL::cuda::FastQueue<int> queue(0, num_inserts);

  BCL::cuda::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  BCL::cuda::global_launch(num_inserts,
                     [] __device__ (size_t idx, BCL::cuda::FastQueue<int>& queue) {
                       bool success = queue.push(idx);
                       if (!success) {
                         printf("AGH! I have failed!\n");
                       }
                     }, queue);

  cudaDeviceSynchronize();

  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  BCL::print("Finished in %lf s (%lf megapushes/s)\n", duration, (num_inserts / duration) / (1024*1024));

  BCL::print("Queue has %lu items (out of %lu)\n", queue.size(), num_inserts);

  BCL::cuda::global_launch(num_inserts,
                     [] __device__ (size_t idx, BCL::cuda::FastQueue<int>& queue) {
                       int value;
                       bool success = queue.pop(value);
                       if (!success || value < 0 || value > NUM_INSERTS) {
                         printf("AGH! I have failed (popping)!\n");
                       }
                     }, queue);
  cudaDeviceSynchronize();
  BCL::cuda::barrier();

  BCL::print("Queue has %lu items (out of %lu)\n", queue.size(), num_inserts);

  BCL::finalize();
  return 0;
}
