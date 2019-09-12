#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/FastQueue.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

int main(int argc, char** argv) {
  BCL::init(16);

  printf("Hello, world! I am rank %lu/%lu\n",
         BCL::rank(), BCL::nprocs());

  BCL::cuda::init();

  BCL::cuda::FastQueue<int> queue(0, 1000);

  BCL::cuda::global_launch(1000,
                     [] __device__ (size_t idx, BCL::cuda::FastQueue<int>& queue) {
                       bool success = queue.push(idx);
                       if (!success) {
                         printf("AGH! I have failed!\n");
                       }
                     }, queue);

  BCL::cuda::barrier();

  BCL::cuda::global_launch(1000,
                     [] __device__ (size_t idx, BCL::cuda::FastQueue<int>& queue) {
                       int value;
                       bool success = queue.pop(value);
                       if (!success || value < 0 || value > 1000) {
                         printf("AGH! I have failed (popping)!\n");
                       } else {
                         printf("Got %d\n", value);
                       }
                     }, queue);
  fflush(stdout);
  fflush(stderr);
  BCL::cuda::barrier();
  fflush(stdout);
  fflush(stderr);
  BCL::cuda::barrier();

  BCL::finalize();
  return 0;
}
