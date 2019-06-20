#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/DArray.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

#include <bcl/containers/experimental/cuda/DHashMap.hpp>

int main(int argc, char** argv) {
  BCL::init(64);
  BCL::cuda::init(1024);
  BCL::cuda::DHashMap<int, int> map(100);

  if (BCL::rank() == 0) {
    BCL::cuda::launch(1, [] __device__ (size_t tid, BCL::cuda::DHashMap<int, int>& map) {
                           for (size_t i = 0; i < 10; i++) {
                             map.insert(i, i);
                           }
                         }, map);
    cudaDeviceSynchronize();
  }
  BCL::cuda::barrier();

  if (BCL::rank() == 1) {
    BCL::cuda::launch(1, [] __device__ (size_t tid, BCL::cuda::DHashMap<int, int>& map) {
                           for (size_t i = 0; i < 10; i++) {
                             int value = map.find(i);
                             printf("{%lu, %d}\n", i, value);
                           }
                         }, map);
    cudaDeviceSynchronize();
  }

  BCL::cuda::finalize();
  BCL::finalize();
  return 0;
}
