#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>
#include <bcl/containers/experimental/cuda/DArray.hpp>

void bcl_array_test() {
  size_t block_size = 4;
  BCL::cuda::DArray<int> array(BCL::nprocs()*block_size);

  BCL::cuda::global_launch(array.size(),
                     [] __device__ (size_t idx, BCL::cuda::DArray<int>& array) {
                       array[idx] = idx;
                     }, array);

  BCL::cuda::barrier();

  BCL::cuda::global_launch(array.size(),
                     [] __device__ (size_t idx, BCL::cuda::DArray<int>& array) {
                       int result = array[idx];
                       printf("%lu: %d\n", idx, result);
                     }, array);

  BCL::cuda::barrier();
}

int main(int argc, char** argv) {
  BCL::init(64);

  BCL::cuda::init();

  bcl_array_test();

  BCL::cuda::finalize();

  BCL::finalize();
  return 0;
}
