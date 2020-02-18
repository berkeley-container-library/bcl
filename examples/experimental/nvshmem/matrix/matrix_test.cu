
#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

template <typename T>
void print_vector(const std::vector<T>& vec, size_t m, size_t n) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      std::cout << vec[i*n + j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init(1024);

  BCL::cuda::Matrix<int> m(16, 16);

  m.apply([] __device__ (int m) { return 12; });

  if (BCL::rank() == 0) {
    m.print_info();

    for (size_t i = 0; i < m.grid_shape()[0]; i++) {
      for (size_t j = 0; j < m.grid_shape()[1]; j++) {
        auto f = m.get_tile({i, j});
        std::vector<int> local_tile(f.size());
        cudaMemcpy(local_tile.data(), f.data(), sizeof(int)*f.size(), cudaMemcpyDeviceToHost);
        f.destroy();
        printf("Block (%lu, %lu)\n", i, j);
        print_vector(local_tile, m.tile_shape({i, j})[0], m.tile_shape({i, j})[1]);
      }
    }
  }

  BCL::cuda::barrier();

  BCL::finalize();
  return 0;
}
