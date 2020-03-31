

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include "cublas_v2.h"

#include <bcl/containers/experimental/cuda/CudaSPMatrix.hpp>

#include <chrono>
#include <essl.h>

void print_gpu() {
  int device;
  cudaGetDevice(&device);
  printf("Rank %lu on %d\n", BCL::rank(), device);
  fflush(stdout);
}

void flush_barrier() {
  BCL::barrier();
  fflush(stdout);
  BCL::barrier();
}

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init(8192);

  using index_type = graphblas::Index;

  using T = float;

  std::string fname = "/autofs/nccs-svm1_home2/b2v/data/chesapeake_general.mtx";

/*
  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  size_t m = matrix_shape[0];
  size_t n = matrix_shape[1];
  assert(m == n);
  size_t k = m;
  */

  BCL::print("Choosing blocks...\n");
  auto blocks = BCL::block_matmul(1024, 1024, 1024);

  BCL::print("Reading matrices...\n");
  BCL::cuda::SPMatrix<T> a(fname, std::move(blocks[0]));
  BCL::cuda::SPMatrix<T> b(fname, std::move(blocks[1]));
  BCL::cuda::SPMatrix<T> c(fname, std::move(blocks[2]));

  BCL::print("Info:\n");
  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_info();
    printf("B:\n");
    b.print_info();
    printf("C:\n");
    c.print_info();
  }
  flush_barrier();

  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  print_gpu();
  flush_barrier();
  flush_barrier();

  BCL::print("Beginning matrix multiply...\n");
    for (size_t i = 0; i < a.grid_shape()[0]; i++) {
      for (size_t j = 0; j < b.grid_shape()[1]; j++) {
        if (c.tile_rank({i, j}) == BCL::rank()) {
          graphblas::Matrix<T> local_c(a.tile_shape({i, 0})[0], b.tile_shape({0, j})[1]);
          for (size_t k = 0; k < a.grid_shape()[1]; k++) {
            auto local_a = a.get_tile({i, k});
            auto local_b = b.get_tile({k, j});
            print_gpu();
            graphblas::Descriptor* desc_ptr = nullptr;
            graphblas::Descriptor desc;
            desc.descriptor_.debug_ = false;
            graphblas::mxm<T, T, T, T>(&local_c, GrB_NULL,
                           GrB_NULL, graphblas::PlusMultipliesSemiring<T>(),
                           local_a, local_b,
                           &desc);
          }
        }
      }
    }

  BCL::finalize();
  return 0;
}
