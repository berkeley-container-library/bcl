
#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>
#include <thrust/sort.h>

#include <bcl/containers/experimental/cuda/CudaSPMatrix.hpp>

#include <unordered_map>

#include "cusparse_util.hpp"
#include "spgemm.hpp"

#include <chrono>
#include <essl.h>

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init();

  using T = float;
  using index_type = int;

  bool verify_result = true;

  std::string fname = std::string(argv[1]);

  // Number of vecs in SpMM (width of multi-vec, matrix)
  size_t num_vecs = 1024;

  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  size_t m = matrix_shape.shape[0];
  size_t k = matrix_shape.shape[1];
  size_t n = num_vecs;

  BCL::print("Choosing blocks...\n");
  auto blocks = BCL::block_matmul(m, n, k);

  BCL::print("Reading matrices...\n");
  BCL::cuda::SPMatrix<T, graphblas::Index> a(fname, std::move(blocks[0]));
  BCL::cuda::Matrix<T> b(k, n, std::move(blocks[1]));
  BCL::cuda::Matrix<T> c(m, n, std::move(blocks[2]));
  b = 1;
  c = 0;

  BCL::cuda::grb_desc_ = new graphblas::Descriptor();

  BCL::print("Info:\n");
  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_info();
    printf("B:\n");
    b.print_info();
    printf("C:\n");
    c.print_info();
  }

  // printf("A taking %lf GB, B %lf GB\n", 1.0e-9*a.my_mem(), 1.0e-9*b.my_mem());

  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  using allocator_type = BCL::cuda::bcl_allocator<T>;

  BCL::cuda::gemm<T, graphblas::Index, allocator_type>(a, b, c);

  if (BCL::rank() == 0 && verify_result) {
  }

  BCL::finalize();
  return 0;
}
