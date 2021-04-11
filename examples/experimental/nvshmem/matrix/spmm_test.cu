// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

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

  bool verify_result = false;

  std::string fname = std::string(argv[1]);

  // Number of vecs in SpMM (width of multi-vec, matrix)
  size_t num_vecs = std::atoi(argv[2]);

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
  BCL::cuda::barrier();

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

  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::gemm(a, b, c);
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  double max_issue = BCL::allreduce(BCL::cuda::duration_issue, BCL::max<double>{});
  double max_sync = BCL::allreduce(BCL::cuda::duration_sync, BCL::max<double>{});
  double max_compute = BCL::allreduce(BCL::cuda::duration_compute, BCL::max<double>{});
  double max_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, BCL::max<double>{});
  double max_barrier = BCL::allreduce(BCL::cuda::duration_barrier, BCL::max<double>{});

  double min_issue = BCL::allreduce(BCL::cuda::duration_issue, BCL::min<double>{});
  double min_sync = BCL::allreduce(BCL::cuda::duration_sync, BCL::min<double>{});
  double min_compute = BCL::allreduce(BCL::cuda::duration_compute, BCL::min<double>{});
  double min_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, BCL::min<double>{});
  double min_barrier = BCL::allreduce(BCL::cuda::duration_barrier, BCL::min<double>{});

  BCL::cuda::duration_issue = BCL::allreduce(BCL::cuda::duration_issue, std::plus<double>{});
  BCL::cuda::duration_sync = BCL::allreduce(BCL::cuda::duration_sync, std::plus<double>{});
  BCL::cuda::duration_compute = BCL::allreduce(BCL::cuda::duration_compute, std::plus<double>{});
  BCL::cuda::duration_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, std::plus<double>{});
  BCL::cuda::duration_barrier = BCL::allreduce(BCL::cuda::duration_barrier, std::plus<double>{});

  BCL::print("SpMM took %lf s\n", duration);

  if (BCL::rank() == 0) {
    printf("duration_issue %lf (%lf -> %lf)\n",
           BCL::cuda::duration_issue / BCL::nprocs(),
           min_issue, max_issue);
    printf("duration_sync %lf (%lf -> %lf)\n",
           BCL::cuda::duration_sync / BCL::nprocs(),
           min_sync, max_sync);
    printf("duration_compute %lf (%lf -> %lf)\n",
           BCL::cuda::duration_compute / BCL::nprocs(),
           min_compute, max_compute);
    printf("duration_accumulate %lf (%lf -> %lf)\n",
           BCL::cuda::duration_accumulate / BCL::nprocs(),
           min_accumulate, max_accumulate);
    printf("duration_barrier %lf (%lf -> %lf)\n",
           BCL::cuda::duration_barrier / BCL::nprocs(),
           min_barrier, max_barrier);
  }

  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  if (BCL::rank() == 0 && verify_result) {
    fprintf(stderr, "Reading in matrix...\n");
    BCL::CSRMatrix<T, graphblas::Index> mat(fname);
    fprintf(stderr, "Copying to GPU...\n");
    auto local_a = BCL::cuda::to_gpu<T, graphblas::Index, allocator_type>(mat);

    fprintf(stderr, "Creating local b...\n");
    BCL::cuda::CudaMatrix<T, allocator_type> local_b({k, n});
    fprintf(stderr, "Creating local c...\n");
    BCL::cuda::CudaMatrix<T, allocator_type> local_c({m, n});

    fprintf(stderr, "Writing to matrices...\n");
    local_b = 1;
    local_c = 0;

    fprintf(stderr, "Doing local spmm...\n");
    BCL::cuda::spmm_cusparse(local_a, local_b, local_c);
    cudaDeviceSynchronize();

    fprintf(stderr, "Getting C matrix...\n");
    auto distributed_c = c.get_matrix();

    std::vector<T> local_data(local_c.size());
    cudaMemcpy(local_data.data(), local_c.data(), sizeof(T)*local_c.size(), cudaMemcpyDeviceToHost);

    assert(distributed_c.size() == local_c.size());
    fprintf(stderr, "Checking accuracy...\n");
    T eps = 1.0e-5;
    size_t matching = 0;
    bool print = false;
    for (size_t i = 0; i < c.shape()[0]; i++) {
      for (size_t j = 0; j < c.shape()[1]; j++) {
        size_t idx = i + j*c.shape()[0];
        if (std::abs(distributed_c[idx] - local_data[idx]) > eps) {
          assert(false);
          if (print) {
            printf("O %2.2lf != %2.2lf ", distributed_c[idx], local_data[idx]);
          }
        } else {
          if (print) {
            printf("X %2.2lf == %2.2lf ", distributed_c[idx], local_data[idx]);
          }
          matching++;
        }
      }
      if (print) {
        printf("\n");
      }
    }
    /*
    for (size_t i = 0; i < distributed_c.size(); i++) {
      if (std::abs(distributed_c[i] - local_data[i]) > eps) {
        // fprintf(stderr, "[%lu] %f != %f\n", i, distributed_c[i], local_data[i]);
      } else {
        matching++;
      }
    }
    */
    printf("%lu / %lu indices match.\n", matching, distributed_c.size());
    printf("OK.\n");
  }

  BCL::finalize();
  return 0;
}
