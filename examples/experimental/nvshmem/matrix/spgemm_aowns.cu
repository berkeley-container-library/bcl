
#define __thrust_compiler_fence() __sync_synchronize()
#include <cusp/io/matrix_market.h>
#include <cusp/csr_matrix.h>
#include <cusp/array2d.h>
#include <cusp/multiply.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

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

template <typename T, typename U>
struct PairHash {
  std::size_t operator()(const std::pair<T, U>& value) const noexcept {
    return std::hash<T>{}(value.first) ^ std::hash<U>{}(value.second);
  }
};

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init();

  using T = float;
  using index_type = int;

  bool verify_result = false;

  std::string fname = std::string(argv[1]);

  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  size_t m = matrix_shape.shape[0];
  size_t n = matrix_shape.shape[1];
  assert(m == n);
  size_t k = m;

  BCL::print("Choosing blocks...\n");
  auto blocks = BCL::block_matmul(m, n, k);

  BCL::print("Reading matrices...\n");
  BCL::cuda::SPMatrix<T, index_type> a(fname, std::move(blocks[0]));
  BCL::cuda::SPMatrix<T, index_type> b(fname, std::move(blocks[1]));
  BCL::cuda::SPMatrix<T, index_type> c(m, n, std::move(blocks[2]));

  BCL::print("Info:\n");
  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_info();
    printf("B:\n");
    b.print_info();
    printf("C:\n");
    c.print_info();
  }

  cusparseStatus_t status = cusparseCreate(&BCL::cuda::bcl_cusparse_handle_);
  BCL::cuda::throw_cusparse(status);
  status = cusparseSetPointerMode(BCL::cuda::bcl_cusparse_handle_, CUSPARSE_POINTER_MODE_HOST);
  BCL::cuda::throw_cusparse(status);

  // printf("A taking %lf GB, B %lf GB\n", 1.0e-9*a.my_mem(), 1.0e-9*b.my_mem());

  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  using allocator_type = BCL::cuda::bcl_allocator<T>;

  BCL::cuda::duration_issue = 0;
  BCL::cuda::duration_sync = 0;
  BCL::cuda::duration_compute = 0;
  BCL::cuda::duration_accumulate = 0;
  BCL::cuda::duration_barrier = 0;

  BCL::print("Beginning SpGEMM...\n");

  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::gemm_aowns<T, index_type, allocator_type>(a, b, c);
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

  BCL::barrier();
  fflush(stdout);
  BCL::barrier();
  fprintf(stderr, "RANK(%lu) A has %lu nnz, B has %lu nnz, C has %lu nnz\n",
          BCL::rank(), a.my_nnzs(), b.my_nnzs(), c.my_nnzs());
  BCL::barrier();
  fflush(stderr);
  BCL::barrier();

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

  BCL::print("Matrix multiply finished in %lf s\n", duration);

  if (BCL::rank() == 0 && verify_result) {
    BCL::CSRMatrix<T, index_type> mat(fname);

    auto local_a = BCL::cuda::to_gpu<T, index_type, allocator_type>(mat);

    auto s_c = spgemm_cusparse(local_a, local_a);

    fprintf(stderr, "Getting COO...\n");
    auto local_c = c.get().get_coo();
    local_c = BCL::cuda::remove_zeros(local_c);

    auto s_c_coo = BCL::cuda::to_cpu(s_c).get_coo();

    fprintf(stderr, "local_computation (%lu nnz), distributed result (%lu nnz)\n", s_c_coo.size(), local_c.size());

/*
    FILE* f = fopen("/gpfs/alpine/bif115/scratch/b2v/data/distributed.binary", "w");
    assert(f != NULL);
    using tuple_type = typename decltype(local_c)::value_type;
    fwrite(&local_c[0], sizeof(tuple_type), local_c.size(), f);
    fclose(f);

    f = fopen("/gpfs/alpine/bif115/scratch/b2v/data/sequential.binary", "w");
    assert(f != NULL);
    using tuple_type = typename decltype(local_c)::value_type;
    fwrite(&s_c_coo[0], sizeof(tuple_type), s_c_coo.size(), f);
    fclose(f);
    */

    if (s_c_coo.size() != local_c.size()) {
      fprintf(stderr, "ERROR: number of nonzeros does not match.\n");
    } else {
      fprintf(stderr, "Nonzeros match %lu == %lu\n", s_c_coo.size(), local_c.size());
    }

/*
    using coord_type = std::pair<index_type, index_type>;
    std::unordered_map<coord_type, T, PairHash<index_type, index_type>> serial_set;
    std::unordered_map<coord_type, T, PairHash<index_type, index_type>> distr_set;

    fprintf(stderr, "Building serial set.\n");
    auto begin = std::chrono::high_resolution_clock::now();
    for (const auto& nz : s_c_coo) {
      auto idx = std::get<0>(nz);
      auto val = std::get<1>(nz);
      serial_set[idx] = val;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    fprintf(stderr, "Took %lf s\n", duration);

    fprintf(stderr, "Building distributed set.\n");
    begin = std::chrono::high_resolution_clock::now();
    for (const auto& nz : local_c) {
      auto idx = std::get<0>(nz);
      auto val = std::get<1>(nz);
      distr_set[idx] = val;
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();
    fprintf(stderr, "Took %lf s\n", duration);

    fprintf(stderr, "Looking through serial set to see if distributed set matches.\n");
    T eps = 1.0e-5;
    for (const auto& nz : s_c_coo) {
      auto idx = std::get<0>(nz);
      auto val = std::get<1>(nz);

      if (distr_set.find(idx) == distr_set.end()) {
        fprintf(stderr, "Serial result contains (%lu, %lu, %f) not in distributed result.\n",
                idx.first, idx.second, val);
      } else if (std::abs(val - distr_set[idx]) > eps) {
        fprintf(stderr, "Distributed result (%lu, %lu, %f) != serial result (%lu, %lu, %f)\n",
                idx.first, idx.second, distr_set[idx],
                idx.first, idx.second, val);
      }
    }

    fprintf(stderr, "Looking through distributed set to see if distributed set matches.\n");
    for (const auto& nz : local_c) {
      auto idx = std::get<0>(nz);
      auto val = std::get<1>(nz);

      if (serial_set.find(idx) == serial_set.end()) {
        fprintf(stderr, "Distributed result contains (%lu, %lu, %f) not in serial result.\n",
                idx.first, idx.second, val);
      }
    }

*/
    T eps = 1.0e-5;
    for (size_t i = 0; i < s_c_coo.size(); i++) {
      auto idx_a = std::get<0>(s_c_coo[i]);
      auto idx_b = std::get<0>(local_c[i]);

      auto val_a = std::get<1>(s_c_coo[i]);
      auto val_b = std::get<1>(local_c[i]);

      assert(idx_a == idx_b);
      if (std::abs((val_a - val_b)/val_b) >= eps) {
        fprintf(stderr, "(%lu, %lu) == (%lu, %lu)\n", idx_a.first, idx_a.second,
                                                      idx_b.first, idx_b.second);
        fprintf(stderr, "%f ~= %f\n", val_a, val_b);
        fflush(stderr);
      }
      assert(std::abs((val_a - val_b)/val_b) < eps);
      // printf("(%lu, %lu) == (%lu, %lu)\n", idx_a.first, idx_a.second,
      //                                    idx_b.first, idx_b.second);
      // printf("%f ~= %f\n", val_a, val_b);
    }
    printf("OK!\n");
  }

  BCL::finalize();
  return 0;
}
