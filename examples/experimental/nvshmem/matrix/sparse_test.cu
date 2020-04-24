

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/CudaMatrix.hpp>
#include <bcl/containers/experimental/cuda/launch_kernel.cuh>

#include <bcl/containers/experimental/cuda/CudaSPMatrix.hpp>

#include "cusparse_util.hpp"
#include "spgemm.hpp"

#include <chrono>
#include <essl.h>

template <typename T>
graphblas::Matrix<T> read_matrix(const std::string& fname) {
  std::vector<graphblas::Index> row_indices, col_indices;
  std::vector<T> values;
  graphblas::Index num_rows, num_cols, num_edges;
  char* dat_name;
  readMtx(fname.c_str(), &row_indices, &col_indices, &values, &num_rows, &num_cols,
          &num_edges, 0, false, &dat_name);
  graphblas::Matrix<T> a(num_rows, num_cols);
  a.build(&row_indices, &col_indices, &values, num_edges, GrB_NULL, NULL);
  return a;
}

int main(int argc, char** argv) {
  BCL::init(16);
  BCL::cuda::init(4000);

  using index_type = graphblas::Index;

  using T = float;

  bool verify_result = true;

  std::string fname = std::string(argv[1]);

  auto matrix_shape = BCL::matrix_io::matrix_info(fname);
  size_t m = matrix_shape.shape[0];
  size_t n = matrix_shape.shape[1];
  assert(m == n);
  size_t k = m;

  BCL::print("Choosing blocks...\n");
  auto blocks = BCL::block_matmul(m, n, k);

  BCL::print("Reading matrices...\n");
  BCL::cuda::SPMatrix<T, graphblas::Index> a(fname, std::move(blocks[0]));
  BCL::cuda::SPMatrix<T, graphblas::Index> b(fname, std::move(blocks[1]));
  BCL::cuda::SPMatrix<T, graphblas::Index> c(m, n, std::move(blocks[2]));

  BCL::print("Info:\n");
  if (BCL::rank() == 0) {
    printf("A:\n");
    a.print_info();
    printf("B:\n");
    b.print_info();
    printf("C:\n");
    c.print_info();
  }

  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  graphblas::Descriptor desc;
  desc.descriptor_.debug_ = false;

  BCL::cuda::duration_issue = 0;
  BCL::cuda::duration_sync = 0;
  BCL::cuda::duration_compute = 0;
  BCL::cuda::duration_accumulate = 0;
  BCL::cuda::duration_barrier = 0;

  BCL::print("Beginning SpGEMM...\n");

  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::gemm(a, b, c, desc);
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  BCL::cuda::duration_issue = BCL::allreduce(BCL::cuda::duration_issue, std::plus<double>{});
  BCL::cuda::duration_sync = BCL::allreduce(BCL::cuda::duration_sync, std::plus<double>{});
  BCL::cuda::duration_compute = BCL::allreduce(BCL::cuda::duration_compute, std::plus<double>{});
  BCL::cuda::duration_accumulate = BCL::allreduce(BCL::cuda::duration_accumulate, std::plus<double>{});
  BCL::cuda::duration_barrier = BCL::allreduce(BCL::cuda::duration_barrier, std::plus<double>{});

  if (BCL::rank() == 0) {
    printf("duration_issue %lf\n", BCL::cuda::duration_issue / BCL::nprocs());
    printf("duration_sync %lf\n", BCL::cuda::duration_sync / BCL::nprocs());
    printf("duration_compute %lf\n", BCL::cuda::duration_compute / BCL::nprocs());
    printf("duration_accumulate %lf\n", BCL::cuda::duration_accumulate / BCL::nprocs());
    printf("duration_barrier %lf\n", BCL::cuda::duration_barrier / BCL::nprocs());
  }

  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  BCL::print("Matrix multiply finished in %lf s\n", duration);

  if (BCL::rank() == 0 && verify_result) {
    BCL::CSRMatrix<T, graphblas::Index> mat(fname);

    T* d_a_vals, *d_b_vals;
    graphblas::Index* d_a_rowptr, *d_b_rowptr;
    graphblas::Index* d_a_colind, *d_b_colind;
    cudaMalloc((void**) &d_a_vals, sizeof(T)*mat.vals_.size());
    cudaMalloc((void**) &d_a_rowptr, sizeof(graphblas::Index)*mat.row_ptr_.size());
    cudaMalloc((void**) &d_a_colind, sizeof(graphblas::Index)*mat.col_ind_.size());
    cudaMalloc((void**) &d_b_vals, sizeof(T)*mat.vals_.size());
    cudaMalloc((void**) &d_b_rowptr, sizeof(graphblas::Index)*mat.row_ptr_.size());
    cudaMalloc((void**) &d_b_colind, sizeof(graphblas::Index)*mat.col_ind_.size());

    if (d_a_vals == nullptr || d_a_rowptr == nullptr || d_b_vals == nullptr ||
        d_b_rowptr == nullptr || d_a_colind == nullptr || d_b_colind == nullptr) {
      throw std::runtime_error("Ran out of memory verifying results.");
    }

    cudaMemcpy(d_a_vals, mat.vals_.data(), sizeof(T)*mat.vals_.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_rowptr, mat.row_ptr_.data(), sizeof(T)*mat.row_ptr_.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_colind, mat.col_ind_.data(), sizeof(T)*mat.col_ind_.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_vals, mat.vals_.data(), sizeof(T)*mat.vals_.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_rowptr, mat.row_ptr_.data(), sizeof(T)*mat.row_ptr_.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_colind, mat.col_ind_.data(), sizeof(T)*mat.col_ind_.size(), cudaMemcpyHostToDevice);

    graphblas::Matrix<T> s_a(mat.m_, mat.n_);
    graphblas::Matrix<T> s_b(mat.m_, mat.n_);

    s_a.build(d_a_rowptr, d_a_colind, d_a_vals, mat.vals_.size());
    s_b.build(d_b_rowptr, d_b_colind, d_b_vals, mat.vals_.size());

    graphblas::Matrix<T> s_c(m, n);

    fprintf(stderr, "Multiplying matrix...\n");
    graphblas::mxm<T, T, T, T>(&s_c, GrB_NULL,
                               GrB_NULL, graphblas::PlusMultipliesSemiring<T>(),
                               &s_a, &s_b, &desc);

    fprintf(stderr, "Getting COO...\n");
    auto local_c = c.get().get_coo();

    fprintf(stderr, "Extracting tuples...\n");
    index_type nnz;
    s_c.nvals(&nnz);
    std::vector<index_type> row_idx(nnz);
    std::vector<index_type> col_idx(nnz);
    std::vector<T> vals(nnz);

    s_c.extractTuples(&row_idx, &col_idx, &vals, &nnz);

    fprintf(stderr, "Extracting COO from GraphBLAS\n");

    auto s_c_coo = BCL::cuda::get_coo(vals, row_idx, col_idx);

    fprintf(stderr, "s_c_coo.size (%lu), local_c.size (%lu)\n", s_c_coo.size(), local_c.size());

    assert(s_c_coo.size() == local_c.size());

    T eps = 1.0e-5;
    for (size_t i = 0; i < s_c_coo.size(); i++) {
      auto idx_a = std::get<0>(s_c_coo[i]);
      auto idx_b = std::get<0>(local_c[i]);

      auto val_a = std::get<1>(s_c_coo[i]);
      auto val_b = std::get<1>(local_c[i]);

      assert(idx_a == idx_b);
      assert(std::abs(val_a - val_b) < eps);
      // printf("(%lu, %lu) == (%lu, %lu)\n", idx_a.first, idx_a.second,
      //                                    idx_b.first, idx_b.second);
      // printf("%f ~= %f\n", val_a, val_b);
    }
    printf("OK!\n");
  }

  BCL::finalize();
  return 0;
}
