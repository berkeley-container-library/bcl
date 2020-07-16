#pragma once

#include <bcl/bcl.hpp>
#include "nsparse_util.hpp"
#include "cusparse_util.hpp"
#include "grb_util.hpp"

namespace BCL {

namespace cuda {

double duration_issue;
double duration_sync;
double duration_compute;
double duration_accumulate;
double duration_barrier;

template <typename T, typename index_type>
bool is_sorted(graphblas::Matrix<T>* a);

template <typename T, typename index_type>
size_t print_matrix(const std::string& fname, graphblas::Matrix<T>* a);

template <typename T, typename index_type>
void print_mtx(const std::string& fname, graphblas::Matrix<T>* a);

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::cuda_allocator<T>>
void gemm(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
          BCL::cuda::SPMatrix<T, index_type>& c) {
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k_offset = i + j;

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile({k_offset % a.grid_shape()[1], j});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();

        using csr_type = decltype(buf_a.get());
        std::vector<csr_type> intermediate_results;

        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          begin = std::chrono::high_resolution_clock::now();
          auto local_a = buf_a.get();
          auto local_b = buf_b.get();
          end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          if (k_+1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile({(k+1) % a.grid_shape()[1], j});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          fprintf(stderr, "RANK(%lu) multiplying...\n", BCL::rank());
          auto result_c = cusparse_spgemm<T, index_type, Allocator>(local_a, local_b);
          if (is_shared_seg(result_c)) {
            fprintf(stderr, "RANK(%lu) result of cusparse SPGEMM is in shared seg.\n", BCL::rank());
          } else {
            fprintf(stderr, "RANK(%lu) result of cusparse SPGEMM is not in shared seg.\n", BCL::rank());
          }
          fprintf(stderr, "RANK(%lu) done multiplying.\n", BCL::rank());
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();

          if (!result_c.empty()) {
            auto begin = std::chrono::high_resolution_clock::now();
            fprintf(stderr, "RANK(%lu) pushing...\n", BCL::rank());
            intermediate_results.push_back(std::move(result_c));
            // TODO: implement sum_tiles_grb
            fprintf(stderr, "RANK(%lu) adding...\n", BCL::rank());
            auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
            fprintf(stderr, "RANK(%lu) done adding.\n", BCL::rank());
            intermediate_results.clear();
            fprintf(stderr, "RANK(%lu) pushing new block...\n", BCL::rank());
            intermediate_results.push_back(std::move(c_block));
            fprintf(stderr, "RANK(%lu) done pushing.\n", BCL::rank());
            auto end = std::chrono::high_resolution_clock::now();
            duration_accumulate += std::chrono::duration<double>(end - begin).count();
          }
        }
        // TODO: also add C block to this.
        begin = std::chrono::high_resolution_clock::now();

        auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
        if (!c_block.empty()) {
          c.assign_tile({i, j}, c_block);
        }
        end = std::chrono::high_resolution_clock::now();
        duration_accumulate += std::chrono::duration<double>(end - begin).count();
      }
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
  c.rebroadcast_tiles();
  auto end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

template <typename T, typename index_type>
size_t num_flops(graphblas::Matrix<T>* a, graphblas::Matrix<T>* b);

template <typename T, typename index_type>
size_t num_flops(graphblas::Matrix<T>* a, graphblas::Matrix<T>* b) {
  T* d_a_vals = a->matrix_.sparse_.d_csrVal_;
  index_type* d_a_row_ptr = a->matrix_.sparse_.d_csrRowPtr_;
  index_type* d_a_col_ind = a->matrix_.sparse_.d_csrColInd_;
  size_t a_nnz = a->matrix_.sparse_.nvals_;
  size_t a_m = a->matrix_.sparse_.nrows_;
  size_t a_n = a->matrix_.sparse_.ncols_;

  T* d_b_vals = b->matrix_.sparse_.d_csrVal_;
  index_type* d_b_row_ptr = b->matrix_.sparse_.d_csrRowPtr_;
  index_type* d_b_col_ind = b->matrix_.sparse_.d_csrColInd_;
  size_t b_nnz = b->matrix_.sparse_.nvals_;
  size_t b_m = b->matrix_.sparse_.nrows_;
  size_t b_n = b->matrix_.sparse_.ncols_;

  std::vector<T> a_vals(a_nnz);
  std::vector<index_type> a_col_ind(a_nnz);
  std::vector<index_type> a_row_ptr(a_m+1);

  std::vector<T> b_vals(b_nnz);
  std::vector<index_type> b_col_ind(b_nnz);
  std::vector<index_type> b_row_ptr(b_m+1);

  cudaMemcpy(a_vals.data(), d_a_vals, a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(b_vals.data(), d_b_vals, b_vals.size() * sizeof(T), cudaMemcpyDefault);

  cudaMemcpy(a_col_ind.data(), d_a_col_ind, a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(b_col_ind.data(), d_b_col_ind, b_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);

  cudaMemcpy(a_row_ptr.data(), d_a_row_ptr, a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(b_row_ptr.data(), d_b_row_ptr, b_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  std::vector<size_t> a_col_counts(a_n, 0);
  std::vector<size_t> b_row_counts(b_m, 0);

  assert(a_col_counts.size() == b_row_counts.size());

  for (index_type i = 0; i < a_m; i++) {
    for (size_t j_ptr = a_row_ptr[i]; j_ptr < a_row_ptr[i+1]; j_ptr++) {
      T val = a_vals[j_ptr];
      index_type j = a_col_ind[j_ptr];
      a_col_counts[j]++;
    }
  }

  for (index_type i = 0; i < b_m; i++) {
    for (size_t j_ptr = b_row_ptr[i]; j_ptr < b_row_ptr[i+1]; j_ptr++) {
      T val = b_vals[j_ptr];
      index_type j = b_col_ind[j_ptr];
      b_row_counts[i]++;
    }
  }

  size_t nflops = 0;
  for (size_t i = 0; i < a_col_counts.size(); i++) {
    nflops += (a_col_counts[i] * b_row_counts[i]);
  }
  return nflops;
}

template <typename T, typename index_type>
bool is_sorted(graphblas::Matrix<T>* a) {
  T* d_a_vals = a->matrix_.sparse_.d_csrVal_;
  index_type* d_a_row_ptr = a->matrix_.sparse_.d_csrRowPtr_;
  index_type* d_a_col_ind = a->matrix_.sparse_.d_csrColInd_;
  size_t a_nnz = a->matrix_.sparse_.nvals_;
  size_t a_m = a->matrix_.sparse_.nrows_;
  size_t a_n = a->matrix_.sparse_.ncols_;

  std::vector<T> a_vals(a_nnz);
  std::vector<index_type> a_col_ind(a_nnz);
  std::vector<index_type> a_row_ptr(a_m+1);

  cudaMemcpy(a_vals.data(), d_a_vals, a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(a_col_ind.data(), d_a_col_ind, a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(a_row_ptr.data(), d_a_row_ptr, a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  for (index_type i = 0; i < a_m; i++) {
    size_t col_num = 0;
    for (size_t j_ptr = a_row_ptr[i]; j_ptr < a_row_ptr[i+1]; j_ptr++) {
      T val = a_vals[j_ptr];
      index_type j = a_col_ind[j_ptr];
      if (col_num != 0) {
        if (a_col_ind[j_ptr] <= a_col_ind[j_ptr-1]) {
          return false;
        }
      }
      col_num++;
    }
  }

  return true;
}

template <typename T, typename index_type>
void print_mtx(const std::string& fname, graphblas::Matrix<T>* a) {
  FILE* f = fopen(fname.c_str(), "w");
  assert(f != nullptr);
  T* d_a_vals = a->matrix_.sparse_.d_csrVal_;
  index_type* d_a_row_ptr = a->matrix_.sparse_.d_csrRowPtr_;
  index_type* d_a_col_ind = a->matrix_.sparse_.d_csrColInd_;
  size_t a_nnz = a->matrix_.sparse_.nvals_;
  size_t a_m = a->matrix_.sparse_.nrows_;
  size_t a_n = a->matrix_.sparse_.ncols_;

  std::vector<T> a_vals(a_nnz);
  std::vector<index_type> a_col_ind(a_nnz);
  std::vector<index_type> a_row_ptr(a_m+1);

  cudaMemcpy(a_vals.data(), d_a_vals, a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(a_col_ind.data(), d_a_col_ind, a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(a_row_ptr.data(), d_a_row_ptr, a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  size_t counted_nnz = 0;

  fprintf(f, "%lu %lu %lu\n", a_m, a_n, a_nnz);

  for (index_type i = 0; i < a_m; i++) {
    size_t col_num = 0;
    for (size_t j_ptr = a_row_ptr[i]; j_ptr < a_row_ptr[i+1]; j_ptr++) {
      T val = a_vals[j_ptr];
      index_type j = a_col_ind[j_ptr];
      fprintf(f, "%d %d %lf", i+1, j+1, val);
      counted_nnz++;
      if (col_num != 0) {
        if (a_col_ind[j_ptr] <= a_col_ind[j_ptr-1]) {
          fprintf(f, " [unsorted]");
        }
      }
      fprintf(f, "\n");
      col_num++;
    }
  }

  printf("Printing matrix to %s (%lu x %lu), %lu nnz (counted %lu nnz)\n",
         fname.c_str(), a_m, a_n, a_nnz, counted_nnz);
  fclose(f);
}

template <typename T, typename index_type>
size_t print_matrix(const std::string& fname, graphblas::Matrix<T>* a) {
  FILE* f = fopen(fname.c_str(), "w");
  assert(f != nullptr);
  T* d_a_vals = a->matrix_.sparse_.d_csrVal_;
  index_type* d_a_row_ptr = a->matrix_.sparse_.d_csrRowPtr_;
  index_type* d_a_col_ind = a->matrix_.sparse_.d_csrColInd_;
  size_t a_nnz = a->matrix_.sparse_.nvals_;
  size_t a_m = a->matrix_.sparse_.nrows_;
  size_t a_n = a->matrix_.sparse_.ncols_;

  std::vector<T> a_vals(a_nnz);
  std::vector<index_type> a_col_ind(a_nnz);
  std::vector<index_type> a_row_ptr(a_m+1);

  cudaMemcpy(a_vals.data(), d_a_vals, a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(a_col_ind.data(), d_a_col_ind, a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(a_row_ptr.data(), d_a_row_ptr, a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  for (index_type i = 0; i < a_m; i++) {
    size_t col_num = 0;
    fprintf(f, "Row(%d): ", i);
    for (size_t j_ptr = a_row_ptr[i]; j_ptr < a_row_ptr[i+1]; j_ptr++) {
      T val = a_vals[j_ptr];
      index_type j = a_col_ind[j_ptr];
      fprintf(f, " (%d, %lf)", j, val);
      if (col_num != 0) {
        if (a_col_ind[j_ptr] <= a_col_ind[j_ptr-1]) {
          fprintf(f, "[unsorted]");
        }
      }
      col_num++;
    }
    fprintf(f, "\n");
  }
  fclose(f);

  return true;
}

} // end cuda

} // end BCL
