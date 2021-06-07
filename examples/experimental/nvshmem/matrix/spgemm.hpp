#pragma once

#include <bcl/bcl.hpp>
#include "cusparse_util.hpp"
#include "grb_util.hpp"
#include "cusp_util.hpp"

#include <bcl/containers/CircularQueue.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>

namespace BCL {

namespace cuda {

double duration_issue;
double duration_sync;
double duration_compute;
double duration_accumulate;
double duration_barrier;
double duration_issue_reduction;
double duration_sync_reduction;

template <typename T>
struct CudaMatrix_ptr {
  BCL::cuda::ptr<T> values;
  size_t m;
  size_t n;
  size_t ld;
  size_t i;
  size_t j;
};

template <typename T, typename I>
struct CudaSparse_ptr {
  BCL::cuda::ptr<T> values;
  BCL::cuda::ptr<I> rowptr;
  BCL::cuda::ptr<I> colind;
  size_t m;
  size_t n;
  size_t nnz;
  size_t i;
  size_t j;
};

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::cuda_allocator<T>>
void gemm(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
          BCL::cuda::SPMatrix<T, index_type>& c) {
  using timer_type = decltype(std::chrono::high_resolution_clock::now());
  timer_type last_request;
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k_offset = i + j;

        auto begin = std::chrono::high_resolution_clock::now();
        last_request = begin;
        auto buf_a = a.arget_tile_exp({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile_exp({k_offset % a.grid_shape()[1], j});
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
          double comm_time = std::chrono::duration<double>(end - last_request).count();

          if (k_+1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            last_request = begin;
            buf_a = a.arget_tile_exp({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile_exp({(k+1) % a.grid_shape()[1], j});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();

          auto result_c = spgemm_cusparse(local_a, local_b);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();

          if (!result_c.empty()) {
            auto begin = std::chrono::high_resolution_clock::now();
            intermediate_results.push_back(std::move(result_c));
            // TODO: implement sum_tiles_grb
            auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
            intermediate_results.clear();
            intermediate_results.push_back(std::move(c_block));
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

template <typename AMatrixType, typename BMatrixType, typename CMatrixType>
void warmup_communicators(AMatrixType& a,
                          BMatrixType& b,
                          CMatrixType& c) {
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k = 0;
        MPI_Comm row_comm = a.row_teams_mpi_[i][0].comm();
        MPI_Comm column_comm = b.column_teams_mpi_[j][0].comm();
        BCL::cuda::ptr<int> data = BCL::cuda::alloc<int>(100);
        MPI_Bcast(data.local(), 100, MPI_INT, k, row_comm);
        MPI_Bcast(data.local(), 100, MPI_INT, k, column_comm);
        BCL::cuda::dealloc(data);
      }
    }
  }
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_mpi_simple(BCL::cuda::SPMatrix<T, index_type>& a,
                     BCL::cuda::SPMatrix<T, index_type>& b,
                     BCL::cuda::SPMatrix<T, index_type>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        using csr_type = decltype(a.arget_tile_exp({0, 0}).get());
        std::vector<csr_type> intermediate_results;
        // fprintf(stderr, "RANK(%lu): Doing tile (%lu, %lu)\n", BCL::rank(), i, j);
        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          auto begin = std::chrono::high_resolution_clock::now();
          MPI_Comm row_comm = a.row_teams_mpi_[i][0].comm();
          MPI_Comm column_comm = b.column_teams_mpi_[j][0].comm();

          T* a_values_data;
          index_type* a_rowptr_data;
          index_type* a_colind_data;

          size_t nnz = a.nnzs_[i*a.grid_shape()[1] + k];
          size_t m = a.tile_shape({i, k})[0];
          size_t n = a.tile_shape({i, k})[1];

          if (BCL::rank() == a.tile_rank({i, k})) {
            auto local_a = a.get_local_tile({i, k});
            a_values_data = local_a.values_data();
            a_rowptr_data = local_a.rowptr_data();
            a_colind_data = local_a.colind_data();
          } else {
            a_values_data = allocate_with<T, Allocator>(nnz);
            a_rowptr_data = allocate_with<index_type, Allocator>(m+1);
            a_colind_data = allocate_with<index_type, Allocator>(nnz);
          }

          size_t root = a.row_teams_[i].resolve(a.tile_rank({i, k}));
          assert(root == k);

          MPI_Bcast(a_values_data, nnz, MPI_FLOAT, root, row_comm);
          MPI_Bcast(a_rowptr_data, m+1, MPI_INT, root, row_comm);
          MPI_Bcast(a_colind_data, nnz, MPI_INT, root, row_comm);

          BCL::cuda::CudaCSRMatrixView<T, index_type> local_a(m, n, nnz,
                                                              a_values_data,
                                                              a_rowptr_data,
                                                              a_colind_data);

          // *** BCAST of B ***

          T* b_values_data;
          index_type* b_rowptr_data;
          index_type* b_colind_data;

          nnz = b.nnzs_[k*b.grid_shape()[1] + j];
          m = b.tile_shape({k, j})[0];
          n = b.tile_shape({k, j})[1];

          if (BCL::rank() == b.tile_rank({k, j})) {
            auto local_b = b.get_local_tile({k, j});
            b_values_data = local_b.values_data();
            b_rowptr_data = local_b.rowptr_data();
            b_colind_data = local_b.colind_data();
          } else {
            b_values_data = allocate_with<T, Allocator>(nnz);
            b_rowptr_data = allocate_with<index_type, Allocator>(m+1);
            b_colind_data = allocate_with<index_type, Allocator>(nnz);
          }

          root = b.column_teams_[j].resolve(b.tile_rank({k, j}));
          assert(root == k);

          MPI_Bcast(b_values_data, nnz, MPI_FLOAT, root, column_comm);
          MPI_Bcast(b_rowptr_data, m+1, MPI_INT, root, column_comm);
          MPI_Bcast(b_colind_data, nnz, MPI_INT, root, column_comm);

          BCL::cuda::CudaCSRMatrixView<T, index_type> local_b(m, n, nnz,
                                                              b_values_data,
                                                              b_rowptr_data,
                                                              b_colind_data);

          auto end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          // *** Local MatMul ***

          begin = std::chrono::high_resolution_clock::now();
          auto result_c = spgemm_cusparse(local_a, local_b);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();

          if (BCL::rank() != a.tile_rank({i, k})) {
            deallocate_with<T, Allocator>(a_values_data);
            deallocate_with<index_type, Allocator>(a_rowptr_data);
            deallocate_with<index_type, Allocator>(a_colind_data);
          }

          if (BCL::rank() != b.tile_rank({k, j})) {
            deallocate_with<T, Allocator>(b_values_data);
            deallocate_with<index_type, Allocator>(b_rowptr_data);
            deallocate_with<index_type, Allocator>(b_colind_data);
          }

          if (!result_c.empty()) {
            auto begin = std::chrono::high_resolution_clock::now();
            intermediate_results.push_back(std::move(result_c));
            // TODO: implement sum_tiles_grb
            auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
            intermediate_results.clear();
            intermediate_results.push_back(std::move(c_block));
            auto end = std::chrono::high_resolution_clock::now();
            duration_accumulate += std::chrono::duration<double>(end - begin).count();
          }
        }

        auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
        if (!c_block.empty()) {
          c.assign_tile({i, j}, c_block);
        }
      }
    }
  }
  auto begin = std::chrono::high_resolution_clock::now();
  c.rebroadcast_tiles();
  auto end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

void check_requests(std::vector<MPI_Request>& requests) {
  for (size_t i = 0; i < requests.size(); ) {
    int flag;
    MPI_Test(&requests[i], &flag, MPI_STATUS_IGNORE);
    if (flag) {
      requests.erase(requests.begin()+i);
    } else {
      i++;
    }
  }
}


/*
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
*/

template <typename T, typename index_type, typename Allocator>
bool is_positive_colind(CudaCSRMatrix<T, index_type, Allocator>& a) {
  std::vector<T> a_vals(a.nnz());
  std::vector<index_type> a_col_ind(a.nnz());
  std::vector<index_type> a_row_ptr(a.m()+1);

  cudaMemcpy(a_vals.data(), a.values_data(), a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(a_col_ind.data(), a.colind_data(), a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(a_row_ptr.data(), a.rowptr_data(), a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  for (index_type i = 0; i < a.m(); i++) {
    for (size_t j_ptr = a_row_ptr[i]; j_ptr < a_row_ptr[i+1]; j_ptr++) {
      T val = a_vals[j_ptr];
      index_type j = a_col_ind[j_ptr];
      if (j < 0) {
        return false;
      }
    }
  }

  return true;
}

template <typename T, typename index_type, typename Allocator>
bool is_sorted(CudaCSRMatrix<T, index_type, Allocator>& a) {
  std::vector<T> a_vals(a.nnz());
  std::vector<index_type> a_col_ind(a.nnz());
  std::vector<index_type> a_row_ptr(a.m()+1);

  cudaMemcpy(a_vals.data(), a.values_data(), a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(a_col_ind.data(), a.colind_data(), a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(a_row_ptr.data(), a.rowptr_data(), a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  for (index_type i = 0; i < a.m(); i++) {
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

template <typename T, typename index_type, typename Allocator>
void print_mtx(const std::string& fname, CudaCSRMatrix<T, index_type, Allocator>& a) {
  FILE* f = fopen(fname.c_str(), "w");
  assert(f != nullptr);
  T* d_a_vals = a.values_data();
  index_type* d_a_row_ptr = a.rowptr_data();
  index_type* d_a_col_ind = a.colind_data();
  size_t a_nnz = a.nnz();
  size_t a_m = a.m();
  size_t a_n = a.n();

  std::vector<T> a_vals(a_nnz);
  std::vector<index_type> a_col_ind(a_nnz);
  std::vector<index_type> a_row_ptr(a_m+1);

  cudaMemcpy(a_vals.data(), d_a_vals, a_vals.size() * sizeof(T), cudaMemcpyDefault);
  cudaMemcpy(a_col_ind.data(), d_a_col_ind, a_col_ind.size() * sizeof(index_type), cudaMemcpyDefault);
  cudaMemcpy(a_row_ptr.data(), d_a_row_ptr, a_row_ptr.size() * sizeof(index_type), cudaMemcpyDefault);

  size_t counted_nnz = 0;

  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%%\n");
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

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_cusp(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
               BCL::cuda::SPMatrix<T, index_type>& c) {
  using timer_type = decltype(std::chrono::high_resolution_clock::now());
  timer_type last_request;
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k_offset = i + j;

        cusp::csr_matrix<index_type, T, cusp::device_memory>
        result_c(c.tile_shape({i, j})[0], c.tile_shape({i, j}), 0);

        auto begin = std::chrono::high_resolution_clock::now();
        last_request = begin;
        auto buf_a = a.arget_tile_exp({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile_exp({k_offset % a.grid_shape()[1], j});
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
          double comm_time = std::chrono::duration<double>(end - last_request).count();
          double bandwidth = comm_time;

          if (k_+1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            last_request = begin;
            buf_a = a.arget_tile_exp({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile_exp({(k+1) % a.grid_shape()[1], j});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          // auto result_c = spgemm_cusparse<T, index_type, Allocator>(local_a, local_b);
          // TODO: is in-place accumulation during SPGEMM the best option?
          spgemm_cusp(local_a, local_b, result_c);

          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
        }
        // TODO: also add C block to this.
        begin = std::chrono::high_resolution_clock::now();

        auto c_block = get_view<T, index_type>(result_c);

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


template <typename T, typename index_type,
          typename Allocator = BCL::cuda::cuda_allocator<T>>
void gemm_aowns_simple(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
                       BCL::cuda::SPMatrix<T, index_type>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  using csr_type = BCL::cuda::CudaCSRMatrix<T, index_type, Allocator>;

  std::list<csr_type> result_cs;

  using queue_type = BCL::ChecksumQueue<CudaSparse_ptr<T, index_type>, BCL::djb2_hash<CudaSparse_ptr<T, index_type>>>;
  std::vector<queue_type> queues;

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    queues.emplace_back(queue_type(i, a.grid_shape()[1]+8));
  }

  for (size_t i = 0; i < a.grid_shape()[0]; i++) {
    for (size_t k = 0; k < a.grid_shape()[1]; k++) {
      if (a.tile_rank({i, k}) == BCL::rank()) {
        auto local_a = a.get_local_tile({i, k});
        for (size_t j = 0; j < c.grid_shape()[1]; j++) {
          auto local_b = b.arget_tile_exp({k, j}).get();

          auto result_c = spgemm_cusparse(local_a, local_b);

          queue_type& queue = queues[c.tile_rank({i, j})];
          queue.push({BCL::cuda::__to_cuda_gptr<T>(result_c.values_data()),
                      BCL::cuda::__to_cuda_gptr<index_type>(result_c.rowptr_data()),
                      BCL::cuda::__to_cuda_gptr<index_type>(result_c.colind_data()),
                      result_c.shape()[0], result_c.shape()[1],
                      result_c.nnz(), i, j});

          result_cs.push_back(std::move(result_c));
        }
      }
    }
  }

  std::vector<csr_type> intermediate_results;
  queue_type& my_queue = queues[BCL::rank()];
  size_t i_, j_;
  bool coord_set = false;
  if (c.my_num_tiles() > 0) {
    for (size_t i = 0; i < a.grid_shape()[1]; i++) {
      CudaSparse_ptr<T, index_type> ptr;
      while (!my_queue.pop(ptr)) {}
      if (!coord_set) {
        i_ = ptr.i;
        j_ = ptr.j;
      } else {
        assert(i_ == ptr.i && j_ == ptr.j);
      }
      BCL::cuda::CudaCSRMatrix<T, index_type, Allocator> x(ptr.m, ptr.n, ptr.nnz);
      BCL::cuda::memcpy(x.values_data(), ptr.values, sizeof(T)*x.nnz());
      BCL::cuda::memcpy(x.rowptr_data(), ptr.rowptr, sizeof(index_type)*(x.m()+1));
      BCL::cuda::memcpy(x.colind_data(), ptr.colind, sizeof(index_type)*x.nnz());
      intermediate_results.push_back(std::move(x));

      if (intermediate_results.size() >= 2) {
        auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
        intermediate_results.clear();
        intermediate_results.push_back(std::move(c_block));
      }
    }
  }

  auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
  if (!c_block.empty()) {
    c.assign_tile({i_, j_}, c_block);
  }

  c.rebroadcast_tiles();
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_aowns(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
                BCL::cuda::SPMatrix<T, index_type>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  using csr_type = BCL::cuda::CudaCSRMatrix<T, index_type, Allocator>;

  std::list<csr_type> result_cs;

  using queue_type = BCL::ChecksumQueue<CudaSparse_ptr<T, index_type>, BCL::djb2_hash<CudaSparse_ptr<T, index_type>>>;
  std::vector<queue_type> queues;

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    queues.emplace_back(queue_type(i, a.grid_shape()[1]+8));
  }

  for (size_t i = 0; i < a.grid_shape()[0]; i++) {
    for (size_t k = 0; k < a.grid_shape()[1]; k++) {
      if (a.tile_rank({i, k}) == BCL::rank()) {

        auto local_a = a.get_local_tile({i, k});
        size_t j_offset = i + k;

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_b = b.arget_tile_exp({k, j_offset % b.grid_shape()[1]});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();
        for (size_t j_ = 0; j_ < c.grid_shape()[1]; j_++) {
          size_t j = (j_ + j_offset) % b.grid_shape()[1];

          auto begin = std::chrono::high_resolution_clock::now();
          auto local_b = buf_b.get();
          auto end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          if (j_ + 1 < b.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            buf_b = b.arget_tile_exp({k, (j+1) % b.grid_shape()[1]});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          auto result_c = spgemm_cusparse(local_a, local_b);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();

          queue_type& queue = queues[c.tile_rank({i, j})];
          queue.push({BCL::cuda::__to_cuda_gptr<T>(result_c.values_data()),
                      BCL::cuda::__to_cuda_gptr<index_type>(result_c.rowptr_data()),
                      BCL::cuda::__to_cuda_gptr<index_type>(result_c.colind_data()),
                      result_c.shape()[0], result_c.shape()[1],
                      result_c.nnz(), i, j});

          result_cs.push_back(std::move(result_c));
        }
      }
    }
  }

  auto begin = std::chrono::high_resolution_clock::now();
  std::vector<csr_type> intermediate_results;
  queue_type& my_queue = queues[BCL::rank()];
  size_t i_, j_;
  bool coord_set = false;
  if (c.my_num_tiles() > 0) {
    for (size_t i = 0; i < a.grid_shape()[1]; i++) {
      CudaSparse_ptr<T, index_type> ptr;
      while (!my_queue.pop(ptr)) {}
      if (!coord_set) {
        i_ = ptr.i;
        j_ = ptr.j;
      } else {
        assert(i_ == ptr.i && j_ == ptr.j);
      }
      if (ptr.nnz > 0) {
        BCL::cuda::CudaCSRMatrix<T, index_type, Allocator> x(ptr.m, ptr.n, ptr.nnz);
        BCL::cuda::memcpy(x.values_data(), ptr.values, sizeof(T)*x.nnz());
        BCL::cuda::memcpy(x.rowptr_data(), ptr.rowptr, sizeof(index_type)*(x.m()+1));
        BCL::cuda::memcpy(x.colind_data(), ptr.colind, sizeof(index_type)*x.nnz());
        intermediate_results.push_back(std::move(x));

        if (intermediate_results.size() >= 2) {
          auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
          intermediate_results.clear();
          intermediate_results.push_back(std::move(c_block));
        }
      }
    }
  }

  auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
  if (!c_block.empty()) {
    c.assign_tile({i_, j_}, c_block);
  }

  auto end = std::chrono::high_resolution_clock::now();
  duration_accumulate += std::chrono::duration<double>(end - begin).count();

  begin = std::chrono::high_resolution_clock::now();
  c.rebroadcast_tiles();
  end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

template <typename T, typename index_type,
          typename queue_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_aowns_ws(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
                   BCL::cuda::SPMatrix<T, index_type>& c,
                   std::vector<queue_type>& queues,
                   std::vector<BCL::GlobalPtr<int>>& ws_grid) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  size_t k_interval = 1;

  using csr_type = BCL::cuda::CudaCSRMatrix<T, index_type, Allocator>;

  std::list<csr_type> result_cs;

  size_t my_block = 0;

  for (size_t i = 0; i < a.grid_shape()[0]; i++) {
    for (size_t k = 0; k < a.grid_shape()[1]; k++) {
      if (a.tile_rank({i, k}) == BCL::rank()) {
        my_block = i*a.grid_shape()[1] + k;

        int my_tile_prog = 0;
        while (my_tile_prog < b.grid_shape()[1]) {
          size_t idx = i*a.grid_shape()[1] + k;
          BCL::GlobalPtr<int> ptr = ws_grid[idx];
          my_tile_prog = BCL::fetch_and_op<int>(ptr, k_interval, BCL::plus<int>{});
          size_t j_offset = i + k;
          for (size_t j_ = my_tile_prog; j_ < std::min(my_tile_prog + k_interval, b.grid_shape()[1]); j_++) {
            size_t j = (j_ + j_offset) % b.grid_shape()[1];

            auto begin = std::chrono::high_resolution_clock::now();
            auto local_a = a.get_local_tile({i, k});
            auto local_b = b.arget_tile_exp({k, j}).get();
            auto end = std::chrono::high_resolution_clock::now();
            duration_sync += std::chrono::duration<double>(end - begin).count();

            begin = std::chrono::high_resolution_clock::now();
            auto result_c = spgemm_cusparse(local_a, local_b);
            end = std::chrono::high_resolution_clock::now();
            duration_compute += std::chrono::duration<double>(end - begin).count();

            queue_type& queue = queues[c.tile_rank({i, j})];

            queue.push({BCL::cuda::__to_cuda_gptr<T>(result_c.values_data()),
                        BCL::cuda::__to_cuda_gptr<index_type>(result_c.rowptr_data()),
                        BCL::cuda::__to_cuda_gptr<index_type>(result_c.colind_data()),
                        result_c.shape()[0], result_c.shape()[1],
                        result_c.nnz(), i, j});

            result_cs.push_back(std::move(result_c));
          }
          my_tile_prog += k_interval;
        }
      }
    }
  }

  auto probe = [](size_t offset) { return offset*2; };
  size_t max_seeks = a.grid_shape()[0];
  size_t seeks = 0;

  for (size_t idx_ = 1; seeks < max_seeks; idx_++) {
    size_t tile_idx = (my_block + probe(idx_)) % (a.grid_shape()[0]*a.grid_shape()[1]);
    size_t i = tile_idx / a.grid_shape()[1];
    size_t k = tile_idx % a.grid_shape()[1];

    int my_tile_prog = 0;
    while (my_tile_prog < b.grid_shape()[1]) {
      size_t idx = i*a.grid_shape()[1] + k;
      BCL::GlobalPtr<int> ptr = ws_grid[idx];
      my_tile_prog = BCL::fetch_and_op<int>(ptr, k_interval, BCL::plus<int>{});
      size_t j_offset = i + k;

      for (size_t j_ = my_tile_prog; j_ < std::min(my_tile_prog + k_interval, b.grid_shape()[1]); j_++) {
        size_t j = (j_ + j_offset) % b.grid_shape()[1];

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile_exp({i, k});
        auto buf_b = b.arget_tile_exp({k, j});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();

        begin = std::chrono::high_resolution_clock::now();
        auto local_a = buf_a.get();
        auto local_b = buf_b.get();
        end = std::chrono::high_resolution_clock::now();
        duration_sync += std::chrono::duration<double>(end - begin).count();

        begin = std::chrono::high_resolution_clock::now();
        auto result_c = spgemm_cusparse(local_a, local_b);
        end = std::chrono::high_resolution_clock::now();
        duration_compute += std::chrono::duration<double>(end - begin).count();

        queue_type& queue = queues[c.tile_rank({i, j})];

        queue.push({BCL::cuda::__to_cuda_gptr<T>(result_c.values_data()),
                    BCL::cuda::__to_cuda_gptr<index_type>(result_c.rowptr_data()),
                    BCL::cuda::__to_cuda_gptr<index_type>(result_c.colind_data()),
                    result_c.shape()[0], result_c.shape()[1],
                    result_c.nnz(), i, j});

        result_cs.push_back(std::move(result_c));
      }
      my_tile_prog += k_interval;
    }
    seeks++;
  }

  auto begin = std::chrono::high_resolution_clock::now();
  std::vector<csr_type> intermediate_results;
  queue_type& my_queue = queues[BCL::rank()];
  size_t i_, j_;
  bool coord_set = false;
  if (c.my_num_tiles() > 0) {
    for (size_t i = 0; i < a.grid_shape()[1]; i++) {
      CudaSparse_ptr<T, index_type> ptr;
      while (!my_queue.pop(ptr)) {}
      if (!coord_set) {
        i_ = ptr.i;
        j_ = ptr.j;
      } else {
        assert(i_ == ptr.i && j_ == ptr.j);
      }
      if (ptr.nnz > 0) {
        BCL::cuda::CudaCSRMatrix<T, index_type, Allocator> x(ptr.m, ptr.n, ptr.nnz);
        BCL::cuda::memcpy(x.values_data(), ptr.values, sizeof(T)*x.nnz());
        BCL::cuda::memcpy(x.rowptr_data(), ptr.rowptr, sizeof(index_type)*(x.m()+1));
        BCL::cuda::memcpy(x.colind_data(), ptr.colind, sizeof(index_type)*x.nnz());
        intermediate_results.push_back(std::move(x));

        if (intermediate_results.size() >= 2) {
          auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
          intermediate_results.clear();
          intermediate_results.push_back(std::move(c_block));
        }
      }
    }
  }

  auto c_block = sum_tiles_cusparse<T, index_type, Allocator>(intermediate_results);
  if (!c_block.empty()) {
    c.assign_tile({i_, j_}, c_block);
  }

  auto end = std::chrono::high_resolution_clock::now();
  duration_accumulate += std::chrono::duration<double>(end - begin).count();

  begin = std::chrono::high_resolution_clock::now();
  c.rebroadcast_tiles();
  end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

} // end cuda

} // end BCL

#include "spmm.hpp"