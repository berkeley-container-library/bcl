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
double duration_issue_reduction;
double duration_sync_reduction;

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

          if (k_+1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile_exp({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile_exp({(k+1) % a.grid_shape()[1], j});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          auto result_c = spgemm_cusparse<T, index_type, Allocator>(local_a, local_b);
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

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::cuda_allocator<T>>
void single_process_gemm(BCL::cuda::SPMatrix<T, index_type>& a,
                         BCL::cuda::SPMatrix<T, index_type>& b,
                         BCL::cuda::SPMatrix<T, index_type>& c) {
  if (BCL::rank() == 0) {
    for (size_t i = 0; i < c.grid_shape()[0]; i++) {
      for (size_t j = 0; j < c.grid_shape()[1]; j++) {
        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          fprintf(stderr, "C[%lu, %lu] += A[%lu, %lu] * B[%lu, %lu]\n",
                  i, j, i, k, k, j);
          auto local_a = a.arget_tile_exp({i, k}).get();
          auto local_b = b.arget_tile_exp({k, j}).get();
          auto result_c = spgemm_nsparse<T, index_type, Allocator>(local_a, local_b);
          if (!is_sorted(result_c)) {
            fprintf(stderr, "  AGH! Error! Result matrix unsorted.\n");
            if (!is_sorted(local_a)) {
              fprintf(stderr, "  A matrix was unsorted.\n");
            }
            if (!is_sorted(local_b)) {
              fprintf(stderr, "  B matrix was unsorted.\n");
            }
            if (!is_positive_colind(local_a)) {
              fprintf(stderr, "  A matrix has negative colind.\n");
            }
            if (!is_positive_colind(local_b)) {
              fprintf(stderr, "  B matrix has negative colind.\n");
            }
            std::string prefix = "/gpfs/alpine/bif115/scratch/b2v/";
            auto cusparse_c = spgemm_cusparse<T, index_type, Allocator>(local_a, local_b);
            print_mtx(prefix + "a_matrix.mtx", local_a);
            print_mtx(prefix + "b_matrix.mtx", local_b);
            print_mtx(prefix + "c_nsparse.mtx", result_c);
            print_mtx(prefix + "c_cusparse.mtx", cusparse_c);
          }
        }
      }
    }
  }
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::cuda_allocator<T>>
void gemm_simple(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
                 BCL::cuda::Matrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        fprintf(stderr, "RANK(%lu): Doing tile (%lu, %lu)\n", BCL::rank(), i, j);
        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          auto local_a = a.arget_tile_exp({i, k}).get();
          auto local_b = b.arget_tile_exp({k, j}).get();
          auto local_c = c.get_local_tile({i, j});

          spmm_cusparse<T, index_type, Allocator>(local_a, local_b, local_c);
        }
      }
    }
  }
  BCL::cuda::barrier();
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
          BCL::cuda::Matrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  if (BCL::rank() == 45 || true) {
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k_offset = i + j;

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile_exp({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile_exp({k_offset % a.grid_shape()[1], j});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();
        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];
          auto begin = std::chrono::high_resolution_clock::now();
          auto local_a = buf_a.get();
          auto local_b = buf_b.get();
          auto local_c = c.get_local_tile({i, j});
          auto end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          if (k_+1 < a.grid_shape()[1]) {
            auto begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile_exp({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile_exp({(k+1) % a.grid_shape()[1], j});
            auto end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          spmm_cusparse(local_a, local_b, local_c);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
        }
      }
    }
  }
  }
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
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

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_bowns_simple(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
                       BCL::cuda::Matrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  std::vector<BCL::cuda::CudaMatrix<T, Allocator>> c_mats;

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    /*
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        */
        c_mats.emplace_back(BCL::cuda::CudaMatrix<T, Allocator>(
                            {c.tile_shape()[0], c.tile_shape()[1]}));
        c_mats[i] = 0;
      // }
    // }
  }

  // std::vector<MPI_Request> requests;
  std::vector<MPI_Request> requests(c.grid_shape()[0], MPI_REQUEST_NULL);

  for (size_t k = 0; k < a.grid_shape()[1]; k++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (b.tile_rank({k, j}) == BCL::rank()) {
        size_t i_offset = k + j;
        // size_t i_offset = 0;
        for (size_t i_ = 0; i_ < a.grid_shape()[0]; i_++) {
          size_t i = (i_ + i_offset) % a.grid_shape()[0];

          auto local_a = a.arget_tile_exp({i, k}).get();
          auto local_b = b.get_local_tile({k, j});
          auto& local_c = c_mats[i];

          local_c.m_ = local_a.shape()[0];
          local_c.n_ = local_b.shape()[1];
          spmm_cusparse(local_a, local_b, local_c);

          // MPI reduce with local_c on column j of B to process c.tile_rank({i, j})
          // Do I need to add a tag?
          if (true || (i == 0 && j == 0)) {
            std::string threads = "";
            for (size_t ii = 0; ii < b.column_teams_[j].nprocs(); ii++) {
              threads += std::to_string(b.column_teams_[j].to_world(ii)) + ", ";
            }
            MPI_Request request;
            BCL::backend::MPICommWrapper& comm = b.column_teams_mpi_[j][i];
            /*
            fprintf(stderr, "(%lu) Calling Ireduce with data %p, size %lu, MPI_FLOAT, MPI_SUM, comm (%s) -> %lu\n",
                             BCL::rank(), local_c.data(), local_c.size(),
                             threads.c_str(),
                             c.tile_rank({i, j}));
                             */
            T* rcv_ptr;
            if (BCL::rank() == c.tile_rank({i, j})) {
              auto local_c = c.get_local_tile({i, j});
              rcv_ptr = local_c.data();
            }
            int rv = MPI_Ireduce(local_c.data(), rcv_ptr, local_c.size(),
                                 MPI_FLOAT, MPI_SUM, b.column_teams_[j].resolve(c.tile_rank({i, j})), comm.comm(), &request);
            /*
            int rv = MPI_Reduce(local_c.data(), rcv_ptr, local_c.size(),
                                MPI_FLOAT, MPI_SUM, c.tile_rank({i, j}), comm.comm());
                                */
            assert(rv == MPI_SUCCESS);
            // fprintf(stderr, "(%lu) Finished Ireduce\n", BCL::rank());
            requests[i] = request;
          }
        }
      }
    }
  }
  // BCL::print("About to request...\n");
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  /*
  for (auto& request : requests) {
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
  */
  BCL::cuda::barrier();
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_bowns(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
                       BCL::cuda::Matrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  std::vector<BCL::cuda::CudaMatrix<T, Allocator>> c_mats;

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    /*
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        */
        c_mats.emplace_back(BCL::cuda::CudaMatrix<T, Allocator>(
                            {c.tile_shape()[0], c.tile_shape()[1]}));
        c_mats[i] = 0;
      // }
    // }
  }

  // std::vector<MPI_Request> requests;
  std::vector<MPI_Request> requests(c.grid_shape()[0], MPI_REQUEST_NULL);

  for (size_t k = 0; k < a.grid_shape()[1]; k++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (b.tile_rank({k, j}) == BCL::rank()) {
        size_t i_offset = k + j;
        // size_t i_offset = 0;
        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile_exp({i_offset % a.grid_shape()[0], k});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();
        for (size_t i_ = 0; i_ < a.grid_shape()[0]; i_++) {
          size_t i = (i_ + i_offset) % a.grid_shape()[0];

          begin = std::chrono::high_resolution_clock::now();
          auto local_a = buf_a.get();
          auto local_b = b.get_local_tile({k, j});
          auto& local_c = c_mats[i];
          end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          if (i_ + 1 < a.grid_shape()[0]) {
            begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile_exp({(i+1) % a.grid_shape()[0], k});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          local_c.m_ = local_a.shape()[0];
          local_c.n_ = local_b.shape()[1];
          spmm_cusparse(local_a, local_b, local_c);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();

          // MPI reduce with local_c on column j of B to process c.tile_rank({i, j})
          // Do I need to add a tag?
          if (true || (i == 0 && j == 0)) {
            begin = std::chrono::high_resolution_clock::now();
            std::string threads = "";
            for (size_t ii = 0; ii < b.column_teams_[j].nprocs(); ii++) {
              threads += std::to_string(b.column_teams_[j].to_world(ii)) + ", ";
            }
            MPI_Request request;
            BCL::backend::MPICommWrapper& comm = b.column_teams_mpi_[j][i];
            /*
            fprintf(stderr, "(%lu) Calling Ireduce with data %p, size %lu, MPI_FLOAT, MPI_SUM, comm (%s) -> %lu\n",
                             BCL::rank(), local_c.data(), local_c.size(),
                             threads.c_str(),
                             c.tile_rank({i, j}));
                             */
            T* rcv_ptr;
            if (BCL::rank() == c.tile_rank({i, j})) {
              auto local_c = c.get_local_tile({i, j});
              rcv_ptr = local_c.data();
            }
            int rv = MPI_Ireduce(local_c.data(), rcv_ptr, local_c.size(),
                                 MPI_FLOAT, MPI_SUM, b.column_teams_[j].resolve(c.tile_rank({i, j})), comm.comm(), &request);
            /*
            int rv = MPI_Reduce(local_c.data(), rcv_ptr, local_c.size(),
                                MPI_FLOAT, MPI_SUM, c.tile_rank({i, j}), comm.comm());
                                */
            assert(rv == MPI_SUCCESS);
            // fprintf(stderr, "(%lu) Finished Ireduce\n", BCL::rank());
            requests[i] = request;
            end = std::chrono::high_resolution_clock::now();
            duration_issue_reduction += std::chrono::duration<double>(end - begin).count();
          }
        }
      }
    }
  }
  // BCL::print("About to request...\n");
  auto begin = std::chrono::high_resolution_clock::now();
  MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
  auto end = std::chrono::high_resolution_clock::now();
  duration_sync_reduction += std::chrono::duration<double>(end - begin).count();
  /*
  for (auto& request : requests) {
    MPI_Wait(&request, MPI_STATUS_IGNORE);
  }
  */
  begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
  end = std::chrono::high_resolution_clock::now();
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
