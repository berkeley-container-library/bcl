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
                 BCL::cuda::SPMatrix<T, index_type>& c, graphblas::Descriptor& desc) {
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k_offset = i + j;

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile_exp({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile_exp({k_offset % a.grid_shape()[1], j});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();

        std::vector<graphblas::Matrix<T>*> intermediate_results;

        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          /*
          begin = std::chrono::high_resolution_clock::now();
          BCL::barrier();
          end = std::chrono::high_resolution_clock::now();
          duration_barrier += std::chrono::duration<double>(end - begin).count();
          */

          begin = std::chrono::high_resolution_clock::now();
          auto local_a = buf_a.get();
          auto local_b = buf_b.get();
          end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          if (BCL::rank() == 1) {
            print_mtx<T, index_type>("/gpfs/alpine/bif115/scratch/b2v/a_mat.mtx", local_a);
            print_mtx<T, index_type>("/gpfs/alpine/bif115/scratch/b2v/b_mat.mtx", local_b);
          }

/*
          printf("RANK(%lu) local A %lf GB, local B %lf GB\n",
                 BCL::rank(),
                 1.0e-9*local_a->nbytes(), 1.0e-9*local_b->nbytes());
                 */

          if (k_+1 < a.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile_exp({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile_exp({(k+1) % a.grid_shape()[1], j});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          graphblas::Index a_nnz, b_nnz;
          {
            graphblas::Index a_m, a_n;
            graphblas::Index b_m, b_n;
            local_a->nvals(&a_nnz);
            local_a->nrows(&a_m);
            local_a->ncols(&a_n);
            local_b->nvals(&b_nnz);
            local_b->nrows(&b_m);
            local_b->ncols(&b_n);
          }

          graphblas::Matrix<T>* result_c = new graphblas::Matrix<T>(c.tile_shape({i, j})[0],
                                                                    c.tile_shape({i, j})[1]);

          begin = std::chrono::high_resolution_clock::now();
          if (a_nnz != 0 && b_nnz != 0) {
            /*
            graphblas::mxm<T, T, T, T>(result_c, GrB_NULL,
                                       GrB_NULL, graphblas::PlusMultipliesSemiring<T>(),
                                       local_a, local_b,
                                       &desc);
                                       */
            auto binary_op = GrB_NULL;
            auto semiring = graphblas::PlusMultipliesSemiring<T>{};
            /*
            graphblas::mxm<T, T, T, T, decltype(binary_op), decltype(semiring),
                           Allocator>
                           (result_c, GrB_NULL,
                            binary_op, semiring,
                            local_a, local_b, &desc);
                            */

            free(result_c);
            result_c = mxm_nsparse<T, index_type, Allocator>(local_a, local_b);
            cudaDeviceSynchronize();

            if (!is_sorted<T, index_type>(result_c)) {
              fprintf(stderr, "RANK(%lu) result C is not sorted.\n", BCL::rank());
            }
                           /*
            printf("RANK(%lu) Resulting C value %lf GB\n",
                   BCL::rank(), 1.0e-9*result_c->nbytes());
                   */
          } else {
            fprintf(stderr, "RANK(%lu) (%lu, %lu)[%lu] is empty.\n",
                    BCL::rank(), i, j, k);
          }
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
          destroy_grb(local_a, "local A");
          destroy_grb(local_b, "local B");

          // TODO: replace with some `accumulator` design.

          BCL::barrier();
          fflush(stdout);
          sleep(1);
          BCL::barrier();
          if (BCL::rank() == 1) {
            fprintf(stderr, "Printing matrix...\n");
            print_mtx<T, index_type>("/gpfs/alpine/bif115/scratch/b2v/c_mat.mtx", result_c);
          }
          BCL::barrier();
          fflush(stdout);
          sleep(1);
          BCL::barrier();
          graphblas::Index nnz;
          result_c->nvals(&nnz);
          if (nnz != 0) {
            auto begin = std::chrono::high_resolution_clock::now();
            intermediate_results.push_back(result_c);
            auto* new_local_c = sum_tiles<T, Allocator>(intermediate_results);
            intermediate_results = {new_local_c};
            auto end = std::chrono::high_resolution_clock::now();
            duration_accumulate += std::chrono::duration<double>(end - begin).count();
          }
        }
        // TODO: also add C block to this.
        begin = std::chrono::high_resolution_clock::now();

        auto* new_local_c = sum_tiles<T, Allocator>(intermediate_results);
        // auto* new_local_c = sum_tiles<T, Allocator>(intermediate_results);
        if (new_local_c != nullptr) {
          c.assign_tile({i, j}, new_local_c);
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

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::cuda_allocator<T>>
void dry_gemm(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
              BCL::cuda::SPMatrix<T, index_type>& c, graphblas::Descriptor& desc) {
  size_t total_flops = 0;
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
          graphblas::Matrix<T>* result_c = new graphblas::Matrix<T>(c.tile_shape({i, j})[0],
                                                                    c.tile_shape({i, j})[1]);
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

          graphblas::Index a_nnz, b_nnz;
          {
            graphblas::Index a_m, a_n;
            graphblas::Index b_m, b_n;
            local_a->nvals(&a_nnz);
            local_a->nrows(&a_m);
            local_a->ncols(&a_n);
            local_b->nvals(&b_nnz);
            local_b->nrows(&b_m);
            local_b->ncols(&b_n);
          }

          begin = std::chrono::high_resolution_clock::now();
          if (a_nnz != 0 && b_nnz != 0) {
            auto binary_op = GrB_NULL;
            auto semiring = graphblas::PlusMultipliesSemiring<T>{};
            /*
            size_t nflops = num_flops<T, index_type>(local_a, local_b);
            fprintf(stderr, "RANK(%lu)[%lu] C(%lu, %lu) %lu flops\n",
                    BCL::rank(), k_, i, j, nflops);
            total_flops += nflops;
            */
            /*
            graphblas::mxm<T, T, T, T, decltype(binary_op), decltype(semiring),
                           Allocator>
                           (result_c, GrB_NULL,
                            binary_op, semiring,
                            local_a, local_b, &desc);
                            */
            free(result_c);

            result_c = mxm_nsparse<T, index_type, Allocator>(local_a, local_b);
            cudaDeviceSynchronize();
          } else {
            fprintf(stderr, "RANK(%lu) (%lu, %lu)[%lu] is empty.\n",
                    BCL::rank(), i, j, k);
          }
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
          destroy_grb(local_a, "local A");
          destroy_grb(local_b, "local B");
          destroy_grb(result_c, "local C");
        }

/*
        fprintf(stderr, "RANK(%lu) C(%lu, %lu) %lf GFlops total\n",
                BCL::rank(), i, j, total_flops / 1e9);
                */
      }
    }
  }
  double min_flops = BCL::allreduce(total_flops, BCL::min<double>{});
  double max_flops = BCL::allreduce(total_flops, BCL::max<double>{});
  double average_flops = BCL::allreduce(total_flops, std::plus<double>{});
  average_flops /= BCL::nprocs();
  BCL::print("%lf GFlops avg, %lf min, %lf max\n",
             average_flops / 1e9, min_flops / 1e9, max_flops / 1e9);
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::cuda_allocator<T>>
void gemm_single_proc(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::SPMatrix<T, index_type>& b,
                      BCL::cuda::SPMatrix<T, index_type>& c, graphblas::Descriptor& desc) {
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  if (BCL::rank() == 0) {
    for (size_t i = 0; i < c.grid_shape()[0]; i++) {
      for (size_t j = 0; j < c.grid_shape()[1]; j++) {

        double compute_duration = 0;

        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          // fprintf(stderr, "Allocating empty result C\n", BCL::rank());
          auto begin = std::chrono::high_resolution_clock::now();
          auto local_a = a.arget_tile_exp({i, k}).get();
          auto local_b = b.arget_tile_exp({k, j}).get();
          auto end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          graphblas::Index a_m, a_n;
          graphblas::Index b_m, b_n;
          graphblas::Index a_nnz, b_nnz;
          {
            local_a->nvals(&a_nnz);
            local_a->nrows(&a_m);
            local_a->ncols(&a_n);
            local_b->nvals(&b_nnz);
            local_b->nrows(&b_m);
            local_b->ncols(&b_n);
          }

/*
          fprintf(stderr, "RANK(%lu) multiplying A(%lu, %lu) (%lu x %lu, %lu nonzeros) * B(%lu, %lu) (%lu x %lu, %lu nonzeros)\n",
                  BCL::rank(),
                  i, k, a_m, a_n, a_nnz,
                  k, j, b_m, b_n, b_nnz);

                  */
          graphblas::Matrix<T>* result_c = new graphblas::Matrix<T>(c.tile_shape({i, j})[0],
                                                                    c.tile_shape({i, j})[1]);
          begin = std::chrono::high_resolution_clock::now();
          if (a_nnz != 0 && b_nnz != 0) {
            auto binary_op = GrB_NULL;
            auto semiring = graphblas::PlusMultipliesSemiring<T>{};
            // fprintf(stderr, "Calling mxm...\n");
            // CUDA_CALL(cudaPeekAtLastError());
            /*
            graphblas::mxm<T, T, T, T, decltype(binary_op), decltype(semiring),
                           Allocator>
                           (result_c, GrB_NULL,
                            binary_op, semiring,
                            local_a, local_b, &desc);
                            */
            free(result_c);
            fprintf(stderr, "RANK Calling mxm sparse... C(%lu, %lu)[%lu]\n", i, j, k);
            result_c = mxm_nsparse<T, index_type>(local_a, local_b);
            // CUDA_CALL(cudaPeekAtLastError());
            cudaDeviceSynchronize();
          } else {
            fprintf(stderr, "RANK(%lu) (%lu, %lu)[%lu] is empty.\n",
                    BCL::rank(), i, j, k);
          }
          end = std::chrono::high_resolution_clock::now();
          compute_duration += std::chrono::duration<double>(end - begin).count();
          duration_compute += std::chrono::duration<double>(end - begin).count();
          fprintf(stderr, "C(%lu, %lu)[%lu] took %lf seconds\n",
                  i, j, k, std::chrono::duration<double>(end - begin).count());
          destroy_grb(local_a, "local A");
          destroy_grb(local_b, "local B");
          // fprintf(stderr, "Destroying local C\n");
          destroy_grb(result_c, "local C");
        }

        fprintf(stderr, "Computing C(%lu, %lu) Took %lf seconds\n", i, j, compute_duration);
      }
    }
  }
}

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
