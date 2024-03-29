// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <bcl/bcl.hpp>
#include "nsparse_util.hpp"
#include "cusparse_util.hpp"
#include "grb_util.hpp"
#include "cusp_util.hpp"

#include <bcl/containers/CircularQueue.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>

namespace BCL {

namespace cuda {

template <typename T, typename index_type,
          typename Indexing,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_mpi_simple(BCL::cuda::SPMatrix<T, index_type>& a,
                     BCL::cuda::Matrix<T, Indexing>& b,
                     BCL::cuda::Matrix<T, Indexing>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        // fprintf(stderr, "RANK(%lu): Doing tile (%lu, %lu)\n", BCL::rank(), i, j);
        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          auto begin = std::chrono::high_resolution_clock::now();
          MPI_Comm row_comm = a.row_teams_mpi_[i][0].comm();
          MPI_Comm column_comm = b.column_teams_mpi_[j][0].comm();
          BCL::cuda::CudaMatrix<T, Allocator, Indexing> local_b({b.tile_shape({k, j})[0],
                                                                 b.tile_shape({k, j})[1]});

          if (BCL::rank() == b.tile_rank({k, j})) {
            local_b = b.get_local_tile({k, j});
          } else {
            // local_b = 0;
          }

          size_t root = b.column_teams_[j].resolve(b.tile_rank({k, j}));

          assert(root == k);
          auto copy_begin = std::chrono::high_resolution_clock::now();
          MPI_Bcast(local_b.data(), local_b.size(), MPI_FLOAT, root, column_comm);
          auto copy_end = std::chrono::high_resolution_clock::now();
          double copied_size = local_b.size()*sizeof(float);
          double duration_copy = std::chrono::duration<double>(copy_end - copy_begin).count();

          T* values_data;
          index_type* rowptr_data;
          index_type* colind_data;

          size_t nnz = a.nnzs_[i*a.grid_shape()[1] + k];
          size_t m = a.tile_shape({i, k})[0];
          size_t n = a.tile_shape({i, k})[1];

          if (BCL::rank() == a.tile_rank({i, k})) {
            auto local_a = a.get_local_tile({i, k});
            values_data = local_a.values_data();
            rowptr_data = local_a.rowptr_data();
            colind_data = local_a.colind_data();
          } else {
            values_data = allocate_with<T, Allocator>(nnz);
            rowptr_data = allocate_with<index_type, Allocator>(m+1);
            colind_data = allocate_with<index_type, Allocator>(nnz);
          }

          root = a.row_teams_[i].resolve(a.tile_rank({i, k}));
          assert(root == k);

          copy_begin = std::chrono::high_resolution_clock::now();
          MPI_Bcast(values_data, nnz, MPI_FLOAT, root, row_comm);
          MPI_Bcast(rowptr_data, m+1, MPI_INT, root, row_comm);
          MPI_Bcast(colind_data, nnz, MPI_INT, root, row_comm);
          copy_end = std::chrono::high_resolution_clock::now();
          copied_size += nnz*sizeof(float) + (m+1+nnz)*sizeof(int);
          copied_size *= 1e-9;
          duration_copy += std::chrono::duration<double>(copy_end - copy_begin).count();
          double bw_gb = copied_size / duration_copy;
          // BCL::print("%lu MPI Bcasts achieved %lf GB/s (%lf s) \n", k, bw_gb, duration_copy);

          BCL::cuda::CudaCSRMatrixView<T, index_type> local_a(m, n, nnz,
                                                              values_data,
                                                              rowptr_data,
                                                              colind_data);

          auto local_c = c.get_local_tile({i, j});
          auto end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          begin = std::chrono::high_resolution_clock::now();
          spmm_cusparse(local_a, local_b, local_c);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();

          if (BCL::rank() != a.tile_rank({i, k})) {
            deallocate_with<T, Allocator>(values_data);
            deallocate_with<index_type, Allocator>(rowptr_data);
            deallocate_with<index_type, Allocator>(colind_data);
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

template <typename MatrixType>
std::vector<BCL::GlobalPtr<int>>
generate_grid(MatrixType& x) {
  std::vector<BCL::GlobalPtr<int>> grid(x.grid_shape()[0]*x.grid_shape()[1], nullptr);
  for (size_t i = 0; i < x.grid_shape()[0]; i++) {
    for (size_t j = 0; j < x.grid_shape()[1]; j++) {
      size_t idx = i*x.grid_shape()[1] + j;
      if (x.tile_rank({i, j}) == BCL::rank()) {
        grid[idx] = BCL::alloc<int>(1);
        *grid[idx].local() = 0;
      }
      grid[idx] = BCL::broadcast(grid[idx], x.tile_rank({i, j}));
    }
  }
  return grid;
}


template <typename T, typename index_type,
          typename queue_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_workstealing(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
                       BCL::cuda::Matrix<T>& c,
                       std::vector<queue_type>& queues,
                       std::vector<BCL::GlobalPtr<int>>& ws_grid,
                       bool print_perf = false) {

  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  size_t k_interval = 1;

  std::list<BCL::cuda::CudaMatrix<T, Allocator>> c_mats;

  size_t my_block = 0;

  // Go through *my own block*.
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
            // printf("(%lu) executing %lu (%lu)\n", BCL::rank(), j_, j);

            /*
            printf("(%lu) executing c[%lu, %lu] += a[%lu, %lu] * b[%lu, %lu] (step %lu of %lu's tile A[%lu, %lu]\n",
                   BCL::rank(), i, j, i, k, k, j,
                   j_, a.tile_rank({i, k}), i, k);
                   */

            auto begin = std::chrono::high_resolution_clock::now();
            auto local_a = a.get_local_tile({i, k});
            auto local_b = b.arget_tile_exp({k, j}).get();
            auto end = std::chrono::high_resolution_clock::now();
            duration_sync += std::chrono::duration<double>(end - begin).count();

            BCL::cuda::CudaMatrix<T, Allocator> local_c({c.tile_shape({i, j})[0], c.tile_shape({i, j})[1]});

            begin = std::chrono::high_resolution_clock::now();
            spmm_cusparse(local_a, local_b, local_c, 1.0, 0.0);
            end = std::chrono::high_resolution_clock::now();
            duration_compute += std::chrono::duration<double>(end - begin).count();

            queue_type& queue = queues[c.tile_rank({i, j})];
            queue.push({BCL::cuda::__to_cuda_gptr<T>(local_c.data()), local_c.shape()[0], local_c.shape()[1], local_c.ld(), i, j});

            c_mats.push_back(std::move(local_c));
          }
          my_tile_prog += k_interval;
        }
      }
    }
  }

  auto probe = [](size_t offset) { return offset*2; };
  size_t max_seeks = a.grid_shape()[0];
  size_t seeks = 0;

  if (BCL::rank() == 10 || true) {
  // Try to steal from others.
  for (size_t idx_ = 1; seeks < max_seeks; idx_++) {
    size_t tile_idx = (my_block + probe(idx_)) % (a.grid_shape()[0]*a.grid_shape()[1]);
    size_t i = tile_idx / a.grid_shape()[1];
    size_t k = tile_idx % a.grid_shape()[1];

    // Try to steal work
    int my_tile_prog = 0;
    while (my_tile_prog < b.grid_shape()[1]) {
      size_t idx = i*a.grid_shape()[1] + k;
      BCL::GlobalPtr<int> ptr = ws_grid[idx];
      my_tile_prog = BCL::fetch_and_op<int>(ptr, k_interval, BCL::plus<int>{});
      size_t j_offset = i + k;
      /*
      printf("Tried to steal (%lu, %lu)[%lu] (Got %lu / %lu, %s)\n",
             i, k, idx,
             my_tile_prog, std::min(my_tile_prog + k_interval, b.grid_shape()[1]),
             (my_tile_prog < std::min(my_tile_prog + k_interval, b.grid_shape()[1])) ? "success" : "failure");
             */
      for (size_t j_ = my_tile_prog; j_ < std::min(my_tile_prog + k_interval, b.grid_shape()[1]); j_++) {
        size_t j = (j_ + j_offset) % b.grid_shape()[1];
        // printf("%lu succeeded in stealing (%lu, %lu)[%lu %%... %lu]\n", BCL::rank(), i, k, j_, j);
        /*
        printf("(%lu) stealing c[%lu, %lu] += a[%lu, %lu] * b[%lu, %lu] (step %lu of %lu's tile A[%lu, %lu]\n",
               BCL::rank(), i, j, i, k, k, j,
               j_, a.tile_rank({i, k}), i, k);
               */

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

        BCL::cuda::CudaMatrix<T, Allocator> local_c({c.tile_shape({i, j})[0], c.tile_shape({i, j})[1]});

        begin = std::chrono::high_resolution_clock::now();
        spmm_cusparse(local_a, local_b, local_c, 1.0, 0.0);
        end = std::chrono::high_resolution_clock::now();
        duration_compute += std::chrono::duration<double>(end - begin).count();

        queue_type& queue = queues[c.tile_rank({i, j})];
        queue.push({BCL::cuda::__to_cuda_gptr<T>(local_c.data()), local_c.shape()[0], local_c.shape()[1], local_c.ld(), i, j});
        /*
        printf("Pushing (%lu, %lu) (%lu x %lu, LD %lu) from %lu (%lu) \n",
               i, j, local_c.shape()[0], local_c.shape()[1], local_c.ld(),
               BCL::rank(), BCL::cuda::__to_cuda_gptr<T>(local_c.data()).ptr_);
               */

        c_mats.push_back(std::move(local_c));
      }
      my_tile_prog += k_interval;
    }

    seeks++;
  }
  }

  queue_type& my_queue = queues[BCL::rank()];
  auto begin = std::chrono::high_resolution_clock::now();
  if (c.my_num_tiles() > 0) {
    for (size_t i = 0; i < b.grid_shape()[1]; i++) {
      CudaMatrix_ptr<T> ptr;
      while (!my_queue.pop(ptr)) {}
      // printf("(%lu) adding (%lu, %lu)\n", BCL::rank(), ptr.i, ptr.j);
      CudaMatrix<T, BCL::cuda::bcl_allocator<T>> x({c.tile_shape({ptr.i, ptr.j})[0], c.tile_shape({ptr.i, ptr.j})[1]});
      BCL::cuda::memcpy(x.data(), ptr.values, sizeof(T)*x.size());
      BCL::cuda::flush();
      assert(c.tile_rank({ptr.i, ptr.j}) == BCL::rank());
      auto local_c = c.get_local_tile({ptr.i, ptr.j});
      if (BCL::rank() == 12) {
        // printf("Got (%lu, %lu) (%lu x %lu, LD %lu) from %lu (%lu) \n", ptr.i, ptr.j, ptr.m, ptr.n, ptr.ld, ptr.values.rank_, ptr.values.ptr_);
        std::vector<T> local_x(x.size());
        cudaMemcpy(local_x.data(), x.data(), sizeof(T)*x.size(), cudaMemcpyDeviceToHost);
        size_t x_m = c.tile_shape({ptr.i, ptr.j})[0];
        size_t x_n = c.tile_shape({ptr.i, ptr.j})[1];
        /*
        for (size_t i = 0; i < x_m; i++) {
          printf("%lu:", i);
          for (size_t j = 0; j < x_n; j++) {
            printf(" %.2f", local_x[i*x_n + j]);
          }
          printf("\n");
        }
        */
      }
      local_c += x;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  duration_accumulate += std::chrono::duration<double>(end - begin).count();
  begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
  end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>,
          typename Indexing>
void gemm(BCL::cuda::SPMatrix<T, index_type>& a,
          BCL::cuda::Matrix<T, Indexing>& b,
          BCL::cuda::Matrix<T, Indexing>& c,
          bool print_perf = false) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  using timer_type = decltype(std::chrono::high_resolution_clock::now());
  timer_type last_issue;
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {
        size_t k_offset = i + j;

        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_a = a.arget_tile_exp({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile_exp({k_offset % a.grid_shape()[1], j});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();
        last_issue = begin;
        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];
          auto begin = std::chrono::high_resolution_clock::now();
          auto local_a = buf_a.get();
          auto local_b = buf_b.get();
          auto local_c = c.get_local_tile({i, j});
          auto end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();
          double duration_comm = std::chrono::duration<double>(end - last_issue).count();
          double data_a = sizeof(index_type)*(local_a.nnz() + local_a.m()+1) + sizeof(T)*local_a.nnz();
          double data_b = sizeof(T)*local_b.size();
          double bandwidth_gb = (1e-9*(data_a + data_b)) / duration_comm;
          if (print_perf) {
            printf("(%lu) k(%lu): %lf GB/s BW\n", BCL::rank(), k_, bandwidth_gb);
          }

          if (k_+1 < a.grid_shape()[1]) {
            auto begin = std::chrono::high_resolution_clock::now();
            buf_a = a.arget_tile_exp({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile_exp({(k+1) % a.grid_shape()[1], j});
            auto end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
            last_issue = begin;
          }

          begin = std::chrono::high_resolution_clock::now();
          spmm_cusparse(local_a, local_b, local_c);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
          double num_gflops = 1e-9*2*local_a.nnz()*local_b.shape()[1];
          double gflop_rate = num_gflops / std::chrono::duration<double>(end - begin).count();
          // (nnz(A)*2*m) / w*(nnz(A) + B + C)
          double data_size = sizeof(index_type)*(local_a.nnz() + local_a.m()+1) + sizeof(T)*(local_a.nnz() + local_b.size() + local_c.size());
          data_size *= 1e-9;
          double arithmetic_intensity = num_gflops / data_size;
          double memory_bw = 900;
          double roofline_peak = arithmetic_intensity * memory_bw;
          if (print_perf) {
            printf("(%lu) k(%lu): %lf GFlop/s compute (%lf%% of peak %lf)\n",
                   BCL::rank(), k_, gflop_rate, 100*(gflop_rate/roofline_peak),
                   roofline_peak);
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

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_bowns_onesided(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
                         BCL::cuda::Matrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);

  using queue_type = BCL::ChecksumQueue<CudaMatrix_ptr<T>, BCL::djb2_hash<CudaMatrix_ptr<T>>>;
  std::vector<queue_type> queues;

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    queues.emplace_back(queue_type(i, a.grid_shape()[1]+8));
  }

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

  std::vector<MPI_Request> requests(c.grid_shape()[0], MPI_REQUEST_NULL);

  for (size_t k = 0; k < a.grid_shape()[1]; k++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (b.tile_rank({k, j}) == BCL::rank()) {
        size_t i_offset = k + j;
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
          
          queue_type& queue = queues[c.tile_rank({i, j})];
          queue.push({BCL::cuda::__to_cuda_gptr<T>(local_c.data()), local_c.shape()[0], local_c.shape()[1], local_c.ld(), i, j});

        }
      }
    }
  }

  queue_type& my_queue = queues[BCL::rank()];
  auto begin = std::chrono::high_resolution_clock::now();
  if (c.my_num_tiles() > 0) {
    for (size_t i = 0; i < a.grid_shape()[1]; i++) {
      CudaMatrix_ptr<T> ptr;
      while (!my_queue.pop(ptr)) {}
      CudaMatrix<T, BCL::cuda::bcl_allocator<T>> x({c.tile_shape()[0], c.tile_shape()[1]});
      BCL::cuda::memcpy(x.data(), ptr.values, sizeof(T)*x.size());
      auto local_c = c.get_local_tile({ptr.i, ptr.j});
      local_c += x;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  duration_sync_reduction += std::chrono::duration<double>(end - begin).count();
  begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
  end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

template <typename T, typename index_type,
          typename queue_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_aowns_onesided(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
                         BCL::cuda::Matrix<T>& c,
                         std::vector<queue_type>& queues) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);


  std::vector<BCL::cuda::CudaMatrix<T, Allocator>> c_mats;

  // std::vector<MPI_Request> requests;
  std::vector<MPI_Request> requests(c.grid_shape()[0], MPI_REQUEST_NULL);

  for (size_t i = 0; i < a.grid_shape()[0]; i++) {
    for (size_t k = 0; k < a.grid_shape()[1]; k++) {
      if (a.tile_rank({i, k}) == BCL::rank()) {

        for (size_t j = 0; j < c.grid_shape()[1]; j++) {
          c_mats.emplace_back(BCL::cuda::CudaMatrix<T, Allocator>(
                              {c.tile_shape({i, j})[0], c.tile_shape({i, j})[1]}));
        }

        size_t j_offset = i + k;
        auto begin = std::chrono::high_resolution_clock::now();
        auto buf_b = b.arget_tile_exp({k, j_offset % b.grid_shape()[1]});
        auto end = std::chrono::high_resolution_clock::now();
        duration_issue += std::chrono::duration<double>(end - begin).count();
        for (size_t j_ = 0; j_ < b.grid_shape()[1]; j_++) {
          size_t j = (j_ + j_offset) % b.grid_shape()[1];

          begin = std::chrono::high_resolution_clock::now();
          auto local_a = a.get_local_tile({i, k});
          auto local_b = buf_b.get();

          auto& local_c = c_mats[j];

          end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

          if (j_ + 1 < b.grid_shape()[1]) {
            begin = std::chrono::high_resolution_clock::now();
            buf_b = b.arget_tile_exp({k, (j+1) % b.grid_shape()[1]});
            end = std::chrono::high_resolution_clock::now();
            duration_issue += std::chrono::duration<double>(end - begin).count();
          }

          begin = std::chrono::high_resolution_clock::now();
          spmm_cusparse(local_a, local_b, local_c, 1.0, 0.0);
          
          queue_type& queue = queues[c.tile_rank({i, j})];
          queue.push({BCL::cuda::__to_cuda_gptr<T>(local_c.data()), local_c.shape()[0], local_c.shape()[1], local_c.ld(), i, j});
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();

        }
      }
    }
  }
  queue_type& my_queue = queues[BCL::rank()];
  auto begin = std::chrono::high_resolution_clock::now();
  if (c.my_num_tiles() > 0) {
    for (size_t i = 0; i < b.grid_shape()[1]; i++) {
      CudaMatrix_ptr<T> ptr;
      while (!my_queue.pop(ptr)) {}
      CudaMatrix<T, BCL::cuda::bcl_allocator<T>> x({c.tile_shape({ptr.i, ptr.j})[0], c.tile_shape({ptr.i, ptr.j})[1]});
      BCL::cuda::memcpy(x.data(), ptr.values, sizeof(T)*x.size());
      auto local_c = c.get_local_tile({ptr.i, ptr.j});
      local_c += x;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  duration_accumulate += std::chrono::duration<double>(end - begin).count();
  begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
  end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

template <typename T, typename index_type,
          typename Allocator = BCL::cuda::bcl_allocator<T>>
void gemm_cusp(BCL::cuda::SPMatrix<T, index_type>& a, BCL::cuda::Matrix<T>& b,
               BCL::cuda::Matrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
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
          spmm_cusp(local_a, local_b, local_c);
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
        }
      }
    }
  }
  auto begin = std::chrono::high_resolution_clock::now();
  BCL::cuda::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  duration_barrier += std::chrono::duration<double>(end - begin).count();
}

} // end cuda

} // end grb
