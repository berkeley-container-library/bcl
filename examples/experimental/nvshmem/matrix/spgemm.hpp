#pragma once

#include <bcl/bcl.hpp>
#include "cusparse_util.hpp"
#include "grb_util.hpp"

namespace BCL {

namespace cuda {

double duration_issue;
double duration_sync;
double duration_compute;
double duration_accumulate;
double duration_barrier;


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
          graphblas::Matrix<T>* result_c = new graphblas::Matrix<T>(c.tile_shape({i, j})[0],
                                                                    c.tile_shape({i, j})[1]);

          begin = std::chrono::high_resolution_clock::now();
          BCL::barrier();
          end = std::chrono::high_resolution_clock::now();
          duration_barrier += std::chrono::duration<double>(end - begin).count();

          begin = std::chrono::high_resolution_clock::now();
          auto local_a = buf_a.get();
          auto local_b = buf_b.get();
          end = std::chrono::high_resolution_clock::now();
          duration_sync += std::chrono::duration<double>(end - begin).count();

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
            graphblas::mxm<T, T, T, T, decltype(binary_op), decltype(semiring),
                           Allocator>
                           (result_c, GrB_NULL,
                            binary_op, semiring,
                            local_a, local_b, &desc);
                           /*
            printf("RANK(%lu) Resulting C value %lf GB\n",
                   BCL::rank(), 1.0e-9*result_c->nbytes());
                   */
            cudaDeviceSynchronize();
          } else {
            fprintf(stderr, "RANK(%lu) (%lu, %lu)[%lu] is empty.\n",
                    BCL::rank(), i, j, k);
          }
          end = std::chrono::high_resolution_clock::now();
          duration_compute += std::chrono::duration<double>(end - begin).count();
          destroy_grb(local_a, "local A");
          destroy_grb(local_b, "local B");

          // TODO: replace with some `accumulator` design.
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

} // end cuda

} // end BCL
