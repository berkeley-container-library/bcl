// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>
#include <bcl/containers/SPMatrix.hpp>

#include "experimental_gemm.hpp"

namespace BCL {

#ifdef USE_MKL
template <typename T, typename I>
void spmm_wrapper(size_t m, size_t n, size_t k, size_t nnz, T* c, T* b,
                  T* a_values, I* a_rowptr, I* a_colind,
                  size_t ldb, size_t ldc) {

  MKL_INT M = m;
  MKL_INT N = n;
  MKL_INT K = k;
  MKL_INT ldb_ = ldb;
  MKL_INT ldc_ = ldc;
  T alpha = 1.0;
  T beta = 1.0;
  char matdescra[6];
  matdescra[0] = 'G';
  matdescra[3] = 'C';
  mkl_scsrmm("N", &M, &N, &K, &alpha,
             matdescra,
             a_values, a_colind, a_rowptr, a_rowptr+1,
             b, &ldb_, &beta,
             c, &ldc_);
}
#else
template <typename T, typename I>
void spmm_wrapper(size_t m, size_t n, size_t k, size_t nnz, T* c, T* b,
                  T* a_values, I* a_rowptr, I* a_colind,
                  size_t ldb, size_t ldc) {
  for (std::size_t i = 0; i < m; i++) {
    for (I k_ptr = a_rowptr[i]; k_ptr < a_rowptr[i+1]; k_ptr++) {
      auto&& v = a_values[k_ptr];
      auto&& k_ = a_colind[k_ptr];

      for (size_t j = 0; j < n; j++) {
        // c[i, j] += a[i, k] * b[k, j]
        c[i*ldc + j] += v * b[k_*ldb + j];
      }
    }
  }
}
#endif

template <typename T, typename I>
void gemm(const SPMatrix<T, I>& a, const DMatrix<T>& b, DMatrix<T>& c) {
  assert(a.grid_shape()[0] == c.grid_shape()[0]);
  assert(b.grid_shape()[1] == c.grid_shape()[1]);
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (BCL::rank() == c.tile_rank({i, j})) {
        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          auto begin = std::chrono::high_resolution_clock::now();
          auto local_a = a.get_tile({i, k});
          auto local_b = b.get_tile({k, j});
          auto end = std::chrono::high_resolution_clock::now();
          double duration = std::chrono::duration<double>(end - begin).count();
          BCL::row_comm += duration;

          // call csrmm
          T* values = local_a.vals_.data();
          I* rowptr = local_a.row_ptr_.data();
          I* colind = local_a.col_ind_.data();

          spmm_wrapper(c.tile_shape({i, j})[0], c.tile_shape({i, j})[1],
                       a.tile_shape({i, k})[1], a.tile_nnz({i, k}),
                       c.tile_ptr({i, j}).local(), local_b.data(),
                       values, rowptr, colind, b.tile_shape()[1], c.tile_shape()[1]);
        }
      }
    }
  }
}

template <
          typename T,
          typename index_type,
          typename Mult = std::multiplies<T>,
          typename Plus = std::plus<T>
          >
void gemm(const BCL::SPMatrix<T, index_type>& a,
          const BCL::SPMatrix<T, index_type>& b,
                BCL::SPMatrix<T, index_type>& c) {
  // accumulated C's: a map of grid coordinates to sparse
  //                  matrices (for now, also maps)
  //

  std::unordered_map<
                     std::pair<size_t, size_t>,
                     // BCL::SparseSPAAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     BCL::SparseHashAccumulator<T, index_type, Plus, BCL::bcl_allocator<T>>,
                     // BCL::SparseHeapAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     // BCL::CombBLASAccumulator<T, index_type>,
                     // BCL::EagerMKLAccumulator<T, index_type>,
                     BCL::djb2_hash<std::pair<size_t, size_t>>
                     >
                     accumulated_cs;

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_rank({i, j}) == BCL::rank()) {

        size_t k_offset = j % a.grid_shape()[1];

        auto buf_a = a.arget_tile({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile({k_offset % a.grid_shape()[1], j});

        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];


          auto my_a = buf_a.get();
          auto my_b = buf_b.get();

          if (k_+1 < a.grid_shape()[1]) {
            buf_a = a.arget_tile({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile({(k+1) % a.grid_shape()[1], j});
          }

          auto c = my_a. template dot<Mult, Plus>(my_b);

          accumulated_cs[{i, j}].accumulate(std::move(c));
        }
      }
    }
  }

  BCL::barrier();

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_locale(i, j) == BCL::rank()) {
        auto cmatrix = accumulated_cs[{i, j}].get_matrix(c.tile_shape(i, j)[0], c.tile_shape(i, j)[1]);
        c.assign_tile({i, j}, cmatrix);
      }
    }
  }

  BCL::barrier();
  c.rebroadcast_tiles();
}

} // end BCL
