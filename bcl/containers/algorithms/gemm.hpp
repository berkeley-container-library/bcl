// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <bcl/bcl.hpp>
#include <bcl/containers/detail/Blocking.hpp>

#include <bcl/containers/algorithms/cblas_wrapper.hpp>
#include <bcl/containers/algorithms/spgemm.hpp>

#include <cassert>

namespace BCL {

template <typename T>
class DMatrix;

template <typename T, typename I>
class SPMatrix;

template <typename E>
class DExpr;

namespace experimental {

enum BlasTrans {
  notrans,
  trans
};

template <typename T>
inline constexpr bool is_distributed_dense_matrix_v = std::is_base_of_v<DExpr<std::decay_t<T>>, std::decay_t<T>>;

template <typename T>
struct is_a_spmatrix_impl_ { static constexpr bool value = false; };

template <typename... Args>
struct is_a_spmatrix_impl_<BCL::SPMatrix<Args...>> { static constexpr bool value = true; };

template <typename T>
inline constexpr bool is_distributed_sparse_matrix_v = is_a_spmatrix_impl_<std::decay_t<T>>::value;

template <typename AMatrixType,
          typename BMatrixType,
          typename CMatrixType,
          __BCL_REQUIRES(is_distributed_dense_matrix_v<AMatrixType> &&
                         is_distributed_dense_matrix_v<BMatrixType> &&
                         is_distributed_dense_matrix_v<CMatrixType>)>
inline void gemm_cowns(AMatrixType&& a, BMatrixType&& b, CMatrixType&& c) {
  static_assert(std::is_same_v<typename std::decay_t<AMatrixType>::scalar_type,
                               typename std::decay_t<BMatrixType>::scalar_type>);
  static_assert(std::is_same_v<typename std::decay_t<BMatrixType>::scalar_type,
                               typename std::decay_t<CMatrixType>::scalar_type>);

  // Inner dimensions of the tiles we're multiplying must match.
  using T = typename std::decay_t<AMatrixType>::scalar_type;

  assert(a.grid_shape()[0] == c.grid_shape()[0] &&
         a.grid_shape()[1] == b.grid_shape()[0] &&
         b.grid_shape()[1] == c.grid_shape()[1]);

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_ptr({i, j}).is_local()) {

        size_t k_offset = i + j;
        auto buf_a = a.arget_tile({i, k_offset % a.grid_shape()[1]});
        auto buf_b = b.arget_tile({k_offset % a.grid_shape()[1], j});
        BCL::GlobalPtr<T> my_c = c.tile_ptr({i, j});
        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto my_a = std::get<0>(buf_a).get();
          auto my_b = std::get<0>(buf_b).get();

          bool a_trans = std::get<1>(buf_a);
          bool b_trans = std::get<1>(buf_b);

          if (k_+1 < a.grid_shape()[1]) {
            buf_a = a.arget_tile({i, (k+1) % a.grid_shape()[1]});
            buf_b = b.arget_tile({(k+1) % a.grid_shape()[1], j});
          }

          CBLAS_TRANSPOSE transa, transb;
          transa = transb = CblasNoTrans;
          if (a_trans) {
            transa = CblasTrans;
          }
          if (b_trans) {
            transb = CblasTrans;
          }

          size_t M, N, K;
          size_t lda = a.tile_shape()[1];
          size_t ldb = b.tile_shape()[1];
          size_t ldc = c.tile_shape()[1];
          M = c.tile_shape({i, j})[0];
          N = c.tile_shape({i, j})[1];
          K = a.tile_shape({i, k})[1];

          cblas_gemm_wrapper_(CblasRowMajor, transa, transb,
                              M, N, K,
                              1.0, my_a.data(), lda,
                              my_b.data(), ldb, 1.0,
                              my_c.local(), ldc);
        }
      }
    }
  }
}

template <typename AMatrixType,
          typename BMatrixType,
          typename CMatrixType>
inline void gemm(AMatrixType&& a, BMatrixType&& b, CMatrixType&& c) {
  static_assert(std::is_same_v<typename std::decay_t<AMatrixType>::scalar_type,
                               typename std::decay_t<BMatrixType>::scalar_type>);
  static_assert(std::is_same_v<typename std::decay_t<BMatrixType>::scalar_type,
                               typename std::decay_t<CMatrixType>::scalar_type>);

  if (!(a.shape()[0] == c.shape()[0] && a.shape()[1] == b.shape()[0] && b.shape()[1] == c.shape()[1])) {
    throw std::runtime_error("gemm: Attempting to multiply matrices with incompatible dimensions");
  }

  gemm_cowns(std::forward<AMatrixType>(a),
             std::forward<BMatrixType>(b),
             std::forward<CMatrixType>(c));
}

template <typename T>
inline void slicing_gemm(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
  assert(a.shape()[0] == c.shape()[0]);
  assert(b.shape()[1] == c.shape()[1]);
  assert(a.shape()[1] == b.shape()[0]);

  size_t k_steps = 16;

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_ptr(i, j).is_local()) {

        size_t m_min = i*c.tile_shape()[0];
        size_t m_max = (i+1)*c.tile_shape()[0];

        size_t n_min = j*c.tile_shape()[1];
        size_t n_max = (j+1)*c.tile_shape()[1];

        size_t k_size = (a.shape()[1] + k_steps - 1) / k_steps;

        size_t k_offset = i + j;

        size_t k = k_offset % k_steps;
        size_t k_min = k_size*k;
        size_t k_max = std::min((k_size)*(k+1), a.shape()[1]);

        auto a_buf = a.arslice({m_min, m_max}, {k_min, k_max});
        auto b_buf = b.arslice({k_min, k_max}, {n_min, n_max});

        T* local_c = c.tile_ptr(i, j).local();
        for (size_t k_ = 0; k_ < k_steps; k_++) {
          size_t k = (k_ + k_offset) % k_steps;

          size_t k_min = k_size*k;
          size_t k_max = std::min((k_size)*(k+1), a.shape()[1]);

          size_t M = c.tile_shape()[0];
          size_t N = c.tile_shape()[1];
          size_t K = k_max - k_min;

          auto local_a = a_buf.get();
          auto local_b = b_buf.get();

          if (k_+1 < k_steps) {
            size_t k = ((k_+1) + k_offset)  % k_steps;
            size_t k_min = k_size*k;
            size_t k_max = std::min((k_size)*(k+1), a.shape()[1]);
            a_buf = a.arslice({m_min, m_max}, {k_min, k_max});
            b_buf = b.arslice({k_min, k_max}, {n_min, n_max});
          }

          size_t lda = K;
          size_t ldb = N;
          size_t ldc = c.tile_shape()[1];

          CBLAS_TRANSPOSE transa, transb;
          transa = transb = CblasNoTrans;
          cblas_gemm_wrapper_(CblasRowMajor, transa, transb,
                              M, N, K,
                              1.0, local_a.data(), lda,
                              local_b.data(), ldb, 1.0,
                              local_c, ldc);
        }
      }
    }
  }
}

template <typename T>
inline void a_owns_gemm(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c,
                        std::vector<BCL::DMatrix<T>>& c_acc) {
  assert(a.shape()[1] == b.shape()[0]);
  // Inner dimensions of the tiles we're multiplying must match.
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  assert(a.tile_shape()[1] == b.tile_shape()[0]);

  if (!(a.shape()[0] == c.shape()[0] && a.shape()[1] == b.shape()[0] && b.shape()[1] == c.shape()[1])) {
    throw std::runtime_error("SUMMA: ruh roh, you gave me matrices with mismatched dimensions.");
  }

  assert(c_acc.size() >= a.grid_shape()[1]);

  std::vector<BCL::future<std::vector<T>>> futures;

  for (size_t i = 0; i < a.grid_shape()[0]; i++) {
    for (size_t k = 0; k < a.grid_shape()[1]; k++) {

      if (a.tile_ptr(i, k).is_local()) {
        T* local_a = a.tile_ptr(i, k).local();

        size_t j_offset = k;

        auto buf_b = b.arget_tile(k, j_offset % c.grid_shape()[1]);

        for (size_t j_ = 0; j_ < c.grid_shape()[1]; j_++) {
          size_t j = (j_ + j_offset) % c.grid_shape()[1];

          std::vector<T> local_c(c.tile_size());

          auto local_b = std::get<0>(buf_b).get();

          if (j_ + 1 < c.grid_shape()[1]) {
            buf_b = b.arget_tile(k, (j+1) % c.grid_shape()[1]);
          }

          size_t M, N, K;
          size_t lda = a.tile_shape()[1];
          size_t ldb = b.tile_shape()[1];
          size_t ldc = c.tile_shape()[1];
          M = c.tile_shape(i, j)[0];
          N = c.tile_shape(i, j)[1];
          K = a.tile_shape(i, k)[1];

          cblas_gemm_wrapper_(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                              M, N, K,
                              1.0, local_a, lda,
                              local_b.data(), ldb, 1.0,
                              local_c.data(), ldc);

          auto future = c_acc[k].arput_tile(i, j, std::move(local_c));
          futures.emplace_back(std::move(future));
        }
      }
    }
  }

  for (auto& future : futures) {
    future.get();
  }

  BCL::barrier();

  for (auto& acc : c_acc) {
    c += acc;
  }
}

template <typename T>
inline void summa(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
  assert(a.shape()[1] == b.shape()[0]);
  // Inner dimensions of the tiles we're multiplying must match.
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  assert(a.tile_shape()[1] == b.tile_shape()[0]);

  if (!(a.shape()[0] == c.shape()[0] && a.shape()[1] == b.shape()[0] && b.shape()[1] == c.shape()[1])) {
    throw std::runtime_error("SUMMA: ruh roh, you gave me matrices with mismatched dimensions.");
  }

  // TODO: should also assert that distribution is pure block, not block cyclic.

  std::vector<T> local_a(a.tile_size());
  std::vector<T> local_b(b.tile_size());

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_ptr(i, j).is_local()) {

        T* local_c = c.tile_ptr(i, j).local();

        for (size_t k = 0; k < a.grid_shape()[1]; k++) {
          size_t row_src = a.tile_ptr(i, k).rank;
          size_t column_src = b.tile_ptr(k, j).rank;

          auto row_team = a.row_teams_[i];
          auto column_team = b.column_teams_[j];

          row_src = row_team.resolve(row_src);
          column_src = column_team.resolve(column_src);

          T* local_a_ptr = local_a.data();
          T* local_b_ptr = local_b.data();

          if (a.tile_ptr(i, k).is_local()) {
            local_a_ptr = a.tile_ptr(i, k).local();
          }
          if (b.tile_ptr(k, j).is_local()) {
            local_b_ptr = b.tile_ptr(k, j).local();
          }

          auto fa = BCL::arbroadcast(local_a_ptr, row_src, a.tile_size(), row_team);
          auto fb = BCL::arbroadcast(local_b_ptr, column_src, b.tile_size(), column_team);

          fa.wait();
          fb.wait();

          CBLAS_TRANSPOSE transa, transb;
          transa = transb = CblasNoTrans;

          size_t M, N, K;
          size_t lda = a.tile_shape()[1];
          size_t ldb = b.tile_shape()[1];
          size_t ldc = c.tile_shape()[1];
          M = c.tile_shape(i, j)[0];
          N = c.tile_shape(i, j)[1];
          K = a.tile_shape(i, k)[1];

          cblas_gemm_wrapper_(CblasRowMajor, transa, transb,
                              M, N, K,
                              1.0, local_a_ptr, lda,
                              local_b_ptr, ldb, 1.0,
                              local_c, ldc);
        }
      }
    }
  }
}

template <typename T>
inline void async_summa(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
  assert(a.shape()[1] == b.shape()[0]);
  // Inner dimensions of the tiles we're multiplying must match.
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  assert(a.tile_shape()[1] == b.tile_shape()[0]);

  if (!(a.shape()[0] == c.shape()[0] &&
        a.shape()[1] == b.shape()[0] &&
        b.shape()[1] == c.shape()[1])) {
    throw std::runtime_error("SUMMA: ruh roh, you gave me matrices with mismatched dimensions.");
  }

  // TODO: should also assert that distribution is pure block, not block cyclic.

  std::vector<T> local_a(a.tile_size());
  std::vector<T> local_b(b.tile_size());

  std::vector<T> buf_a(a.tile_size());
  std::vector<T> buf_b(b.tile_size());

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_ptr(i, j).is_local()) {

        T* local_c = c.tile_ptr(i, j).local();

        T* buf_a_ptr;
        T* buf_b_ptr;

        T* local_a_ptr;
        T* local_b_ptr;

        size_t k = 0;
        auto row_team = a.row_teams_[i];
        auto column_team = b.column_teams_[j];

        size_t row_src = a.tile_ptr(i, k).rank;
        size_t column_src = b.tile_ptr(k, j).rank;

        row_src = row_team.resolve(row_src);
        column_src = column_team.resolve(column_src);

        buf_a_ptr = buf_a.data();
        buf_b_ptr = buf_b.data();

        if (a.tile_ptr(i, k).is_local()) {
          buf_a_ptr = a.tile_ptr(i, k).local();
        }
        if (b.tile_ptr(k, j).is_local()) {
          buf_b_ptr = b.tile_ptr(k, j).local();
        }

        auto fa = BCL::arbroadcast(buf_a_ptr, row_src, a.tile_size(), row_team);
        auto fb = BCL::arbroadcast(buf_b_ptr, column_src, b.tile_size(), column_team);
        for (size_t k = 0; k < a.grid_shape()[1]; k++) {

          fa.wait();
          fb.wait();

          local_a_ptr = buf_a_ptr;
          local_b_ptr = buf_b_ptr;

          std::swap(local_a, buf_a);
          std::swap(local_b, buf_b);

          if (k+1 < a.grid_shape()[1]) {
            size_t row_src = a.tile_ptr(i, k+1).rank;
            size_t column_src = b.tile_ptr(k+1, j).rank;

            auto row_team = a.row_teams_[i];
            auto column_team = b.column_teams_[j];

            row_src = row_team.resolve(row_src);
            column_src = column_team.resolve(column_src);

            buf_a_ptr = buf_a.data();
            buf_b_ptr = buf_b.data();

            if (a.tile_ptr(i, k+1).is_local()) {
              buf_a_ptr = a.tile_ptr(i, k+1).local();
            }
            if (b.tile_ptr(k+1, j).is_local()) {
              buf_b_ptr = b.tile_ptr(k+1, j).local();
            }

            fa = BCL::arbroadcast(buf_a_ptr, row_src, a.tile_size(), row_team);
            fb = BCL::arbroadcast(buf_b_ptr, column_src, b.tile_size(), column_team);
          }

          CBLAS_TRANSPOSE transa, transb;
          transa = transb = CblasNoTrans;

          size_t M, N, K;
          size_t lda = a.tile_shape()[1];
          size_t ldb = b.tile_shape()[1];
          size_t ldc = c.tile_shape()[1];
          M = c.tile_shape(i, j)[0];
          N = c.tile_shape(i, j)[1];
          K = a.tile_shape(i, k)[1];

          cblas_gemm_wrapper_(CblasRowMajor, transa, transb,
                              M, N, K,
                              1.0, local_a_ptr, lda,
                              local_b_ptr, ldb, 1.0,
                              local_c, ldc);
        }
      }
    }
  }
}


template <typename AMatrixType,
          typename BMatrixType>
inline auto gemm_two_args_impl_(AMatrixType&& a, BMatrixType&& b) {
  static_assert(std::is_same_v<typename std::decay_t<AMatrixType>::scalar_type,
                               typename std::decay_t<BMatrixType>::scalar_type>);

  // Inner dimensions must match.
  assert(a.shape()[1] == b.shape()[0]);
  using T = typename std::decay_t<AMatrixType>::scalar_type;

  if constexpr(is_distributed_dense_matrix_v<std::decay_t<AMatrixType>> &&
               is_distributed_dense_matrix_v<std::decay_t<BMatrixType>>) {

    DMatrix<T> result({a.shape()[0], b.shape()[1]},
                      BCL::BlockCustom({a.tile_shape()[0], b.tile_shape()[1]},
                                       {a.pgrid_shape()[0], b.pgrid_shape()[1]}));
    result = 0;

    BCL::experimental::gemm(std::forward<AMatrixType>(a), std::forward<BMatrixType>(b), result);
    return result;
  } else if constexpr(is_distributed_dense_matrix_v<std::decay_t<BMatrixType>>) {
    // Sparse times dense
    DMatrix<T> result({a.shape()[0], b.shape()[1]},
                      BCL::BlockCustom({a.tile_shape()[0], b.tile_shape()[1]},
                                       {a.pgrid_shape()[0], b.pgrid_shape()[1]}));
    result = 0;

    BCL::gemm(std::forward<AMatrixType>(a), std::forward<BMatrixType>(b), result);
    return result;
    assert(a.shape()[1] == b.shape()[0]);
  } else if constexpr (is_distributed_dense_matrix_v<std::decay_t<AMatrixType>>) {
    assert(false);
    // Dense times sparse
    // (Not implemented)
  } else {
    // Sparse times sparse
    static_assert(std::is_same_v<typename std::decay_t<AMatrixType>::index_type,
                                 typename std::decay_t<BMatrixType>::index_type>);
    using I = typename std::decay_t<AMatrixType>::index_type;
    SPMatrix<T, I> result({a.shape()[0], b.shape()[1]},
                           BCL::BlockCustom({a.tile_shape()[0], b.tile_shape()[1]},
                                            {a.pgrid_shape()[0], b.pgrid_shape()[1]}));

    BCL::gemm(std::forward<AMatrixType>(a), std::forward<BMatrixType>(b), result);
    return result;
  }
  static_assert(!(is_distributed_dense_matrix_v<std::decay_t<AMatrixType>> &&
                  is_distributed_sparse_matrix_v<std::decay_t<BMatrixType>>),
                "Dense times sparse matrix multiplication not yet implemented.");
}

} // end experimental
} // end BCL
