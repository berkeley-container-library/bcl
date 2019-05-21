
#pragma once

#include <bcl/bcl.hpp>
#include <bcl/containers/DMatrix.hpp>

#include <bcl/containers/algorithms/cblas_wrapper.hpp>

#include <cassert>

namespace BCL {

template <typename T>
class DMatrix;

template <typename E>
class DExpr;

namespace experimental {

enum BlasTrans {
  notrans,
  trans
};

template <typename T, typename E, typename V>
void gemm_notrans_impl_(const BCL::DExpr<E>& a, const BCL::DExpr<V>& b, BCL::DMatrix<T>& c);

template <typename T, typename E, typename V>
void gemm(const BCL::DExpr<E>& a, const BCL::DExpr<V>& b, BCL::DMatrix<T>& c) {
  gemm_notrans_impl_(a, b, c);
}

template <typename T, typename E, typename V>
void gemm_notrans_impl_(const BCL::DExpr<E>& a, const BCL::DExpr<V>& b, BCL::DMatrix<T>& c) {
  assert(a.shape()[1] == b.shape()[0]);
  // Inner dimensions of the tiles we're multiplying must match.
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  assert(a.tile_shape()[1] == b.tile_shape()[0]);

  if (!(a.shape()[0] == c.shape()[0] && a.shape()[1] == b.shape()[0] && b.shape()[1] == c.shape()[1])) {
    throw std::runtime_error("SUMMA: ruh roh, you gave me matrices with mismatched dimensions.");
  }

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_ptr(i, j).is_local()) {

        size_t k_offset = j;
        auto buf_a = a.arget_tile(i, k_offset % a.grid_shape()[1]);
        auto buf_b = b.arget_tile(k_offset % a.grid_shape()[1], j);
        BCL::GlobalPtr<T> my_c = c.tile_ptr(i, j);
        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto my_a = std::get<0>(buf_a).get();
          auto my_b = std::get<0>(buf_b).get();

          bool a_trans = std::get<1>(buf_a);
          bool b_trans = std::get<1>(buf_b);

          if (k_+1 < a.grid_shape()[1]) {
            buf_a = a.arget_tile(i, (k+1) % a.grid_shape()[1]);
            buf_b = b.arget_tile((k+1) % a.grid_shape()[1], j);
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
          M = c.tile_shape(i, j)[0];
          N = c.tile_shape(i, j)[1];
          K = a.tile_shape(i, k)[1];

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

template <typename T>
void slicing_gemm(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
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

        size_t k_offset = j;

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
void a_owns_gemm(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c,
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
void summa(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
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
void async_summa(const BCL::DMatrix<T>& a, const BCL::DMatrix<T>& b, BCL::DMatrix<T>& c) {
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

}
}
