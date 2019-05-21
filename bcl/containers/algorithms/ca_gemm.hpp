
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

template <typename T, typename E, typename V, typename TeamType>
void ca_gemm(const BCL::DExpr<E>& a, const BCL::DExpr<V>& b, BCL::DMatrix<T>& c,
             const std::vector<TeamType>& teams) {
  assert(a.shape()[1] == b.shape()[0]);
  // Inner dimensions of the tiles we're multiplying must match.
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  assert(a.tile_shape()[1] == b.tile_shape()[0]);

  if (!(a.shape()[0] == c.shape()[0] && a.shape()[1] == b.shape()[0] && b.shape()[1] == c.shape()[1])) {
    throw std::runtime_error("SUMMA: ruh roh, you gave me matrices with mismatched dimensions.");
  }

  std::vector<BCL::DMatrix<T>> c_acc;

  for (size_t i = 0; i < teams.size(); i++) {
    c_acc.emplace_back(std::move(c.apply([](T value) -> T { return 0.0; })));
  }

  size_t num_teams = teams.size();
  size_t sub_muls = a.grid_shape()[1];

  size_t my_team;
  for (size_t i = 0; i < teams.size(); i++) {
    if (teams[i].in_team()) {
      my_team = i;
    }
  }

  size_t mul_per_team = (sub_muls + num_teams - 1) / num_teams;

  size_t mul_start = std::min(mul_per_team * my_team, sub_muls);
  size_t mul_end = std::min(mul_per_team * (my_team+1), sub_muls);

  std::vector<BCL::future<std::vector<T>>> futures;

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.team_ptr_->resolve(c.submatrix(i, j).rank) == BCL::rank(teams[my_team])) {

        size_t k_offset = j;
        auto buf_a = a.arget_submatrix(i, (mul_start + k_offset) % a.grid_shape()[1]);
        auto buf_b = b.arget_submatrix((mul_start + k_offset) % a.grid_shape()[1], j);
        std::vector<T> my_c(c.tile_size(), 0);
        for (size_t k_ = mul_start; k_ < mul_end; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto my_a = std::get<0>(buf_a).get();
          auto my_b = std::get<0>(buf_b).get();

          bool a_trans = std::get<1>(buf_a);
          bool b_trans = std::get<1>(buf_b);

          if (k_+1 < a.grid_shape()[1]) {
            buf_a = a.arget_submatrix(i, (k+1) % a.grid_shape()[1]);
            buf_b = b.arget_submatrix((k+1) % a.grid_shape()[1], j);
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
                              my_c.data(), ldc);
        }
        futures.emplace_back(BCL::arput(c_acc[my_team].submatrix(i, j), std::move(my_c)));
      }
    }
  }
  for (auto& future : futures) {
    future.get();
  }
  futures.clear();
  BCL::barrier();

  for (auto& mat : c_acc) {
    c += mat;
  }
}

template <typename T, typename TeamType>
void ca_gemm(const std::vector<BCL::DMatrix<T>>& as, const std::vector<BCL::DMatrix<T>>& bs, BCL::DMatrix<T>& c,
             const std::vector<TeamType>& teams) {
  size_t num_teams = teams.size();
  size_t sub_muls = as[0].grid_shape()[1];

  size_t my_team;
  for (size_t i = 0; i < teams.size(); i++) {
    if (teams[i].in_team()) {
      my_team = i;
    }
  }

  size_t mul_per_team = (sub_muls + num_teams - 1) / num_teams;

  size_t mul_start = std::min(mul_per_team * my_team, sub_muls);
  size_t mul_end = std::min(mul_per_team * (my_team+1), sub_muls);

  const auto& a = as[my_team];
  const auto& b = bs[my_team];

  assert(a.shape()[1] == b.shape()[0]);
  // Inner dimensions of the tiles we're multiplying must match.
  assert(a.grid_shape()[1] == b.grid_shape()[0]);
  assert(a.tile_shape()[1] == b.tile_shape()[0]);

  if (!(a.shape()[0] == c.shape()[0] && a.shape()[1] == b.shape()[0] && b.shape()[1] == c.shape()[1])) {
    throw std::runtime_error("SUMMA: ruh roh, you gave me matrices with mismatched dimensions.");
  }

  std::vector<BCL::DMatrix<T>> c_acc;

  auto begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < a.grid_shape()[1]; i++) {
    c_acc.emplace_back(std::move(c.apply([](T value) -> T { return 0.0; })));
  }
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  BCL::print("%lf initializing accumulators\n", duration);

  std::vector<BCL::future<std::vector<T>>> futures;

  begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.team_ptr_->resolve(c.submatrix(i, j).rank) == BCL::rank(teams[my_team])) {

        size_t k_offset = j;
        auto buf_a = a.arget_submatrix(i, (mul_start + k_offset) % a.grid_shape()[1]);
        auto buf_b = b.arget_submatrix((mul_start + k_offset) % a.grid_shape()[1], j);
        for (size_t k_ = mul_start; k_ < mul_end; k_++) {
          std::vector<T> my_c(c.tile_size(), 0);
          size_t k = (k_ + k_offset) % a.grid_shape()[1];

          auto my_a = std::get<0>(buf_a).get();
          auto my_b = std::get<0>(buf_b).get();

          bool a_trans = std::get<1>(buf_a);
          bool b_trans = std::get<1>(buf_b);

          if (k_+1 < mul_end) {
            buf_a = a.arget_submatrix(i, (k+1) % a.grid_shape()[1]);
            buf_b = b.arget_submatrix((k+1) % a.grid_shape()[1], j);
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
                              my_c.data(), ldc);

          futures.emplace_back(BCL::arput(c_acc[k].submatrix(i, j), std::move(my_c)));
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  BCL::print("%lf computing.\n", duration);
  begin = std::chrono::high_resolution_clock::now();
  for (auto& future : futures) {
    future.get();
  }
  futures.clear();
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  BCL::print("%lf syncing futures.\n", duration);
  BCL::barrier();

  begin = std::chrono::high_resolution_clock::now();
  for (auto& mat : c_acc) {
    c += mat;
  }
  end = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  BCL::print("%lf accumulating.\n", duration);
}

}
}
