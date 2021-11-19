// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdio>
#include <stdexcept>

#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <numeric>

#include <bcl/core/detail/hash_functions.hpp>
#include <bcl/containers/sequential/CSRMatrix.hpp>

#include <bcl/containers/sequential/SimpleHash.hpp>

/*
 This is added for GPU usage on Summit, since the full
 SparseAccumulator.hpp requires MKL.
*/

namespace BCL {

template <
          typename T,
          typename index_type,
          typename Plus = std::plus<T>,
          typename Allocator = std::allocator<T>>
struct SparseVecHashAccumulator {
  using value_type = T;

  using allocator_traits = std::allocator_traits<Allocator>;
  using HashAllocator = typename allocator_traits:: template rebind_alloc<std::pair<index_type, T>>;

  BCL::SimpleHash<
                  index_type, T,
                  BCL::nagasaka_hash<index_type>,
                  HashAllocator
                 > values_;

  SparseVecHashAccumulator() = default;
  SparseVecHashAccumulator(const SparseVecHashAccumulator&) = default;
  SparseVecHashAccumulator& operator=(const SparseVecHashAccumulator&) = default;
  SparseVecHashAccumulator(SparseVecHashAccumulator&&) = default;
  SparseVecHashAccumulator& operator=(SparseVecHashAccumulator&&) = default;

  void accumulate(index_type idx, T value) {
    // values_[idx] += value;
    values_[idx] = Plus{}(values_[idx], value);
  }

  std::vector<std::pair<index_type, T>, HashAllocator> get() const {
    std::vector<std::pair<index_type, T>, HashAllocator> vec(values_.begin(), values_.end());

    auto lambda = [](const std::pair<index_type, T>& v1, const std::pair<index_type, T>& v2) { return v1.first < v2.first; };
    // std::sort(vec.begin(), vec.end(), [](const auto& v1, const auto& v2) { return v1.first < v2.first; });
    std::sort(vec.begin(), vec.end(), lambda);

    return vec;
  }

  void resize(size_t size) {}

  void reserve(size_t size) {
    values_.reserve(size);
  }

  void clear() {
    values_.clear();
  }

  size_t nnz() const {
    return values_.size();
  }
};

template <
          typename T,
          typename index_type = int,
          typename Plus = std::plus<T>,
          typename Allocator = std::allocator<T>
          >
struct SparseHashAccumulator {
  using value_type = T;

  using coord_type = std::pair<index_type, index_type>;

  std::vector<std::pair<BCL::CSRMatrix<T, index_type, Allocator>, coord_type>> mats_;

  SparseHashAccumulator() = default;
  SparseHashAccumulator(const SparseHashAccumulator&) = default;
  SparseHashAccumulator(SparseHashAccumulator&&) = default;

  void accumulate(BCL::CSRMatrix<T, index_type, Allocator>&& mat, const coord_type& offset = {0, 0}) {
    mats_.push_back({std::move(mat), offset});
  }

  BCL::CSRMatrix<T, index_type> get_matrix(size_t m, size_t n) {
    SparseVecHashAccumulator<T,
                             index_type,
                             Plus /*,
                             tbb::scalable_allocator<T> */> acc;

    std::vector<decltype(acc.get())> rows(m);
    #pragma omp parallel for default(shared) private(acc)
    for (size_t i = 0; i < m; i++) {
      size_t flop_estimate = 0;

      for (const auto& mat_ : mats_) {
        const auto& mat = mat_.first;
        const auto& offset = mat_.second;
        index_type bound_min = offset.first;
        index_type bound_max = bound_min + mat.shape()[0];
        if (index_type(i) >= bound_min && index_type(i) < bound_max) {
          flop_estimate += mat.row_ptr_[i+1 - bound_min] - mat.row_ptr_[i - bound_min];
        }
      }

      acc.reserve(flop_estimate);

      for (const auto& mat_ : mats_) {
        const auto& mat = mat_.first;
        const auto& offset = mat_.second;

        index_type bound_min = offset.first;
        index_type bound_max = bound_min + mat.shape()[0];
        if (index_type(i) >= bound_min && index_type(i) < bound_max) {
          for (index_type j = mat.row_ptr_[i - bound_min]; j < mat.row_ptr_[i+1 - bound_min]; j++) {
            if (mat.col_ind_[j] + offset.second >= 0 && mat.col_ind_[j] + offset.second < n) {
              acc.accumulate(mat.col_ind_[j] + offset.second, mat.vals_[j]);
            }
          }
        }
      }
      rows[i] = acc.get();
      acc.clear();
    }

    // TODO: replace with std::transform_exclusive_scan
    //       when available.
    std::vector<size_t> row_starts(rows.size());
    row_starts[0] = 0;
    for (size_t i = 1; i < row_starts.size(); i++) {
      row_starts[i] = rows[i-1].size() + row_starts[i-1];
    }

    // TODO: use partial_sum to take a prefix sum

    size_t nnz = std::accumulate(rows.begin(), rows.end(), 0,
                                 [](size_t sum, auto& row) {
                                   return sum + row.size();
                                 });
    // size_t nnz = row_starts.back() + rows.back().size();

    if (BCL::rank() == 0) {
      // printf("(%lu) scan %lf\n", BCL::rank(), duration);
    }

    BCL::CSRMatrix<T, index_type> rv(m, n, nnz);

    #pragma omp parallel for
    for (size_t i = 0; i < m; i++) {
      size_t new_i = row_starts[i];
      rv.row_ptr_[i] = new_i;
      for (size_t j = 0; j < rows[i].size(); j++) {
        rv.col_ind_[new_i] = rows[i][j].first;
        rv.vals_[new_i] = rows[i][j].second;
        new_i++;
      }
    }

    rv.row_ptr_[m] = nnz;

    if (BCL::rank() == 0) {
      // printf("%lf final\n", duration);
    }

    return rv;
  }
};

} // end BCL
