#pragma once

#include <cstdio>
#include <stdexcept>

#include <unordered_map>
// #include <experimental/numeric>
#include <unordered_set>
#include <queue>
#include <numeric>

#include <bcl/core/detail/hash_functions.hpp>
#include <bcl/containers/detail/mkl/mkl_error_handle.hpp>

#include <bcl/containers/sequential/SimpleHash.hpp>

#include <mkl.h>

namespace BCL {

// An Eager Sparse Accumulator
template <typename T, typename index_type = int>
struct SparseAccumulator {
  using coord_type = std::pair<index_type, index_type>;
  using value_type = T;

  std::unordered_map<coord_type, value_type, djb2_hash<coord_type>> vals_;

  SparseAccumulator() = default;
  SparseAccumulator(const SparseAccumulator&) = delete;
  SparseAccumulator(SparseAccumulator&&) = default;

  void accumulate(const coord_type& coords, const value_type& value) {
    vals_[coords] += value;
  }

  template <typename CSRType>
  void accumulate(const CSRType& csr_mat,
                  const coord_type& offset = {0, 0}) {
    for (size_t i = 0; i < csr_mat.m_; i++) {
      for (index_type j = csr_mat.row_ptr_[i]; j < csr_mat.row_ptr_[i+1]; j++) {
        accumulate({std::get<0>(offset) + i,
                    std::get<1>(offset) + csr_mat.col_ind_[j]},
                   csr_mat.vals_[j]);
      }
    }
  }

  template <typename Allocator = std::allocator<T>>
  BCL::CSRMatrix<T, index_type, Allocator> get_matrix(size_t m, size_t n) const {
    if (vals_.empty()) {
      return BCL::CSRMatrix<T, index_type, Allocator>(m, n);
    }
    std::vector<std::pair<coord_type, value_type>> elems(vals_.begin(), vals_.end());
    std::sort(elems.begin(), elems.end());

    std::vector<index_type> rows;
    std::vector<index_type> cols;
    std::vector<value_type> vals;
    rows.reserve(elems.size());
    cols.reserve(elems.size());
    vals.reserve(elems.size());
    for (const auto& elem : elems) {
      rows.push_back(std::get<0>(std::get<0>(elem)));
      cols.push_back(std::get<1>(std::get<0>(elem)));
      vals.push_back(std::get<1>(elem));
    }

    sparse_matrix_t c_coo, c_csr;
    auto status = mkl_sparse_s_create_coo(&c_coo, SPARSE_INDEX_BASE_ZERO,
                            m, n, vals.size(),
                            rows.data(), cols.data(), vals.data());
    mkl_error_handle(status, "create COO");
    status = mkl_sparse_convert_csr(c_coo, SPARSE_OPERATION_NON_TRANSPOSE, &c_csr);
    mkl_error_handle(status, "Convert CSR");

    BCL::CSRMatrix<value_type, index_type, Allocator> c_mat(c_csr);

    mkl_sparse_destroy(c_coo);
    mkl_sparse_destroy(c_csr);

    return c_mat;
  }

  void reserve(size_t count) {
    vals_.reserve(count);
  }
};

template <typename T, typename index_type = int>
struct LazyAccumulator {
  using coord_type = std::pair<index_type, index_type>;
  using value_type = T;

  using matrix_type = BCL::CSRMatrix<T, index_type, BCL::bcl_allocator<T>>;
  using lazy_matrix_type = std::pair<coord_type, matrix_type>;

  std::unordered_map<coord_type, value_type, djb2_hash<coord_type>> vals_;
  std::vector<lazy_matrix_type> lazy_matrices_;

  LazyAccumulator() = default;
  LazyAccumulator(const LazyAccumulator&) = delete;
  LazyAccumulator(LazyAccumulator&&) = default;

  void accumulate(const coord_type& coords, const value_type& value) {
    vals_[coords] += value;
  }

  void accumulate(matrix_type&& csr_mat,
                  const coord_type& offset = {0, 0}) {
    lazy_matrices_.emplace_back(std::move(std::make_pair(offset, std::move(csr_mat))));
  }

  template <typename CSRType>
  void eagerly_accumulate(const CSRType& csr_mat,
                          const coord_type& offset = {0, 0}) {
    for (size_t i = 0; i < csr_mat.m_; i++) {
      for (index_type j = csr_mat.row_ptr_[i]; j < csr_mat.row_ptr_[i+1]; j++) {
        accumulate({std::get<0>(offset) + i,
                    std::get<1>(offset) + csr_mat.col_ind_[j]},
                   csr_mat.vals_[j]);
      }
    }
  }

  void finish_accumulation_() {
    for (const auto& matrix : lazy_matrices_) {
      const matrix_type& csr_mat = std::get<1>(matrix);
      const coord_type& offset = std::get<0>(matrix);
      for (size_t i = 0; i < csr_mat.m_; i++) {
        for (index_type j = csr_mat.row_ptr_[i]; j < csr_mat.row_ptr_[i+1]; j++) {
          accumulate({std::get<0>(offset) + i,
                      std::get<1>(offset) + csr_mat.col_ind_[j]},
                     csr_mat.vals_[j]);
        }
      }
    }
    lazy_matrices_.clear();
  }

  template <typename Allocator = std::allocator<T>>
  BCL::CSRMatrix<T, index_type, Allocator> get_matrix(size_t m, size_t n) {
    finish_accumulation_();
    if (vals_.empty()) {
      return BCL::CSRMatrix<T, index_type, Allocator>(m, n);
    }
    std::vector<std::pair<coord_type, value_type>> elems(vals_.begin(), vals_.end());
    std::sort(elems.begin(), elems.end());

    std::vector<index_type> rows;
    std::vector<index_type> cols;
    std::vector<value_type> vals;
    rows.reserve(elems.size());
    cols.reserve(elems.size());
    vals.reserve(elems.size());
    for (const auto& elem : elems) {
      rows.push_back(std::get<0>(std::get<0>(elem)));
      cols.push_back(std::get<1>(std::get<0>(elem)));
      vals.push_back(std::get<1>(elem));
    }

    sparse_matrix_t c_coo, c_csr;
    auto status = mkl_sparse_s_create_coo(&c_coo, SPARSE_INDEX_BASE_ZERO,
                            m, n, vals.size(),
                            rows.data(), cols.data(), vals.data());
    mkl_error_handle(status, "create COO");
    status = mkl_sparse_convert_csr(c_coo, SPARSE_OPERATION_NON_TRANSPOSE, &c_csr);
    mkl_error_handle(status, "Convert CSR");

    BCL::CSRMatrix<value_type, index_type, Allocator> c_mat(c_csr);

    mkl_sparse_destroy(c_coo);
    mkl_sparse_destroy(c_csr);

    return c_mat;
  }
};

// An Eager Sparse Accumulator
template <typename T, typename index_type = int>
struct EagerSumAccumulator {
  using coord_type = std::pair<index_type, index_type>;
  using value_type = T;

  BCL::CSRMatrix<T, index_type, std::allocator<T>> mat_;
  bool filled_ = false;

  // XXX: intentionally creating an invalid CSRMatrix
  //      (it will be overwritten)
  EagerSumAccumulator() : mat_(1, 1) {}
  EagerSumAccumulator(size_t m, size_t n) : mat_(m, n) {}
  EagerSumAccumulator(const EagerSumAccumulator&) = delete;
  EagerSumAccumulator(EagerSumAccumulator&&) = default;

  template <typename CSRType>
  void accumulate(CSRType& csr_mat) {
    if (filled_) {
      mat_ = mat_ + csr_mat;
    } else {
      mat_ = csr_mat;
      filled_ = true;
    }
  }

  auto get_matrix(size_t m, size_t n) const {
    if (!filled_) {
      return BCL::CSRMatrix<T, index_type, std::allocator<T>>(m, n);
    } else {
      return mat_;
    }
  }
};

// An Eager Sparse Accumulator
template <typename T, typename index_type = int>
struct EagerMKLAccumulator {
  using coord_type = std::pair<index_type, index_type>;
  using value_type = T;

  BCL::mkl::spmatrix<T> mat_;
  bool filled_ = false;

  // XXX: intentionally creating an invalid CSRMatrix
  //      (it will be overwritten)
  EagerMKLAccumulator() : mat_(1, 1) {}
  EagerMKLAccumulator(size_t m, size_t n) : mat_(m, n) {}
  EagerMKLAccumulator(const EagerMKLAccumulator&) = delete;
  EagerMKLAccumulator(EagerMKLAccumulator&&) = default;

  template <typename CSRType>
  void accumulate(const CSRType& csr_mat) {
    if (filled_) {
      mat_ = mat_ + csr_mat;
    } else {
      mat_ = csr_mat;
      filled_ = true;
    }
  }

  auto get_matrix(size_t m, size_t n) const {
    if (!filled_) {
      return BCL::CSRMatrix<T, index_type, std::allocator<T>>(m, n);
    } else {
      //if (mat_.has_matrix_) {
        return BCL::CSRMatrix<T, index_type, std::allocator<T>>(mat_.matrix_);
        /*
      } else {
        return BCL::CSRMatrix<T, index_type, std::allocator<T>>(mat_.m_, mat_.n_);
      }
      */
    }
  }
};

template <typename T, typename index_type, typename Allocator = std::allocator<T>>
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
    values_[idx] += value;
  }

  std::vector<std::pair<index_type, T>, HashAllocator> get() const {
    std::vector<std::pair<index_type, T>, HashAllocator> vec(values_.begin(), values_.end());

    std::sort(vec.begin(), vec.end(), [](const auto& v1, const auto& v2) { return v1.first < v2.first; });

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

template <typename T,
          typename index_type = int,
          typename Allocator = std::allocator<T>>
struct SparseVecAccumulator {
  using value_type = T;

  using allocator_traits = std::allocator_traits<Allocator>;
  using IAllocator = typename allocator_traits:: template rebind_alloc<index_type>;
  using BAllocator = typename allocator_traits:: template rebind_alloc<bool>;
  using PairAllocator = typename allocator_traits:: template rebind_alloc<std::pair<index_type, T>>;

  size_t size_ = 0;
  size_t nnz_ = 0;
  std::vector<T, Allocator> values_;
  std::vector<index_type, IAllocator> indices_;
  std::vector<bool, BAllocator> nonzeros_;

  SparseVecAccumulator(size_t size)
     : size_(size), nnz_(0), values_(size), nonzeros_(size, false) {}

  SparseVecAccumulator() = default;
  SparseVecAccumulator(const SparseVecAccumulator&) = default;
  SparseVecAccumulator& operator=(const SparseVecAccumulator&) = default;

  void accumulate(index_type idx, T value) {
    if (!nonzeros_[idx]) {
      nonzeros_[idx] = true;
      indices_.push_back(idx);
      values_[idx] = value;
      nnz_++;
    } else {
      values_[idx] += value;
    }
  }

  std::vector<std::pair<index_type, T>, PairAllocator> get() {
    std::vector<std::pair<index_type, T>, PairAllocator> vec(nnz());

    std::sort(indices_.begin(), indices_.end());

    for (size_t i = 0; i < nnz(); i++) {
      vec[i] = {indices_[i], values_[indices_[i]]};
    }

    return std::move(vec);
  }

  // XXX: Shrinking not currently supported.
  void resize(size_t size) {
    if (size > size_) {
      values_.resize(size);
      nonzeros_.resize(size, false);
      size_ = size;
    }
  }

  void reserve(size_t count) {}

  void clear() {
    for (size_t i = 0; i < nnz(); i++) {
      nonzeros_[indices_[i]] = false;
    }
    indices_.clear();
    nnz_ = 0;
  }

  size_t nnz() const {
    return nnz_;
  }
};

// TODO: refactor the "SparseAccumulator" concept
//       handle signed/unsigned integer stuff.

template <typename T, typename index_type = int, typename Allocator = std::allocator<T>>
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
    auto begin = std::chrono::high_resolution_clock::now();
    SparseVecHashAccumulator<T,
                             index_type,
                             tbb::scalable_allocator<T>> acc;

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

    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - begin).count();

    // TODO: replace with std::transform_exclusive_scan
    //       when available.
    begin = std::chrono::high_resolution_clock::now();
    std::vector<size_t> row_starts(rows.size());
    row_starts[0] = 0;
    for (size_t i = 1; i < row_starts.size(); i++) {
      row_starts[i] = rows[i-1].size() + row_starts[i-1];
    }
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(end - begin).count();

    // TODO: use partial_sum to take a prefix sum

    begin = std::chrono::high_resolution_clock::now();
    size_t nnz = std::accumulate(rows.begin(), rows.end(), 0,
                                 [](size_t sum, auto& row) {
                                   return sum + row.size();
                                 });
    // size_t nnz = row_starts.back() + rows.back().size();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("(%lu) scan %lf\n", BCL::rank(), duration);
    }

    begin = std::chrono::high_resolution_clock::now();

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

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("%lf final\n", duration);
    }

    return rv;
  }
};


template <typename T, typename index_type = int, typename Allocator = std::allocator<T>>
struct SparseSPAAccumulator {
  using value_type = T;

  using coord_type = std::pair<index_type, index_type>;

  std::vector<std::pair<BCL::CSRMatrix<T, index_type, Allocator>, coord_type>> mats_;

  SparseSPAAccumulator() = default;
  SparseSPAAccumulator(const SparseSPAAccumulator&) = default;
  SparseSPAAccumulator(SparseSPAAccumulator&&) = default;

  void accumulate(BCL::CSRMatrix<T, index_type, Allocator>&& mat, const coord_type& offset = {0, 0}) {
    mats_.push_back({std::move(mat), offset});
  }

  BCL::CSRMatrix<T, index_type> get_matrix(size_t m, size_t n) {
    auto begin = std::chrono::high_resolution_clock::now();
    SparseVecAccumulator<T,
                         index_type,
                         tbb::scalable_allocator<T>> acc;

    std::vector<decltype(acc.get())> rows(m);
    #pragma omp parallel for default(shared) private(acc)
    for (size_t i = 0; i < m; i++) {

      acc.resize(n);

      for (const auto& mat_ : mats_) {
        const auto& mat = mat_.first;
        const auto& offset = mat_.second;

        size_t bound_min = offset.first;
        size_t bound_max = bound_min + mat.shape()[0];
        if (i >= bound_min && i < bound_max) {
          for (index_type j = mat.row_ptr_[i - bound_min]; j < mat.row_ptr_[i+1 - bound_min]; j++) {
            acc.accumulate(mat.col_ind_[j] + offset.second, mat.vals_[j]);
          }
        }
      }
      rows[i] = acc.get();
      acc.clear();
    }

    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("(%lu) accumulate %lf\n", BCL::rank(), duration);
    }

    // TODO: replace with std::transform_exclusive_scan
    //       when available.
    begin = std::chrono::high_resolution_clock::now();
    std::vector<size_t> row_starts(rows.size());
    row_starts[0] = 0;
    for (size_t i = 1; i < row_starts.size(); i++) {
      row_starts[i] = rows[i-1].size() + row_starts[i-1];
    }
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(end - begin).count();

    // TODO: use partial_sum to take a prefix sum

    begin = std::chrono::high_resolution_clock::now();
    size_t nnz = std::accumulate(rows.begin(), rows.end(), 0,
                                 [](size_t sum, auto& row) {
                                   return sum + row.size();
                                 });
    // size_t nnz = row_starts.back() + rows.back().size();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("(%lu) scan %lf\n", BCL::rank(), duration);
    }

    begin = std::chrono::high_resolution_clock::now();

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

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("%lf final\n", duration);
    }

    return rv;
  }
};

template <typename T, typename index_type = int, typename Allocator = std::allocator<T>>
struct SparseHeapAccumulator {
  using value_type = T;

  using coord_type = std::pair<index_type, index_type>;

  std::vector<std::pair<BCL::CSRMatrix<T, index_type, Allocator>, coord_type>> mats_;

  SparseHeapAccumulator() = default;
  SparseHeapAccumulator(const SparseHeapAccumulator&) = default;
  SparseHeapAccumulator(SparseHeapAccumulator&&) = default;

  void accumulate(BCL::CSRMatrix<T, index_type, Allocator>&& mat, const coord_type& offset = {0, 0}) {
    mats_.push_back({std::move(mat), offset});
  }

  struct Entry {
    T value_;
    index_type index_;
    size_t matrix_id_;

    Entry(const Entry&) = default;
    Entry& operator=(const Entry&) = default;

    bool operator<(const Entry& other) const noexcept {
      return index_ < other.index_;
    }

    bool operator>(const Entry& other) const noexcept {
      return index_ > other.index_;
    }
  };

  BCL::CSRMatrix<T, index_type> get_matrix(size_t m, size_t n) {
    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::pair<index_type, T>,
                            tbb::scalable_allocator<std::pair<index_type, T>>
                            >
                > rows(m);

    #pragma omp parallel for default(shared)
    for (size_t i = 0; i < m; i++) {
      std::priority_queue<Entry,
                          std::vector<Entry,
                                      tbb::scalable_allocator<Entry>
                                      >,
                          std::greater<Entry>> entries;
      std::vector<index_type,
                  tbb::scalable_allocator<index_type>
                  > row_indices_(mats_.size());

      for (size_t mat_id = 0; mat_id < mats_.size(); mat_id++) {
        const auto& mat = mats_[mat_id].first;
        index_type& row_index = row_indices_[mat_id];
        row_index = mat.row_ptr_[i];
        if (row_index++ < mat.row_ptr_[i+1]) {
          entries.push(Entry{mat.vals_[row_index-1],
                             mat.col_ind_[row_index-1],
                             mat_id});
        }
      }

      while (!entries.empty()) {
        Entry value = entries.top();
        entries.pop();

        index_type& row_index = row_indices_[value.matrix_id_];
        const auto& mat = mats_[value.matrix_id_].first;

        if (row_index++ < mat.row_ptr_[i+1]) {
          entries.push(Entry{mat.vals_[row_index-1],
                             mat.col_ind_[row_index-1],
                             value.matrix_id_});
        }

        if (rows[i].empty() || value.index_ > rows[i].back().first) {
          rows[i].push_back({value.index_, value.value_});
        } else if (value.index_ == rows[i].back().first) {
          rows[i].back().second += value.value_;
        } else {
          throw std::runtime_error("SparseHeapAccumulator requires sorted CSR.  Index " +
                                   std::to_string(value.index_) + " < " +
                                   std::to_string(rows[i].back().first));
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("(%lu) accumulate %lf\n", BCL::rank(), duration);
    }

    // TODO: replace with std::transform_exclusive_scan
    //       when available.
    begin = std::chrono::high_resolution_clock::now();
    std::vector<size_t> row_starts(rows.size());
    row_starts[0] = 0;
    for (size_t i = 1; i < row_starts.size(); i++) {
      row_starts[i] = rows[i-1].size() + row_starts[i-1];
    }
    end = std::chrono::high_resolution_clock::now();

    duration = std::chrono::duration<double>(end - begin).count();

    // TODO: use partial_sum to take a prefix sum

    begin = std::chrono::high_resolution_clock::now();
    size_t nnz = std::accumulate(rows.begin(), rows.end(), 0,
                                 [](size_t sum, auto& row) {
                                   return sum + row.size();
                                 });
    // size_t nnz = row_starts.back() + rows.back().size();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("(%lu) scan %lf\n", BCL::rank(), duration);
    }

    begin = std::chrono::high_resolution_clock::now();

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

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();

    if (BCL::rank() == 0) {
      // printf("%lf final\n", duration);
    }

    return rv;
  }
};


/*
// An accumulator using CombBLAS
template <typename T, typename index_type = int>
struct CombBLASAccumulator {
  using value_type = T;

  using tuple_type = std::tuple<index_type, index_type, T>;
  using coo_t = std::vector<tuple_type>;

  size_t nnz_ = 0;

  std::vector<coo_t> mats_;
  std::vector<combblas::SpTuples<index_type, T>*> cb_mats_;

  // XXX: intentionally creating an invalid CSRMatrix
  //      (it will be overwritten)
  CombBLASAccumulator() = default;
  CombBLASAccumulator(size_t m, size_t n) {}
  CombBLASAccumulator(const CombBLASAccumulator&) = delete;
  CombBLASAccumulator(CombBLASAccumulator&&) = default;

  ~CombBLASAccumulator() {
    // XXX: not currently deleting cb_mat;
    //      it attempts to delete the pointer
    //      to a coo_t, which it thinks it owns.
    for (auto& cb_mat : cb_mats_) {
      // delete cb_mat;
    }
  }

  template <typename CSRType>
  void accumulate(const CSRType& csr_mat) {
    mats_.push_back(csr_mat.get_combblas_coo());

    nnz_ += csr_mat.nnz_;
    cb_mats_.push_back(new combblas::SpTuples<index_type, T>(csr_mat.nnz_, csr_mat.n_, csr_mat.m_,
                                                             mats_.back().data(), true, false));
  }

  // Adaptor to transform any accumulator functor `Accumulator`
  // into a CombBLAS accumulator class.
  template <typename Accumulator>
  struct cb_accumulator {
    template <typename T, typename U>
    static auto add(const T& a, const U& b) noexcept
    -> decltype(std::declval<Accumulator>()(std::declval<T>(), std::declval<U>()))
    {
      return Accumulator{}(a, b);
    }
  };

  template <typename Allocator = std::allocator<T>>
  auto get_matrix(size_t m, size_t n) {
    if (nnz_ <= 0) {
      return BCL::CSRMatrix<T, index_type, Allocator>(m, n);
    }

    static_assert(std::is_same<Allocator, std::allocator<T>>::value);

    auto begin = std::chrono::high_resolution_clock::now();
    auto output_mat = combblas::MultiwayMerge<cb_accumulator<std::plus<T>>,
                                              index_type,
                                              T>(cb_mats_, n, m);
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    if (BCL::rank() == 0) {
      // printf("(%lu) %lf MultiwayMerge\n", BCL::rank(), duration);
    }
    begin = std::chrono::high_resolution_clock::now();
    output_mat->SortColBased();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();
    if (BCL::rank() == 0) {
      // printf("(%lu) %lf COO Sort\n", BCL::rank(), duration);
      // printf("Out of multi-way merge! got %lld nnnz\n", output_mat->getnnz());
      // fflush(stdout);
    }

    begin = std::chrono::high_resolution_clock::now();
    std::vector<index_type> row_ptr_(m+1);
    std::vector<T> vals_(output_mat->getnnz());
    std::vector<index_type> col_ind_(output_mat->getnnz());

    size_t rp = 0;
    row_ptr_[rp] = 0;
    for (size_t i = 0; i < output_mat->getnnz(); i++) {
      while (rp < output_mat->colindex(i)) {
        row_ptr_[rp+1] = i;
        rp++;
      }
      vals_[i] = output_mat->numvalue(i);
      col_ind_[i] = output_mat->rowindex(i);
    }

    for ( ; rp < m; rp++) {
      row_ptr_[rp+1] = output_mat->getnnz();
    }

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(end - begin).count();

    // printf("(%lu) %lf custom COO -> CSR\n", BCL::rank(),
    //        duration);
    //        fflush(stdout);

    delete output_mat;

    return CSRMatrix<value_type, index_type, Allocator>(m, n, vals_.size(),
                     std::move(vals_), std::move(row_ptr_), std::move(col_ind_));
  }
};
*/

}
