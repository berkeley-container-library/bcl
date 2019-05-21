#pragma once

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <regex.h>
#include <cassert>
#include <array>
#include <memory>
#include <limits>
#include <algorithm>

#include <vector>

#include <unistd.h>

// TODO: move this unordered_map code out of here
//       it's really ugly
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/detail/mkl/mkl_error_handle.hpp>
#include <bcl/containers/detail/mkl/spmatrix.hpp>

#include <mtspgemmlib/utility.h>
#include <mtspgemmlib/CSC.h>
#include <mtspgemmlib/CSR.h>
#include <mtspgemmlib/multiply.h>
#include <mtspgemmlib/heap_mult.h>
#include <mtspgemmlib/hash_mult_hw.h>

#include <mkl.h>

namespace BCL {

template <typename T, typename index_type>
struct SparseAccumulator;

enum FileFormat {
  MatrixMarket,
  MatrixMarketZeroIndexed,
  Binary,
  Unknown
};

template <
          typename T,
          typename index_t = MKL_INT,
          typename Allocator = std::allocator<T>
          >
struct CSRMatrix {

  using size_type = size_t;
  using value_type = T;
  using index_type = index_t;

  using allocator_traits = std::allocator_traits<Allocator>;
  using IAllocator = typename allocator_traits:: template rebind_alloc<index_type>;

  size_type m_;
  size_type n_;
  size_type nnz_;

  std::vector<T, Allocator> vals_;
  std::vector<index_type, IAllocator> row_ptr_;
  std::vector<index_type, IAllocator> col_ind_;

  CSRMatrix(const CSRMatrix&) = default;
  CSRMatrix(CSRMatrix&&) = default;
  CSRMatrix& operator=(CSRMatrix&& m) = default;
  CSRMatrix& operator=(const CSRMatrix& m) = default;

  template <typename A>
  CSRMatrix& operator=(const CSRMatrix<T, index_t, A>& other) {
    m_ = other.m_;
    n_ = other.n_;
    nnz_ = other.nnz_;
    vals_.resize(other.vals_.size());
    vals_.assign(other.vals_.begin(), other.vals_.end());
    row_ptr_.resize(other.row_ptr_.size());
    row_ptr_.assign(other.row_ptr_.begin(), other.row_ptr_.end());
    col_ind_.resize(other.col_ind_.size());
    col_ind_.assign(other.col_ind_.begin(), other.col_ind_.end());
    return *this;
  }

  bool operator==(const CSRMatrix& other) const {
    if (m_ != other.m_ || n_ != other.n_ || nnz_ != other.nnz_) {
      return false;
    }

    if (vals_ != other.vals_) {
      return false;
    }

    if (row_ptr_ != other.row_ptr_) {
      return false;
    }

    if (col_ind_ != other.col_ind_) {
      return false;
    }

    return true;
  }

  std::array<size_type, 2> shape() const noexcept {
    return {m_, n_};
  }

  void read_MatrixMarket(const std::string& fname, bool one_indexed = true);
  void read_Binary(const std::string& fname);
  void write_Binary(const std::string& fname);

  CSRMatrix(size_type m, size_type n, size_type nnz, std::vector<T, Allocator>&& vals,
            std::vector<index_type, IAllocator>&& row_ptr,
            std::vector<index_type, IAllocator>&& col_ind)
              : m_(m), n_(n), nnz_(nnz), vals_(std::move(vals)),
                row_ptr_(std::move(row_ptr)), col_ind_(std::move(col_ind)) {}

  CSRMatrix(size_type m, size_type n, size_type nnz) : m_(m), n_(n), nnz_(nnz) {
    vals_.resize(nnz_);
    col_ind_.resize(nnz_);
    row_ptr_.resize(m_+1);
  }

  CSRMatrix(size_type m, size_type n) : m_(m), n_(n), nnz_(0) {
    row_ptr_.resize(m+1, 0);
  }

  CSRMatrix(const std::string& fname, FileFormat format = FileFormat::MatrixMarket) {
    if (format == FileFormat::MatrixMarket) {
      read_MatrixMarket(fname);
    } else if (format == FileFormat::MatrixMarketZeroIndexed) {
      read_MatrixMarket(fname, false);
    } else if (format == FileFormat::Binary) {
      read_Binary(fname);
    }
  }

  template <typename Permuter>
  CSRMatrix permute(const Permuter& permuter) const {
    assert(m_ == n_);
    BCL::SparseAccumulator<T, index_type> acc;

    for (size_t i = 0 ; i < m_; i++) {
      for (size_t j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
        acc.accumulate({permuter[i], permuter[col_ind_[j]]}, vals_[j]);
      }
    }
    return acc.get_matrix(m_, n_);
  }

  // Return a CSRMatrix representing the submatrix at coordinates [imin, imax), [jmin, jmax)
  CSRMatrix get_slice_impl_(size_type imin, size_type imax, size_type jmin,
                            size_type jmax) {
    std::vector<T, Allocator> vals;
    std::vector<index_type, IAllocator> row_ptr;
    std::vector<index_type, IAllocator> col_ind;

    imin = std::max(imin, size_type(0));
    imax = std::min(imax, m_);
    jmax = std::min(jmax, n_);
    jmin = std::max(jmin, size_type(0));

    size_type m = imax - imin;
    size_type n = jmax - jmin;

    row_ptr.resize(m+1);

    assert(imin <= imax && jmin <= jmax);

    // TODO: there's an early exit possible when
    //       column indices are sorted.

    size_type new_i = 0;
    for (size_type i = imin; i < imax; i++) {
      row_ptr[i - imin] = new_i;
      for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
        if (col_ind_[j] >= jmin && col_ind_[j] < jmax) {
          vals.push_back(vals_[j]);
          col_ind.push_back(col_ind_[j] - jmin);
          new_i++;
        }
      }
    }

    size_type nnz = vals.size();
    row_ptr[m] = nnz;

    return CSRMatrix(m, n, nnz, std::move(vals), std::move(row_ptr),
                     std::move(col_ind));
  }

  void print();
  void print_details();
  size_t count_nnz();
  size_t count_indices();
  void verify();
  T count_values();

  template <typename U>
  void print_vec_(const std::vector<U>& vec, const std::string& lbl) const {
    std::cout << lbl << ":";
    for (const auto& v : vec) {
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }

  size_t count_nonzeros_() const {
    size_t num_nonzeros = 0;
    for (size_t i = 0; i < m_; i++) {
      for (size_t j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
        num_nonzeros++;
      }
    }
    return num_nonzeros;
  }

  template <typename ITYPE, typename HashType>
  void accumulate_me_to_map_(std::unordered_map<std::pair<ITYPE, ITYPE>, T, HashType>& map) const {
    for (size_type i = 0; i < m_; i++) {
      for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
        map[std::make_tuple(i, col_ind_[j])] = vals_[j];
      }
    }
  }

  CSRMatrix(sparse_matrix_t mat_descr) {
    sparse_index_base_t indexing;
    MKL_INT rows, cols;
    MKL_INT *rows_start, *rows_end, *col_ind;
    T* vals;
    auto status = mkl_sparse_s_export_csr(mat_descr, &indexing, &rows, &cols,
                            &rows_start, &rows_end, &col_ind, &vals);

    mkl_error_handle(status, "Export C");

    size_t ind = (indexing == SPARSE_INDEX_BASE_ZERO) ? 0 : 1;
    m_ = rows;
    n_ = cols;

    row_ptr_.resize(m_+1);

    nnz_ = 0;
    for (size_t i = 0; i < rows; i++) {
      row_ptr_[i] = rows_start[i-ind]-ind;
      row_ptr_[i+1] = rows_end[i-ind]-ind;
      for (MKL_INT j = rows_start[i-ind]-ind; j < rows_end[i-ind]-ind; j++) {
        col_ind_.push_back(col_ind[j-ind]-ind);
        vals_.push_back(vals[j-ind]-ind);
        nnz_++;
      }
    }
  }

  CSRMatrix dot(CSRMatrix& b) {
    CSRMatrix& a = *this;

    if (a.nnz_ == 0 || b.nnz_ == 0) {
      return CSRMatrix(a.m_, b.n_);
    }

    // do a*b

    CSR<index_type, T> a_mat, b_mat, c_mat;


    a_mat.rows = a.m_;
    a_mat.cols = a.n_;
    a_mat.nnz = a.nnz_;
    a_mat.rowptr = a.row_ptr_.data();
    a_mat.colids = a.col_ind_.data();
    a_mat.values = a.vals_.data();
    a_mat.zerobased = true;

    b_mat.rows = b.m_;
    b_mat.cols = b.n_;
    b_mat.nnz = b.nnz_;
    b_mat.rowptr = b.row_ptr_.data();
    b_mat.colids = b.col_ind_.data();
    b_mat.values = b.vals_.data();
    b_mat.zerobased = true;

    HashSpGEMM<false, true>(a_mat, b_mat, c_mat, std::multiplies<T>(), std::plus<T>());

    CSRMatrix c(c_mat.rows, c_mat.cols, c_mat.nnz);

    c.row_ptr_.assign(c_mat.rowptr, c_mat.rowptr + c.row_ptr_.size());
    c.col_ind_.assign(c_mat.colids, c_mat.colids + c.col_ind_.size());
    c.vals_.assign(c_mat.values, c_mat.values + c.vals_.size());

    a_mat.nnz = 0;
    a_mat.rows = 0;
    a_mat.rowptr = nullptr;
    a_mat.colids = nullptr;
    a_mat.values = nullptr;
    b_mat.nnz = 0;
    b_mat.rows = 0;
    b_mat.rowptr = nullptr;
    b_mat.colids = nullptr;
    b_mat.values = nullptr;

    return c;

    /*
    sparse_matrix_t a_descr, b_descr;
    sparse_matrix_t c_descr = NULL;

    using a_type = typename std::remove_reference<decltype(a)>::type;
    using b_type = typename std::remove_reference<decltype(b)>::type;
    static_assert(std::is_same<typename a_type::index_type, MKL_INT>::value);
    static_assert(std::is_same<typename b_type::index_type, MKL_INT>::value);

    auto status = mkl_sparse_s_create_csr(&a_descr, SPARSE_INDEX_BASE_ZERO, a.m_, a.n_,
                                          a.row_ptr_.data(), a.row_ptr_.data()+1,
                                          a.col_ind_.data(), a.vals_.data());
    mkl_error_handle(status, "Creating A");
    status = mkl_sparse_s_create_csr(&b_descr, SPARSE_INDEX_BASE_ZERO, b.m_, b.n_,
                                     b.row_ptr_.data(), b.row_ptr_.data()+1,
                                     b.col_ind_.data(), b.vals_.data());
    mkl_error_handle(status, "Creating B");

    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, a_descr, b_descr, &c_descr);
    mkl_error_handle(status, "SpMM");

    auto c = CSRMatrix(c_descr);

    mkl_sparse_destroy(a_descr);
    mkl_sparse_destroy(b_descr);
    mkl_sparse_destroy(c_descr);

    return CSRMatrix(c);
    */
  }

  CSRMatrix dot_mkl(CSRMatrix& b) {
    CSRMatrix& a = *this;

    if (a.nnz_ == 0 || b.nnz_ == 0) {
      return CSRMatrix(a.m_, b.n_);
    }

    // do a*b

    sparse_matrix_t a_descr, b_descr;
    sparse_matrix_t c_descr = NULL;

    using a_type = typename std::remove_reference<decltype(a)>::type;
    using b_type = typename std::remove_reference<decltype(b)>::type;
    static_assert(std::is_same<typename a_type::index_type, MKL_INT>::value);
    static_assert(std::is_same<typename b_type::index_type, MKL_INT>::value);

    auto status = mkl_sparse_s_create_csr(&a_descr, SPARSE_INDEX_BASE_ZERO, a.m_, a.n_,
                                          a.row_ptr_.data(), a.row_ptr_.data()+1,
                                          a.col_ind_.data(), a.vals_.data());
    mkl_error_handle(status, "Creating A");
    status = mkl_sparse_s_create_csr(&b_descr, SPARSE_INDEX_BASE_ZERO, b.m_, b.n_,
                                     b.row_ptr_.data(), b.row_ptr_.data()+1,
                                     b.col_ind_.data(), b.vals_.data());
    mkl_error_handle(status, "Creating B");

    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, a_descr, b_descr, &c_descr);
    mkl_error_handle(status, "SpMM");

    auto c = CSRMatrix(c_descr);

    mkl_sparse_destroy(a_descr);
    mkl_sparse_destroy(b_descr);
    mkl_sparse_destroy(c_descr);

    return CSRMatrix(c);
  }

  BCL::mkl::spmatrix<value_type> dot_mkl_raw(CSRMatrix& b) {
    CSRMatrix& a = *this;

    if (a.nnz_ == 0 || b.nnz_ == 0) {
      return BCL::mkl::spmatrix<value_type>(a.m_, b.n_);
    }

    // do a*b
    sparse_matrix_t a_descr, b_descr;
    BCL::mkl::spmatrix<value_type> c(a.m_, b.n_);

    auto status = mkl_sparse_s_create_csr(&a_descr, SPARSE_INDEX_BASE_ZERO, a.m_, a.n_,
                                          a.row_ptr_.data(), a.row_ptr_.data()+1,
                                          a.col_ind_.data(), a.vals_.data());
    mkl_error_handle(status, "Creating A");
    status = mkl_sparse_s_create_csr(&b_descr, SPARSE_INDEX_BASE_ZERO, b.m_, b.n_,
                                          b.row_ptr_.data(), b.row_ptr_.data()+1,
                                          b.col_ind_.data(), b.vals_.data());
    mkl_error_handle(status, "Creating B");

    status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, a_descr, b_descr, &c.matrix_);
    mkl_error_handle(status, "SpMM");

    c.has_matrix_ = true;

    mkl_sparse_destroy(a_descr);
    mkl_sparse_destroy(b_descr);

    return c;
  }

  template <typename A>
  CSRMatrix operator+(CSRMatrix<T, index_type, A>& b) {
    CSRMatrix& a = *this;

    if (a.nnz_ == 0 && b.nnz_ == 0) {
      return CSRMatrix(a.m_, b.n_);
    } else if (a.nnz_ == 0) {
      CSRMatrix b_ret(1, 1);
      b_ret = b;
      return b_ret;
    } else if (b.nnz_ == 0) {
      return a;
    }

    assert(a.m_ == b.m_);
    assert(a.n_ == b.n_);

    // do a*b
    sparse_matrix_t a_descr, b_descr;
    sparse_matrix_t c_descr = NULL;

    using a_type = typename std::remove_reference<decltype(a)>::type;
    using b_type = typename std::remove_reference<decltype(b)>::type;
    static_assert(std::is_same<typename a_type::index_type, MKL_INT>::value);
    static_assert(std::is_same<typename b_type::index_type, MKL_INT>::value);

    auto status = mkl_sparse_s_create_csr(&a_descr, SPARSE_INDEX_BASE_ZERO, a.m_, a.n_,
                                          a.row_ptr_.data(), a.row_ptr_.data()+1,
                                          a.col_ind_.data(), a.vals_.data());
    mkl_error_handle(status, "Creating A");
    status = mkl_sparse_s_create_csr(&b_descr, SPARSE_INDEX_BASE_ZERO, b.m_, b.n_,
                                          b.row_ptr_.data(), b.row_ptr_.data()+1,
                                          b.col_ind_.data(), b.vals_.data());
    mkl_error_handle(status, "Creating B");

    status = mkl_sparse_s_add(SPARSE_OPERATION_NON_TRANSPOSE, a_descr, 1.0, b_descr, &c_descr);
    mkl_error_handle(status, "SpAdd");

    auto c = CSRMatrix(c_descr);

    mkl_sparse_destroy(a_descr);
    mkl_sparse_destroy(b_descr);
    mkl_sparse_destroy(c_descr);

    return CSRMatrix(c);
  }

  // TODO: this should be relatively efficient, but
  //       a dedicated COOMatrix class would be more
  //       elegant.
  auto get_coo() const {
    using coord_type = std::pair<index_type, index_type>;
    using tuple_type = std::pair<coord_type, value_type>;
    using coo_t = std::vector<tuple_type>;

    coo_t values(nnz_);
    #pragma omp parallel for default(shared)
    for (size_t i = 0; i < m_; i++) {
      for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
        values[j] = {{i, col_ind_[j]}, vals_[j]};
      }
    }

    std::sort(values.begin(), values.end(),
              [](const auto& a, const auto& b) -> bool {
                return std::get<0>(a) < std::get<0>(b);
              });

    return values;
  }

  // Returns sorted by column!
  auto get_combblas_coo() const {
    using tuple_type = std::tuple<index_type, index_type, value_type>;
    using coo_t = std::vector<tuple_type>;

    coo_t values(nnz_);
    #pragma omp parallel for default(shared)
    for (size_t i = 0; i < m_; i++) {
      for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
        values[j] = tuple_type{col_ind_[j], i, vals_[j]};
      }
    }
    /*
    std::sort(values.begin(), values.end(), [](const auto& a, const auto& b) {
      if (std::get<1>(a) < std::get<1>(b)) {
        return true;
      } else if (std::get<1>(a) == std::get<1>(b) && std::get<0>(a) < std::get<0>(b)) {
        return true;
      } else {
        return false;
      }
    });
    */
    return values;
  }
};

// NOTE: this does not work with random newlines at the end of the file.
template <typename T, typename index_type, typename Allocator>
void CSRMatrix<T, index_type, Allocator>::read_MatrixMarket(const std::string& fname, bool one_indexed) {
  std::ifstream f;

  f.open(fname.c_str());

  if (!f.is_open()) {
    throw std::runtime_error("CSRMatrix::read_MatrixMarket cannot open " + fname);
  }

  std::string buf;

  bool outOfComments = false;

  while (!outOfComments) {
    getline(f, buf);
    regex_t regex;
    int reti;

    reti = regcomp(&regex, "^%", 0);
    reti = regexec(&regex, buf.c_str(), 0, NULL, 0);

    if (reti == REG_NOMATCH) {
      outOfComments = true;
    }
  }

  size_t m, n, nnz;
  sscanf(buf.c_str(), "%lu %lu %lu", &m, &n, &nnz);

  m_ = m;
  n_ = n;
  nnz_ = nnz;

  vals_.resize(nnz_);
  row_ptr_.resize(m_+1);
  col_ind_.resize(nnz_);

  row_ptr_[0] = 0;
  size_t r = 0;
  size_t c = 0;
  size_t i0 = 0;
  size_t j0 = 0;
  while (getline(f, buf)) {
    size_t i, j;
    double v;
    sscanf(buf.c_str(), "%lu %lu %lf", &i, &j, &v);
    if (one_indexed) {
      i--;
      j--;
    }

    if ((i == i0 && j < j0) || i < i0) {
      throw std::runtime_error("CSRMatrix::read_MatrixMarket " + fname + " is not sorted.");
    }
    i0 = i;
    j0 = j;

    assert(c < nnz_);

    vals_[c] = v;
    col_ind_[c] = j;

    while (r < i) {
      if (r+1 >= m_+1) {
        printf("Trying to write to %lu >= %lu\n", r+1, m_+1);
        printf("Trying to write to %lu >= %lu indices %lu, %lu\n",
               r+1, m_+1,
               i, j);
        fflush(stdout);
      }
      assert(r+1 < m_+1);
      row_ptr_[r+1] = c;
      r++;
    }
    c++;
  }

  for ( ; r < m_; r++) {
    row_ptr_[r+1] = nnz_;
  }

  f.close();
}

template <typename T, typename index_type, typename Allocator>
void CSRMatrix<T, index_type, Allocator>::print_details() {
  printf("CSRMatrix of dimensions %lu x %lu, %lu nnz\n",
         m_, n_, nnz_);
  printf("  Count %lu nnzs\n", count_nnz());
}

template <typename T, typename index_type, typename Allocator>
void CSRMatrix<T, index_type, Allocator>::print() {
  printf("Printing out sparse matrix (%d, %d) %d nnz\n",
          m_, n_, nnz_);
  printf("Values:\n");
  for (size_type i = 0; i < m_; i++) {
    for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
      printf("%d %d %f\n", i+1, col_ind_[j]+1, vals_[j]);
      // std::cout << "(" << i << "," << col_ind_[j] << ") " << vals_[j] << std::endl;
    }
  }
}

template <typename T, typename index_type, typename Allocator>
size_t CSRMatrix<T, index_type, Allocator>::count_nnz() {
  size_type count = 0;
  for (size_type i = 0; i < m_; i++) {
    for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
      count++;
    }
  }
  return count;
}

template <typename T, typename index_type, typename Allocator>
T CSRMatrix<T, index_type, Allocator>::count_values() {
  T count = 0;
  for (size_type i = 0; i < m_; i++) {
    assert(i+1 < row_ptr_.size());
    for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
      assert(j < vals_.size());
      count += vals_[j];
    }
  }
  return count;
}

template <typename T, typename index_type, typename Allocator>
size_t CSRMatrix<T, index_type, Allocator>::count_indices() {
  size_type count = 0;
  for (size_type i = 0; i < m_; i++) {
    assert(i+1 < row_ptr_.size());
    for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
      assert(j < col_ind_.size());
      count += col_ind_[j];
    }
  }
  return count;
}

template <typename T, typename index_type, typename Allocator>
void CSRMatrix<T, index_type, Allocator>::verify() {
  assert(vals_.size() == nnz_);
  assert(col_ind_.size() == nnz_);
  assert(row_ptr_.size() == m_+1);
  assert(m_-1 <= std::numeric_limits<index_type>::max());
  assert(n_-1 <= std::numeric_limits<index_type>::max());

  for (size_type i = 0; i < m_; i++) {
    assert(i+1 < row_ptr_.size());
    for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
      assert(j >= 0 && j < nnz_);
      index_type colind = col_ind_[j];
      assert(colind >= 0 && colind < n_);
    }
  }
}

template <typename T, typename index_type, typename Allocator>
void CSRMatrix<T, index_type, Allocator>::read_Binary(const std::string& fname) {
  FILE* f = fopen(fname.c_str(), "r");
  assert(f != NULL);
  size_t items_read = fread(&m_, sizeof(size_t), 1, f);
  assert(items_read == 1);
  items_read = fread(&n_, sizeof(size_t), 1, f);
  assert(items_read == 1);
  items_read = fread(&nnz_, sizeof(size_t), 1, f);
  assert(items_read == 1);

  vals_.resize(nnz_);
  col_ind_.resize(nnz_);
  row_ptr_.resize(m_+1);

  items_read = fread(vals_.data(), sizeof(T), vals_.size(), f);
  assert(items_read == vals_.size());
  items_read = fread(col_ind_.data(), sizeof(index_type), col_ind_.size(), f);
  assert(items_read == col_ind_.size());
  items_read = fread(row_ptr_.data(), sizeof(index_type), row_ptr_.size(), f);
  assert(items_read == row_ptr_.size());

  fclose(f);
}

template <typename T, typename index_type, typename Allocator>
void CSRMatrix<T, index_type, Allocator>::write_Binary(const std::string& fname) {
  FILE* f = fopen(fname.c_str(), "w");
  assert(f != NULL);
  fwrite(&m_, sizeof(size_t), 1, f);
  fwrite(&n_, sizeof(size_t), 1, f);
  fwrite(&nnz_, sizeof(size_t), 1, f);

  vals_.resize(nnz_);
  col_ind_.resize(nnz_);
  row_ptr_.resize(m_+1);

  fwrite(vals_.data(), sizeof(T), vals_.size(), f);
  fwrite(col_ind_.data(), sizeof(index_type), col_ind_.size(), f);
  fwrite(row_ptr_.data(), sizeof(index_type), row_ptr_.size(), f);

  fclose(f);
}

}
