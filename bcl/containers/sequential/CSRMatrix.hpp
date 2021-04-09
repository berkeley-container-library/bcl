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
#include "matrix_io.hpp"

namespace BCL {

template <
          typename T,
          typename index_t = int,
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

  template <typename... Args>
  CSRMatrix(const CSRMatrix<Args...>& other)
    : m_(other.m_), n_(other.n_), nnz_(other.nnz_),
      vals_(other.vals_.begin(), other.vals_.end()),
      row_ptr_(other.row_ptr_.begin(), other.row_ptr_.end()),
      col_ind_(other.col_ind_.begin(), other.col_ind_.end())
  {}

  size_type m() const {
    return m_;
  }

  size_type n() const {
    return n_;
  }

  size_type nnz() const {
    return nnz_;
  }

  T* values_data() {
    return vals_.data();
  }

  index_type* rowptr_data() {
    return row_ptr_.data();
  }

  index_type* colind_data() {
    return col_ind_.data();
  }

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
  void write_MatrixMarket(const std::string& fname);

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

  CSRMatrix(const std::string& fname, FileFormat format = FileFormat::Unknown) {
    // If file format not given, attempt to detect.
    if (format == FileFormat::Unknown) {
      format = matrix_io::detect_file_type(fname);
    }
    if (format == FileFormat::MatrixMarket) {
      read_MatrixMarket(fname);
    } else if (format == FileFormat::MatrixMarketZeroIndexed) {
      read_MatrixMarket(fname, false);
    } else if (format == FileFormat::Binary) {
      read_Binary(fname);
    } else {
      throw std::runtime_error("CSRMatrix: Could not detect file format for \""
                               + fname + "\"");
    }
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
                if (std::get<0>(a) != std::get<0>(b)) {
                  return std::get<0>(a) < std::get<0>(b);
                } else {
                  return std::get<1>(a) < std::get<1>(b);
                }
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

template <typename T, typename index_type, typename Allocator>
void CSRMatrix<T, index_type, Allocator>::write_MatrixMarket(const std::string& fname) {
  fprintf(stderr, "Opening up %s for writing.\n", fname.c_str());
  FILE* f = fopen(fname.c_str(), "w");
  assert(f != NULL);

  fprintf(f, "%%%%MatrixMarket matrix coordinate integer general\n");
  fprintf(f, "%lu %lu %lu\n", m_, n_, nnz_);

  for (size_type i = 0; i < m_; i++) {
    for (index_type j = row_ptr_[i]; j < row_ptr_[i+1]; j++) {
      fprintf(f, "%d %d %f\n", i+1, col_ind_[j]+1, vals_[j]);
    }
  }

  fclose(f);
}

}
