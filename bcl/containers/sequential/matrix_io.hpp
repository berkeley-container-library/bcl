// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once
#include <sstream>
#include <bcl/containers/detail/index.hpp>

namespace BCL {

enum FileFormat {
  MatrixMarket,
  MatrixMarketZeroIndexed,
  Binary,
  Unknown
};

namespace matrix_io {

inline auto detect_file_type(const std::string& fname) {
  size_t suffix_spot = 0;

  for (int64_t i = fname.size()-1; i >= 0; i--) {
    if (fname[i] == '.') {
      suffix_spot = i;
      break;
    }
  }

  std::string suffix = fname.substr(suffix_spot);

  if (suffix == ".mtx") {
    return BCL::FileFormat::MatrixMarket;
  } else if (suffix == ".binary") {
    return BCL::FileFormat::Binary;
  } else {
    assert(false);
    return BCL::FileFormat::Unknown;
  }
}

struct matrix_shape {
  std::array<size_t, 2> shape;
  size_t nnz;
};

inline matrix_shape matrix_market_info(const std::string& fname, bool one_indexed = true);
inline matrix_shape binary_info(const std::string& fname);

inline matrix_shape matrix_info(const std::string& fname,
                                BCL::FileFormat format = BCL::FileFormat::Unknown) {
  if (format == FileFormat::Unknown) {
    format = detect_file_type(fname);
  }

  if (format == FileFormat::MatrixMarket) {
    return matrix_market_info(fname);
  } else if (format == FileFormat::MatrixMarketZeroIndexed) {
    return matrix_market_info(fname, false);
  } else if (format == FileFormat::Binary) {
    return binary_info(fname);
  } else {
    throw std::runtime_error("matrix_info: Could not detect file format for \""
                             + fname + "\"");
  }
}

inline matrix_shape matrix_market_info(const std::string& fname, bool one_indexed) {
  std::ifstream f;

  f.open(fname.c_str());

  if (!f.is_open()) {
    throw std::runtime_error("matrix_market_size: cannot open " + fname);
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

  f.close();

  return matrix_shape{{m, n}, nnz};
}

inline matrix_shape binary_info(const std::string& fname) {
  FILE* f = fopen(fname.c_str(), "r");
  size_t m, n, nnz;
  assert(f != NULL);
  size_t items_read = fread(&m, sizeof(size_t), 1, f);
  assert(items_read == 1);
  items_read = fread(&n, sizeof(size_t), 1, f);
  assert(items_read == 1);
  items_read = fread(&nnz, sizeof(size_t), 1, f);
  assert(items_read == 1);
  fclose(f);
  return matrix_shape{{m, n}, nnz};
}

template <typename T,
          typename I>
class coo_matrix {
public:
  using value_type = std::pair<std::pair<I, I>, T>;
  using index_type = I;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using key_type = std::pair<I, I>;
  using map_type = T;

  using backend_type = std::vector<value_type>;

  using iterator = typename backend_type::iterator;
  using const_iterator = typename backend_type::const_iterator;

  using reference = typename backend_type::reference;
  using const_reference = typename backend_type::const_reference;

  coo_matrix(BCL::index<I> shape) : shape_(shape) {}

  BCL::index<I> shape() const noexcept {
    return shape_;
  }

  size_type size() const noexcept {
    return tuples_.size();
  }

  void reserve(size_type new_cap) {
    tuples_.reserve(new_cap);
  }

  iterator begin() noexcept {
    return tuples_.begin();
  }

  const_iterator begin() const noexcept {
    return tuples_.begin();
  }

  iterator end() noexcept {
    return tuples_.end();
  }

  const_iterator end() const noexcept {
    return tuples_.end();
  }

  std::pair<iterator, bool>
  insert(value_type&& value) {
    auto&& [insert_index, insert_value] = value;
    for (auto iter = begin(); iter != end(); ++iter) {
      auto&& [index, v] = *iter;
      if (index == insert_index) {
        return {iter, false};
      }
    }
    tuples_.push_back(value);
    return {--tuples_.end(), true};
  }

  template <class M>
  std::pair<iterator, bool>
  insert_or_assign(key_type k, M&& obj) {
    for (auto iter = begin(); iter != end(); ++iter) {
      auto&& [index, v] = *iter;
      if (index == k) {
        v = std::forward<M>(obj);
        return {iter, false};
      }
    }
    tuples_.push_back({k, std::forward<M>(obj)});
    return {--tuples_.end(), true};
  }

  iterator find(key_type key) noexcept {
    return std::find_if(begin(), end(), [&](auto&& v) {
                                         auto&& [i, v_] = v;
                                         return i == key;
                                        });
  }

  const_iterator find(key_type key) const noexcept {
    return std::find_if(begin(), end(), [&](auto&& v) {
                                          auto&& [i, v_] = v;
                                          return i == key;
                                        });
  }

  coo_matrix() = default;
  ~coo_matrix() = default;
  coo_matrix(const coo_matrix&) = default;
  coo_matrix(coo_matrix&&) = default;
  coo_matrix& operator=(const coo_matrix&) = default;
  coo_matrix& operator=(coo_matrix&&) = default;

private:
  BCL::index<I> shape_;
  backend_type tuples_;
};

// Read in a matrix market file `fname`, returning a sorted vector
// of nonzero tuples.
template <typename T, typename I = std::size_t>
inline
coo_matrix<T, I>
mmread(std::string fname, bool one_indexed = true) {
  using index_type = I;
  using size_type = std::size_t;

  std::ifstream f;

  f.open(fname.c_str());

  if (!f.is_open()) {
    // TODO better choice of exception.
    throw std::runtime_error("mmread: cannot open " + fname);
  }

  std::string buf;

  // Make sure the file is matrix market matrix, coordinate, and check whether
  // it is symmetric. If the matrix is symmetric, non-diagonal elements will
  // be inserted in both (i, j) and (j, i).  Error out if skew-symmetric or
  // Hermitian.
  std::getline(f, buf);
  std::istringstream ss(buf);
  std::string item;
  ss >> item;
  if (item != "%%MatrixMarket") {
    throw std::runtime_error(fname + " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "matrix") {
    throw std::runtime_error(fname + " could not be parsed as a Matrix Market file.");
  }
  ss >> item;
  if (item != "coordinate") {
    throw std::runtime_error(fname + " could not be parsed as a Matrix Market file.");
  }
  bool pattern;
  ss >> item;
  if (item == "pattern") {
    pattern = true;
  } else {
    pattern = false;
  }
  // TODO: do something with real vs. integer vs. pattern?
  ss >> item;
  bool symmetric;
  if (item == "general") {
    symmetric = false;
  } else if (item == "symmetric") {
    symmetric = true;
  } else {
    throw std::runtime_error(fname + " has an unsupported matrix type");
  }

  bool outOfComments = false;
  while (!outOfComments) {
    std::getline(f, buf);

    if (buf[0] != '%') {
      outOfComments = true;
    }
  }

  I m, n, nnz;
  // std::istringstream ss(buf);
  ss.clear();
  ss.str(buf);
  ss >> m >> n >> nnz;

  // NOTE for symmetric matrices: `nnz` holds the number of stored values in
  // the matrix market file, while `matrix.nnz_` will hold the total number of
  // stored values (including "mirrored" symmetric values).
  coo_matrix<T, I> matrix({m, n});
  if (symmetric) {
    matrix.reserve(2*nnz);
  } else {
    matrix.reserve(nnz);
  }
  /*
    TODO: reserve? (for general and for symmetric)
  */

  size_type c = 0;
  while (std::getline(f, buf)) {
    I i, j;
    T v;
    std::istringstream ss(buf);
    if (!pattern) {
      ss >> i >> j >> v;
    } else {
      ss >> i >> j;
      v = T(1);
    }
    if (one_indexed) {
      i--;
      j--;
    }

    if (i >= m || j >= n) {
      throw std::runtime_error("read_MatrixMarket: file has nonzero out of bounds.");
    }

    matrix.insert({{i, j}, v});

    if (symmetric && i != j) {
      matrix.insert({{j, i}, v});
    }

    c++;
    if (c > nnz) {
      throw std::runtime_error("read_MatrixMarket: error reading Matrix Market file, file has more nonzeros than reported.");
    }
  }

  auto sort_fn = [](auto&& a, auto&& b) {
                   auto&& [a_index, a_value] = a;
                   auto&& [b_index, b_value] = b;
                   auto&& [a_i, a_j] = a_index;
                   auto&& [b_i, b_j] = b_index;
                   if (a_i < b_i) {
                     return true;
                   }
                   else if (a_i == b_i) {
                     if (a_j < b_j) {
                      return true;
                     }
                   }
                   return false;
                 };

  std::sort(matrix.begin(), matrix.end(), sort_fn);

  f.close();

  return matrix;
}

} // end matrix_io

} // end BCL
