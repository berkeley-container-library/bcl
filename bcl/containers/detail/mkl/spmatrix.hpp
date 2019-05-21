#pragma once

#include <mkl.h>

#include <bcl/containers/detail/mkl/mkl_sparse_wrapper.hpp>

namespace BCL {

template <
          typename T,
          typename index_t,
          typename Allocator
          >
struct CSRMatrix;

namespace mkl {

template <typename T>
class spmatrix {
public:
  sparse_matrix_t matrix_ = NULL;
  bool has_matrix_ = false;

  size_t m_;
  size_t n_;
  size_t nnz_ = 0;

  using value_type = T;
  using index_type = MKL_INT;

public:

  spmatrix() = delete;

  spmatrix& operator=(spmatrix&& other) {
    matrix_ = std::move(other.matrix_);
    has_matrix_ = std::move(other.has_matrix_);
    m_ = std::move(other.m_);
    n_ = std::move(other.n_);
    nnz_ = std::move(other.nnz_);
    other.has_matrix_ = false;
    other.matrix_ = NULL;
    return *this;
  }

  spmatrix(spmatrix&& other) : matrix_(std::move(other.matrix_)),
                               has_matrix_(std::move(other.has_matrix_)),
                               m_(std::move(other.m_)), n_(std::move(other.n_)),
                               nnz_(std::move(other.nnz_)) {
    other.has_matrix_ = false;
    other.matrix_ = NULL;
  }

  spmatrix(const spmatrix& other) : m_(other.m_), n_(other.n_), nnz_(other.nnz_),
                                    has_matrix_(other.has_matrix_) {
    if (other.has_matrix_) {
      struct matrix_descr descr;
      auto status = mkl_sparse_copy(other.matrix_, descr, &matrix_);
      mkl_error_handle(status, "BCL::mkl::spmatrix: copy constructor");
    }
  }

  spmatrix& operator=(const spmatrix& other) {
    m_ = other.m_;
    n_ = other.n_;
    nnz_ = other.nnz_;
    has_matrix_ = other.has_matrix_;
    if (other.has_matrix_) {
      struct matrix_descr descr;
      descr.type = SPARSE_MATRIX_TYPE_GENERAL;
      auto status = mkl_sparse_copy(other.matrix_, descr, &matrix_);
      mkl_error_handle(status, "BCL::mkl::spmatrix: copy assignment");
    }
    return *this;
  }

  spmatrix(size_t m, size_t n) : m_(m), n_(n) {}

  ~spmatrix() {
    if (has_matrix_) {
      auto status = mkl_sparse_destroy(matrix_);
      mkl_error_handle(status, "BCL::mkl::spmatrix: mkl_sparse_destroy");
    }
  }

  template <typename Allocator>
  spmatrix(const CSRMatrix<value_type, index_type, Allocator>& matrix)
           : m_(matrix.m_), n_(matrix.n_), nnz_(matrix.nnz_) {
    if (matrix.nnz_ > 0) {
      sparse_matrix_t m_handle;
      auto status = mkl_sparse_create_csr_wrapper_(&m_handle, SPARSE_INDEX_BASE_ZERO, matrix.m_, matrix.n_,
                                                   matrix.row_ptr_.data(), matrix.row_ptr_.data()+1,
                                                   matrix.col_ind_.data(), matrix.vals_.data());
      mkl_error_handle(status, "BCL::mkl::spmatrix: mkl_sparse_create_csr");
      struct matrix_descr descr;
      descr.type = SPARSE_MATRIX_TYPE_GENERAL;
      status = mkl_sparse_copy(m_handle, descr, &matrix_);
      mkl_error_handle(status, "BCL::mkl::spmatrix: mkl_sparse_copy");
      mkl_sparse_destroy(m_handle);
      has_matrix_ = true;
    }
  }

  spmatrix dot(const spmatrix& other) {
    if (has_matrix_ && other.has_matrix_) {
      spmatrix c(m_, other.n_);
      auto status = mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, matrix_, other.matrix_, &c.matrix_);
      mkl_error_handle(status, "BCL::mkl::spmatrix: dot()");
      c.has_matrix_ = true;
      // TODO: update c.nnz_ here.
      return c;
    } else {
      return spmatrix(m_, other.n_);
    }
  }

  spmatrix operator+(const spmatrix& other) const {
    if (has_matrix_ && other.has_matrix_) {
      assert(m_ == other.m_);
      assert(n_ == other.n_);
      spmatrix c(m_, n_);
      auto status = mkl_sparse_add_wrapper_(SPARSE_OPERATION_NON_TRANSPOSE, matrix_,
                                            value_type(1.0), other.matrix_, &c.matrix_);
      mkl_error_handle(status, "BCL::mkl::spmatrix: operator+()");
      c.has_matrix_ = true;
      // TODO: update c.nnz_ here.
      return c;
    } else if (has_matrix_) {
      return *this;
    } else if (other.has_matrix_) {
      return other;
    } else {
      throw std::runtime_error("WHAT NO ONE HAS A MATRIX");
      return *this;
    }
  }

  // TODO: this function doesn't really work.
  template <typename Allocator>
  spmatrix operator+(const CSRMatrix<value_type, index_type, Allocator>& other) const {
    if (other.nnz_ > 0) {
      spmatrix other_(other);
      return this->operator+(other_);
    } else {
      return *this;
    }
  }
};

}
}
