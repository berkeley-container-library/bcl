#pragma once

namespace BCL {

namespace cuda {

template <typename T, typename index_type, typename Allocator = BCL::cuda::cuda_allocator<T>>
struct CudaCSRMatrix {

  struct matrix_dim;

  CudaCSRMatrix(size_t m, size_t n, size_t nnz) : m_(m), n_(n), nnz_(nnz) {
  	d_vals_ = allocate_with<T, Allocator>(nnz);
  	d_rowptr_ = allocate_with<index_type, Allocator>(m+1);
  	d_colind_ = allocate_with<index_type, Allocator>(nnz);
  }

  CudaCSRMatrix(matrix_dim shape, size_t nnz) : m_(shape[0]), n_(shape[1]), nnz_(nnz) {
  	d_vals_ = allocate_with<T, Allocator>(nnz);
  	d_rowptr_ = allocate_with<index_type, Allocator>(m()+1);
  	d_colind_ = allocate_with<index_type, Allocator>(nnz);
  }

  CudaCSRMatrix(matrix_dim shape, size_t nnz, T* values, index_type* rowptr,
                index_type* colind)
    : m_(shape[0]), n_(shape[1]), nnz_(nnz), d_vals_(values), d_rowptr_(rowptr),
      d_colind_(colind)
  {}

  CudaCSRMatrix(matrix_dim shape) : m_(shape[0]), n_(shape[1]), nnz_(0) {
    d_rowptr_ = allocate_with<index_type, Allocator>(m()+1);
    cudaMemset(d_rowptr_, 0, sizeof(index_type)*(m()+1));
  }

  CudaCSRMatrix(const CudaCSRMatrix&) = delete;
  CudaCSRMatrix& operator=(const CudaCSRMatrix&) = delete;

  CudaCSRMatrix(CudaCSRMatrix&& other) {
    move_(std::move(other));
  }

  CudaCSRMatrix& operator=(CudaCSRMatrix&& other) {
    move_(std::move(other));
    return *this;
  }

  void move_(CudaCSRMatrix&& other) {
    m_ = other.m_;
    n_ = other.n_;
    nnz_ = other.nnz_;
    d_vals_ = other.d_vals_;
    d_rowptr_ = other.d_rowptr_;
    d_colind_ = other.d_colind_;
    other.nnz_ = 0;
    other.d_vals_ = nullptr;
    other.d_rowptr_ = nullptr;
    other.d_colind_ = nullptr;
  }

  ~CudaCSRMatrix() {
  	deallocate_with<T, Allocator>(d_vals_);
  	deallocate_with<index_type, Allocator>(d_rowptr_);
  	deallocate_with<index_type, Allocator>(d_colind_);
  }

  size_t nnz() const {
  	return nnz_;
  }

  size_t m() const {
  	return m_;
  }

  size_t n() const {
  	return n_;
  }

  T* values_data() {
  	return d_vals_;
  }

  index_type* rowptr_data() {
  	return d_rowptr_;
  }

  index_type* colind_data() {
  	return d_colind_;
  }

  matrix_dim shape() const {
  	return {m(), n()};
  }

  bool empty() const {
    return nnz() == 0;
  }

  size_t m_;
  size_t n_;
  size_t nnz_;

  T* d_vals_ = nullptr;
  index_type* d_rowptr_ = nullptr;
  index_type* d_colind_ = nullptr;

  struct matrix_dim {
    size_t m, n;
    __device__ __host__ size_t operator[](size_t dim_num) {
      if (dim_num == 0) {
        return m;
      } else {
        return n;
      }
    }
  };

};

} // end cuda
} // end BCL