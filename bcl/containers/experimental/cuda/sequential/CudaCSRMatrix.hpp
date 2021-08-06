#pragma once

#include <bcl/containers/sequential/CSRMatrix.hpp>

namespace BCL {

namespace cuda {

template <typename T, typename I>
struct CudaCSRMatrixView;

template <typename T, typename I, typename Allocator = BCL::cuda::cuda_allocator<T>>
struct CudaCSRMatrix {
  using value_type = T;
  using index_type = I;
  using allocator_type = Allocator;

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

/*
  CudaCSRMatrix(matrix_dim shape) : m_(shape[0]), n_(shape[1]), nnz_(0) {
    d_rowptr_ = allocate_with<index_type, Allocator>(m()+1);
    cudaMemset(d_rowptr_, 0, sizeof(index_type)*(m()+1));
  }
  */

  CudaCSRMatrix(const CudaCSRMatrix&) = delete;
  CudaCSRMatrix& operator=(const CudaCSRMatrix&) = delete;

  CudaCSRMatrix(CudaCSRMatrix&& other) {
    move_(std::move(other));
  }

  CudaCSRMatrix& operator=(CudaCSRMatrix&& other) {
    move_(std::move(other));
    return *this;
  }

  operator CudaCSRMatrixView<T, I>() {
    return view();
  }

  CudaCSRMatrixView<T, I> view() {
    return CudaCSRMatrixView<T, I>(m(), n(), nnz(), d_vals_, d_rowptr_, d_colind_);
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

template <typename T, typename I>
struct CudaCSRMatrixView {
  using value_type = T;
  using index_type = I;

  struct matrix_dim;

  CudaCSRMatrixView(size_t m, size_t n, size_t nnz,
                    T* d_vals, I* d_rowptr, I* d_colind)
    : m_(m), n_(n), nnz_(nnz),
      d_vals_(d_vals), d_rowptr_(d_rowptr), d_colind_(d_colind) {}

  CudaCSRMatrixView(const CudaCSRMatrixView&) = default;
  CudaCSRMatrixView& operator=(const CudaCSRMatrixView&) = default;

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

template <typename T, typename index_type, typename Allocator = BCL::bcl_allocator<T>>
CudaCSRMatrix<T, index_type, Allocator> to_gpu(CSRMatrix<T, index_type>& mat) {
  CudaCSRMatrix<T, index_type, Allocator> mat_gpu({mat.m(), mat.n()}, mat.nnz());
  cudaMemcpy(mat_gpu.values_data(), mat.values_data(), sizeof(T)*mat.nnz(), cudaMemcpyHostToDevice);
  cudaMemcpy(mat_gpu.rowptr_data(), mat.rowptr_data(), sizeof(index_type)*(mat.m()+1), cudaMemcpyHostToDevice);
  cudaMemcpy(mat_gpu.colind_data(), mat.colind_data(), sizeof(index_type)*mat.nnz(), cudaMemcpyHostToDevice);
  return mat_gpu;
}

template <typename T, typename index_type, typename Allocator>
CSRMatrix<T, index_type> to_cpu(CudaCSRMatrix<T, index_type, Allocator>& mat) {
  std::vector<T> values(mat.nnz());
  std::vector<index_type> rowptr(mat.m()+1);
  std::vector<index_type> colind(mat.nnz());

  cudaMemcpy(values.data(), mat.values_data(), sizeof(T)*mat.nnz(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rowptr.data(), mat.rowptr_data(), sizeof(index_type)*(mat.m()+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(colind.data(), mat.colind_data(), sizeof(index_type)*mat.nnz(), cudaMemcpyDeviceToHost);

  return CSRMatrix<T, index_type>(mat.m(), mat.n(), mat.nnz(),
                                  std::move(values), std::move(rowptr),
                                  std::move(colind));
}

template <typename MatrixType>
CSRMatrix<typename MatrixType::value_type, typename MatrixType::index_type> to_cpu_generic(MatrixType& mat) {
  using T = typename MatrixType::value_type;
  using index_type = typename MatrixType::index_type;
  std::vector<T> values(mat.nnz());
  std::vector<index_type> rowptr(mat.m()+1);
  std::vector<index_type> colind(mat.nnz());

  cudaMemcpy(values.data(), mat.values_data(), sizeof(T)*mat.nnz(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rowptr.data(), mat.rowptr_data(), sizeof(index_type)*(mat.m()+1), cudaMemcpyDeviceToHost);
  cudaMemcpy(colind.data(), mat.colind_data(), sizeof(index_type)*mat.nnz(), cudaMemcpyDeviceToHost);

  return CSRMatrix<T, index_type>(mat.m(), mat.n(), mat.nnz(),
                                  std::move(values), std::move(rowptr),
                                  std::move(colind));
}

template <typename T, typename index_type>
auto get_coo(const std::vector<T>& values,
             const std::vector<index_type>& row_indices,
             const std::vector<index_type>& col_indices)
{
    using coord_type = std::pair<index_type, index_type>;
    using tuple_type = std::pair<coord_type, T>;
    using coo_t = std::vector<tuple_type>;

    coo_t coo_values(values.size());

    for (size_t i = 0; i < values.size(); i++) {
      coo_values[i] = {{row_indices[i], col_indices[i]}, values[i]};
    }

    std::sort(coo_values.begin(), coo_values.end(),
              [](const auto& a, const auto& b) -> bool {
                if (std::get<0>(a) != std::get<0>(b)) {
                  return std::get<0>(a) < std::get<0>(b);
                } else {
                  return std::get<1>(a) < std::get<1>(b);
                }
              });

    return coo_values;
}

template <typename T, typename index_type>
auto remove_zeros(const std::vector<std::pair<std::pair<index_type, index_type>, T>>& coo_values) {
    using coord_type = std::pair<index_type, index_type>;
    using tuple_type = std::pair<coord_type, T>;
    using coo_t = std::vector<tuple_type>;

    coo_t new_coo;

    for (const auto& nz : coo_values) {
      auto val = std::get<1>(nz);
      if (val != 0.0) {
        new_coo.push_back(nz);
      }
    }
    return new_coo;
}

template <typename T>
void print_coo(const T& coo, size_t max_idx = std::numeric_limits<size_t>::max()) {
  for (size_t i = 0; i < std::min(coo.size(), max_idx); i++) {
    auto idx = std::get<0>(coo[i]);
    auto val = std::get<1>(coo[i]);
    printf("(%lu, %lu) %f\n", idx.first, idx.second, val);
  }
}

} // end cuda
} // end BCL