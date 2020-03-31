
#pragma once

#define GRB_USE_CUDA
#define private public

#include <bcl/bcl.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <list>
#include <memory>

#include <bcl/containers/detail/Blocking.hpp>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>
#include <bcl/containers/experimental/cuda/util/cuda_future.hpp>
#include <bcl/containers/sequential/CSRMatrix.hpp>

#include <graphblas/graphblas.hpp>

namespace BCL {

namespace cuda {

template <typename T, typename index_type = int>
class SPMatrix {
public:

  using value_type = T;

  std::vector<BCL::cuda::ptr<T>> vals_;
  std::vector<BCL::cuda::ptr<index_type>> col_ind_;
  std::vector<std::size_t> nnzs_;

  std::vector<BCL::cuda::ptr<index_type>> row_ptr_;

  // Size of *matrix* (in elements)
  size_t m_, n_;
  // Size of *processor* grid
  size_t pm_, pn_;
  // Size (in elements) of a *tile*
  size_t tile_size_m_, tile_size_n_;

  // Size of *tile* grid (in tiles)
  size_t grid_dim_m_, grid_dim_n_;

  // Total number of nonzeros across the whole matrix
  size_t nnz_;

  SPMatrix(const SPMatrix&) = default;
  SPMatrix(SPMatrix&&) = default;

  void init(const std::string& fname, Block&& blocking, FileFormat format = FileFormat::MatrixMarket) {
    CSRMatrix<T, index_type> mat(fname, format);

    m_ = mat.m_;
    n_ = mat.n_;
    nnz_ = mat.nnz_;

    size_t m = m_;
    size_t n = n_;

    blocking.seed(mat.m_, mat.n_, BCL::nprocs());
    pm_ = blocking.pgrid_shape()[0];
    pn_ = blocking.pgrid_shape()[1];

    tile_size_m_ = blocking.tile_shape()[0];
    tile_size_n_ = blocking.tile_shape()[1];

    if (pm_*pn_ > BCL::nprocs()) {
      throw std::runtime_error("SPMatrix: tried to create a SPMatrix with a too large p-grid.");
    }

    if (tile_size_m_ == 0) {
      tile_size_m_ = (m + pm_ - 1) / pm_;
    }
    if (tile_size_n_ == 0) {
      tile_size_n_ = (n + pn_ - 1) / pn_;
    }

    grid_dim_m_ = (m + tile_shape()[0] - 1) / tile_shape()[0];
    grid_dim_n_ = (n + tile_shape()[1] - 1) / tile_shape()[1];


    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        size_t lpi = i % pm_;
        size_t lpj = j % pn_;
        size_t proc = lpj + lpi*pn_;

        size_t nnz;
        BCL::cuda::ptr<T> values;
        BCL::cuda::ptr<index_type> col_ind;
        BCL::cuda::ptr<index_type> row_ptr;
        if (BCL::rank() == proc) {
          auto slc = mat.get_slice_impl_(i*tile_size_m_, (i+1)*tile_size_m_,
                                         j*tile_size_n_, (j+1)*tile_size_n_);
          nnz = slc.nnz_;
          values = BCL::cuda::alloc<T>(slc.vals_.size());
          if (values != nullptr) {
            cudaMemcpy(values.local(), slc.vals_.data(), sizeof(T)*slc.vals_.size(),
                       cudaMemcpyHostToDevice);
          }
          col_ind = BCL::cuda::alloc<index_type>(slc.col_ind_.size());
          if (col_ind != nullptr) {
            cudaMemcpy(col_ind.local(), slc.col_ind_.data(), sizeof(index_type)*slc.col_ind_.size(),
                       cudaMemcpyHostToDevice);
          }
          row_ptr = BCL::cuda::alloc<index_type>(slc.row_ptr_.size());
          if (row_ptr != nullptr) {
            cudaMemcpy(row_ptr.local(), slc.row_ptr_.data(), sizeof(index_type)*slc.row_ptr_.size(),
                       cudaMemcpyHostToDevice);
          }
        }
        nnz = BCL::broadcast(nnz, proc);
        values = BCL::broadcast(values, proc);
        col_ind = BCL::broadcast(col_ind, proc);
        row_ptr = BCL::broadcast(row_ptr, proc);
        if (values == nullptr || col_ind == nullptr || row_ptr == nullptr) {
          throw std::runtime_error("SPMatrix: ran out of memory!");
        }
        nnzs_.push_back(nnz);
        vals_.push_back(values);
        col_ind_.push_back(col_ind);
        row_ptr_.push_back(row_ptr);
      }
    }
  }

  SPMatrix(const std::string& fname, Block&& blocking = BCL::BlockOpt(),
           FileFormat format = FileFormat::MatrixMarket)
  {
    init(fname, std::move(blocking), format);
  }

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

  __device__ __host__ matrix_dim shape() const {
    return matrix_dim {m_, n_};
  }

  __device__ __host__ matrix_dim tile_shape() const {
    return matrix_dim {tile_size_m_, tile_size_n_};
  }

  template <typename U, typename V>
  auto min(U a, V b) const {
    if (a < b) {
      return a;
    } else {
      return b;
    }
  }

  __device__ __host__ matrix_dim tile_shape(matrix_dim idx) const {
    size_t tile_shape_m = min(tile_shape()[0], shape()[0] - idx[0]*tile_shape()[0]);
    size_t tile_shape_n = min(tile_shape()[1], shape()[1] - idx[1]*tile_shape()[1]);
    return matrix_dim {tile_shape_m, tile_shape_n};
  }

  __device__ __host__ matrix_dim grid_shape() const {
    return matrix_dim {grid_dim_m_, grid_dim_n_};
  }

  __device__ __host__ size_t tile_size() const {
    return tile_shape()[0] * tile_shape()[1];
  }

  __host__ void print_info(bool print_pgrid = true) const {
    printf("=== MATRIX INFO ===\n");
    printf("%lu x %lu matrix\n", shape()[0], shape()[1]);
    printf("  * Consists of %lu x %lu tiles\n", tile_shape()[0], tile_shape()[1]);
    printf("  * Arranged in a %lu x %lu grid\n", grid_shape()[0], grid_shape()[1]);

    if (print_pgrid) {
      for (size_t i = 0; i < grid_shape()[0]; i++) {
        printf("   ");
        for (size_t j = 0; j < grid_shape()[1]; j++) {
          printf("%2lu ", vals_[i*grid_shape()[1] + j].rank_);
        }
        printf("\n");
      }
    }
  }

  __host__ graphblas::Matrix<T>* get_tile(matrix_dim idx) {
    size_t i = idx[0];
    size_t j = idx[1];
    size_t m = tile_shape(idx)[0];
    size_t n = tile_shape(idx)[1];

    size_t vals_idx = i*grid_shape()[1] + j;
    size_t nnz = nnzs_[vals_idx];

    auto vals_ptr = vals_[vals_idx];
    auto row_ptr = row_ptr_[vals_idx];
    auto col_ind = col_ind_[vals_idx];

    auto d_vals_ptr = BCL::cuda::alloc<T>(nnz);
    auto d_row_ptr = BCL::cuda::alloc<index_type>(m+1);
    auto d_col_ind = BCL::cuda::alloc<index_type>(nnz);

    if (d_vals_ptr == nullptr || d_row_ptr == nullptr || d_col_ind == nullptr) {
      throw std::runtime_error("Cuda::SPMatrix ran out of memory");
    }

    BCL::cuda::memcpy(d_vals_ptr, vals_ptr, sizeof(T)*nnz);
    BCL::cuda::memcpy(d_row_ptr, row_ptr, sizeof(index_type)*(m+1));
    BCL::cuda::memcpy(d_col_ind, col_ind, sizeof(index_type)*nnz);

    BCL::cuda::flush();
    graphblas::Matrix<T>* local_mat = new graphblas::Matrix<T>(m, n);
    local_mat->build(d_row_ptr.local(), d_col_ind.local(), d_vals_ptr.local(), nnz);
    return local_mat;
  }

  __host__ size_t tile_rank(matrix_dim idx) const {
    size_t i = idx[0];
    size_t j = idx[1];
    size_t vals_idx = i*grid_shape()[1] + j;
    return vals_[vals_idx].rank_;
  }

};

} // end cuda

} // end BCL
