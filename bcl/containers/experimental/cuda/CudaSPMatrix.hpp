
#pragma once

#define GRB_USE_CUDA
#define private public

#include <bcl/bcl.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <list>
#include <memory>
#include <numeric>

#include <bcl/containers/detail/Blocking.hpp>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>
#include <bcl/containers/experimental/cuda/util/cuda_future.hpp>

#include <bcl/containers/sequential/CSRMatrix.hpp>
#include <bcl/containers/sequential/SparseHashAccumulator.hpp>

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

  void init(const std::string& fname, Block&& blocking, FileFormat format) {
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
           FileFormat format = FileFormat::Unknown)
  {
    init(fname, std::move(blocking), format);
  }

  void init_with_zero(size_t m, size_t n, Block&& blocking) {
    m_ = m;
    n_ = n;
    nnz_ = 0;

    blocking.seed(m_, n_, BCL::nprocs());
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
          values = BCL::cuda::alloc<T>(1);
          col_ind = BCL::cuda::alloc<index_type>(1);
          row_ptr = BCL::cuda::alloc<index_type>(tile_shape({i, j})[0]+1);
          T v_data = 0;
          // *values = 0;
          BCL::cuda::memcpy(values, &v_data, sizeof(T));
          // *col_ind = 0;
          index_type c_data = 0;
          BCL::cuda::memcpy(col_ind, &c_data, sizeof(index_type));
          /*
          row_ptr[0] = 0;
          row_ptr[1] = 1;
          */
          std::vector<index_type> row_ptr_data(tile_shape({i, j})[0] + 1, 0);
          BCL::cuda::memcpy(row_ptr, row_ptr_data.data(), sizeof(index_type)*row_ptr_data.size());
          nnz = 0;
          BCL::cuda::flush();
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

  SPMatrix(size_t m, size_t n, Block&& blocking = BCL::BlockOpt())
  {
    init_with_zero(m, n, std::move(blocking));
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
  __host__ __device__ auto min(U a, V b) const {
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

    if (nnz > 0) {
      BCL::cuda::memcpy(d_vals_ptr, vals_ptr, sizeof(T)*nnz);
    }
    BCL::cuda::memcpy(d_row_ptr, row_ptr, sizeof(index_type)*(m+1));
    BCL::cuda::memcpy(d_col_ind, col_ind, sizeof(index_type)*nnz);

    BCL::cuda::flush();
    graphblas::Matrix<T>* local_mat = new graphblas::Matrix<T>(m, n);
    local_mat->build(d_row_ptr.local(), d_col_ind.local(), d_vals_ptr.local(), nnz);
    return local_mat;
  }

  __host__ auto arget_tile(matrix_dim idx) {
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

    if (nnz > 0) {
      BCL::cuda::memcpy(d_vals_ptr, vals_ptr, sizeof(T)*nnz);
    }
    BCL::cuda::memcpy(d_row_ptr, row_ptr, sizeof(index_type)*(m+1));
    BCL::cuda::memcpy(d_col_ind, col_ind, sizeof(index_type)*nnz);

    graphblas::Matrix<T>* local_mat = new graphblas::Matrix<T>(m, n);
    local_mat->build(d_row_ptr.local(), d_col_ind.local(), d_vals_ptr.local(), nnz);
    return cuda_future<graphblas::Matrix<T>*>(std::move(local_mat), cuda_request());
  }

  __host__ auto arget_tile_exp(matrix_dim idx) {
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

    std::thread memcpy_thread([](auto d_vals_ptr, auto vals_ptr,
                                 auto d_row_ptr, auto row_ptr,
                                 auto d_col_ind, auto col_ind,
                                 auto nnz, auto m) {
                                if (nnz > 0) {
                                  BCL::cuda::memcpy(d_vals_ptr, vals_ptr, sizeof(T)*nnz);
                                }
                                BCL::cuda::memcpy(d_row_ptr, row_ptr, sizeof(index_type)*(m+1));
                                BCL::cuda::memcpy(d_col_ind, col_ind, sizeof(index_type)*nnz);
                              }, d_vals_ptr, vals_ptr,
                                 d_row_ptr, row_ptr,
                                 d_col_ind, col_ind,
                                 nnz, m);

    using request_type = cuda_thread_request<decltype(memcpy_thread)>;

    graphblas::Matrix<T>* local_mat = new graphblas::Matrix<T>(m, n);
    local_mat->build(d_row_ptr.local(), d_col_ind.local(), d_vals_ptr.local(), nnz);
    return cuda_future<graphblas::Matrix<T>*, request_type>
                      (std::move(local_mat),
                       cuda_thread_request<decltype(memcpy_thread)>
                                            (std::move(memcpy_thread)));
  }

  __host__ BCL::CSRMatrix<T, index_type> get_tile_cpu(matrix_dim idx) {
    size_t i = idx[0];
    size_t j = idx[1];
    size_t m = tile_shape(idx)[0];
    size_t n = tile_shape(idx)[1];

    size_t vals_idx = i*grid_shape()[1] + j;
    size_t nnz = nnzs_[vals_idx];

    auto vals_ptr = vals_[vals_idx];
    auto row_ptr = row_ptr_[vals_idx];
    auto col_ind = col_ind_[vals_idx];

    std::vector<T> l_vals(nnz);
    std::vector<index_type> l_col_ind(nnz);
    std::vector<index_type> l_row_ptr(m+1);

    if (nnz > 0) {
      BCL::cuda::memcpy(l_vals.data(), vals_ptr, sizeof(T)*nnz);
    }
    BCL::cuda::memcpy(l_row_ptr.data(), row_ptr, sizeof(index_type)*(m+1));
    if (nnz > 0) {
      BCL::cuda::memcpy(l_col_ind.data(), col_ind, sizeof(index_type)*nnz);
    }

    BCL::cuda::flush();

    return CSRMatrix<T, index_type> (m, n, nnz,
                                     std::move(l_vals),
                                     std::move(l_row_ptr),
                                     std::move(l_col_ind));
  }

  __host__ void assign_tile(matrix_dim idx, graphblas::Matrix<T>* mat_) {
    graphblas::Matrix<T>& mat = *mat_;
    size_t i = idx[0];
    size_t j = idx[1];
    assert(vals_[j + i*grid_shape()[1]].is_local());

    size_t nnz = mat.matrix_.sparse_.nvals_;
    size_t nrows = mat.matrix_.sparse_.nrows_;
    T* vals_ptr = mat.matrix_.sparse_.d_csrVal_;
    index_type* row_ptr_ptr = mat.matrix_.sparse_.d_csrRowPtr_;
    index_type* col_ind_ptr = mat.matrix_.sparse_.d_csrColInd_;

    BCL::cuda::ptr<T> vals = BCL::cuda::alloc<T>(std::max<size_t>(1, nnz));
    BCL::cuda::ptr<index_type> col_ind = BCL::cuda::alloc<index_type>(std::max<size_t>(1, nnz));
    BCL::cuda::ptr<index_type> row_ptr = BCL::cuda::alloc<index_type>(std::max<size_t>(1, nrows+1));

    if (vals == nullptr || col_ind == nullptr || row_ptr == nullptr) {
      throw std::runtime_error("assign_tail(): Ran out of memory!");
    }

    cudaMemcpy(vals.local(), vals_ptr, sizeof(T)*nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(col_ind.local(), col_ind_ptr, sizeof(index_type)*nnz, cudaMemcpyDeviceToDevice);
    cudaMemcpy(row_ptr.local(), row_ptr_ptr, sizeof(index_type)*(nrows+1), cudaMemcpyDeviceToDevice);

    std::swap(vals, vals_[j + i*grid_shape()[1]]);
    std::swap(col_ind, col_ind_[j + i*grid_shape()[1]]);
    std::swap(row_ptr, row_ptr_[j + i*grid_shape()[1]]);
    nnzs_[j + i*grid_shape()[1]] = mat.matrix_.sparse_.nvals_;

    BCL::cuda::dealloc<T>(vals);
    BCL::cuda::dealloc<index_type>(col_ind);
    BCL::cuda::dealloc<index_type>(row_ptr);
  }

  void rebroadcast_tiles(const std::vector<size_t>& locales = {}) {
    for (size_t i = 0; i < vals_.size(); i++) {
      vals_[i] = BCL::broadcast(vals_[i], vals_[i].rank_);
    }

    for (size_t i = 0; i < col_ind_.size(); i++) {
      col_ind_[i] = BCL::broadcast(col_ind_[i], col_ind_[i].rank_);
    }

    for (size_t i = 0; i < row_ptr_.size(); i++) {
      row_ptr_[i] = BCL::broadcast(row_ptr_[i], row_ptr_[i].rank_);
    }

    for (size_t i = 0; i < nnzs_.size(); i++) {
      nnzs_[i] = BCL::broadcast(nnzs_[i], vals_[i].rank_);
    }

    nnz_ = std::accumulate(nnzs_.begin(), nnzs_.end(), size_t(0));
  }

  __host__ size_t tile_rank(matrix_dim idx) const {
    size_t i = idx[0];
    size_t j = idx[1];
    size_t vals_idx = i*grid_shape()[1] + j;
    return vals_[vals_idx].rank_;
  }

  auto get() {
    BCL::SparseHashAccumulator<T, index_type> acc;

    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        // TODO: This if statement shouldn't be required
        if (nnzs_[i*grid_shape()[1] + j] > 0) {
          auto local_tile = get_tile_cpu({i, j});

          size_t offset_i = i*tile_shape()[0];
          size_t offset_j = j*tile_shape()[1];

          acc.accumulate(std::move(local_tile), {offset_i, offset_j});
        }
      }
    }

    auto mat = acc.get_matrix(shape()[0], shape()[1]);
    return mat;
  }

  // Return the amount of memory in bytes used on this process
  __host__ size_t my_mem() {
    size_t n_bytes = 0;
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_rank({i, j}) == BCL::rank()) {
          size_t vals_idx = i*grid_shape()[1] + j;
          // col_ind
          n_bytes += nnzs_[vals_idx]*sizeof(index_type);
          // vals
          n_bytes += nnzs_[vals_idx]*sizeof(T);
          // row_ptr
          n_bytes += (tile_shape({i, j})[0]+1)*sizeof(index_type);
        }
      }
    }
    return n_bytes;
  }

};

} // end cuda

} // end BCL
