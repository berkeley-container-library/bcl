

#pragma once

#include <bcl/bcl.hpp>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <list>
#include <memory>

#include <bcl/containers/detail/Blocking.hpp>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>
#include <bcl/containers/experimental/cuda/util/cuda_future.hpp>

namespace BCL {

namespace cuda {

template <typename T, typename Fn>
__global__ void apply_matrix_impl_(BCL::cuda::ptr<T> data, size_t size, Fn fn) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    data.local()[tid] = fn(data.local()[tid]);
  }
}

__global__ void dummy() {
}

template <typename T, typename Fn>
__global__ void
apply_matrix_2d_idx_impl_(BCL::cuda::ptr<T> data,
                          size_t tile_shape_m, size_t tile_shape_n,
                          size_t tile_size_m, size_t tile_size_n,
                          size_t tile_offset_m, size_t tile_offset_n,
                          Fn fn) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  size_t tidy = threadIdx.y + blockIdx.y * blockDim.y;
  if (tidx < tile_shape_m && tidy < tile_shape_n) {
    // Column-major indexing, here.
    size_t idx = tidx + tidy*tile_size_m;
    data.local()[idx] = fn(data.local()[idx], tidx + tile_offset_m, tidy + tile_offset_n);
  }
}

template <typename T>
class Matrix {
public:

  using value_type = T;

  std::vector<BCL::cuda::ptr<value_type>> ptrs_;

  // Size of *matrix* (in elements)
  size_t m_, n_;
  // Size of *processor* grid
  size_t pm_, pn_;
  // Size (in elements) of a *tile*
  size_t tile_size_m_, tile_size_n_;

  // Size of *tile* grid (in tiles)
  size_t grid_dim_m_, grid_dim_n_;

  Matrix(const Matrix&) = default;
  Matrix(Matrix&&) = default;

  void init(size_t m, size_t n, Block&& blocking) {
    blocking.seed(m, n, BCL::nprocs());
    pm_ = blocking.pgrid_shape()[0];
    pn_ = blocking.pgrid_shape()[1];

    tile_size_m_ = blocking.tile_shape()[0];
    tile_size_n_ = blocking.tile_shape()[1];

    if (pm_*pn_ > BCL::nprocs()) {
      throw std::runtime_error("Matrix: tried to create a Matrix with a too large p-grid.");
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
        BCL::cuda::ptr<T> ptr;
        if (BCL::rank() == proc) {
          ptr = BCL::cuda::alloc<T>(tile_size());
        }
        ptr = BCL::broadcast(ptr, proc);
        if (ptr == nullptr) {
          throw std::runtime_error("Matrix: ran out of memory!");
        }
        ptrs_.push_back(ptr);
      }
    }
    BCL::barrier();
  }

  Matrix(size_t m, size_t n, Block&& blocking = BCL::BlockOpt())
          : m_(m), n_(n)
  {
    init(m, n, std::move(blocking));
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

  __device__ __host__ BCL::cuda::ptr<T> tile_ptr(matrix_dim idx) const {
    return ptrs_[idx[0]*grid_shape()[1] + idx[1]];
  }

  __host__ auto get_tile(matrix_dim idx) const {
    BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>> x(tile_size());
    BCL::cuda::memcpy(x.data(), tile_ptr(idx), sizeof(T) * tile_size());
    BCL::cuda::flush();
    return x;
  }

  __host__ auto arget_tile(matrix_dim idx) const {
    using no_init = typename BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>>::no_init;
    BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>> x(tile_size(), no_init{});
    if (x.data() == nullptr) {
      printf("%lu has nullptr x!\n", BCL::rank());
      assert(false);
    }
    auto begin = std::chrono::high_resolution_clock::now();
    BCL::cuda::memcpy(x.data(), tile_ptr(idx), sizeof(T) * tile_size());
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    if (BCL::rank() == 0) {
      printf("memcpy: %lf\n", duration);
    }
    return cuda_future<BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>>>
                      (std::move(x), cuda_request());
  }

  // Apply fn, a device function, elementwise to the matrix.
  template <typename Fn>
  __host__ void apply(Fn fn) {
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_ptr({i, j}).rank_ == BCL::rank()) {
          size_t block_dim = 1024;
          size_t block_size = std::min(std::size_t(block_dim), tile_size());
          size_t num_blocks = (tile_size() + block_size - 1) / block_size;
          apply_matrix_impl_<<<num_blocks, block_size>>>(tile_ptr({i, j}), tile_size(), fn);
        }
      }
    }
  }

  template <typename Fn>
  __host__ void apply_by_index(Fn fn) {
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_ptr({i, j}).rank_ == BCL::rank()) {
          size_t extent_m = tile_shape()[0];
          size_t extent_n = tile_shape()[1];
          size_t block_dim = 32;
          dim3 num_blocks((extent_m + block_dim - 1) / block_dim, (extent_n + block_dim - 1) / block_dim);
          dim3 block_size(std::min(std::size_t(block_dim), extent_m), std::min(std::size_t(block_dim), extent_n));
          apply_matrix_2d_idx_impl_<<<num_blocks, block_size>>>
             (tile_ptr({i, j}), tile_shape({i, j})[0], tile_shape({i, j})[1],
                                tile_shape()[0], tile_shape()[1],
                                i*tile_shape()[0], j*tile_shape()[1],
                                fn);
        }
      }
    }
  }

  __host__ std::vector<T> get_matrix() const {
    std::vector<T> local_matrix(shape()[0]*shape()[1]);
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        auto f = get_tile({i, j});
        std::vector<T> local_tile(f.size());
        cudaMemcpy(local_tile.data(), f.data(), sizeof(T)*f.size(), cudaMemcpyDeviceToHost);
        f.destroy();

        size_t offset_idx_m = tile_shape()[0]*i;
        size_t offset_idx_n = tile_shape()[1]*j;
        for (size_t ii = 0; ii < tile_shape({i, j})[0]; ii++) {
          for (size_t jj = 0; jj < tile_shape({i, j})[1]; jj++) {
            size_t li = ii + offset_idx_m;
            size_t lj = jj + offset_idx_n;
            local_matrix[li*shape()[1] + lj] = local_tile[index({ii, jj}, tile_shape())];
          }
        }
      }
    }
    return local_matrix;
  }

  __host__ __device__ std::size_t index(matrix_dim idx, matrix_dim dims) const {
    // Column-major indexing, here.
    return idx[0] + idx[1]*dims[0];
  }

  __host__ void randomize() {
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_ptr({i, j}).rank_ == BCL::rank()) {
          std::vector<T> local(tile_size());
          for (size_t k = 0; k < local.size(); k++) {
            local[k] = (lrand48() % 101) - 50;
          }
          cudaMemcpy(tile_ptr({i, j}).local(), local.data(), sizeof(T)*tile_size(), cudaMemcpyHostToDevice);
        }
      }
    }
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
          printf("%2lu ", tile_ptr({i, j}).rank_);
        }
        printf("\n");
      }
    }
  }

};

} // end cuda

} // end BCL
