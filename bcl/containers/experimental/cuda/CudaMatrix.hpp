

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
#include <bcl/backends/mpi/team_conv.hpp>
#include <bcl/containers/experimental/cuda/util/matrix_indexing.hpp>

namespace BCL {

namespace cuda {

template <typename T, typename Allocator, typename Indexing>
struct CudaMatrix;

template <typename T, typename Indexing>
struct CudaMatrixView;

template <typename T, typename Fn>
__global__ void apply_fn_1d_impl_(T* a, T* b, size_t size, Fn fn) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < size) {
    a[tid] = fn(a[tid], b[tid]);
  }
}

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
    size_t idx = tidx*tile_size_n + tidy;
    data.local()[idx] = fn(data.local()[idx], tidx + tile_offset_m, tidy + tile_offset_n);
  }
}

template <typename T,
          typename Allocator = BCL::cuda::bcl_allocator<T>,
          typename Indexing = RowMajorIndexing>
struct CudaMatrix;

template <typename T,
          typename Indexing = RowMajorIndexing>
struct CudaMatrixView;

template <typename T, typename Indexing = RowMajorIndexing>
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
          ptr = BCL::cuda::alloc<T>(tile_size({i, j}));
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

  std::vector<BCL::UserTeam> column_teams_;
  std::vector<BCL::UserTeam> row_teams_;

  void init_teams() {
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      std::vector<size_t> row_procs;
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        row_procs.push_back(tile_rank({i, j}));
      }
      /*
      if (BCL::rank() == 0) {
        printf("Row team %lu: ", i);
        print_vec(row_procs);
      }
      */
      row_teams_.push_back(BCL::UserTeam(row_procs));
    }

    for (size_t j = 0; j < grid_shape()[1]; j++) {
      std::vector<size_t> column_procs;
      for (size_t i = 0; i < grid_shape()[0]; i++) {
        column_procs.push_back(tile_rank({i, j}));
      }
      /*
      if (BCL::rank() == 0) {
        printf("Column team %lu: ", j);
        print_vec(column_procs);
      }
      */
      column_teams_.push_back(BCL::UserTeam(column_procs));
    }
  }

  std::vector<std::vector<BCL::backend::MPICommWrapper>> column_teams_mpi_;
  std::vector<std::vector<BCL::backend::MPICommWrapper>> row_teams_mpi_;

  void init_comms(size_t i_block) {
    if (column_teams_.empty()) {
      init_teams();
    }
    for (size_t i = 0; i < column_teams_.size(); i++) {
      column_teams_mpi_.push_back(std::vector<BCL::backend::MPICommWrapper>());
      for (size_t j = 0; j < i_block; j++) {
        BCL::backend::MPICommWrapper comm(column_teams_[i]);
        column_teams_mpi_[i].emplace_back(std::move(comm));
      }
    }
    for (size_t i = 0; i < row_teams_.size(); i++) {
      row_teams_mpi_.push_back(std::vector<BCL::backend::MPICommWrapper>());
      for (size_t j = 0; j < i_block; j++) {
        BCL::backend::MPICommWrapper comm(row_teams_[i]);
        row_teams_mpi_[i].emplace_back(std::move(comm));
      }
    }
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
  __device__ __host__ auto min(U a, V b) const {
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

  __device__ __host__ size_t tile_size(matrix_dim idx) const {
    return tile_shape(idx)[0]*tile_shape(idx)[1];
  }

  __device__ __host__ BCL::cuda::ptr<T> tile_ptr(matrix_dim idx) const {
    return ptrs_[idx[0]*grid_shape()[1] + idx[1]];
  }

  __host__ size_t tile_rank(matrix_dim idx) const {
    size_t i = idx[0];
    size_t j = idx[1];
    size_t vals_idx = i*grid_shape()[1] + j;
    return ptrs_[vals_idx].rank_;
  }

  __host__ auto get_tile(matrix_dim idx) const {
    BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>> x(tile_size(idx));
    BCL::cuda::memcpy(x.data(), tile_ptr(idx), sizeof(T) * tile_size(idx));
    BCL::cuda::flush();
    return x;
  }

  __host__ auto get_local_tile(matrix_dim idx) const {
    T* data = tile_ptr(idx).local();
    return CudaMatrixView<T, Indexing>({tile_shape(idx)[0], tile_shape(idx)[1]}, data);
  }

  template <typename Allocator>
  __host__ void get_tile_no_alloc_(matrix_dim idx,
                         BCL::cuda::device_vector<T, Allocator>& vec) const {
    nvshmem_getmem_nbi(vec.data(), tile_ptr(idx).rptr(), sizeof(T)*tile_size(idx),
                       tile_ptr(idx).rank_);
  }

  __host__ auto arget_tile(matrix_dim idx) const {
    using no_init = typename BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>>::no_init;
    BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>> x(tile_size(idx), no_init{});
    if (x.data() == nullptr) {
      printf("%lu has nullptr x!\n", BCL::rank());
      assert(false);
    }
    // BCL::cuda::memcpy(x.data(), tile_ptr(idx), sizeof(T) * tile_size(idx));
    nvshmem_getmem_nbi(x.data(), tile_ptr(idx).rptr(), sizeof(T)*tile_size(idx),
                       tile_ptr(idx).rank_);
    return cuda_future<BCL::cuda::device_vector<T, BCL::cuda::bcl_allocator<T>>>
                      (std::move(x), cuda_request());
  }

  __host__ auto arget_tile_exp(matrix_dim idx) const {
    auto data = BCL::cuda::alloc<T>(tile_size(idx));
    if (data == nullptr) {
      printf("(%lu has nullptr x!\n", BCL::rank());
      assert(false);
    }
    CudaMatrix<T, BCL::cuda::bcl_allocator<T>, Indexing> x({tile_shape(idx)[0], tile_shape(idx)[1]},
                                                           data.local());

    std::thread memcpy_thread([](auto dst, auto src, auto n) {
      BCL::cuda::memcpy(dst, src, n);
    }, data, tile_ptr(idx), sizeof(T) * tile_size(idx));
    using request_type = cuda_thread_request<decltype(memcpy_thread)>;
    return cuda_future<CudaMatrix<T, BCL::cuda::bcl_allocator<T>, Indexing>, request_type>
                      (std::move(x),
                       cuda_thread_request<decltype(memcpy_thread)>
                                                    (std::move(memcpy_thread)));
  }

  // Apply fn, a device function, elementwise to the matrix.
  template <typename Fn>
  __host__ void apply(Fn fn) {
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_ptr({i, j}).rank_ == BCL::rank()) {
          size_t block_dim = 1024;
          size_t block_size = std::min(std::size_t(block_dim), tile_size({i, j}));
          size_t num_blocks = (tile_size({i, j}) + block_size - 1) / block_size;
          apply_matrix_impl_<<<num_blocks, block_size>>>(tile_ptr({i, j}), tile_size({i, j}), fn);
        }
      }
    }
  }

  __host__ Matrix& operator=(T value) {
    auto assign_value = [=] __device__ (T current_val) { return value; };
    apply(assign_value);
    return *this;
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
            // local_matrix[li*shape()[1] + lj] = local_tile[index({ii, jj}, tile_shape())];
            local_matrix[li*shape()[1] + lj] =
                   local_tile[Indexing().index(ii,
                                               jj,
                                               tile_shape({i, j})[0],
                                               tile_shape({i, j})[1])];
          }
        }
      }
    }
    return local_matrix;
  }
  
  __host__ size_t my_num_tiles() {
    size_t num_tiles = 0;
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_rank({i, j}) == BCL::rank()) {
          num_tiles++;
        }
      }
    }
    return num_tiles;
  }

  __host__ __device__ std::size_t index(matrix_dim idx, matrix_dim dims) const {
    // Column-major indexing, here.
    // return idx[0] + idx[1]*dims[0];
    // Row-major indexing, here.
    return idx[0]*dims[1] + idx[1];
  }

  __host__ void randomize() {
    for (size_t i = 0; i < grid_shape()[0]; i++) {
      for (size_t j = 0; j < grid_shape()[1]; j++) {
        if (tile_ptr({i, j}).rank_ == BCL::rank()) {
          std::vector<T> local(tile_size({i, j}));
          for (size_t k = 0; k < local.size(); k++) {
            local[k] = (lrand48() % 101) - 50;
          }
          cudaMemcpy(tile_ptr({i, j}).local(), local.data(), sizeof(T)*tile_size({i, j}), cudaMemcpyHostToDevice);
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

template <typename T>
__global__ void cudamatrix_copy_impl_(T* data, size_t size, T value) {
  size_t tid = threadIdx.x + blockIdx.x*blockDim.x;

  if (tid < size) {
    data[tid] = value;
  }
}

// GPU matrix
template <typename T,
          typename Allocator,
          typename Indexing>
struct CudaMatrix {
  using value_type = T;
  using allocator_type = Allocator;
  // XXX: not to be confused with index_type
  using indexing_type = Indexing;

  struct matrix_dim;

  CudaMatrix(matrix_dim shape, size_t ld = 0) : m_(shape[0]), n_(shape[1]), ld_(ld) {
    if (ld_ == 0) {
      ld_ = Indexing().default_ld(m(), n());
    }
    data_ = allocate_with<T, Allocator>(size());
  }

  CudaMatrix(matrix_dim shape, T* data, size_t ld = 0) : m_(shape[0]),
             n_(shape[1]), data_(data), ld_(ld)
  {
    if (ld_ == 0) {
      ld_ = Indexing().default_ld(m(), n());
    }
  }

  CudaMatrix() = delete;
  // CudaMatrix(const CudaMatrix&) = delete;
  // CudaMatrix& operator=(const CudaMatrix&) = delete;

  CudaMatrix(const CudaMatrix& other) {
    m_ = other.m_;
    n_ = other.n_;
    ld_ = other.ld_;
    data_ = allocate_with<T, Allocator>(size());
    cudaMemcpy(data_, other.data_, sizeof(T)*other.size(), cudaMemcpyDeviceToDevice);
  }

  CudaMatrix& operator=(const CudaMatrix& other) {
    deallocate_with<T, Allocator>(data_);
    m_ = other.m_;
    n_ = other.n_;
    ld_ = other.ld_;
    data_ = allocate_with<T, Allocator>(size());
    cudaMemcpy(data_, other.data_, sizeof(T)*other.size(), cudaMemcpyDeviceToDevice);
    return *this;
  }

  CudaMatrix& operator=(const CudaMatrixView<T, Indexing>& other) {
    deallocate_with<T, Allocator>(data_);
    m_ = other.m_;
    n_ = other.n_;
    ld_ = other.ld_;
    data_ = allocate_with<T, Allocator>(size());
    cudaMemcpy(data_, other.data_, sizeof(T)*other.size(), cudaMemcpyDeviceToDevice);
    return *this;
  }

  CudaMatrix(CudaMatrix&& other) {
    move(std::move(other));
  }

  CudaMatrix& operator=(CudaMatrix&& other) {
    move(std::move(other));
    return *this;
  }

  operator CudaMatrixView<T, Indexing>() {
    return view();
  }

  CudaMatrixView<T, Indexing> view() {
    return CudaMatrixView<T, Indexing>({shape()[0], shape()[1]}, data(), ld());
  }

  void move(CudaMatrix&& other) {
    data_ = other.data_;
    m_ = other.m_;
    n_ = other.n_;
    ld_ = other.ld_;
    other.data_ = nullptr;
    other.m_ = 0;
    other.n_ = 0;
    other.ld_ = 0;
  }

  ~CudaMatrix() {
    deallocate_with<T, Allocator>(data_);
  }

  size_t m() const {
    return m_;
  }

  size_t n() const {
    return n_;
  }

  // Physical size of leading dimension
  size_t ld() const {
    return ld_;
  }

  // Logical shape of matrix
  matrix_dim shape() const {
    return {m(), n()};
  }

  // Size of matrix data in number of elements
  size_t size() const {
    return Indexing().size(m(), n(), ld());
  }

  T* data() {
    return data_;
  }

  CudaMatrix& operator+=(CudaMatrix& other) {
    auto binary_add = [] __device__ (T x, T y) -> T { return x + y; };
    return apply_binary_op(other, binary_add);
  }

  template <typename Fn>
  CudaMatrix& apply_binary_op(CudaMatrix& other, Fn&& fn) {
    assert(ld() == other.ld());
    size_t block_dim = 1024;
    size_t block_size = std::min(block_dim, size());
    size_t num_blocks = (size() + block_size - 1) / block_size;
    apply_fn_1d_impl_<<<num_blocks, block_size>>>(data(), other.data(), size(), fn);
    cudaDeviceSynchronize();
    return *this;
  }

  CudaMatrix& operator=(T value) {
    size_t block_dim = 1024;
    size_t block_size = std::min(std::size_t(block_dim), size());
    size_t num_blocks = (size() + block_size - 1) / block_size;
    cudamatrix_copy_impl_<<<num_blocks, block_size>>>(data(), size(), value);
    cudaDeviceSynchronize();
    return *this;
  }

  T* data_;
  size_t m_;
  size_t n_;
  size_t ld_;

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

// Row-major GPU view
template <typename T, typename Indexing>
struct CudaMatrixView {
  using value_type = T;
  // XXX: not to be confused with index_type.
  using indexing_type = Indexing;
  
  struct matrix_dim;

  CudaMatrixView(matrix_dim shape, T* data, size_t ld = 0) : m_(shape[0]),
                 n_(shape[1]), data_(data), ld_(ld)
  {
    if (ld_ == 0) {
      ld_ = Indexing().default_ld(m(), n());
    }
  }

  CudaMatrixView() = default;
  ~CudaMatrixView() = default;

  CudaMatrixView(const CudaMatrixView&) = default;
  CudaMatrixView& operator=(const CudaMatrixView&) = default;
  CudaMatrixView(CudaMatrixView&& other) = default;
  CudaMatrixView& operator=(CudaMatrixView&& other) = default;

  size_t m() const {
    return m_;
  }

  size_t n() const {
    return n_;
  }

  // Physical size of leading dimension
  size_t ld() const {
    return ld_;
  }

  // Logical shape of matrix
  matrix_dim shape() const {
    return {m(), n()};
  }

  // Size of matrix data in number of elements
  size_t size() const {
    return Indexing().size(m(), n(), ld());
  }

  T* data() {
    return data_;
  }

  CudaMatrixView& operator+=(CudaMatrixView other) {
    auto binary_add = [] __device__ (T x, T y) -> T { return x + y; };
    return apply_binary_op(other, binary_add);
  }

  template <typename Fn>
  CudaMatrixView& apply_binary_op(CudaMatrixView other, Fn&& fn) {
    assert(ld() == other.ld());
    size_t block_dim = 1024;
    size_t block_size = std::min(block_dim, size());
    size_t num_blocks = (size() + block_size - 1) / block_size;
    apply_fn_1d_impl_<<<num_blocks, block_size>>>(data(), other.data(), size(), fn);
    cudaDeviceSynchronize();
    return *this;
  }

  T* data_ = nullptr;
  size_t m_ = 0;
  size_t n_ = 0;
  size_t ld_ = 0;

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
