// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>

#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>

// TODO: implement API for block read/write

namespace BCL {

namespace cuda {

template <typename D>
class DArray_reference_type;

template <typename D>
class DArray_const_reference_type;

template <typename T>
struct DArray {

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference_type = DArray_reference_type<DArray<T>>;
  using const_reference_type = DArray_const_reference_type<DArray<T>>;

  __host__ __device__ DArray(const DArray& other) {
    #ifdef __CUDA_ARCH__
      d_ptrs_.shallow_copy(other.d_ptrs_);
      size_ = other.size_;
      local_size_ = other.local_size_;
    #else
      d_ptrs_.shallow_copy(other.d_ptrs_);
      size_ = other.size_;
      local_size_ = other.local_size_;
    #endif
  }

  __host__ DArray(size_type size) : size_(size) {
    std::vector<BCL::cuda::ptr<T>> ptrs_(BCL::nprocs(), nullptr);
    local_size_ = (size_ + BCL::nprocs() - 1) / BCL::nprocs();
    ptrs_[BCL::rank()] = BCL::cuda::alloc<T>(local_size_);
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      ptrs_[i] = BCL::broadcast(ptrs_[i], i);
      if (ptrs_[i] == nullptr) {
        throw std::runtime_error("Error! ran out of memory.");
      }
    }

    d_ptrs_.assign(ptrs_.begin(), ptrs_.end());
  }

  __host__ __device__ ~DArray() {
  }

  __device__ T read(size_type idx) const {
    size_type rank = idx / local_size();
    size_type local_idx = idx - rank*local_size();
    T value;
    BCL::cuda::ptr<T> ptr = d_ptrs_.operator[](rank);
    ptr += local_idx;
    BCL::cuda::memcpy(&value,
                      ptr,
                      sizeof(T));
    return value;
  }

  __device__ reference_type operator[](size_type idx) {
    return reference_type(idx, this);
  }

  __device__ void write(size_type idx, T value) {
    size_type rank = idx / local_size();
    size_type local_idx = idx - rank*local_size();
    BCL::cuda::memcpy(static_cast<BCL::cuda::ptr<T>>(d_ptrs_[rank]) + local_idx,
                      &value, sizeof(T));
  }

  __host__ __device__ size_type size() const noexcept {
    return size_;
  }

  __host__ __device__ size_type local_size() const noexcept {
    return local_size_;
  }

  BCL::cuda::device_vector<BCL::cuda::ptr<T>> d_ptrs_;
  size_type size_;
  size_t local_size_;
};

template <typename D>
class DArray_reference_type {
public:
  using size_type = typename D::size_type;
  using value_type = typename D::value_type;

  __host__ __device__ DArray_reference_type(size_t idx, D* vec) : idx_(idx), vec_(vec) {}

  __device__ value_type get() const {
    return static_cast<value_type>(*this);
  }

  __device__ operator value_type() const {
    return vec_->read(idx_);
  }

  __device__ value_type operator=(const value_type& value) {
    vec_->write(idx_, value);
    return value;
  }

private:
  size_type idx_ = 0;
  D* vec_ = nullptr;
};

template <typename D>
class DArray_const_reference_type {
public:
  using size_type = typename D::size_type;
  using value_type = typename D::value_type;

  __host__ __device__ DArray_const_reference_type(size_t idx, const D* vec) : idx_(idx), vec_(vec) {}

  __device__ value_type get() const {
    return static_cast<value_type>(*this);
  }

  __device__ operator value_type() const {
    return vec_->read(idx_);
  }

private:
  size_type idx_ = 0;
  const D* vec_ = nullptr;
};

} // end cuda
} // end BCL
