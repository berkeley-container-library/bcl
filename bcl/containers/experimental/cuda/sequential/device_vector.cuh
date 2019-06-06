#pragma once

#include <unistd.h>
#include <cuda.h>
#include <thrust/device_vector.h>

/*
  NOTE: This is a `device_vector`, a simple vector type that is allocated
        on the host side and refers to memory on a GPU.  Memory can be
        accessed (e.g. with `operator[]`) either on the host side or
        inside a GPU kernel on the GPU which owns the memory.

        Usage: note that `device_vector`'s copy constructor creates
               *shallow copies*, and the destructor does not release
               the resources.  YOU MUST EXPLICITLY CALL `destroy`
               to release resources.

               This design is necessary in order to allow passing the
               `device_vector` (by value) to the GPU (which requires copy
               constructing). The other possible choice would have been
               defining this as a `Managed` type and passing to CUDA kernels
               by reference.  However, this would have required allocating all
               `device_vector`'s with `new`.

*/

namespace BCL {

namespace cuda {

template <typename D>
class device_vector_reference;

template <typename D>
class const_device_vector_reference;

template <typename T>
class device_vector {
public:

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference_type = device_vector_reference<device_vector<value_type>>;
  using const_reference_type = const_device_vector_reference<device_vector<value_type>>;

  __host__ __device__ device_vector(const device_vector& other) {
    shallow_copy(other);
  }

  __host__ __device__ void shallow_copy(const device_vector& other) {
    d_ptr_ = other.d_ptr_;
    capacity_ = other.capacity_;
    size_ = other.size_;
  }

  device_vector(device_vector&&) = default;
  device_vector& operator=(device_vector&&) = default;

  // Constructors

  device_vector() = default;

  template <typename InputIt>
  device_vector(InputIt first, InputIt last) {
    assign(first, last);
  }

  explicit device_vector(size_type count,
                         const T& value = T()) {
    capacity_ = count;
    size_ = count;
    cudaMalloc(&d_ptr_, sizeof(value_type)*count);
    std::vector<value_type> v(count, value);

    cudaMemcpy(d_ptr_, v.data(), sizeof(value_type)*v.size(), cudaMemcpyHostToDevice);
  }

  void resize(size_type count) {
    if (count != capacity()) {
      value_type* new_device_ptr;
      cudaMalloc(&new_device_ptr, sizeof(value_type)*count);
      cudaMemcpy(new_device_ptr, d_ptr_, sizeof(value_type)*std::min(count, capacity()),
                 cudaMemcpyDeviceToDevice);

      value_type* old_ptr = d_ptr_;
      d_ptr_ = new_device_ptr;
      cudaFree(old_ptr);
      capacity_ = count;
      size_ = count;
    }
  }

  template <typename InputIt>
  void assign(InputIt first, InputIt last) {
    std::vector<value_type> v(first, last);
    if (capacity() != v.size()) {
      resize(v.size());
    }
    cudaMemcpy(d_ptr_, v.data(), sizeof(value_type)*v.size(), cudaMemcpyHostToDevice);
  }

  __host__ __device__ reference_type operator[](size_type idx) {
    return reference_type(idx, this);
  }

  __host__ __device__ const_reference_type operator[](size_type idx) const {
    return const_reference_type(idx, this);
  }

  __host__ __device__ size_type capacity() const noexcept {
    return capacity_;
  }

  __host__ __device__ size_type size() const noexcept {
    return size_;
  }

  __host__ __device__ value_type* data() {
    return d_ptr_;
  }

  __host__ __device__ value_type* data() const {
    return d_ptr_;
  }

  __host__ __device__ ~device_vector() {
    #ifdef __CUDA_ARCH__
      // XXX
    #else
      // XXX
    #endif
  }

  void destroy() {
    cudaFree(d_ptr_);
  }

private:
  value_type* d_ptr_ = nullptr;
  size_type capacity_ = 0;
  size_type size_ = 0;
};

template <typename D>
class device_vector_reference {
public:
  using size_type = typename D::size_type;
  using value_type = typename D::value_type;

  __host__ __device__ device_vector_reference(size_t idx, D* vec) : idx_(idx), vec_(vec) {}

  __device__ value_type get() const {
    return vec_->data()[idx_];
  }

  __host__ __device__ operator value_type() const {
    #ifdef __CUDA_ARCH__
      return vec_->data()[idx_];
    #else
      value_type value;
      cudaMemcpy(&value, vec_->data() + idx_, sizeof(value_type),
                 cudaMemcpyDeviceToHost);
      return value;
    #endif
  }

  __host__ __device__ value_type operator=(const value_type& value) {
    #ifdef __CUDA_ARCH__
      /*
      TODO:
      if (vec_ == nullptr) {
        printf("AGH! got a nullptr\n");
      }
      */
      vec_->data()[idx_] = value;
      return value;
    #else
      cudaMemcpy(vec_->data() + idx_, &value, sizeof(value_type),
                 cudaMemcpyHostToDevice);
      return value;
    #endif
  }

private:
  size_type idx_ = 0;
  D* vec_ = nullptr;
};

template <typename D>
class const_device_vector_reference {
public:
  using size_type = typename D::size_type;
  using value_type = typename D::value_type;

  __host__ __device__ const_device_vector_reference(size_t idx, const D* vec) : idx_(idx), vec_(vec) {}

  __device__ value_type get() const {
    return vec_->data()[idx_];
  }

  __host__ __device__ operator value_type() const {
    #ifdef __CUDA_ARCH__
      return vec_->data()[idx_];
    #else
      value_type value;
      cudaMemcpy(&value, vec_->data() + idx_, sizeof(value_type),
                 cudaMemcpyDeviceToHost);
      return value;
    #endif
  }

private:
  size_type idx_ = 0;
  const D* vec_ = nullptr;
};

} // end cuda
} // end BCL
