#pragma once

namespace BCL {
namespace cuda {

extern size_t shared_segment_size;
extern char* smem_base_ptr;
extern __device__ char* device_smem_base_ptr;

template <typename T>
struct ptr {
  size_t rank_ = 0;
  size_t ptr_ = 0;

  typedef T type;

  __host__ __device__ ptr(uint64_t rank = 0, uint64_t ptr = 0)
                         : rank_(rank), ptr_(ptr) {}

  ptr(const ptr<T>& other) = default;
  ptr<T>& operator=(const ptr<T>& other) = default;

  __host__ __device__ ptr(const std::nullptr_t) {
    this->rank_ = 0;
    this->ptr_ = 0;
  }

  __host__ __device__ ptr <T> &operator=(const std::nullptr_t null) {
    this->rank_ = 0;
    this->ptr_ = 0;
    return *this;
  }

  __host__ __device__ bool operator==(const std::nullptr_t null) const {
    return rank_ == 0 && ptr_ == 0;
  }

  __host__ __device__ bool operator!=(const std::nullptr_t null) const {
    return !(*this == null);
  }

  __host__ __device__ bool is_local() const {
    return (*this != nullptr) && rank_ == BCL::rank();
  }

  __host__ __device__ operator const ptr<void>() const noexcept {
    return ptr<void>(rank_, ptr_);
  }

  // Local pointer to somewhere in my shared memory segment.
  __host__ __device__ T *local() const {
    return rptr();
    /*
    if (rank_ != BCL::rank()) {
      // TODO: Exception?
      // fprintf(stderr, "error: calling local() on a remote ptr\n");
      return nullptr;
    } else {
      return rptr();
    }
    */
  }

  // Pointer to shared memory segment on another node.
  // Users should not use this unless they're writing
  // custom SHMEM.
  __host__ __device__ T *rptr() const {
    if (*this == nullptr) {
      return nullptr;
    }
    #ifdef  __CUDA_ARCH__
      return (T *) (((char *) BCL::cuda::device_smem_base_ptr) + ptr_);
    #else
      return (T *) (((char *) BCL::cuda::smem_base_ptr) + ptr_);
    #endif
  }

  __host__ __device__ ptr <T> operator+(const size_t offset) const {
    return ptr <T> (rank_, ptr_ + offset*sizeof(T));
  }

  __host__ __device__ ptr <T> operator-(const size_t offset) const {
    return ptr <T> (rank_, ptr_ - offset*sizeof(T));
  }

  __host__ __device__ std::ptrdiff_t operator-(const ptr <T> &ptr) const {
    return (this->ptr_ - ptr.ptr_) / sizeof(T);
  }

  __host__ __device__ ptr <T> &operator++(int) {
    ptr_ += sizeof(T);
    return *this;
  }

  __host__ __device__ ptr <T> &operator++() {
    ptr_ += sizeof(T);
    return *this;
  }

  __host__ __device__ ptr <T> &operator--(int) {
    ptr_ -= sizeof(T);
    return *this;
  }

  __host__ __device__ ptr <T> &operator--() {
    ptr_ -= sizeof(T);
    return *this;
  }

  __host__ __device__ ptr <T> &operator+=(const size_t offset) {
    ptr_ += offset*sizeof(T);
    return *this;
  }

  __host__ __device__ ptr <T> &operator-=(const size_t offset) {
    ptr_ -= offset*sizeof(T);
    return *this;
  }

  __host__ __device__ bool operator==(const ptr <T> &ptr) const {
    return (this->ptr_ == ptr.ptr_ && this->rank_ == ptr.rank_);
  }

  __host__ __device__ bool operator!=(const ptr <T> &ptr) const {
    return this->ptr_ != ptr.ptr_ || this->rank_ != ptr.rank_;
  }

  std::string str() const {
    if (*this != nullptr) {
      return "ptr(" + std::to_string(rank_) + ": " +
        std::to_string(ptr_) + ")";
    } else {
      return "ptr(nullptr)";
    }
  }
  void print() const {
    printf("%s\n", str().c_str());
  }
};

template <typename T, typename U>
__host__ __device__ ptr<T> reinterpret_pointer_cast(const BCL::cuda::ptr<U> ptr) noexcept {
  return BCL::cuda::ptr<T>(ptr.rank_, ptr.ptr_);
}

}

template <typename T, typename U>
__host__ __device__ BCL::cuda::ptr<T> reinterpret_pointer_cast(const BCL::cuda::ptr<U> ptr) noexcept {
  return BCL::cuda::ptr<T>(ptr.rank_, ptr.ptr_);
}
}
