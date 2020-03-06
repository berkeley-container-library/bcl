#include <cuda.h>

namespace BCL {
namespace cuda {

template <typename T>
class cuda_allocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  cuda_allocator() = default;
  cuda_allocator(const cuda_allocator&) = default;

  pointer allocate(size_type n) {
    T* ptr;
    cudaMalloc(&ptr, n*sizeof(value_type));
    return ptr;
  }

  void deallocate(pointer ptr, size_type n = 0) {
    cudaFree(ptr);
  }

  template<typename... Args>
  void construct(pointer ptr, Args&&... args) {
    new(ptr) T(std::forward<Args>(args)...);
  }

  void destroy(pointer ptr) {
    ptr->~T();
  }

  bool operator==(const cuda_allocator&) const {
    return true;
  }

  bool operator!=(const cuda_allocator& other) const {
    return !operator==(other);
  }
};

template <typename T>
class bcl_allocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  bcl_allocator() = default;
  bcl_allocator(const bcl_allocator&) = default;

  pointer allocate(size_type n) {
    auto lptr_ = BCL::cuda::alloc<value_type>(n);
    if (lptr_ == nullptr) {
      throw std::bad_alloc();
    }
    return lptr_.local();
  }

  void deallocate(pointer ptr, size_type n = 0) {
    auto gptr = __to_cuda_gptr<value_type>(ptr);
    BCL::cuda::dealloc<value_type>(gptr);
  }

  template<typename... Args>
  void construct(pointer ptr, Args&&... args) {
    new(ptr) T(std::forward<Args>(args)...);
  }

  void destroy(pointer ptr) {
    ptr->~T();
  }

  bool operator==(const bcl_allocator&) const {
    return true;
  }

  bool operator!=(const bcl_allocator& other) const {
    return !operator==(other);
  }
};

} // end cuda

} // end BCL
