#include <cuda.h>

namespace BCL {
namespace cuda {
  
// Rebind Allocator<U> as type T
template <typename Allocator, typename T>
using rebind_allocator_t = typename Allocator::rebind<T>::other;

template <typename T, typename Allocator>
T* allocate_with(size_t size) {
  return rebind_allocator_t<Allocator, T>{}.allocate(size);
}

template <typename T, typename Allocator>
void deallocate_with(T* ptr) {
  return rebind_allocator_t<Allocator, T>{}.deallocate(ptr);
}

template <typename T>
class cuda_allocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using is_always_equal = std::true_type;

  template <class U> struct rebind {
    typedef cuda_allocator<U> other;
  };

  cuda_allocator() = default;
  cuda_allocator(const cuda_allocator&) = default;

  pointer allocate(size_type n) {
    if (n == 0) {
      return nullptr;
    }
    T* ptr;
    cudaError_t error = cudaMalloc(&ptr, n*sizeof(value_type));
    if (error != cudaSuccess || ptr == nullptr) {
      throw std::bad_alloc();
    } else {
      return ptr;
    }
  }

  void deallocate(pointer ptr, size_type n = 0) {
    if (ptr != nullptr) {
      cudaFree(ptr);
    }
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
  using is_always_equal = std::true_type;

  template <class U> struct rebind {
    typedef bcl_allocator<U> other;
  };

  bcl_allocator() = default;
  bcl_allocator(const bcl_allocator&) = default;

  pointer allocate(size_type n) {
    if (n == 0) {
      return nullptr;
    }
    auto lptr_ = BCL::cuda::alloc<value_type>(n);
    if (lptr_ == nullptr) {
      throw std::bad_alloc();
    }
    return lptr_.local();
  }

  void deallocate(pointer ptr, size_type n = 0) {
    if (ptr != nullptr) {
      auto gptr = __to_cuda_gptr<value_type>(ptr);
      BCL::cuda::dealloc<value_type>(gptr);
    }
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
