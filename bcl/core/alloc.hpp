#pragma once

#include <bcl/bcl.hpp>

// TODO: Exceptions, new() allocator which
// calls constructor, rdealloc()

namespace BCL {

template <typename T>
inline GlobalPtr <T> alloc(const size_t size) {
  return local_malloc <T> (size);
}

template <typename T>
inline void dealloc(GlobalPtr <T> ptr) {
  local_free <T> (ptr);
}

template <typename T, typename... Args>
inline GlobalPtr<T> new_(Args&& ...args) {
  BCL::GlobalPtr<T> ptr = BCL::alloc<T>(1);
  new (ptr.local()) T(std::forward<Args>(args)...);
  return ptr;
}

template <typename T>
inline void delete_(BCL::GlobalPtr<T> ptr) {
  ptr.local()->~T();
  BCL::dealloc<T>(ptr);
}

template <typename T>
inline BCL::GlobalPtr<T> __to_global_ptr(T* ptr) {
  // TODO: Hrmmm... what to do for local(NULL) -> Global()?
  if (ptr == nullptr) {
    return nullptr;
  }

  size_t offset = sizeof(T)*(ptr - reinterpret_cast<T*>(BCL::smem_base_ptr));

  if (ptr < BCL::smem_base_ptr || offset >= BCL::shared_segment_size) {
    // XXX: alternative would be returning nullptr
    throw std::runtime_error("BCL::__to_global_ptr(): given pointer is outside shared segment.");
  }

  return BCL::GlobalPtr<T>(BCL::rank(), offset);
}

template <typename T>
class bcl_allocator {
public:
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  bcl_allocator() = default;
  bcl_allocator(const bcl_allocator&) = default;

  pointer allocate(size_type n) {
    auto lptr_ = alloc<value_type>(n);
    if (lptr_ == nullptr) {
      throw std::bad_alloc();
    }
    return lptr_.local();
  }

  void deallocate(pointer ptr, size_type n) {
    auto gptr = __to_global_ptr(ptr);
    dealloc<value_type>(gptr);
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

  template <typename U> struct rebind { typedef bcl_allocator<U> other; };
};

}
