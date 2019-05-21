
#pragma once

#include <mpi.h>

namespace BCL {

template <typename T>
class mpi_allocator {
public:
  using value_type = T;
  using size_type = std::size_t;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;

  mpi_allocator() = default;
  mpi_allocator(const mpi_allocator&) = default;

  pointer allocate(size_type n) {
    T* ptr;
    MPI_Alloc_mem(n*sizeof(value_type), MPI_INFO_NULL, &ptr);
    return ptr;
  }

  void deallocate(pointer ptr, size_type n) {
    MPI_Free_mem(ptr);
  }

  template<typename... Args>
  void construct(pointer ptr, Args&&... args) {
    new(ptr) T(std::forward<Args>(args)...);
  }

  void destroy(pointer ptr) {
    ptr->~T();
  }

  bool operator==(const mpi_allocator&) const {
    return true;
  }

  bool operator!=(const mpi_allocator& other) const {
    return !operator==(other);
  }
};

template <typename T>
using async_allocator = mpi_allocator<T>;

/*
template <typename T>
using async_allocator = std::allocator<T>;
*/

}
