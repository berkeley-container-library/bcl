// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <type_traits>

#include <bcl/core/teams.hpp>
#include <bcl/core/GlobalRef.hpp>

namespace BCL {

extern uint64_t shared_segment_size;
extern void *smem_base_ptr;

extern inline size_t rank(BCL::Team &&team);

template <class T, class M> M get_member_type(M T:: *);
#define GET_MEMBER_TYPE(mem) decltype(BCL::get_member_type(mem))

// For GlobalPtr <T> object_ptr, a global pointer to a struct of type T,
// and item, a type M member of struct T, return a GlobalPtr <M> which
// points to the member item within object_ptr
#define pointerto(item, object_ptr) \
  BCL::reinterpret_pointer_cast <GET_MEMBER_TYPE(&std::remove_reference<decltype(object_ptr)>::type::type::item)> \
  (BCL::reinterpret_pointer_cast <char> (object_ptr) +\
   offsetof(typename std::remove_reference<decltype(object_ptr)>::type::type, item))

/// Global pointer class.  Provides a way to point to remote memory.
template <typename T>
struct GlobalPtr {

  // TODO: replace with requires() for C++20
  static_assert(std::is_trivially_copyable_v<T> || std::is_same_v<std::decay_t<T>, void>);

  std::size_t rank = 0;
  std::size_t ptr = 0;

  typedef T type;

  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = BCL::GlobalPtr<T>;
  using const_pointer = BCL::GlobalPtr<std::add_const_t<T>>;
  using reference = BCL::GlobalRef<T>;
  using iterator_category = std::random_access_iterator_tag;

  GlobalPtr() = default;
  ~GlobalPtr() = default;
  GlobalPtr(const GlobalPtr &) = default;
  GlobalPtr &operator=(const GlobalPtr &) = default;
  GlobalPtr(GlobalPtr&&) = default;
  GlobalPtr& operator=(GlobalPtr&&) = default;

  GlobalPtr(std::size_t rank, std::size_t ptr) : rank(rank), ptr(ptr) {}

  GlobalPtr(std::nullptr_t null) : rank(0), ptr(0) {}

  // TODO: convert SFINAE to requires() for C++20
  template <__BCL_REQUIRES(!std::is_same_v<std::decay_t<T>, void> && !std::is_const_v<T>)>
  operator GlobalPtr<void>() const noexcept {
    return GlobalPtr<void>(rank, ptr);
  }

  // TODO: convert SFINAE to requires() for C++20
  template <__BCL_REQUIRES(!std::is_same_v<std::decay_t<T>, void>)>
  operator GlobalPtr<const void>() const noexcept {
    return GlobalPtr<const void>(rank, ptr);
  }

  template <__BCL_REQUIRES(!std::is_const_v<T>)>
  operator const_pointer() const noexcept {
    return const_pointer(rank, ptr);
  }

  GlobalPtr& operator=(std::nullptr_t null) {
    rank = 0;
    ptr = 0;
    return *this;
  }

  bool operator==(pointer other) const noexcept {
    return (rank == other.rank && ptr == other.ptr);
  }

  bool operator!=(pointer other) const noexcept {
    return !(*this == other);
  }

  bool operator==(std::nullptr_t null) const noexcept {
    return (rank == 0 && ptr == 0);
  }

  bool operator!=(std::nullptr_t null) const noexcept {
    return !(*this == null);
  }

  /// Dereference the global pointer, returning a global reference `GlobalRef`
  /// that can be used to read or write to the memory location.
  reference operator*() const noexcept {
    return reference(*this);
  }

  reference operator[](difference_type offset) const noexcept {
    return *(*this + offset);
  }

  pointer operator+(difference_type offset) const noexcept {
    return pointer(rank, ptr + offset*sizeof(T));
  }

  pointer operator-(difference_type offset) const noexcept {
    return pointer(rank, ptr - offset*sizeof(T));
  }

  difference_type operator-(pointer other) const noexcept {
    return (ptr - difference_type(other.ptr)) / sizeof(T);
  }

  pointer& operator++() noexcept {
    ptr += sizeof(T);
    return *this;
  }

  pointer operator++(int) noexcept {
    pointer other(*this);
    ++(*this);
    return other;
  }

  pointer& operator--() noexcept {
    ptr -= sizeof(T);
    return *this;
  }

  pointer operator--(int) noexcept {
    pointer other(*this);
    --(*this);
    return other;
  }

  pointer& operator+=(difference_type offset) noexcept {
    ptr += offset*sizeof(T);
    return *this;
  }

  pointer& operator-=(difference_type offset) noexcept {
    ptr -= offset*sizeof(T);
    return *this;
  }

  /// Check if `GlobalPtr` points to memory on the local rank.
  bool is_local() const noexcept {
    return rank == BCL::rank();
  }

  // Local pointer to somewhere in my shared memory segment.
  // XXX: should we alloc result to be undefined if called on remote pointer?
  //      Currently defined to nullptr.
  /// Return a local pointer of type T* to the location pointed to by the
  /// global pointer.
  /// 
  /// If pointer is pointing to remove memory, returns `nullptr`.
  T *local() const {
    if (rank != BCL::rank()) {
      BCL_DEBUG(throw debug_error("calling local() on a remote GlobalPtr\n"));
      return nullptr;
    } else {
      return (T *) (((char *) BCL::smem_base_ptr) + ptr);
    }
  }

  // Pointer to shared memory segment on another node.
  // Users should not use this unless they're writing
  // custom SHMEM.
  T *rptr() const noexcept {
    return (T *) (((char *) BCL::smem_base_ptr) + ptr);
  }

  std::string str() const {
    if (*this != nullptr) {
      return "GlobalPtr(" + std::to_string(rank) + ": " +
        std::to_string(ptr) + ")";
    } else {
      return "GlobalPtr(nullptr)";
    }
  }
  void print() const {
    printf("%s\n", str().c_str());
  }
};

/// Cast a `GlobalPtr` to point to memory of another type.
template <typename T, typename U>
inline GlobalPtr<T> reinterpret_pointer_cast(GlobalPtr<U> ptr) noexcept {
  // TODO: replace with requires() for C++20
  static_assert(!std::is_const_v<U> || std::is_const_v<T>);
  return GlobalPtr<T>(ptr.rank, ptr.ptr);
}

} // end BCL
