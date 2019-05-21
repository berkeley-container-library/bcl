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

template <typename T>
struct GlobalPtr;

template <typename T>
struct GlobalPtr {
  uint64_t rank = 0;
  uint64_t ptr = 0;

  typedef T type;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using value_type = T;
  using pointer = BCL::GlobalPtr<T>;
  using reference = BCL::GlobalRef<T>;
  using iterator_category = std::random_access_iterator_tag;

  GlobalPtr(const GlobalPtr <T> &ptr) = default;
  GlobalPtr <T> &operator=(const GlobalPtr <T> &ptr) = default;

  // This allows us to use normal move constructors.
  GlobalPtr(GlobalPtr&& other) : rank(std::move(other.rank)), ptr(std::move(other.ptr)) {
    other = nullptr;
  }

  GlobalPtr& operator=(GlobalPtr&& other) {
    rank = std::move(other.rank);
    ptr = std::move(other.ptr);
    other = nullptr;
    return *this;
  }

  GlobalPtr(const uint64_t rank = 0, const uint64_t ptr = 0) :
    rank(rank), ptr(ptr) {}

  GlobalPtr(const std::nullptr_t null) {
    this->rank = 0;
    this->ptr = 0;
  }

  GlobalPtr <T> &operator=(const std::nullptr_t null) {
    this->rank = 0;
    this->ptr = 0;
    return *this;
  }

  bool operator==(const std::nullptr_t null) const {
    return (rank == 0 && ptr == 0);
  }

  bool operator!=(const std::nullptr_t null) const {
    return !(*this == null);
  }

  /*
  template <typename U>
  operator GlobalPtr <U> () const {
    return GlobalPtr <U> (this->rank, this->ptr);
  }
  */

  operator GlobalPtr<void>() noexcept {
    return GlobalPtr<void>(rank, ptr);
  }

  operator const GlobalPtr<void>() const noexcept {
    return GlobalPtr<void>(rank, ptr);
  }

  bool is_local() const {
    return rank == BCL::rank();
  }

  // Local pointer to somewhere in my shared memory segment.
  // XXX: should we alloc result to be undefined if called on remote pointer?
  //      Currently defined to nullptr.
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
  T *rptr() const {
    return (T *) (((char *) BCL::smem_base_ptr) + ptr);
  }

  GlobalRef<T> operator*() {
    return GlobalRef<T>(*this);
  }

  const GlobalRef<T> operator*() const {
    return GlobalRef<T>(*this);
  }

  GlobalRef<T> operator[](size_type idx) {
    return *(*this + idx);
  }

  const GlobalRef<T> operator[](size_type idx) const {
    return *(*this + idx);
  }

  GlobalPtr <T> operator+(difference_type offset) {
    return GlobalPtr <T> (rank, ptr + offset*sizeof(T));
  }

  const GlobalPtr <T> operator+(difference_type offset) const {
    return GlobalPtr <T> (rank, ptr + offset*sizeof(T));
  }

  GlobalPtr <T> operator-(difference_type offset) {
    return GlobalPtr <T> (rank, ptr - offset*sizeof(T));
  }

  const GlobalPtr <T> operator-(difference_type offset) const {
    return GlobalPtr <T> (rank, ptr - offset*sizeof(T));
  }

  difference_type operator-(const GlobalPtr <T> &ptr) const {
    return (this->ptr - ptr.ptr) / sizeof(T);
  }

  GlobalPtr <T> &operator++(int) {
    ptr += sizeof(T);
    return *this;
  }

  GlobalPtr <T> &operator++() {
    ptr += sizeof(T);
    return *this;
  }

  GlobalPtr <T> &operator--(int) {
    ptr -= sizeof(T);
    return *this;
  }

  GlobalPtr <T> &operator--() {
    ptr -= sizeof(T);
    return *this;
  }

  GlobalPtr <T> &operator+=(difference_type offset) {
    ptr += offset*sizeof(T);
    return *this;
  }

  GlobalPtr <T> &operator-=(difference_type offset) {
    ptr -= offset*sizeof(T);
    return *this;
  }

  bool operator==(const GlobalPtr <T> &ptr) const {
    return (this->ptr == ptr.ptr && this->rank == ptr.rank);
  }

  bool operator!=(const GlobalPtr <T> &ptr) const {
    return this->ptr != ptr.ptr || this->rank != ptr.rank;
  }

  template <typename U>
  class CallHandler {
  public:
    CallHandler(const BCL::GlobalPtr<U>& ptr) : ptr_(ptr) {}

    U* operator->() {
      // *reinterpret_cast<U*>(obj_) = *ptr_;
      BCL::rget(ptr_, reinterpret_cast<U*>(obj_), 1);
      return reinterpret_cast<U*>(obj_);
    }

    // TODO: does not currently support -> with const
    const U* operator->() const {
      // *reinterpret_cast<U*>(obj_) = *ptr_;
      BCL::rget(ptr_, reinterpret_cast<U*>(obj_), 1);
      return reinterpret_cast<U*>(obj_);
    }

    ~CallHandler() {
      *ptr_ = *reinterpret_cast<U*>(obj_);
    }

    BCL::GlobalPtr<U> ptr_;
    char obj_[sizeof(U)];
  };

  CallHandler<T> operator->() {
    return CallHandler<T>(*this);
  }

  const CallHandler<T> operator->() const {
    return CallHandler<T>(*this);
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

template <typename T, typename U>
inline GlobalPtr <T> reinterpret_pointer_cast(const GlobalPtr <U> &ptr) noexcept {
  return GlobalPtr <T> (ptr.rank, ptr.ptr);
}

} // end BCL
