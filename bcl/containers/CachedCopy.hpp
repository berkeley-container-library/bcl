#pragma once

#include <bcl/bcl.hpp>

namespace BCL {

template <typename T>
struct CachedCopy {

  CachedCopy() = default;
  CachedCopy(const CachedCopy&) = default;
  CachedCopy(CachedCopy&&) = default;

  CachedCopy& operator=(const CachedCopy& other) = default;

  CachedCopy(BCL::GlobalPtr<T> ptr) : ptr_(ptr) {
    refresh();
  }

  void refresh() {
    BCL::rget(ptr_, reinterpret_cast<T*>(buf_), 1);
  }

  T& operator*() {
    return *reinterpret_cast<T*>(buf_);
  }

  T* operator->() {
    return reinterpret_cast<T*>(buf_);
  }

  BCL::GlobalPtr<T> ptr_ = nullptr;
  char buf_[sizeof(T)];
};

}
