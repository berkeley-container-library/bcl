// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdexcept>

namespace BCL {


template <typename T>
struct GlobalPtr;

template <typename T>
extern inline std::remove_const_t<T> rget(GlobalPtr<T> src);

template <typename T>
extern inline void rput(const T &src, GlobalPtr<T> dst);

template <typename T>
class GlobalRef {
public:

  GlobalRef() = delete;
  ~GlobalRef() = default;
  GlobalRef(const GlobalRef&) = default;
  GlobalRef& operator=(const GlobalRef&) = default;
  GlobalRef(GlobalRef&&) = default;
  GlobalRef& operator=(GlobalRef&&) = default;

  using value_type = T;
  using pointer = GlobalPtr<T>;
  using reference = GlobalRef<T>;
  using const_reference = GlobalRef<std::add_const_t<T>>;

  GlobalRef(BCL::GlobalPtr<T> ptr) : ptr_(ptr) {
    BCL_DEBUG(
      if (ptr_ == nullptr) {
        throw debug_error("GlobalRef() constructor created a null reference.");
      }
    )
  }

  template <__BCL_REQUIRES(!std::is_const_v<T>)>
  operator const_reference() const {
    return const_reference(ptr_);
  }

  operator T() const {
    return BCL::rget(ptr_);
  }

  reference operator=(const T& value) const {
    // TODO: replace static_assert with requires() for C++20
    static_assert(!std::is_const_v<T>);
    BCL::rput(value, ptr_);
    return *this;
  }

  pointer operator&() const noexcept {
    return ptr_;
  }

private:
  BCL::GlobalPtr<T> ptr_ = nullptr;
};

}
