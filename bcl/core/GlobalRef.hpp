
#pragma once

#include <stdexcept>

namespace BCL {


template <typename T>
struct GlobalPtr;

template <typename T>
extern inline T rget(const GlobalPtr <T> &src);

template <typename T>
extern inline void rput(const T &src, const GlobalPtr <T> &dst);

template <typename T>
class GlobalRef {
public:

  BCL::GlobalPtr<T> ptr_;

  GlobalRef(BCL::GlobalPtr<T> ptr) : ptr_(ptr) {
    BCL_DEBUG(
      if (ptr_ == nullptr) {
        throw debug_error("GlobalRef() constructor created a null reference.");
      }
    )
  }

  operator T() const {
    return BCL::rget(ptr_);
  }

  GlobalRef &operator=(const T& value) {
    BCL::rput(value, ptr_);
    return *this;
  }

  BCL::GlobalPtr<T> operator&() {
    return ptr_;
  }

  const BCL::GlobalPtr<T> operator&() const {
    return ptr_;
  }
};

}
