#pragma once

#include <mpp/shmem.h>
#include <bcl/core/GlobalPtr.hpp>

#include "detail/compare_and_swap_impl.hpp"

namespace BCL {

template <typename T>
T compare_and_swap(GlobalPtr<T> ptr, T old_val, T new_val) {
  static_assert(std::is_integral<T>::value, "BCL::compare_and_swap(): only integral types are supported");
  return compare_and_swap_impl_<T>::op(ptr, old_val, new_val);
}

}
