// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace BCL
{

template <typename T = std::size_t,
          __BCL_REQUIRES(std::is_integral_v<T> && !std::is_reference_v<T>)>
class index {
public:
  using index_type = T;

  using first_type = T;
  using second_type = T;

  index_type operator[](index_type dim) const noexcept {
    if (dim == 0) {
      return first;
    } else {
      return second;
    }
  }

  template <typename U,
            __BCL_REQUIRES(std::is_integral_v<U> &&
                 std::numeric_limits<U>::max() >= std::numeric_limits<T>::max())
            >
  operator index<U>() const noexcept {
    return index<U>(first, second);
  }

  index(index_type first, index_type second) : first(first), second(second) {}

  bool operator==(index other) const noexcept {
    return first == other.first && second == other.second;
  }

  index() = default;
  ~index() = default;
  index(const index&) = default;
  index& operator=(const index&) = default;
  index(index&&) = default;
  index& operator=(index&&) = default;

  index_type first;
  index_type second;
};

} // end BCL

namespace std {

template <std::size_t I, typename T,
          __BCL_REQUIRES(I <= 1)>
size_t get(BCL::index<T> idx)
{
  return idx[I];
}

} // end std
