// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace BCL {

template <typename T>
inline void rput(const T &src, GlobalPtr<T> dst) {
  BCL::write(&src, dst, 1);
}

template <typename T>
inline void rput(const T *src, GlobalPtr<T> dst, std::size_t size) {
  BCL::write(src, dst, size);
}

template <typename T>
inline void rget(GlobalPtr<std::add_const_t<T>> src, T *dst, std::size_t size) {
  BCL::read(src, dst, size);
}

template <typename T>
inline void rget_atomic(GlobalPtr<std::add_const_t<T>> src, T *dst, std::size_t size) {
  BCL::atomic_read(src, dst, size);
}

template <typename T>
inline std::remove_const_t<T> rget_atomic(GlobalPtr<T> src) {
  std::remove_const_t<T> rv;
  BCL::atomic_read(src, &rv, 1);
  return rv;
}

template <typename T>
inline std::remove_const_t<T> rget(GlobalPtr<T> src) {
  std::remove_const_t<T> rv;
  BCL::read(src, &rv, 1);
  return rv;
}

template <typename T, typename Allocator = BCL::bcl_allocator<T>>
inline future<std::vector<T, Allocator>> arget(GlobalPtr<std::add_const_t<T>> src, size_t size) {
  std::vector<T, Allocator> dst(size);
  BCL::request request = async_read(src, dst.data(), size);
  return BCL::future<std::vector<T, Allocator>>(std::move(dst), std::move(request));
}

// TODO: should this also accept an allocator?
template <typename T>
inline BCL::future<T> arget(GlobalPtr<T> src) {
  future<T> fut;
  BCL::request request = async_read(src, fut.value_.get(), 1);
  fut.update(request);
  return std::move(fut);
}

template <typename T>
inline BCL::request arget(GlobalPtr<std::add_const_t<T>> src, T* dst, size_t size) {
  return async_read(src, dst, size);
}

template <typename T, typename Allocator>
inline future<std::vector<T, Allocator>> arput(GlobalPtr<T> dst,
                                               std::vector<T, Allocator>&& src) {
  BCL::request request = async_write(src.data(), dst, src.size());
  return BCL::future<std::vector<T, Allocator>>(std::move(src), std::move(request));
}

template <typename T>
inline BCL::request arput(GlobalPtr<T> dst,
                          const T* src, size_t n_elem) {
  return async_write(src, dst, n_elem);
}

inline void memcpy(GlobalPtr<void> dst, const void* src, std::size_t n) {
  BCL::write(static_cast<const char*>(src),
             BCL::reinterpret_pointer_cast<char>(dst),
             n);
}

inline void memcpy(void* dst, GlobalPtr<const void> src, std::size_t n) {
  BCL::read(BCL::reinterpret_pointer_cast<const char>(src),
            static_cast<char*>(dst),
            n);
}

} // end BCL
