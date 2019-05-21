#pragma once

namespace BCL {

template <typename T>
inline void rput(const T &src, const GlobalPtr <T> &dst) {
  BCL::write(&src, dst, 1);
}

template <typename T>
inline void rput(const T *src, const GlobalPtr <T> &dst, const size_t size) {
  BCL::write(src, dst, size);
}

template <typename T>
inline void rget(const GlobalPtr <T> &src, T *dst, const size_t size) {
  BCL::read(src, dst, size);
}

template <typename T>
inline void rget_atomic(const GlobalPtr <T> &src, T *dst, const size_t size) {
  BCL::atomic_read(src, dst, size);
}

template <typename T>
inline T rget_atomic(const GlobalPtr <T> &src) {
  T rv;
  BCL::atomic_read(src, &rv, 1);
  return rv;
}

template <typename T>
inline T rget(const GlobalPtr <T> &src) {
  T rv;
  BCL::read(src, &rv, 1);
  return rv;
}

template <typename T, typename Allocator = BCL::bcl_allocator<T>>
inline future<std::vector<T, Allocator>> arget(const GlobalPtr<T>& src, size_t size) {
  std::vector<T, Allocator> dst(size);
  BCL::request request = async_read(src, dst.data(), size);
  return BCL::future<std::vector<T, Allocator>>(std::move(dst), std::move(request));
}

// TODO: should this also accept an allocator?
template <typename T>
inline BCL::future<T> arget(const GlobalPtr<T>& src) {
  future<T> fut;
  BCL::request request = async_read(src, fut.value_.get(), 1);
  fut.update(request);
  return std::move(fut);
}

template <typename T>
inline BCL::request arget(const GlobalPtr<T>& src, T* dst, size_t size) {
  return async_read(src, dst, size);
}

template <typename T, typename Allocator>
inline future<std::vector<T, Allocator>> arput(const GlobalPtr<T>& dst,
                                               std::vector<T, Allocator>&& src) {
  BCL::request request = async_write(src.data(), dst, src.size());
  return BCL::future<std::vector<T, Allocator>>(std::move(src), std::move(request));
}

template <typename T>
inline BCL::request arput(const GlobalPtr<T>& dst,
                          const T* src, size_t n_elem) {
  return async_write(src, dst, n_elem);
}

inline void memcpy(const GlobalPtr<void>& dst, const void* src, size_t n) {
  BCL::write(static_cast<const char*>(src),
             BCL::reinterpret_pointer_cast<char>(dst),
             n);
}

inline void memcpy(void* dst, const GlobalPtr<void>& src, size_t n) {
  BCL::read(BCL::reinterpret_pointer_cast<char>(src),
            static_cast<char*>(dst),
            n);
}

} // end BCL
