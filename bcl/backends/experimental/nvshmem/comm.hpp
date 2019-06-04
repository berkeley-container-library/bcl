#pragma once

#include <cassert>

namespace BCL {

namespace cuda {

void flush();

template <typename T>
inline void copy_cross_gpu_(const BCL::cuda::ptr<T>& dst, const BCL::cuda::ptr<T>& src, size_t count) {
  assert(dst.rank_ == BCL::rank() || src.rank_ == BCL::rank());
  if (dst.rank_ == BCL::rank()) {
    nvshmem_getmem(dst.local(), src.rptr(), sizeof(T)*count, src.rank_);
  } else if (src.rank_ == BCL::rank()) {
    nvshmem_putmem(dst.rptr(), src.local(), sizeof(T)*count, dst.rank_);
  } else {
    assert(false);
  }
}

template <typename T>
inline __host__ __device__ void read(const BCL::cuda::ptr<T>& src, T* dst, size_t count) {
  #ifdef __CUDA_ARCH__
    nvshmem_getmem(dst, src.rptr(), sizeof(T)*count, src.rank_);
  #else
    if (src.rank_ == BCL::rank()) {
      cudaMemcpy(dst, src.rptr(), sizeof(T)*count, cudaMemcpyDeviceToHost);
    } else {
      BCL::cuda::ptr<T> local_ptr = BCL::cuda::alloc<T>(count);

      copy_cross_gpu_(local_ptr, src, count);
      BCL::cuda::flush();

      cudaMemcpy(dst, local_ptr.rptr(), sizeof(T)*count, cudaMemcpyDeviceToHost);
      // XXX: how do I ensure this is complete?
      //      is cudaDeviceSynchronize really necessary?
      cudaDeviceSynchronize();

      BCL::cuda::dealloc(local_ptr);
    }
  #endif
}

template <typename T>
inline __host__ __device__ void write(const T* src, const BCL::cuda::ptr<T>& dst, size_t count) {
  #ifdef __CUDA_ARCH__
    nvshmem_putmem(dst.rptr(), src, sizeof(T)*count, dst.rank_);
  #else
    if (dst.rank_ == BCL::rank()) {
      cudaMemcpy(dst.rptr(), src, sizeof(T)*count, cudaMemcpyHostToDevice);
    } else {
      BCL::cuda::ptr<T> local_ptr = BCL::cuda::alloc<T>(count);
      cudaMemcpy(local_ptr.rptr(), src, sizeof(T)*count, cudaMemcpyHostToDevice);
      // XXX: how do I ensure this is complete?

      copy_cross_gpu_(dst, local_ptr, count);
      BCL::cuda::flush();

      BCL::cuda::dealloc(local_ptr);
    }
  #endif
}

template <typename T>
inline T rget(const BCL::cuda::ptr<T>& dst) {
  T rv;
  read(dst, &rv, 1);
  return rv;
}

template <typename T>
inline void rput(const T& src, const BCL::cuda::ptr<T>& dst) {
  write(&src, dst, 1);
}

inline void memcpy(const BCL::cuda::ptr<void>& dst, const BCL::cuda::ptr<void>& src, size_t n) {
  copy_cross_gpu_(reinterpret_pointer_cast<char>(dst),
                  reinterpret_pointer_cast<char>(src),
                  n);
}

inline __host__ __device__ void memcpy(void* dst, const BCL::cuda::ptr<void>& src, size_t n) {
  read(reinterpret_pointer_cast<char>(src),
       (char*) dst,
       n);
}

inline __host__ __device__ void memcpy(const BCL::cuda::ptr<void>& dst, const void* src, size_t n) {
  write((char*) src,
        reinterpret_pointer_cast<char>(dst),
        n);
}

inline __device__ void memcpy_warp(const BCL::cuda::ptr<void>& dst, const void* src, size_t n) {
  nvshmemx_putmem_warp(dst.rptr(), src, n, dst.rank_);
}

inline __device__ void memcpy_warp(void* dst, const BCL::cuda::ptr<void>& src, size_t n) {
  nvshmemx_getmem_warp(dst, src.rptr(), n, src.rank_);
}

inline __device__ void memcpy_block(const BCL::cuda::ptr<void>& dst, const void* src, size_t n) {
  nvshmemx_putmem_block(dst.rptr(), src, n, dst.rank_);
}

inline __device__ void memcpy_block(void* dst, const BCL::cuda::ptr<void>& src, size_t n) {
  nvshmemx_getmem_block(dst, src.rptr(), n, src.rank_);
}

} // end cuda

} // end BCL
