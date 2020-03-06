
#pragma once

#include <map>
#include <unordered_map>

#include "ptr.hpp"

namespace BCL {

namespace cuda {

template <typename T>
inline bool __is_valid_cuda_gptr(T* ptr) {
  if (ptr == nullptr) {
    return true;
  }

  if (ptr < BCL::cuda::smem_base_ptr) {
    return false;
  }

  int64_t offset = sizeof(T)*(ptr - reinterpret_cast<T*>(BCL::cuda::smem_base_ptr));

  if (offset < BCL::cuda::shared_segment_size) {
    return true;
  } else {
    return false;
  }
}

template <typename T>
inline BCL::cuda::ptr<T> __to_cuda_gptr(T* ptr) {
  // TODO: Hrmmm... what to do for local(NULL) -> Global()?
  if (ptr == nullptr) {
    return nullptr;
  }

  size_t offset = sizeof(T)*(ptr - reinterpret_cast<T*>(BCL::cuda::smem_base_ptr));

  if ((char *) ptr < BCL::cuda::smem_base_ptr || offset >= BCL::cuda::shared_segment_size) {
    // XXX: alternative would be returning nullptr
    throw std::runtime_error("BCL::__to_cuda_gptr(): given pointer is outside shared segment.");
  }

  return BCL::cuda::ptr<T>(BCL::rank(), offset);
}

struct fchunk_t {
  size_t ptr_ = 0;
  size_t size_ = 0;

  fchunk_t() {}
  fchunk_t(const size_t ptr, const size_t size) : ptr_(ptr), size_(size) {}
};

size_t shared_segment_size;
char* smem_base_ptr = nullptr;

constexpr size_t CUDA_SMALLEST_MEM_UNIT = 1024;

std::unordered_map <size_t, fchunk_t> allocd_chunks;
std::map <size_t, fchunk_t> free_chunks;

size_t heap_ptr = CUDA_SMALLEST_MEM_UNIT;

template <typename T>
ptr<T> alloc(size_t size) {
  size = sizeof(T) * size;
  size = CUDA_SMALLEST_MEM_UNIT * ((size + CUDA_SMALLEST_MEM_UNIT - 1) / CUDA_SMALLEST_MEM_UNIT);

  size_t chunk_ptr = 0;
  for (const auto& free_chunk : free_chunks) {
    if (free_chunk.second.size_ >= size) {
      chunk_ptr = free_chunk.first;
      break;
    }
  }

  fchunk_t chunk;
  if (chunk_ptr != 0) {
    // Found a free chunk!
    chunk = free_chunks[chunk_ptr];
    free_chunks.erase(chunk_ptr);
    if (chunk.size_ - size >= CUDA_SMALLEST_MEM_UNIT) {
      free_chunks.insert(std::make_pair(chunk.ptr_ + size,
                         fchunk_t(chunk.ptr_ + size, chunk.size_ - size)));
      chunk.size_ = size;
    }
  } else {
    // There is no chunk.  Advance the heap pointer.
    if (heap_ptr + size > shared_segment_size) {
      return nullptr;
    }
    chunk = fchunk_t(heap_ptr, size);
    heap_ptr += size;
  }
  allocd_chunks.insert(std::make_pair(chunk.ptr_, chunk));
  return ptr<T> (BCL::rank(), chunk.ptr_);
}

template <typename T>
void dealloc(ptr<T> alloc) {
  if (alloc == nullptr) {
    return;
  }

  if (allocd_chunks.find(alloc.ptr_) == allocd_chunks.end()) {
    throw std::runtime_error("BCL malloc(): attempted to free unallocated chunk");
  }

  if (alloc.rank_ != BCL::rank()) {
    throw std::runtime_error("BCL malloc(): attempted to free someone else's ptr");
  }

  fchunk_t chunk = allocd_chunks[alloc.ptr_];
  allocd_chunks.erase(chunk.ptr_);
  free_chunks.insert(std::make_pair(chunk.ptr_, chunk));
  return;
}

} // end cuda

} // end BCL
