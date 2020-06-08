
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

  if ((char *) ptr < BCL::cuda::smem_base_ptr) {
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
void print_free_list(FILE* f);

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
      size_t freelist_size = 0;
      for (const auto& val : free_chunks) {
        freelist_size += val.second.size_;
      }
      fprintf(stderr, "RANK(%lu) failed Alloc(%lf GB) Heap(%lf GB) Freelist(%lf GB)\n",
              BCL::rank(), size / 1000.0 / 1000.0 / 1000.0,
              (shared_segment_size - heap_ptr) / 1000.0 / 1000.0 / 1000.0,
              freelist_size / 1000.0 / 1000.0 / 1000.0);
      if (size > freelist_size) {
        fprintf(stderr, "RANK(%lu) truly OOM.\n", BCL::rank());
      } else {
        fprintf(stderr, "RANK(%lu) is FOM.\n", BCL::rank());
        if (BCL::rank() == 487) {
          print_free_list(stderr);
        }
        std::string fname = std::string("memory_state_") + std::to_string(BCL::rank()) + ".txt";
        FILE* f = fopen(fname.c_str(), "w");
        print_free_list(f);
        fclose(f);
        fprintf(stderr, "RANK(%lu) wrote freelist.\n", BCL::rank());
      }
      return nullptr;
    }
    chunk = fchunk_t(heap_ptr, size);
    heap_ptr += size;
  }
  allocd_chunks.insert(std::make_pair(chunk.ptr_, chunk));
  return ptr<T> (BCL::rank(), chunk.ptr_);
}

void compress_free_list();

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
  /*
  fprintf(stderr, "RANK(%lu) free'ing chunk of size %lu at pos %lu\n",
          BCL::rank(), chunk.size_, chunk.ptr_);
          */
  compress_free_list();
  return;
}

void compress_free_list() {
  for (auto iter = free_chunks.begin(); iter != free_chunks.end(); ) {
    auto next_chunk = iter;
    next_chunk++;
    if (next_chunk != free_chunks.end()) {
      if (next_chunk->second.ptr_ == iter->second.ptr_ + iter->second.size_) {
        iter->second.size_ += next_chunk->second.size_;
        free_chunks.erase(next_chunk);
        continue;
      }
    }
    iter++;
  }
}

#include <cassert>

void print_free_list(FILE* f) {
  std::map<size_t, fchunk_t> all_chunks;

  all_chunks.insert(free_chunks.begin(), free_chunks.end());
  all_chunks.insert(allocd_chunks.begin(), allocd_chunks.end());

  fprintf(f, "==== RANK(%lu)'s memory space ====\n", BCL::rank());
  for (const auto& i : all_chunks) {
    bool is_free_list = (free_chunks.find(i.first) != free_chunks.end());
    bool is_used_list = (allocd_chunks.find(i.first) != allocd_chunks.end());
    assert(is_free_list || is_used_list && (!(is_free_list && is_used_list)));
    fprintf(f, "   R(%lu) (%lu, %lu) %s\n",
            BCL::rank(),
            i.first, i.second.size_, (is_free_list) ? "FREE" : "USED");
  }
}

} // end cuda

} // end BCL
