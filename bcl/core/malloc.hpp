#pragma once

#include <mutex>
#include <cstdio>
#include <cstdlib>

#include <bcl/bcl.hpp>
#include <bcl/core/GlobalPtr.hpp>

namespace BCL {

extern uint64_t shared_segment_size;
extern void *smem_base_ptr;

extern bool bcl_finalized;

std::mutex malloc_mutex;

typedef struct chunk_t {
  size_t size;
  struct chunk_t *last = NULL;
  struct chunk_t *next = NULL;
} chunk_t;

const size_t SMALLEST_MEM_UNIT = 64;

// Round sizeof(chunk_t) to 64 bytes so
// malloc'd memory region is aligned
inline size_t chunk_t_size() {
  return ((sizeof(chunk_t) + SMALLEST_MEM_UNIT - 1) / SMALLEST_MEM_UNIT) * SMALLEST_MEM_UNIT;
}

chunk_t *flist;
chunk_t *smem_heap;

inline void init_malloc() {
  // Start allocating at 64 bytes past the smem_base_ptr;
  BCL::smem_heap = (chunk_t *) (((char *) BCL::smem_base_ptr) + SMALLEST_MEM_UNIT);
  flist = NULL;
}

inline void print_chunk(chunk_t *chunk) {
  printf("%p chunk of size %lu\n", chunk, chunk->size);
  printf("  last = %p\n", chunk->last);
  printf("  next = %p\n", chunk->next);
}

inline void print_free_list() {
  for (chunk_t *chunk = flist; chunk != NULL; chunk = chunk->next) {
    print_chunk(chunk);
  }
}

// TODO: Error check that memory is within shared segment bounds.
template <typename T>
inline GlobalPtr <T> local_malloc(size_t size) {
  if (bcl_finalized) {
    return nullptr;
  }
  BCL::malloc_mutex.lock();
  size = size * sizeof(T);
  // Align size
  size = ((size + SMALLEST_MEM_UNIT - 1) / SMALLEST_MEM_UNIT) * SMALLEST_MEM_UNIT;

  chunk_t *chunk = flist;

  while (chunk != NULL) {
    if (chunk->size >= size) {
      break;
    }
    chunk = chunk->next;
  }

  char *allocd = nullptr;
  if (chunk != NULL && chunk->size >= size) {
    // I have a chunk!
    if (chunk->size > SMALLEST_MEM_UNIT + size + chunk_t_size()) {
      // Chunk is too big; carve off my piece and
      // replace it with free_chunk in the flist.
      chunk_t *free_chunk = (chunk_t *) (((char *) chunk) + chunk_t_size() + size);
      free_chunk->size = chunk->size - size - chunk_t_size();
      free_chunk->last = chunk->last;
      free_chunk->next = chunk->next;
      if (chunk->last != NULL) {
        chunk->last->next = free_chunk;
      }
      if (chunk->next != NULL) {
        chunk->next->last = free_chunk;
      }
      chunk->size = size;
      chunk->next = NULL;
      chunk->last = NULL;
      if (chunk == flist) {
        flist = free_chunk;
      }
    } else {
      // Chunk is just the right size; carve it
      // off and close the hole in the flist.
      if (chunk->last != NULL) {
        chunk->last->next = chunk->next;
      }
      if (chunk->next != NULL) {
        chunk->next->last = chunk->last;
      }
      if (chunk == flist) {
        flist = chunk->next;
      }
      chunk->last = NULL;
      chunk->next = NULL;
    }
    allocd = ((char *) chunk) + chunk_t_size();
  } else {
    // No free chunk, carve one off the heap.
    chunk = BCL::smem_heap;
    chunk->size = size;
    chunk->next = NULL;
    chunk->last = NULL;
    allocd = ((char *) chunk) + chunk_t_size();
    if (allocd + size > ((char *) BCL::smem_base_ptr) + shared_segment_size) {
      BCL::malloc_mutex.unlock();
      return nullptr;
    } else {
      BCL::smem_heap = (chunk_t *) (((char *) allocd) + size);
    }
  }

  BCL::malloc_mutex.unlock();

  return GlobalPtr <T> (BCL::rank(), (uint64_t) (allocd - (char *) BCL::smem_base_ptr));
}

// TODO: double frees not resulting in infinite loop would be nice.
// Technically undefined, but this behavior is particularly... spicy.
template <typename T>
inline void local_free(const GlobalPtr <T> &ptr) {
  if (bcl_finalized) {
    return;
  }
  BCL::malloc_mutex.lock();
  char *vptr = (char *) ptr.local();

  if (vptr == nullptr) {
    BCL::malloc_mutex.unlock();
    return;
  }

  chunk_t *my_chunk = (chunk_t *) (vptr - chunk_t_size());
  chunk_t *neighbor_chunk = flist;

  // Keep free list in order to
  // make compaction easy
  while (neighbor_chunk != NULL) {
    if (my_chunk > neighbor_chunk) {
      break;
    }
    neighbor_chunk = neighbor_chunk->next;
  }

  // I am the primordial chunk.
  if (neighbor_chunk == NULL) {
    my_chunk->next = flist;
    my_chunk->last = NULL;
    if (flist != NULL) {
      flist->last = my_chunk;
    }
    flist = my_chunk;
  } else {
    if (my_chunk > neighbor_chunk) {
      my_chunk->next = neighbor_chunk->next;
      my_chunk->last = neighbor_chunk;
      if (my_chunk->next != NULL) {
        my_chunk->next->last = my_chunk;
      }
      my_chunk->last->next = my_chunk;
    } else {
      throw std::runtime_error("AGH!!!! What a world, what a world! This was never supposed to happen.\n");
      // TODO: there's a bug here--the "primordial chunk" can be lost.
      my_chunk->next = neighbor_chunk;
      my_chunk->last = neighbor_chunk->last;
      my_chunk->next->last = my_chunk;
      if (my_chunk->last != NULL) {
        my_chunk->last->next = my_chunk;
      }
      if (flist == neighbor_chunk) {
        flist = my_chunk;
      }
    }
  }

  // Compact!

  // If chunk directly after is free, compact.
  if ((char *) my_chunk->next == ((char *) my_chunk) + chunk_t_size() + my_chunk->size) {
    my_chunk->size += chunk_t_size() + my_chunk->next->size;
    my_chunk->next = my_chunk->next->next;
    if (my_chunk->next != NULL) {
      my_chunk->next->last = my_chunk;
    }
  }

  // If chunk directly before is free, compact.
  if (my_chunk->last != NULL &&
    (char *) my_chunk->last->next == ((char *) my_chunk->last) + chunk_t_size() + my_chunk->last->size &&
    my_chunk->last->next == my_chunk) {
      my_chunk->last->size += chunk_t_size() + my_chunk->size;
      my_chunk->last->next = my_chunk->next;
      if (my_chunk->last->next != NULL) {
        my_chunk->last->next->last = my_chunk->last;
      }
  }
  BCL::malloc_mutex.unlock();
}

} // end BCL
