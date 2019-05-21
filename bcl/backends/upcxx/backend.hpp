#pragma once

#include <upcxx/upcxx.hpp>
#include "comm.hpp"

namespace BCL {
extern uint64_t shared_segment_size;
extern void* smem_base_ptr;

extern void init_malloc();

uint64_t my_rank;
uint64_t my_nprocs;

bool bcl_finalized;

uint64_t rank() {
  return BCL::my_rank;
}

uint64_t nprocs() {
  return BCL::my_nprocs;
}

void barrier() {
  upcxx::barrier();
}

template <typename T>
upcxx::global_ptr<T> upcxx_resolve_address(GlobalPtr<T> ptr) {
  return upcxx::global_ptr<T>(upcxx::detail::internal_only(),
                              ptr.rank, ptr.rptr());
}

void init(uint64_t shared_segment_size = 256) {
  BCL::shared_segment_size = 1024*1024*shared_segment_size;

  // TODO: check if UPC++ is already initialized?
  upcxx::init();

  upcxx::global_ptr<char> ptr = upcxx::allocate<char>(BCL::shared_segment_size);

  if (ptr == NULL) {
    throw std::runtime_error("BCL: Could not allocate shared memory segment.");
  }

  smem_base_ptr = ptr.local();

  my_rank = upcxx::rank_me();
  my_nprocs = upcxx::rank_n();


  init_malloc();

  bcl_finalized = false;

  BCL::barrier();
}

void finalize() {
  BCL::barrier();
  upcxx::deallocate(BCL::smem_base_ptr);
  upcxx::finalize();
  bcl_finalized = true;
}

}
