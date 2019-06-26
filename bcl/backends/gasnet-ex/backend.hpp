#pragma once

#include <gasnetex.h>
#include "detail/gasnet_ad.hpp"
#include "comm.hpp"

namespace BCL {

extern uint64_t shared_segment_size;
extern void* smem_base_ptr;

gex_Client_t client;
gex_EP_t ep;
gex_TM_t tm;
const char* clientName = "BCL";

gasnet_seginfo_t* gasnet_seginfo;

extern inline void init_malloc();

uint64_t my_rank;
uint64_t my_nprocs;

bool bcl_finalized;

namespace backend {

inline uint64_t rank() {
  return BCL::my_rank;
}

inline uint64_t nprocs() {
  return BCL::my_nprocs;
}

} // end backend

inline void barrier() {
  gex_Event_t event = gex_Coll_BarrierNB(tm, 0);
  gex_Event_Wait(event);
}

inline void flush() {}

template <typename T>
inline void* gasnet_resolve_address(const GlobalPtr<T> ptr) {
  return reinterpret_cast<char*>(BCL::gasnet_seginfo[ptr.rank].addr) + ptr.ptr;
}

inline void init(uint64_t shared_segment_size = 256, bool thread_safe = false) {
  BCL::shared_segment_size = 1024*1024*shared_segment_size;

  gex_Client_Init(&client, &ep, &tm, clientName, NULL, NULL, 0);

  if (thread_safe) {
    #ifndef GASNET_PAR
      throw BCL::error("Need to use a par build of GASNet-EX");
    #endif
  }

  gex_Segment_t segment;
  gex_Segment_Attach(&segment, tm, BCL::shared_segment_size);

  smem_base_ptr = gex_Segment_QueryAddr(segment);

  if (smem_base_ptr == NULL) {
    throw std::runtime_error("BCL: Could not allocate shared memory segment.");
  }

  my_rank = gex_System_QueryJobRank();
  my_nprocs = gex_System_QueryJobSize();

  gasnet_seginfo = (gasnet_seginfo_t*) malloc(sizeof(gasnet_seginfo_t) * nprocs());
  gasnet_getSegmentInfo(gasnet_seginfo, BCL::nprocs());

  init_malloc();
  init_atomics();

  bcl_finalized = false;

  BCL::barrier();
}

inline void finalize() {
  BCL::barrier();
  finalize_atomics();
  free(gasnet_seginfo);
  bcl_finalized = true;
}

} // end BCL
