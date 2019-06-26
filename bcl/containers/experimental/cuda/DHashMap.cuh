#pragma once

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>

#include "util/comm.cuh"
#include "util/error.cuh"
#include "util/hash.cuh"

#define USED ((unsigned short int)1)
#define FREE ((unsigned short int)0)
#define DEAD ((unsigned short int)2)

#define NO_FREE ((char)0)
#define FREE_FAIL ((char)1)
#define FREE_SUCCESS ((char)2)

namespace BCL {

namespace cuda {

template <typename T>
struct hash;

template <>
struct hash<uint32_t>
{
    __device__ __host__ std::size_t operator()(const uint32_t& value) const noexcept
    {
      return value;
    }
};

template <typename Key, typename T>
struct HashMapEntry {
  Key key;
  T value;
};

template <typename Key, typename T>
struct DHashMap {
  using HME = HashMapEntry<Key, T>;

  using size_type = size_t;

  DHashMap(size_type capacity) {
    local_capacity_ = (capacity + BCL::nprocs() - 1) / BCL::nprocs();
    local_capacity_ = align_up(local_capacity_, 2);
    capacity_ = local_capacity_*BCL::nprocs();

    std::vector<BCL::cuda::ptr<HME>> ptrs(BCL::nprocs(), nullptr);
    std::vector<BCL::cuda::ptr<unsigned short int>> flags(BCL::nprocs(), nullptr);
    ptrs[BCL::rank()] = BCL::cuda::alloc<HME>(local_capacity_);
    flags[BCL::rank()] = BCL::cuda::alloc<unsigned short int>(local_capacity_);
    CUDA_CHECK(cudaMemset(flags[BCL::rank()].rptr(), 0, sizeof(unsigned short int)*local_capacity_));

    for (size_t i = 0; i < BCL::nprocs(); i++) {
      ptrs[i] = BCL::broadcast(ptrs[i], i);
      flags[i] = BCL::broadcast(flags[i], i);
    }

    ptrs_.assign(ptrs.begin(), ptrs.end());
    flags_.assign(flags.begin(), flags.end());
  }

  __host__ __device__ auto get_slot_ptr(size_type idx) {
    size_type proc_id = idx / local_capacity_;
    size_type local_idx = idx % local_capacity_;
    return ptrs_[proc_id].get() + local_idx;
  }

  __device__ auto get_flag_ptr(size_type idx) {
    size_type proc_id = idx / local_capacity_;
    size_type local_idx = idx % local_capacity_;
    return flags_[proc_id].get() + local_idx;
  }
  __host__ __device__ HME get_slot(size_type idx) {
    auto ptr = get_slot_ptr(idx);
    HME entry;
    // TODO: use pinned memory for entry *BUG on Infiniband*
    BCL::cuda::memcpy(&entry, ptr, sizeof(HME));
    BCL::cuda::flush();
    return entry;
  }

  __device__ uint32_t get_flag_int(size_type idx) {
    auto ptr = get_flag_ptr(idx*2);
    return nvshmem_int_g((int *)(ptr.rptr()), ptr.rank_);
  }

  __device__ bool request_slot(size_type idx, uint32_t comp, uint32_t value, bool p) {
//    auto ptr = pointerto(reserved, get_slot_ptr(idx*2));
    auto ptr = get_flag_ptr(idx*2);
    //int old_value = BCL::cuda::compare_and_swap(ptr, comp, value);
    uint32_t old_value = nvshmem_int_cswap(((int *)ptr.rptr()), comp, value, ptr.rank_);
    if(p)
      printf("old_value %d\n", old_value);
    return old_value == comp;
  }

  __device__ char try_free_slot_warp(unsigned mask, HME tuple, uint32_t prob_entry, int pe)
  {
    uint32_t flag = get_flag_int((prob_entry+LANE)%(capacity_/2));
    bool ifFree = (   ((flag&flagMask0) != 0)  & ((flag&flagMask2)!=0)   )==0;
    if(pe == 1 && WARPID==49 )
      printf("LANE %d, flag %d, ifFree %d, part1 %d, part2 %d\n", LANE, flag, int(ifFree), (flag&flagMask0), (flag&flagMask2));
    unsigned ifFreeMask = __ballot_sync(mask, ifFree);
    if(ifFreeMask!=0)
    {
      int laneID = __ffs(ifFreeMask)-1;
      bool success = 0;
      if(LANE == laneID)
      {
        uint32_t new_flag = flag|flagMask5;
        size_type slot = ((prob_entry+LANE)%(capacity_/2))*2+1;
        if((flag&flagMask0) == 0)
        {
          new_flag = flag|flagMask4;
          slot = slot-1;
        }
        success = request_slot((prob_entry+LANE)%(capacity_/2), flag, new_flag, (pe == 1 && WARPID==49));
        if(pe == 1 && WARPID==49 )
          printf("lane %d, flag %d, new_flag %d, slot %d, success %d, success %d\n", LANE, flag, new_flag, slot, success, success);
        if(success)
        {
            BCL::cuda::memcpy(get_slot_ptr(slot), &tuple, sizeof(HME));
            BCL::cuda::flush();
        }
      }
      __syncwarp(mask);
      success = __shfl_sync(mask, success, laneID);
      __syncwarp(mask);

      if(success)
        return FREE_SUCCESS;
      else return FREE_FAIL;
    }
    return NO_FREE;
  }

  __device__ bool insert_warp(const Key &key, const T& value, int pe)
  {
    HME entry{key, value};
    unsigned mask = __activemask();
    size_t hash = BCL::cuda::hash<Key>{}(key);
    uint32_t prob_entry = align_down(hash%capacity_, 2);
    if(WARPID == 49 && LANE == 0 && pe == 1)
      printf("hash %d, prob_entry %d, %d\n", uint32_t(hash), prob_entry, uint32_t(align_down(hash%capacity_,2)));
    uint32_t prob = 0;
    char result = NO_FREE;
    do {
      result = try_free_slot_warp(mask, entry, prob_entry/2+prob, pe);
      if(result == NO_FREE)
        prob = prob + __popc(mask);
    } while(result!=FREE_SUCCESS && prob < capacity_/2);
    return (result = FREE_SUCCESS);
  }

  __host__ __device__ size_t capacity() const noexcept {
    return capacity_;
  }

  __host__ __device__ size_t get_probe(size_t probe) const {
    return probe;
  }

  BCL::cuda::device_vector<BCL::cuda::ptr<HME>> ptrs_;
  BCL::cuda::device_vector<BCL::cuda::ptr<unsigned short int>> flags_;
  size_t capacity_;
  size_t local_capacity_;
};

} // end cuda
} // end BCL
