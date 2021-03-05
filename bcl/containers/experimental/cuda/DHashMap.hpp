#pragma once

#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>

namespace BCL {

namespace cuda {

template <typename T>
struct hash;

template <>
struct hash<int>
{
    __device__ __host__ std::size_t operator()(const int& value) const noexcept
    {
      return value;
    }
};

template <typename Key, typename T>
struct HashMapEntry {
  Key key;
  T value;
  int reserved = 0;
};

template <typename Key, typename T>
struct DHashMap {
  using HME = HashMapEntry<Key, T>;

  using size_type = size_t;

  static constexpr int free_flag = 0;
  static constexpr int used_flag = 1;

  DHashMap(size_type capacity) {
    local_capacity_ = (capacity + BCL::nprocs() - 1) / BCL::nprocs();
    capacity_ = local_capacity_*BCL::nprocs();

    std::vector<BCL::cuda::ptr<HME>> ptrs(BCL::nprocs(), nullptr);
    ptrs[BCL::rank()] = BCL::cuda::alloc<HME>(local_capacity_);

    for (size_t i = 0; i < BCL::nprocs(); i++) {
      ptrs[i] = BCL::broadcast(ptrs[i], i);
    }

    ptrs_.assign(ptrs.begin(), ptrs.end());
  }

  __host__ __device__ auto get_slot_ptr(size_type idx) {
    size_type proc_id = idx / local_capacity_;
    size_type local_idx = idx % local_capacity_;
    return ptrs_[proc_id].get() + local_idx;
  }

  __host__ __device__ HME get_slot(size_type idx) {
    auto ptr = get_slot_ptr(idx);
    HME entry;
    // TODO: use pinned memory for entry *BUG on Infiniband*
    BCL::cuda::memcpy(&entry, ptr, sizeof(HME));
    BCL::cuda::flush();
    return entry;
  }

  __device__ bool request_slot(size_type idx) {
    auto ptr = pointerto(reserved, get_slot_ptr(idx));
    int old_value = BCL::cuda::compare_and_swap(ptr, free_flag, used_flag);
    return old_value == free_flag;
  }

  __device__ bool insert(const Key& key, const T& value) {
    HME entry{key, value, used_flag};
    size_t hash = BCL::cuda::hash<Key>{}(key);
    size_type probe = 0;
    bool success = false;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      success = request_slot(slot);
      if (success) {
        BCL::cuda::memcpy(get_slot_ptr(slot), &entry, offsetof(HME, reserved));
        BCL::cuda::flush();
      }
    } while (!success && probe < capacity());
    return success;
  }

  __device__ T find(const Key& key) {
    HME entry;
    size_t hash = BCL::cuda::hash<Key>{}(key);
    size_type probe = 0;
    bool success = false;
    do {
      size_type slot = (hash + get_probe(probe++)) % capacity();
      entry = get_slot(slot);
      success = entry.key == key || entry.reserved == free_flag;
    } while (!success && probe < capacity());
    return entry.value;
  }

  __host__ __device__ size_t capacity() const noexcept {
    return capacity_;
  }

  __host__ __device__ size_t get_probe(size_t probe) const {
    return probe;
  }

  BCL::cuda::device_vector<BCL::cuda::ptr<HME>> ptrs_;
  size_t capacity_;
  size_t local_capacity_;
};

} // end cuda
} // end BCL
