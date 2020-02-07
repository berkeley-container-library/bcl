#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>
#include <bcl/containers/experimental/cuda/sequential/device_vector.cuh>

namespace BCL {

namespace cuda {

template <typename T>
struct DuplQueue {
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  DuplQueue() {}

  DuplQueue(size_t host, size_t capacity) :
            host_(host), capacity_(capacity) {
    std::vector<BCL::cuda::ptr<value_type>> ptrs(BCL::nprocs(), nullptr);
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      if (BCL::rank() == host) {
        ptrs[i] = BCL::cuda::alloc<value_type>(capacity);
      }
      ptrs[i] = BCL::broadcast(ptrs[i], host);
      if (ptrs[i] == nullptr) {
        throw std::runtime_error("AGH! Ran out of memory with request for " + std::to_string(1e-9*capacity*sizeof(value_type)) + "GB");
      }
    }
    ptrs_.assign(ptrs.begin(), ptrs.end());

    if (BCL::rank() == host) {
      limits_ = BCL::cuda::alloc<int>(BCL::nprocs()*2);

      std::vector<int> limits(BCL::nprocs()*2, 0);
      BCL::cuda::memcpy(limits_, limits.data(), sizeof(int)*BCL::nprocs()*2);
    }

    limits_ = BCL::broadcast(limits_, host);
    if (limits_ == nullptr) {
      throw std::runtime_error("AGH! Ran out of memory with request for " + std::to_string(1e-9*capacity*sizeof(value_type)) + "GB");
    }

    head = BCL::cuda::alloc<int>(4);
    tail = head + 1;
    tail_mutex = head + 2;
    send_mutex = head + 3;
    int zero = 0;
    cudaMemcpy(head.local(), &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tail.local(), &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(tail_mutex.local(), &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(send_mutex.local(), &zero, sizeof(int), cudaMemcpyHostToDevice);
  }

  __device__ __host__ size_type host() const noexcept {
    return host_;
  }

  __device__ __host__ size_type capacity() const noexcept {
    return capacity_;
  }

  __device__ __host__ BCL::cuda::ptr<value_type> data_ptr(size_type rank = BCL::cuda::rank()) {
    return ptrs_[rank];
  }

  __device__ __host__ BCL::cuda::ptr<int> tail_ptr(size_type rank = BCL::cuda::rank()) {
    return limits_ + BCL::cuda::nprocs() + rank;
  }

  __device__ __host__ BCL::cuda::ptr<int> head_ptr(size_type rank = BCL::cuda::rank()) {
    return limits_ + rank;
  }

  template <typename U>
  __host__ __device__ U max(const U& a, const U& b) {
    if (a < b) {
      return b;
    } else {
      return a;
    }
  }

  /*
     XXX: prototypes
  __device__ bool push_warp(bool ifpush, T value) {
    size_t warp_id = threadIdx.x % 32;
    unsigned mask =  __ballot_sync(0xffffffff, (ifpush == 1));
    uint32_t total = __popc(mask);
    int rank = __popc(mask & lanemask_lt());

    T* values;
    int pos;
    if(warp_id == 0)
    {
      values = emalloc::emalloc(sizeof(T)*total);
      pos = atomicAdd(tail.local(), size);
    }
    values = __shfl_sync(0xffffffff, values, 0);
    pos = __shlf_sync(0xffffffff, pos, 0);
    if(ifpush)
    values[rank] = value;
    __syncwarp();


    bcl::cuda::memcpy_warp(remote_ptr+pos, values, total);

    if (warp_id == 0) {
      BCL::cuda::memcpy(...);
    }

  }
  */

  // REQUIRES: data is in pinned memory
  //           called as a collective function by a warp
  __device__ bool push_warp(value_type* data, size_t size) {
    size_t warp_id = threadIdx.x % 32;
    int pos;
    if (warp_id == 0) {
      pos = atomicAdd(tail.local(), size);
    }

    pos = __shfl_sync(0xffffffff, pos, 0);

    BCL::cuda::ptr<value_type> ptr = ptrs_[BCL::cuda::rank()];
    BCL::cuda::memcpy_warp(ptr + (pos % capacity()), data, size*sizeof(value_type));
    BCL::cuda::flush();

    if (warp_id == 0) {
      // Get lock
      while (atomicCAS(tail_mutex.local(), 0, 1) != 0) {}

      // Copy value
      int* new_tail_value_ = (int *) emalloc::emalloc(sizeof(int));
      int& new_tail_value = *new_tail_value_;
      new_tail_value = max(*tail.local(), pos);

      BCL::cuda::memcpy(tail_ptr(), &new_tail_value, sizeof(int));
      BCL::cuda::flush();

      // Unlock
      atomicCAS(tail_mutex.local(), 1, 0);
    }
    return true;
  }

  // TODO
  //      1) Handle cases where the write is split on either side of the buffer
  //      2) Don't do unnecessary memcpy's of tail value
  //      2) Handle cases where there isn't enough space

  // REQUIRES: data is in pinned memory
  __device__ bool push(value_type* data, size_t size) {
    // TODO: bounds check
    int pos = atomicAdd(tail.local(), size);
    BCL::cuda::ptr<value_type> ptr = ptrs_[BCL::cuda::rank()];
    BCL::cuda::memcpy(ptr + (pos % capacity()), data, size*sizeof(value_type));
    BCL::cuda::flush();

    while (atomicCAS(tail_mutex.local(), 0, 1) != 0) {}

    int* new_tail_value_ = (int *) emalloc::emalloc(sizeof(int));

    if (new_tail_value_ == nullptr) {
      printf("Out of memory!\n");
    }

    int& new_tail_value = *new_tail_value_;
    new_tail_value = max(*tail.local(), pos);

    BCL::cuda::memcpy(tail_ptr(), &new_tail_value, sizeof(int));
    BCL::cuda::flush();

    atomicCAS(tail_mutex.local(), 1, 0);
    return true;
  }

  // TODO: handle atomic insertions.

  __device__ bool local_pop(value_type& data) {
    int offset = threadIdx.x;
    for (size_t i = 0; i < ptrs_.size(); i++) {
      size_t queue_id = (i + offset) % ptrs_.size();
      int loc = atomicAdd(head_ptr(queue_id).local(), 1);
      int max = *tail_ptr(queue_id).local();
      if (loc < max) {
        BCL::cuda::ptr<value_type> ptr = ptrs_[queue_id];
        data = ptr.local()[loc % capacity()];
        return true;
      } else {
        atomicAdd(head_ptr(queue_id).local(), -1);
      }
    }
    return false;
  }

  BCL::cuda::ptr<int> head;
  BCL::cuda::ptr<int> tail;
  BCL::cuda::ptr<int> tail_mutex;
  BCL::cuda::ptr<int> send_mutex;

  size_t capacity_;
  size_t host_;

  BCL::cuda::device_vector<BCL::cuda::ptr<value_type>> ptrs_;
  BCL::cuda::ptr<int> limits_;
};

} // end cuda
} // end BCL
