#include <bcl/bcl.hpp>

namespace BCL {

template <typename T>
struct DuplQueue {
  using value_type = T;

  BCL::GlobalPtr<T> data;
  BCL::GlobalPtr<int> pointers;

  size_t capacity_;
  size_t host_;

  size_t head_buffer_ = 0;
  size_t tail_buffer_ = 0;

  size_t capacity() const noexcept {
    return capacity_;
  }

  DuplQueue(size_t host, size_t capacity) : host_(host), capacity_(capacity) {
    if (host == BCL::rank()) {
      data = BCL::alloc<T>(capacity*BCL::nprocs());
      pointers = BCL::alloc<int>(2 * capacity);

      if (data == nullptr || pointers == nullptr) {
        throw std::runtime_error("AGH! ran out of memory.");
      }

      for (size_t i = 0; i < 2*capacity; i++) {
        pointers[i] = 0;
      }
    }

    data = BCL::broadcast(data, host);
    pointers = BCL::broadcast(pointers, host);
  }

  BCL::GlobalPtr<T> data_ptr(size_t rank = BCL::rank()) {
    return data + capacity()*rank;
  }

  BCL::GlobalPtr<int> head_ptr(size_t rank = BCL::rank()) {
    return pointers + rank*2;
  }

  BCL::GlobalPtr<int> tail_ptr(size_t rank = BCL::rank()) {
    return pointers + rank*2 + 1;
  }

  bool push_nonatomic(const T& value) {
    if (tail_buffer_ - head_buffer_ >= capacity()) {
      head_buffer_ = *head_ptr();
      if (tail_buffer_ - head_buffer_ >= capacity()) {
        return false;
      }
    }
    data_ptr()[tail_buffer_ % capacity()] = value;
    tail_buffer_++;
    *tail_ptr() = tail_buffer_;
    return true;
  }

  bool push_atomic(const T& value) {
    if (tail_buffer_ - head_buffer_ >= capacity()) {
      head_buffer_ = *head_ptr();
      if (tail_buffer_ - head_buffer_ >= capacity()) {
        return false;
      }
    }
    data_ptr()[tail_buffer_ % capacity()] = value;
    BCL::flush();
    tail_buffer_++;
    *tail_ptr() = tail_buffer_;
  }

  bool pop(T& value) {
    size_t offset = lrand48() % BCL::nprocs();
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      size_t remote_rank = (i + offset) % BCL::nprocs();
      if (pop_from_queue_impl_(value, remote_rank)) {
        return true;
      }
    }
    return false;
  }

  bool pop_from_queue_impl_(T& value, size_t rank) {
    size_t tail = *tail_ptr(rank);
    size_t head = *head_ptr(rank);
    if (tail > head) {
      value = data_ptr(rank)[head];
      *head_ptr(rank) = head+1;
      return true;
    } else {
      return false;
    }
  }

};

} // end BCL
