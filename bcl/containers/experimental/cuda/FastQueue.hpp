
namespace BCL {

namespace cuda {

template <typename T>
struct FastQueue {
  using value_type = T;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  FastQueue(size_t host, size_t capacity) :
            capacity_(capacity), host_(host) {
    if (BCL::rank() == host) {
      data_ = BCL::cuda::alloc<value_type>(capacity);
      head_ = BCL::cuda::alloc<int>(1);
      tail_ = BCL::cuda::alloc<int>(1);
      int value = 0;
      BCL::cuda::memcpy(head_, &value, sizeof(int));
      BCL::cuda::memcpy(tail_, &value, sizeof(int));
    }
    data_ = BCL::broadcast(data_, host);
    head_ = BCL::broadcast(head_, host);
    tail_ = BCL::broadcast(tail_, host);

    if (data_ == nullptr || head_ == nullptr || tail_ == nullptr) {
      throw std::runtime_error("FastQueue ran out of memory with request for " + std::to_string(capacity*sizeof(value_type)) + " bytes.");
    }
  }

  __device__ bool push(const value_type& value) {
    int loc = BCL::cuda::fetch_and_add(tail_, 1);
    int new_tail = loc+1;
    // TODO: update tail_buffer
    if (new_tail - head_buf_ > capacity()) {
      // head_buf_ = BCL::cuda::rget(head_);
      BCL::cuda::memcpy(&head_buf_, head_, sizeof(int));
      if (new_tail - head_buf_ > capacity()) {
        BCL::cuda::fetch_and_add(tail_, -1);
        return false;
      }
    }

    // data_[loc % capacity()] = value;
    BCL::cuda::memcpy(data_ + (loc % capacity()), &value, sizeof(value_type));

    return true;
  }

  __device__ bool pop(value_type& value) {
    int loc = BCL::cuda::fetch_and_add(head_, 1);
    int new_head = loc+1;
    // TODO: update head_buffer?
    if (new_head > tail_buf_) {
      // tail_buf_ = BCL::cuda::rget(tail_);
      BCL::cuda::memcpy(&tail_buf_, tail_, sizeof(int));
      if (new_head > tail_buf_) {
        BCL::cuda::fetch_and_add(head_, -1);
        return false;
      }
    }

    // value = data_[loc % capacity()];
    BCL::cuda::memcpy(&value, data_ + (loc % capacity()), sizeof(value_type));

    return true;
  }

  __host__ __device__ size_t host() const noexcept {
    return host_;
  }

  __host__ __device__ size_t capacity() const noexcept {
    return capacity_;
  }

  __host__ __device__ size_t size() const noexcept {
    int head;
    int tail;
    BCL::cuda::memcpy(&head, head_, sizeof(int));
    BCL::cuda::memcpy(&tail, tail_, sizeof(int));
    return tail - head;
  }

  BCL::cuda::ptr<value_type> data_;
  BCL::cuda::ptr<int> head_;
  BCL::cuda::ptr<int> tail_;
  int head_buf_ = 0;
  int tail_buf_ = 0;
  size_t host_;
  size_t capacity_;
};

} // end cuda
} // end BCL
