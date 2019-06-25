#pragma once

#include <stdexcept>

#include <bcl/bcl.hpp>
#include <bcl/containers/Container.hpp>
#include <bcl/containers/Array.hpp>
#include <bcl/core/util/Backoff.hpp>

#include <unistd.h>

namespace BCL {

template <typename T>
struct hashedData {
  T data;
  size_t hash;
};

/*
 * This ChecksumQueue can only be used when there is only one poper.
*/
template <
  typename T,
  typename Hash = std::hash<T>,
  typename TSerialize = BCL::serialize <hashedData<T>>
  >
struct ChecksumQueue {
  BCL::Array <hashedData<T>, TSerialize> data;
  BCL::GlobalPtr<int> head;
  BCL::GlobalPtr<int> tail;

  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using hasher = Hash;

  Hash hash_fn;

  uint64_t my_host;
  size_t my_capacity;

  size_t reserve_end = 0;
  int head_buf = 0;
  int tail_buf = 0;

  void print(const bool print_elements = false) {
    printf("Checksum Queue, size %d, capacity %d, hosted on %d\n", size(),
      capacity(), host());
    printf("Head at %d (%d), tail at %d (%d)\n", BCL::rget(head),
      BCL::rget(head) % capacity(), BCL::rget(tail), BCL::rget(tail) % capacity());

    if (print_elements) {
      for (int i = BCL::rget(head); i < BCL::rget(tail); i++) {
        std::cout << "slot " << i % capacity() << " " << data[i % capacity()].get().data << std::endl;
      }
    }
  }

  ChecksumQueue(const uint64_t host, const size_t capacity, const size_t reserve = 1) {
    this->my_host = host;
    this->my_capacity = capacity;
    this->reserve_end = reserve;
    try {
      this->data = std::move(BCL::Array <hashedData<T>, TSerialize>(host, capacity));
    } catch (std::runtime_error e) {
      throw std::runtime_error("BCL::Array: could not allocate data.");
    }

    if (BCL::rank() == host) {
      head = BCL::alloc <int>(1);
      tail = BCL::alloc <int>(1);

      if (head == nullptr || tail == nullptr) {
        throw std::runtime_error("BCL: ChecksumQueue does not have enough memory");
      }

      *head.local() = 0;
      *tail.local() = 0;
    }

    head = BCL::broadcast(head, host);
    tail = BCL::broadcast(tail, host);
  }


  ChecksumQueue(const ChecksumQueue &queue) = delete;

  ChecksumQueue &operator=(ChecksumQueue &&queue) {
    this->data = std::move(queue.data);
    this->head = queue.head;
    this->tail = queue.tail;
    this->my_capacity = queue.my_capacity;
    this->my_host = queue.my_host;
    this->reserve_end = queue.reserve_end;
    this->head_buf = queue.head_buf;

    queue.head = nullptr;
    queue.tail = nullptr;
    queue.my_capacity = 0;
    queue.my_host = 0;
    queue.head_buf = 0;
    return *this;
  }

  ChecksumQueue(ChecksumQueue && queue) {
    this->data = std::move(queue.data);
    this->head = queue.head;
    this->tail = queue.tail;
    this->my_capacity = queue.my_capacity;
    this->reserve_end = queue.reserve_end;
    this->my_host = queue.my_host;
    this->head_buf = queue.head_buf;

    queue.head = nullptr;
    queue.tail = nullptr;
    queue.my_capacity = 0;
    queue.my_host = 0;
    queue.head_buf = 0;
  }

  ~ChecksumQueue() {
    if (BCL::rank() == host() && !BCL::bcl_finalized) {
      if (head != nullptr) {
        dealloc(head);
      }
      if (tail != nullptr) {
        dealloc(tail);
      }
    }
  }

  // XXX: this is the current total size of the queue.
  //      It includes all elements which will be present
  //      in the queue after the next barrier (some may
  //      not be finished inserting).
  size_t size() const {
    return BCL::rget(tail) - BCL::rget(head);
  }

  bool empty() const {
    return size() == 0;
  }

  uint64_t host() const noexcept {
    return my_host;
  }

  size_t capacity() const noexcept {
    return my_capacity;
  }

  size_t get_hash(const T& val, int loc) {
    size_t hashed = hash_fn(val);
    bool sense = (loc / capacity()) % 2;
    // This occurs if you add an int=0. This is problematic since the
    // default data is zero, so processors can pop unwritten data if zero hashes to zero.
    // TODO preferably a more robust method of dealing with this...
    // XXX: I believe the best solution is just to use a more robust hash fn.
    //      There are some options in /bcl/core/detail/hash_functions,
    //      and I'll try to add some more.
    if (hashed == 0) {
      hashed = 42;
    }
    if (sense) {
      hashed = ~hashed;
    }
    return hashed;
  }

  bool __force_push(const T& val) {
    int old_tail = BCL::fetch_and_op<int>(tail, 1, BCL::plus<int>{});
    data[old_tail % capacity()] = {
      val, get_hash(val, old_tail)
    };
    BCL::flush();
    return true;
  }

  bool push(const T& val, bool wait_on_overrun = true) {
    if (wait_on_overrun == false)
      std::cout << "Wait on overrun cannot be false. Please use SafeChecksumQueue" << std::endl;
    int old_tail = BCL::fetch_and_op<int>(tail, 1, BCL::plus<int>{});
    if (old_tail >= capacity() + head_buf - reserve_end) {
      head_buf = BCL::fetch_and_op<int>(head, 0, BCL::plus<int>{});
      Backoff backoff;
      while (old_tail >= capacity() + head_buf - reserve_end) {
        backoff.backoff();
        head_buf = BCL::fetch_and_op<int>(head, 0, BCL::plus<int>{});
      }
    }
    data[old_tail % capacity()] = {
      val, get_hash(val, old_tail)
    };
    BCL::flush();
    return true;
  }

  bool push_atomic_impl_(const std::vector <T>& vals, bool synchronized = true) {
    if (synchronized == false)
      std::cout << "Please use SafeChecksumQueue" << std::endl;
    return push(vals);
  }

  bool push(const std::vector<T> &vals, bool wait_on_overrun = true) {
    if (wait_on_overrun == false)
      std::cout << "Wait on overrun cannot be false. Please use SafeChecksumQueue" << std::endl;
    if (vals.size() == 0) { return true;  }
    int old_tail = BCL::fetch_and_op<int>(tail, vals.size(), BCL::plus<int>{});
    int new_tail = old_tail + vals.size();
    //printf("Pushing %d elements; tail at %d/%d.\n", vals.size(), old_tail, head_buf);
    if (new_tail > capacity() + head_buf - reserve_end) {
      head_buf = BCL::fetch_and_op<int>(head, 0, BCL::plus<int>{});
      Backoff backoff;
      while (new_tail > capacity() + head_buf - reserve_end) {
        backoff.backoff();
        head_buf = BCL::fetch_and_op<int>(head, 0, BCL::plus<int>{});
      }
    }
    std::vector<hashedData<T>> hvals;
    hvals.resize(vals.size());
    for (int ii = 0; ii < vals.size(); ++ii) {
      hvals[ii] = {
        vals[ii], get_hash(vals[ii], old_tail + ii)
      };
      //printf("Pushing @%d v%d h%d\n", old_tail + ii, hvals[ii].data, hvals[ii].hash);
    }
    if ((old_tail % capacity()) + vals.size() <= capacity()) {
      // Contiguous write
      data.put(old_tail % capacity(), hvals);
      //printf("Contig write: (%d->%d), %d %d\n", old_tail, old_tail + hvals.size(),
      //	hvals[0].data, hvals[0].hash);
    } else {
      // Split write
      size_t first_put_nelem = capacity() - old_tail % capacity();
      //printf("Splitting write: (%d->%d), (0->%d), %d %d\n", old_tail, capacity(),
      //	first_put_nelem, hvals[0].data, hvals[0].hash);
      data.put(old_tail % capacity(), hvals.data(), first_put_nelem);
      data.put(0, hvals.data() + first_put_nelem, hvals.size() - first_put_nelem);
    }
    BCL::flush();
    return true;
  }

  using queue_type = ChecksumQueue;
  class push_future {
  public:
    push_future(std::vector<T>&& value, size_t old_tail, size_t new_tail, queue_type& queue)
      : value_(new std::vector<T>(std::move(value))), old_tail_(old_tail),
        new_tail_(new_tail), queue_(&queue)
    {
    }

    push_future() = delete;
    push_future(const push_future&) = delete;
    push_future& operator=(const push_future&) = delete;
    push_future(push_future&&) = default;
    push_future& operator=(push_future&&) = default;

    bool is_ready() {
      if (finished_) {
        return true;
      }

      if (new_tail_ > queue_->capacity() + queue_->head_buf - queue_->reserve_end) {
        queue_->head_buf = BCL::fetch_and_op<int>(queue_->head, 0, BCL::plus<int>{});
        if (new_tail_ > queue_->capacity() + queue_->head_buf - queue_->reserve_end) {
          return finished_;
        }
      }

      finished_ = true;

      std::vector<hashedData<T>> hvals;
      hvals.resize((*value_).size());
      for (int ii = 0; ii < (*value_).size(); ++ii) {
        hvals[ii] = {
          (*value_)[ii], queue_->get_hash((*value_)[ii], old_tail_ + ii)
        };
        //printf("Pushing @%d v%d h%d\n", old_tail + ii, hvals[ii].data, hvals[ii].hash);
      }
      if ((old_tail_ % queue_->capacity()) + (*value_).size() <= queue_->capacity()) {
        // Contiguous write
        queue_->data.put(old_tail_ % queue_->capacity(), hvals);
        //printf("Contig write: (%d->%d), %d %d\n", old_tail, old_tail + hvals.size(),
        //	hvals[0].data, hvals[0].hash);
      } else {
        // Split write
        size_t first_put_nelem = queue_->capacity() - old_tail_ % queue_->capacity();
        //printf("Splitting write: (%d->%d), (0->%d), %d %d\n", old_tail, capacity(),
        //	first_put_nelem, hvals[0].data, hvals[0].hash);
        queue_->data.put(old_tail_ % queue_->capacity(), hvals.data(), first_put_nelem);
        queue_->data.put(0, hvals.data() + first_put_nelem, hvals.size() - first_put_nelem);
      }
      BCL::flush();

      return finished_;
    }

  private:
    std::unique_ptr<std::vector<T>> value_;
    queue_type* queue_;
    int old_tail_, new_tail_;
    bool finished_ = false;
  };

  auto async_push(std::vector<T>&& vals) {
    int old_tail = BCL::fetch_and_op<int>(tail, vals.size(), BCL::plus<int>{});
    int new_tail = old_tail + vals.size();
    return push_future(std::move(vals), old_tail, new_tail, *this);
  }

  bool pop(T& val, bool wait_on_underrun = false) {
    int old_head = BCL::fetch_and_op<int>(head, 1, BCL::plus<int>{});

    if (old_head >= tail_buf) {
      tail_buf = BCL::fetch_and_op<int>(tail, 0, BCL::plus<int>{});
      if (wait_on_underrun) {
        Backoff backoff;
        while (old_head >= tail_buf) {
          //printf("Couldn't pop; not enough space; waiting (h: %d, t: %d)\n", old_head, tail_buf);
          backoff.backoff();
          tail_buf = BCL::fetch_and_op<int>(tail, 0, BCL::plus<int>{});
        }
      } else {
        if (old_head >= tail_buf) {
          BCL::fetch_and_op<int>(head, -1, BCL::plus<int>{});
          //printf("Couldn't pop; not enough space (h: %d, t: %d)\n", old_head, tail_buf);
          return false;
        }
      }
    }

    Backoff backoff;
    //It is necessary to retry pops until they work. Consider the case:
    //push push pop pop (in this order, almost simultaneous, by different machines)
    //where the first push executes slowly, so the first pop reads bad data,
    //but the second pop works. In this case the first pop cannot abort
    //without putting the queue in a questionable state.
    while (1) {
      hashedData<T> hd = *data[old_head % capacity()];
      if (hd.hash == get_hash(hd.data, old_head)) {
        val = std::move(hd.data);
        return true;
      }
      //If this prints with a low size-value and sense-disamb true, then you are popping
      //while someone is still writing. This is safe.
      //If this prints with a high size-value and sense-disamb true, then someone has
      //pushed to the data before you popped it. This is deadly.
      //printf("Read incorrect value! Retrying. (Sense-disambiguated: %d) (Size: %d)\n",
      //	hd.hash == get_hash(hd.data, old_head+capacity()), tail_buf - old_head);
      backoff.backoff();
    }
    //This is not reached.
    return false;
  }

  void __internal_pop_many(hashedData<T>* hvals, size_t n_pop, size_t from) {
    if ((from % capacity()) + n_pop <= capacity()) {
      data.get(from % capacity(), hvals, n_pop);
    } else {
      size_t first_put_nelem = capacity() - (from % capacity());
      data.get(from % capacity(), hvals, first_put_nelem);
      data.get(0, hvals + first_put_nelem, n_pop - first_put_nelem);
    }
  }

  //take_fewer is probably unsafe on multi-pops
  bool pop(std::vector<T> &vals, size_t n_pop, bool take_fewer = true) {
    int old_head = BCL::fetch_and_op<int>(head, n_pop, BCL::plus<int>{});
    int new_head = old_head + n_pop;

    if (new_head > tail_buf) {
      tail_buf = BCL::fetch_and_op<int>(tail, 0, BCL::plus<int>{});
      if (new_head > tail_buf) {
        if (take_fewer) {
          BCL::fetch_and_op<int>(head, tail_buf - new_head, BCL::plus<int>{});
          n_pop = tail_buf - old_head;
          if (n_pop == 0) {
            //printf("Nothing to pop (h: %d, t: %d)\n", old_head, tail_buf);
            return false;
          }
          //printf("Popping fewer elements: %d->%d\n", new_head - old_head, n_pop);
        } else {
          BCL::fetch_and_op<int>(head, -n_pop, BCL::plus<int>{});
          //printf("Couldn't batch-pop; not enough space (h: %d, t: %d)\n", old_head, tail_buf);
          return false;
        }
      }
    }
    std::vector<hashedData<T>> hvals;
    vals.resize(n_pop);
    hvals.resize(n_pop);
    Backoff backoff;
    //Retry pops until they work, but only re-request the bad data.
    size_t s_err = n_pop;
    size_t h_err = 0;
    size_t n_pop_part = n_pop;
    while (1) {
      //printf("Reading %d vals (%d->%d) (%d)\n", n_pop_part,
      //	h_err, s_err, old_head);
      __internal_pop_many(hvals.data() + h_err, n_pop_part, old_head + h_err);
      size_t ubound = s_err;
      for (int ii = h_err; ii < ubound; ++ii) {
        if (hvals[ii].hash == get_hash(hvals[ii].data, old_head + ii)) {
          vals[ii] = std::move(hvals[ii].data);
        } else {
          //printf("Read incorrect @ %d(%d). (Sense-disambiguated: %d) (%d %d)\n", ii, old_head + ii,
          //	hvals[ii].hash == get_hash(hvals[ii].data, old_head + ii + capacity()),
          //		hvals[ii].data, hvals[ii].hash);

          s_err = (ii < s_err) ? ii : s_err;
          h_err = ii + 1;
        }
      }
      if (ubound == s_err) { //No errors
        return true;
      }
      n_pop_part = h_err;
      h_err = s_err;
      s_err = n_pop_part;
      n_pop_part = s_err - h_err;
      printf("Read up to %lu incorrect values (%lu->%lu) (%d->%d)\n", n_pop_part, h_err, s_err,
        old_head, tail_buf);
      backoff.backoff();
    };
    //This is not reached.
    return false;
  }

};

} // end BCL
