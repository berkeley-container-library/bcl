#include <cstdlib>
#include <cstdio>
#include <vector>
#include <stdexcept>

#include <bcl/bcl.hpp>
#include <bcl/containers/Container.hpp>
#include <bcl/containers/Array.hpp>
#include <bcl/core/util/Backoff.hpp>

#include <unistd.h>

namespace BCL {

struct CircularQueueAL {
  constexpr static int none = 0x0;
  constexpr static int push = (0x1 << 0);
  constexpr static int pop = (0x1 << 1);
  constexpr static int push_pop = (0x1 << 0) | (0x1 << 1);

  int val;

  CircularQueueAL(int val) : val(val) {}
  CircularQueueAL& operator=(const CircularQueueAL&) = default;

  operator int() const {
    return val;
  }

  CircularQueueAL& operator=(int val) {
    this->val = val;
    return *this;
  }

  bool operator==(int val) const {
    return this->val == val;
  }
};

template <typename T, typename TSerialize = BCL::serialize <T>>
struct CircularQueue {
  BCL::Array <T, TSerialize> data;
  BCL::GlobalPtr<int> head;
  BCL::GlobalPtr<int> tail;

  BCL::GlobalPtr<int> reserved_head;
  BCL::GlobalPtr<int> reserved_tail;

  uint64_t my_host;
  size_t my_capacity;

  // Buffered location of head.
  // This allows us to get a strict overestimation
  // of the current queue size without an AMO!
  int head_buf = 0;
  int tail_buf = 0;

  void print(const bool print_elements = false) {
    printf("Ring Buffer Queue, size %d, capacity %d, hosted on %d\n", size(),
      capacity(), host());
    printf("Head at %d (%d), tail at %d (%d)\n", BCL::rget(head),
      BCL::rget(head) % capacity(), BCL::rget(tail), BCL::rget(tail) % capacity());

    if (print_elements) {
      for (int i = BCL::rget(head); i < BCL::rget(tail); i++) {
        std::cout << "slot " << i % capacity() << " " << data[i % capacity()].get() << std::endl;
      }
    }
  }

  CircularQueue(const uint64_t host, const size_t capacity) {
    this->my_host = host;
    this->my_capacity = capacity;
    try {
      this->data = std::move(BCL::Array <T, TSerialize> (host, capacity));
    } catch (std::runtime_error e) {
      throw std::runtime_error("BCL::Array: could not allocate data.");
    }

    if (BCL::rank() == host) {
      head = BCL::alloc <int> (1);
      tail = BCL::alloc <int> (1);
      reserved_head = BCL::alloc <int> (1);
      reserved_tail = BCL::alloc <int> (1);

      if (head == nullptr || tail == nullptr || reserved_head == nullptr
          || reserved_tail == nullptr) {
        throw std::runtime_error("BCL: CircularQueue does not have enough memory");
      }

      *head.local() = 0;
      *tail.local() = 0;
      *reserved_head.local() = 0;
      *reserved_tail.local() = 0;
    }

    head = BCL::broadcast(head, host);
    tail = BCL::broadcast(tail, host);
    reserved_head = BCL::broadcast(reserved_head, host);
    reserved_tail = BCL::broadcast(reserved_tail, host);
  }

  CircularQueue(const CircularQueue &queue) = delete;

  CircularQueue &operator=(CircularQueue &&queue) {
    this->data = std::move(queue.data);
    this->head = queue.head;
    this->tail = queue.tail;
    this->reserved_head = queue.reserved_head;
    this->reserved_tail = queue.reserved_tail;
    this->my_capacity = queue.my_capacity;
    this->my_host = queue.my_host;
    this->head_buf = queue.head_buf;

    queue.head = nullptr;
    queue.tail = nullptr;
    queue.reserved_head = nullptr;
    queue.reserved_tail = nullptr;
    queue.my_capacity = 0;
    queue.my_host = 0;
    queue.head_buf = 0;
    return *this;
  }

  CircularQueue(CircularQueue && queue) {
    this->data = std::move(queue.data);
    this->head = queue.head;
    this->tail = queue.tail;
    this->reserved_head = queue.reserved_head;
    this->reserved_tail = queue.reserved_tail;
    this->my_capacity = queue.my_capacity;
    this->my_host = queue.my_host;
    this->head_buf = queue.head_buf;

    queue.head = nullptr;
    queue.tail = nullptr;
    queue.reserved_head = nullptr;
    queue.reserved_tail = nullptr;
    queue.my_capacity = 0;
    queue.my_host = 0;
    queue.head_buf = 0;
  }

  ~CircularQueue() {
    if (BCL::rank() == host() && !BCL::bcl_finalized) {
      if (head != nullptr) {
        dealloc(head);
      }
      if (tail != nullptr) {
        dealloc(tail);
      }
      if (reserved_head != nullptr) {
        dealloc(reserved_head);
      }
      if (reserved_tail != nullptr) {
        dealloc(reserved_tail);
      }
    }
  }

  // XXX: experimental, for Colin Wahl.
  T* get_ptr_() {
    return (BCL::decay_container<T>(data.data) + *head).local();
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

  bool push(const T& val, CircularQueueAL atomicity_level =
            CircularQueueAL::push | CircularQueueAL::pop) {
    if (atomicity_level & CircularQueueAL::pop) {
      return push_atomic_impl_(val);
    } else {
      return push_nonatomic_impl_(val);
    }
  }

  bool push_atomic_impl_(const T &val, bool synchronized = false) {
    int old_tail = BCL::fetch_and_op<int>(tail, 1, BCL::plus<int>{});
    int new_tail = old_tail + 1;

    if (new_tail - head_buf > capacity()) {
      // head_buf = BCL::rget(reserved_head);
      if (synchronized) {
        Backoff backoff;
        while (new_tail - head_buf > capacity()) {
          head_buf = BCL::fetch_and_op<int>(reserved_head, 0, BCL::plus<int>{});
          if (new_tail - head_buf > capacity()) {
            backoff.backoff();
          }
        }
      } else {
        head_buf = BCL::fetch_and_op<int>(reserved_head, 0, BCL::plus<int>{});
      }
      if (new_tail - head_buf > capacity()) {
        BCL::fetch_and_op<int>(tail, -1, BCL::plus<int>{});
        return false;
      }
    }
    data[old_tail % capacity()] = val;
    BCL::flush();
    int rv;
    Backoff backoff;
    do {
      rv = BCL::compare_and_swap<int>(reserved_tail, old_tail, new_tail);
      if (rv != old_tail) {
        backoff.backoff();
      }
    } while (rv != old_tail);
    return true;
  }

  bool push_nonatomic_impl_(const T &val) {
    int old_tail = BCL::fetch_and_op<int>(tail, 1, BCL::plus<int>{});
    int new_tail = old_tail + 1;

    if (new_tail - head_buf > capacity()) {
      // head_buf = BCL::rget(reserved_head);
      head_buf = BCL::fetch_and_op<int>(reserved_head, 0, BCL::plus<int>{});
      if (new_tail - head_buf > capacity()) {
        BCL::fetch_and_op<int>(tail, -1, BCL::plus<int>{});
        return false;
      }
    }
    data[old_tail % capacity()] = val;
    // XXX: flush not required here.
    BCL::fetch_and_op<int>(reserved_tail, 1, BCL::plus<int>{});
    return true;
  }

  bool push(const std::vector <T> &vals, CircularQueueAL atomicity_level =
            CircularQueueAL::push | CircularQueueAL::pop) {
    if (vals.size() == 0) {
      return true;
    }
    if (atomicity_level & CircularQueueAL::pop) {
      return push_atomic_impl_(vals);
    } else {
      return push_nonatomic_impl_(vals);
    }
  }

  using queue_type = CircularQueue;
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
        return finished_;
      }

      if (!reserved_) {
        if (new_tail_ - queue_->head_buf > queue_->capacity()) {
          queue_->head_buf = BCL::fetch_and_op<int>(queue_->reserved_head, 0, BCL::plus<int>{});
          if (new_tail_ - queue_->head_buf > queue_->capacity()) {
            return false;
          }
        }
        reserved_ = true;

        if ((old_tail_ % queue_->capacity()) + value_->size() < queue_->capacity()) {
          // One contiguous write to 1's
          // 00000000000001111
          queue_->data.put(old_tail_ % queue_->capacity(), *value_);
        } else {
          // "Split write:" to 1's
          // 11111000000001111
          size_t first_put_nelem = queue_->capacity() - old_tail_ % queue_->capacity();
          queue_->data.put(old_tail_ % queue_->capacity(), value_->data(), first_put_nelem);
          queue_->data.put(0, value_->data() + first_put_nelem, value_->size() - first_put_nelem);
        }
        BCL::flush();
      }

      int rv = BCL::compare_and_swap<int>(queue_->reserved_tail, old_tail_, new_tail_);
      if (rv == old_tail_) {
        finished_ = true;
      }
      return finished_;
    }

  private:
    std::unique_ptr<std::vector<T>> value_;
    queue_type* queue_;
    int old_tail_, new_tail_;
    bool reserved_ = false;
    bool finished_ = false;
  };

  auto async_push(std::vector<T>&& vals) {
    int old_tail = BCL::fetch_and_op<int>(tail, vals.size(), BCL::plus<int>{});
    int new_tail = old_tail + vals.size();
    return push_future(std::move(vals), old_tail, new_tail, *this);
  }

  bool push_atomic_impl_(const std::vector <T>& vals, bool synchronized = false) {
    if (vals.size() == 0) {
      return true;
    }

    int old_tail = BCL::fetch_and_op<int>(tail, vals.size(), BCL::plus<int>{});
    int new_tail = old_tail + vals.size();

    if (new_tail - head_buf > capacity()) {
      // head_buf = BCL::rget(reserved_head);
      if (synchronized) {
        Backoff backoff;
        while (new_tail - head_buf > capacity()) {
          head_buf = BCL::fetch_and_op<int>(reserved_head, 0, BCL::plus<int>{});
          if (new_tail - head_buf > capacity()) {
            backoff.backoff();
          }
        }
      } else {
        head_buf = BCL::fetch_and_op<int>(reserved_head, 0, BCL::plus<int>{});
      }
      if (new_tail - head_buf > capacity()) {
        BCL::fetch_and_op<int>(tail, -vals.size(), BCL::plus<int>{});
        return false;
      }
    }
    if ((old_tail % capacity()) + vals.size() < capacity()) {
      // One contiguous write to 1's
      // 00000000000001111
      data.put(old_tail % capacity(), vals);
    } else {
      // "Split write:" to 1's
      // 11111000000001111
      size_t first_put_nelem = capacity() - old_tail % capacity();
      data.put(old_tail % capacity(), vals.data(), first_put_nelem);
      data.put(0, vals.data() + first_put_nelem, vals.size() - first_put_nelem);
    }
    BCL::flush();
    int rv;
    Backoff backoff;
    do {
      /*
      fprintf(stderr, "(%lu) in CAS loop. reserved_tail %lu -> %lu\n", BCL::rank(),
              old_tail, new_tail);
              */
      rv = BCL::compare_and_swap<int>(reserved_tail, old_tail, new_tail);
      if (rv != old_tail) {
        backoff.backoff();
      }
    } while (rv != old_tail);
    return true;
  }

  bool push_nonatomic_impl_(const std::vector <T> &vals) {
    int old_tail = BCL::fetch_and_op<int>(tail, vals.size(), BCL::plus<int>{});
    int new_tail = old_tail + vals.size();

    if (new_tail - head_buf > capacity()) {
      head_buf = BCL::rget(reserved_head);
      if (new_tail - head_buf > capacity()) {
        BCL::fetch_and_op<int>(tail, -vals.size(), BCL::plus<int>{});
        return false;
      }
    }
    data.put(old_tail % capacity(), vals);
    // XXX: flush not required here.
    BCL::fetch_and_op<int>(reserved_tail, vals.size(), BCL::plus<int>{});
    return true;
  }

  bool pop(T& val, CircularQueueAL atomicity_level = CircularQueueAL::push |
           CircularQueueAL::pop) {
    if (atomicity_level & CircularQueueAL::push) {
      return pop_atomic_impl_(val);
    } else if (atomicity_level & CircularQueueAL::pop){
      return pop_nonatomic_impl_(val);
    } else if (atomicity_level == CircularQueueAL::none) {
      return local_nonatomic_pop(val);
    }
    return false;
  }

  bool pop_atomic_impl_(T &val) {
    int old_head = BCL::fetch_and_op<int>(head, 1, BCL::plus<int>{});
    int new_head = old_head + 1;

    if (new_head > tail_buf) {
      // tail_buf = BCL::rget(reserved_tail);
      tail_buf = BCL::fetch_and_op<int>(reserved_tail, 0, BCL::plus<int>{});
      if (new_head > tail_buf) {
        BCL::fetch_and_op<int>(head, -1, BCL::plus<int>{});
        return false;
      }
    }

    val = *data[old_head % capacity()];
    // BCL::flush(); (implicit)

    int rv;
    size_t backoff_value;
    if (BCL::rank() == host()) {
      backoff_value = 100;
    } else {
      backoff_value = 100;
    }
    Backoff backoff(1, backoff_value);
    do {
      rv = BCL::compare_and_swap<int>(reserved_head, old_head, old_head + 1);
      if (rv != old_head) {
        backoff.backoff();
      }
    } while (rv != old_head);
    return true;
  }

  bool pop_nonatomic_impl_(T &val) {
    int old_head = BCL::fetch_and_op<int>(head, 1, BCL::plus<int>{});
    int new_head = old_head + 1;

    if (new_head > tail_buf) {
      tail_buf = BCL::rget(tail);
      if (new_head > tail_buf) {
        BCL::fetch_and_op<int>(head, -1, BCL::plus<int>{});
        return false;
      }
    }

    val = *data[old_head % capacity()];
    // XXX: flush not required here.
    BCL::fetch_and_op<int>(reserved_head, 1, BCL::plus<int>{});
    return true;
  }

  bool pop(std::vector<T>& vals, size_t n_to_pop,
           CircularQueueAL atomicity_level = CircularQueueAL::push |
           CircularQueueAL::pop) {
    if (atomicity_level & CircularQueueAL::push) {
      return pop_atomic_impl_(vals);
    } else {
      return pop_nonatomic_impl_(vals);
    }
  }

  bool pop_atomic_impl_(std::vector <T> &vals, size_t n_to_pop) {
    vals.resize(n_to_pop);

    int old_head = BCL::fetch_and_op<int>(head, n_to_pop, BCL::plus<int>());
    int new_head = old_head + 1;

    if (new_head > tail_buf) {
      tail_buf = BCL::fetch_and_op<int>(reserved_tail, 0, BCL::plus<int>{});
      if (new_head > tail_buf) {
        BCL::fetch_and_op <int> (head, -n_to_pop, BCL::plus <int> ());
        return false;
      }
    } else {

      if ((old_head % capacity()) + vals.size() < capacity()) {
        data.get(old_head % capacity(), vals, n_to_pop);
      } else {
        size_t first_get_nelem = capacity() - (old_head % capacity());
        data.get(0, vals.data(), n_to_pop - first_get_nelem);
      }

      // BCL::flush() (implicit)

      int rv;
      Backoff backoff;
      do {
        rv = BCL::compare_and_swap<int>(reserved_head, old_head, old_head + n_to_pop);
        if (rv != old_head) {
          backoff.backoff();
        }
      } while (rv != old_head);
      return true;
    }
  }

  bool pop_nonatomic_impl_(std::vector <T> &vals, size_t n_to_pop) {
    vals.resize(n_to_pop);
    int my_tail = BCL::rget(tail);
    int old_head = BCL::fetch_and_op <int> (head, n_to_pop, BCL::plus <int> ());
    if (my_tail - old_head < n_to_pop) {
      BCL::fetch_and_op <int> (head, -n_to_pop, BCL::plus <int> ());
      return false;
    } else {
      data.get(old_head % capacity(), vals, n_to_pop);
      BCL::fetch_and_op<int>(reserved_head, n_to_pop, BCL::plus<int>{});
      return true;
    }
  }

  // Nonatomic with respect to remote pops or pushes
  bool local_nonatomic_pop(T &val) {
    if (BCL::rank() != host()) {
      return false;
    }
    int *head_ptr = head.local();
    int *tail_ptr = tail.local();
    if (*head_ptr + 1 > *tail_ptr) {
      return false;
    }
    val = *data[*head_ptr % capacity()];
    *head_ptr += 1;
    return true;
  }

  // TODO: deal properly with queues that wrap around.
  std::vector <T> as_vector() {
    if (BCL::rank() != host()) {
      throw std::runtime_error("rank " + std::to_string(BCL::rank()) +
        " tried to collective local vector from remote queue on rank " +
        std::to_string(host()));
    }

    int head_val = *head.local();
    int tail_val = *tail.local();

    if (head_val != 0) {
      throw std::runtime_error("rank " + std::to_string(BCL::rank()) +
        " called as_vector() on a local queue with non-zero head.  This is" +
        "not support yet.");
    }

    T *queue_ptr = BCL::decay_container(data.data).local();
    return std::vector <T> (queue_ptr, queue_ptr + tail_val);
  }

  void resize(const size_t new_capacity) {
    BCL::barrier();

    if (new_capacity == capacity()) {
      return;
    }

    BCL::Array <T, TSerialize> new_data(host(), new_capacity);

    if (BCL::rank() == host()) {
      BCL::GlobalPtr <int> new_head = BCL::alloc <int> (1);
      BCL::GlobalPtr <int> new_tail = BCL::alloc <int> (1);

      *new_head.local() = 0;
      *new_tail.local() = std::min(size(), new_capacity);

      for (int i = *head.local(), j = 0; i < *tail.local() && j < *new_tail.local(); i++, j++) {
        new_data[j] = *data[i % capacity()];
      }

      // Maybe replace with this?
      /*
      memcpy(new_data.local(), data.local() + *head.local(),
        sizeof(BCL::Container <T, TSerialize>) * (*new_tail.local());
      */

      BCL::dealloc(head);
      BCL::dealloc(tail);

      head = new_head;
      tail = new_tail;
    }

    std::swap(data, new_data);

    head = BCL::broadcast(head, host());
    tail = BCL::broadcast(tail, host());
    my_capacity = new_capacity;

    BCL::barrier();
  }

  void print_info() {
    printf("Scanning through copy... at %s\n", data.str().c_str());
    for (int i = *head.local(); i < *tail.local(); i++) {
      printf("%d %s %d\n", i, data.local()[i].ptr.str().c_str(),
        data.local()[i].len);
    }
  }

  // Deal with migrating containers more intelligently
  void migrate(const uint64_t new_host) {
    BCL::barrier();

    if (new_host == host()) {
      return;
    }

    BCL::Array <T, TSerialize> new_data(new_host, capacity());

    BCL::GlobalPtr<int> new_head;
    BCL::GlobalPtr<int> new_tail;

    if (BCL::rank() == new_host) {
      new_head = BCL::alloc<int>(1);
      new_tail = BCL::alloc<int>(1);
    }

    new_head = BCL::broadcast(new_head, new_host);
    new_tail = BCL::broadcast(new_tail, new_host);

    if (BCL::rank() == host()) {
      for (int i = *head.local(); i < *tail.local(); i++) {
        new_data[i % capacity()] = *data[i % capacity()];
      }

      BCL::rput(*head.local(), new_head);
      BCL::rput(*tail.local(), new_tail);

      BCL::dealloc(head);
      BCL::dealloc(tail);

      head = new_head;
      tail = new_tail;
    }

    std::swap(data, new_data);

    head = BCL::broadcast(head, host());
    tail = BCL::broadcast(tail, host());
    my_host = new_host;

    BCL::barrier();
  }
};

} // end BCL
