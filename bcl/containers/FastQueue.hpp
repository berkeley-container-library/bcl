#include <cstdlib>
#include <cstdio>
#include <vector>
#include <bcl/core/detail/optional.hpp>
#include <stdexcept>

#include <bcl/bcl.hpp>
#include <bcl/containers/Container.hpp>
#include <bcl/containers/Array.hpp>

#include <unistd.h>

namespace BCL {

template <typename T, typename TSerialize = BCL::serialize <T>>
struct FastQueue {
  BCL::Array <T, TSerialize> data;
  BCL::GlobalPtr <int> head;
  BCL::GlobalPtr <int> tail;

  using value_type = T;
  using iterator = BCL::GlobalPtr<T>;

  uint64_t my_host;
  size_t my_capacity;

  // Buffered location of head, tail.
  // This allows us to get a strict overestimation
  // of the current queue size without an AMO!
  int head_buf = 0;
  int tail_buf = 0;

  FastQueue() = default;

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

  FastQueue(size_t capacity) {
    this->my_host = BCL::rank();
    this->my_capacity = capacity;
    try {
      this->data = std::move(BCL::Array<T, TSerialize>(capacity));
    } catch (std::runtime_error e) {
      throw std::runtime_error("BCL: FastQueue could not allocate data.");
    }

    head = BCL::alloc<int>(1);
    tail = BCL::alloc<int>(1);

    if (head == nullptr || tail == nullptr) {
      throw std::runtime_error("BCL: FastQueue does not have enough memory");
    }

    *head.local() = 0;
    *tail.local() = 0;
  }

  FastQueue(const uint64_t host, const size_t capacity) {
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

      if (head == nullptr || tail == nullptr) {
        throw std::runtime_error("BCL: FastQueue does not have enough memory");
      }

      *head.local() = 0;
      *tail.local() = 0;
    }

    head = BCL::broadcast(head, host);
    tail = BCL::broadcast(tail, host);
  }

  FastQueue(const FastQueue&) = delete;
  FastQueue& operator=(const FastQueue&) = delete;

  FastQueue(FastQueue&&) = default;
  FastQueue& operator=(FastQueue&&) = default;

  ~FastQueue() {
    if (BCL::rank() == host() && !BCL::bcl_finalized) {
      if (head != nullptr) {
        dealloc(head);
      }
      if (tail != nullptr) {
        dealloc(tail);
      }
    }
  }

  // XXX: experimental, for Colin Wahl.
  T* get_ptr_() {
    return (BCL::decay_container(data.data) + *head).local();
  }

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

  bool push(const T &val) {
    int old_tail = BCL::fetch_and_op <int> (tail, 1, BCL::plus <int> ());
    int new_tail = old_tail + 1;

    if (new_tail - head_buf > capacity()) {
      head_buf = BCL::rget(head);
      if (new_tail - head_buf > capacity()) {
        BCL::fetch_and_op <int> (tail, -1, BCL::plus <int> ());
        return false;
      }
    }
    data[old_tail % capacity()] = val;
    return true;
  }

  bool pop(T &val) {
    int old_head = BCL::fetch_and_op <int> (head, 1, BCL::plus <int> ());
    int new_head = old_head + 1;

    if (new_head > tail_buf) {
      tail_buf = BCL::rget(tail);
      if (new_head > tail_buf) {
        BCL::fetch_and_op<int>(head, -1, BCL::plus<int>{});
        return false;
      }
    }

    val = *data[old_head % capacity()];
    return true;
  }

  bool push(const std::vector <T> &vals) {
    int old_tail = BCL::fetch_and_op <int> (tail, vals.size(), BCL::plus <int> ());
    int new_tail = old_tail + vals.size();

    if (new_tail - head_buf > capacity()) {
      head_buf = BCL::rget(head);
      if (new_tail - head_buf > capacity()) {
        BCL::fetch_and_op <int> (tail, -vals.size(), BCL::plus <int> ());
        return false;
      }
    }
    data.put(old_tail % capacity(), vals);
    return true;
  }

  bool pop(std::vector <T> &vals, const size_t n_to_pop) {
    vals.resize(n_to_pop);
    int my_tail = BCL::rget(tail);
    int old_head = BCL::fetch_and_op <int> (head, n_to_pop, BCL::plus <int> ());
    if (tail_buf - old_head < n_to_pop) {
      tail_buf = BCL::rget(tail);
      if (tail_buf - old_head < n_to_pop) {
        BCL::fetch_and_op <int> (head, -n_to_pop, BCL::plus <int> ());
        return false;
      }
    }
    data.get(old_head % capacity(), vals, n_to_pop);
    return true;
  }

  std::experimental::optional<future<std::vector<T>>> push(std::vector<T>&& vals) {
    int old_tail = BCL::fetch_and_op <int> (tail, vals.size(), BCL::plus <int> ());
    int new_tail = old_tail + vals.size();

    if (new_tail - head_buf > capacity()) {
      head_buf = BCL::rget(head);
      if (new_tail - head_buf > capacity()) {
        BCL::fetch_and_op <int> (tail, -vals.size(), BCL::plus <int> ());
        return {};
      }
    }
    BCL::GlobalPtr<T> data_ptr = BCL::decay_container(data.data);
    return BCL::arput(&data_ptr[old_tail % capacity()], std::move(vals));
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

  BCL::GlobalPtr<T> begin() const {
    return BCL::decay_container(data.data) + *head;
  }

  BCL::GlobalPtr<T> end() const {
    return BCL::decay_container(data.data) + *tail;
  }


  bool pop() {
    int my_tail = BCL::rget(tail);
    int old_head = BCL::fetch_and_op <int> (head, 1, BCL::plus <int> ());
    if (my_tail - old_head < 1) {
      BCL::fetch_and_op <int> (head, -1, BCL::plus <int> ());
      return false;
    } else {
      data[old_head % capacity()].free();
      return true;
    }
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

  void warmup_() {
    size_t n_vals = 64;
    srand48(BCL::rank());
    for (size_t i = 0; i < n_vals; i++) {
      size_t idx = lrand48() % capacity();
      this->begin()[idx] = i;
    }

    for (size_t i = 0; i < n_vals; i++) {
      size_t idx = lrand48() % capacity();
      T retrv = this->begin()[idx];
    }
  }
};

}
