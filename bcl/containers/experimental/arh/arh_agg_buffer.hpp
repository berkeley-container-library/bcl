//
// Created by Jiakun Yan on 10/25/19.
//

#ifndef ARH_AGG_BUFFER_HPP
#define ARH_AGG_BUFFER_HPP

#include <atomic>
#include <vector>

namespace ARH {
  template <typename T>
  class AggBuffer {
    alignas(alignof_cacheline) std::vector<T> buffer;
    alignas(alignof_cacheline) std::atomic<size_t> tail;
    alignas(alignof_cacheline) std::atomic<size_t> reserved_tail;
    alignas(alignof_cacheline) size_t len;
    alignas(alignof_cacheline) std::mutex mutex_pop;

  public:
    enum class status_t {
      FAIL,
      SUCCESS,
      SUCCESS_AND_FULL
    };

    AggBuffer(): len(0), tail(0), reserved_tail(0) {
      ARH_Assert_Align(buffer, alignof_cacheline);
      ARH_Assert_Align(tail, alignof_cacheline);
      ARH_Assert_Align(reserved_tail, alignof_cacheline);
      ARH_Assert_Align(len, alignof_cacheline);
    }

    explicit AggBuffer(size_t _size): len(_size), tail(0), reserved_tail(0) {
      buffer = std::vector<T>(len);
      ARH_Assert_Align(buffer, alignof_cacheline);
      ARH_Assert_Align(tail, alignof_cacheline);
      ARH_Assert_Align(reserved_tail, alignof_cacheline);
      ARH_Assert_Align(len, alignof_cacheline);
    }

    void init(size_t _size) {
      len = _size;
      tail = 0;
      reserved_tail = 0;
      buffer = std::vector<T>(len);
    }

    size_t size() const {
      return reserved_tail.load();
    }

    status_t push(T val) {
      size_t current_tail = tail++;
      if (current_tail >= len) {
        if (current_tail == std::numeric_limits<size_t>::max()) {
          throw std::overflow_error("ARH::AggBuffer::push: tail overflow");
        }
        return status_t::FAIL;
      }
      buffer[current_tail] = std::move(val);
      size_t temp = ++reserved_tail;
      if (temp == len) {
        return status_t::SUCCESS_AND_FULL;
      } else {
        return status_t::SUCCESS;
      }
    }

    size_t pop_all(std::vector<T>& receiver) {
      if (mutex_pop.try_lock()) {
        size_t real_tail = std::min(tail.fetch_add(len), len); // prevent others from begining pushing
        // wait until those who is pushing finish
        while (reserved_tail != real_tail) {
        }

        receiver = std::move(buffer);
        buffer = std::vector<T>(len);

        ARH_Assert(real_tail <= len, "");
        ARH_Assert(real_tail == reserved_tail, "");
        for (int i = 0; i < len - real_tail; ++i) {
          receiver.pop_back();
        }

        reserved_tail = 0;
        tail = 0;

        mutex_pop.unlock();
        return real_tail;
      } else {
        // someone is poping
        return 0;
      }
    }

  };
}

#endif //ARH_AGG_BUFFER_HPP
