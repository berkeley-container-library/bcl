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
    size_t len;
    std::vector<T> buffer;
    std::atomic<size_t> tail;
    std::atomic<size_t> reserved_tail;

  public:
    enum class status_t {
      FAIL,
      SUCCESS,
      SUCCESS_AND_FULL
    };

    AggBuffer(): len(0), tail(0), reserved_tail(0) {}

    explicit AggBuffer(size_t _size): len(_size), tail(0), reserved_tail(0) {
      buffer = std::vector<T>(len);
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

    // not concurrently safe with other pop_all
    bool pop_full(std::vector<T>& receiver) {
      if (reserved_tail != len) {
        return false; // not full
      }
#ifdef ARH_DEBUG
      if (reserved_tail != len) {
        std::printf("Warning (ARH::AggBuffer::pop_all): Call pop_all when buffer is not full\n");
      }
#endif
      receiver = std::move(buffer);
      buffer = std::vector<T>(len);
#ifdef ARH_DEBUG
      if (receiver.size() != len) {
        std::printf("Error (ARH::AggBuffer::pop_all): receiver.size() != len after move\n");
      }
#endif

      reserved_tail = 0;
      tail = 0;

      return true;
    }

    // not concurrently safe at all
    size_t pop_nofull(std::vector<T>& receiver) {
      receiver = std::move(buffer);
      buffer = std::vector<T>(len);

      size_t tmp = reserved_tail.load();
      reserved_tail = 0;
      tail = 0;

      return tmp;
    }

  };
}

#endif //ARH_AGG_BUFFER_HPP
