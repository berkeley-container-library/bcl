#pragma once

#include <atomic>
#include <functional>
#include <utility>

namespace ARH {
  struct ThreadBarrier {

    void init(size_t thread_num, std::function<void(void)> do_something = []{}) {
      thread_num_ = thread_num;
      do_something_ = std::move(do_something);
    }

    void wait() {
      size_t mstep = step.load();

      if (++waiting == thread_num_) {
        waiting = 0;
        step++;
      }
      else {
        while (step == mstep) {
          do_something_();
        }
      }
    }

  private:
    std::atomic<size_t> waiting = 0;
    std::atomic<size_t> step = 0;
    size_t thread_num_;
    std::function<void(void)> do_something_;
  };
}
