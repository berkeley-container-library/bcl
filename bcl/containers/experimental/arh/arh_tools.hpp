//
// Created by Jiakun Yan on 10/21/19.
//

#ifndef ARH_BENCHMARK_TOOLS_HPP
#define ARH_BENCHMARK_TOOLS_HPP

#include <sys/time.h>
#include <time.h>

namespace ARH {
  typedef uint64_t tick_t;

  // microseconds, 12ns
  tick_t ticks_now() {
    return gasneti_ticks_now();
  }

  uint64_t ticks_to_ns(tick_t val) {
    return gasneti_ticks_to_ns(val);
  }

  // microseconds, 370ns
//  tick_t ticks_now() {
//    timespec temp;
//    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &temp);
//    return temp.tv_sec * 1e6 + temp.tv_nsec / 1e3;
//  }

  inline void update_average(double &average, uint64_t val, uint64_t num) {
    average += (val - average) / num;
//      if (my_worker() == 0) {
//        std::printf("val=%lu; num=%lu; average=%.2lf\n", val, num, average);
//      }
  }
}


#endif //BCL_BENCHMARK_TOOLS_HPP
