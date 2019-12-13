//
// Created by Jiakun Yan on 10/21/19.
//

#ifndef ARH_BENCHMARK_TOOLS_HPP
#define ARH_BENCHMARK_TOOLS_HPP

#include <gasnetex.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>

#ifndef ARH_DEBUG
#   define ARH_Assert(Expr, Msg) \
    __ARH_Assert(#Expr, Expr, __FILE__, __LINE__, Msg)
#else
#   define ARH_Assert(Expr, Msg) ;
#endif
void __ARH_Assert(const char* expr_str, bool expr, const char* file, int line, const char* msg)
{
  if (!expr)
  {
    std::cerr << "Assert failed:\t" << msg << "\n"
              << "Expected:\t" << expr_str << "\n"
              << "Source:\t\t" << file << ", line " << line << "\n";
    abort();
  }
}

#define ARH_Assert_Align(Val, alignof_size) ARH_Assert(alignof(Val) % alignof_size == 0, "alignment check failed!")

namespace ARH {
  typedef uint64_t tick_t;

  // microseconds, 12ns
  tick_t ticks_now() {
    return gasneti_ticks_now();
  }

  uint64_t ticks_to_ns(tick_t val) {
    return gasneti_ticks_to_ns(val);
  }

  uint64_t ticks_to_us(tick_t val) {
    return ticks_to_ns(val) / 1e3;
  }

  double ticks_to_s(tick_t val) {
    return ticks_to_ns(val) / 1e9;
  }

  // microseconds, 370ns
//  tick_t ticks_now() {
//    timespec temp;
//    clock_gettime(CLOCK_MONOTONIC, &temp);
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
