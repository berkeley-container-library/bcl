#pragma once

#include <upcxx/upcxx.hpp>

namespace BUtil {
  template <typename ...Args>
  void print(std::string format, Args... args) {
    fflush(stdout);
    upcxx::barrier();
    if (upcxx::rank_me() == 0) {
      printf(format.c_str(), args...);
    }
    fflush(stdout);
    upcxx::barrier();
  }
}
