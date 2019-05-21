#pragma once

#include <bcl/bcl.hpp>
#include <string>
#include <unistd.h>

namespace BCL {

template <typename ...Args>
void print(std::string format, Args... args) {
  fflush(stdout);
  BCL::barrier();
  if (BCL::rank() == 0) {
    printf(format.c_str(), args...);
  }
  fflush(stdout);
  BCL::barrier();
}

std::string hostname() {
  constexpr size_t MH = 2048;
  char buf[MH+1];
  gethostname(buf, MH);
  return std::string(buf, MH);
}

} // end BCL
