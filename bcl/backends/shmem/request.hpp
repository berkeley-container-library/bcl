
#pragma once

#include <stdexcept>
#include <mpp/shmem.h>

namespace BCL {

class request {
public:
  request() = default;
  request(const request&) = default;

  void wait() {
    shmem_quiet();
  }

  bool check() const {
    return true;
  }
};

}

#include <bcl/core/future.hpp>
