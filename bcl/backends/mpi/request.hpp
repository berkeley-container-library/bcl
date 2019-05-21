
#pragma once

#include <stdexcept>
#include "backend.hpp"

namespace BCL {

class request {
  MPI_Request request_ = MPI_REQUEST_NULL;
public:
  request() = default;

  request(const request&) = default;
  request& operator=(const request&) = default;

  request(request&&) = default;
  request& operator=(request&&) = default;

  request(const MPI_Request& request) : request_(request) {}

  void wait() {
    if (request_ == MPI_REQUEST_NULL) {
      return;
      // throw std::runtime_error("request: waiting on an expired request");
    }
    MPI_Wait(&request_, MPI_STATUS_IGNORE);
    request_ = MPI_REQUEST_NULL;
  }

  bool check() {
    if (request_ == MPI_REQUEST_NULL) {
      return true;
    }
    int status;
    MPI_Test(const_cast<MPI_Request*>(&request_), &status, MPI_STATUS_IGNORE);
    if (status) {
      request_ = MPI_REQUEST_NULL;
    }
    return status;
  }
};

}

#include <bcl/core/future.hpp>
