
#pragma once

#include <iostream>

#include <chrono>
#include <stdexcept>
#include "backend.hpp"

namespace BCL {

class request {
  gex_Event_t request_ = GEX_EVENT_INVALID;
public:
  request() = default;
  request(const request&) = default;

  request(const gex_Event_t& request) : request_(request) {}

  void wait() {
    if (request_ != GEX_EVENT_INVALID) {
      gex_Event_Wait(request_);
      request_ = GEX_EVENT_INVALID;
    }
  }

  bool check() {
    if (request_ == GEX_EVENT_INVALID) {
      return true;
    } else {
      int success = !gex_Event_Test(request_);

      if (success) {
        request_ = GEX_EVENT_INVALID;
      } else {
        gasnet_AMPoll();

        success = !gex_Event_Test(request_);

        if (success) {
          request_ = GEX_EVENT_INVALID;
        }
      }

      return success;
    }
  }
};

}

#include <bcl/core/future.hpp>
