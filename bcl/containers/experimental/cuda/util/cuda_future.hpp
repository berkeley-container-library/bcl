// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

namespace BCL {

namespace cuda {

class cuda_request {
public:
  cuda_request() = default;
  cuda_request(const cuda_request&) = default;

  void wait() {
    BCL::cuda::flush();
  }

  bool check() const {
    return true;
  }
};

template <typename Thread>
class cuda_thread_request {
public:
  using thread_type = Thread;

  cuda_thread_request(thread_type&& thread) : thread_(std::move(thread)) {}

  void wait() {
    thread_.join();
    BCL::cuda::flush();
  }

  bool check() const {
    return true;
  }

  thread_type thread_;
};

template <typename T, typename Request = cuda_request>
class cuda_future {
  using request_type = Request;
  std::vector<request_type> requests_;

public:
  std::unique_ptr<T> value_;

  cuda_future() : value_(new T()) {}

  /*
  cuda_future(T&& value, const request_type& request)
         : value_(new T(std::move(value))), requests_({request}) {}

  cuda_future(T&& value, const std::vector<request_type>& requests)
         : value_(new T(std::move(value))), requests_(requests) {}

  cuda_future(T&& value, std::vector<request_type>&& requests)
         : value_(new T(std::move(value))), requests_(std::move(requests)) {}

  cuda_future(const T& value, const request_type& request)
         : value_(new T(value)), requests_({request}) {}
         */

  cuda_future(T&& value, request_type&& request)
         : value_(new T(std::move(value))) {
    requests_.push_back(std::move(request));
  }

         /*
  void update(const request_type& request) {
    requests_.push_back(request);
  }
  */

  void update(request_type&& request) {
    requests_.push_back(std::move(request));
  }

  cuda_future(cuda_future&&) = default;
  cuda_future& operator=(cuda_future&&) = default;
  cuda_future(const cuda_future&) = delete;

  T get() {
    for (auto& request : requests_) {
      request.wait();
    }
    return std::move(*value_);
  }

  void wait() {
    for (auto& request : requests_) {
      request.wait();
    }
  }

  template <class Rep, class Period>
  std::future_status wait_for(const std::chrono::duration<Rep,Period>& timeout_duration) {
    for (auto& request : requests_) {
      if (!request.check()) {
        return std::future_status::timeout;
      }
    }
    return std::future_status::ready;
  }
};

} // end cuda

} // end BCL
