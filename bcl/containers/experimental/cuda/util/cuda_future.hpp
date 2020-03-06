
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

template <typename T>
class cuda_future {
  std::vector<cuda_request> requests_;

public:
  std::unique_ptr<T> value_;

  cuda_future() : value_(new T()) {}

  cuda_future(T&& value, const cuda_request& request)
         : value_(new T(std::move(value))), requests_({request}) {}

  cuda_future(T&& value, const std::vector<cuda_request>& requests)
         : value_(new T(std::move(value))), requests_(requests) {}

  cuda_future(T&& value, std::vector<cuda_request>&& requests)
         : value_(new T(std::move(value))), requests_(std::move(requests)) {}

  cuda_future(const T& value, const cuda_request& request)
         : value_(new T(value)), requests_({request}) {}

  void update(const cuda_request& request) {
    requests_.push_back(request);
  }

  void update(cuda_request&& request) {
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
