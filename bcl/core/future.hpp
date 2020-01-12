#pragma once

#include <future>
#include <vector>
#include <memory>

namespace BCL {

template <typename T>
class future {
  std::vector<BCL::request> requests_;

public:
  std::unique_ptr<T> value_;

  future() : value_(new T()) {}

  future(T&& value, const BCL::request& request)
         : value_(new T(std::move(value))), requests_({request}) {}

  future(T&& value, const std::vector<BCL::request>& requests)
         : value_(new T(std::move(value))), requests_(requests) {}

  future(T&& value, std::vector<BCL::request>&& requests)
         : value_(new T(std::move(value))), requests_(std::move(requests)) {}

  future(const T& value, const BCL::request& request)
         : value_(new T(value)), requests_({request}) {}

  void update(const BCL::request& request) {
    requests_.push_back(request);
  }

  void update(BCL::request&& request) {
    requests_.push_back(std::move(request));
  }

  future(future&&) = default;
  future& operator=(future&&) = default;
  future(const future&) = delete;

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

  bool check() {
    return wait_for(std::chrono::seconds(0)) == std::future_status::ready;
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

}
