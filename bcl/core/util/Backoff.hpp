#pragma once

namespace BCL {

size_t double_backoff(size_t sleep) {
  return sleep *= 2;
}

template <typename BackoffFn>
class Backoff {
public:
  Backoff(size_t init_sleep = 1, size_t max_sleep = 1,
          BackoffFn&& backoff_fn = double_backoff)
          : sleep_time_(init_sleep), max_sleep_(max_sleep),
            init_sleep_(init_sleep), backoff_fn_(backoff_fn) {}

  void backoff() {
    usleep(sleep_time_);
    increase_backoff_impl_();
  }

  void increase_backoff_impl_() {
    sleep_time_ = backoff_fn_(sleep_time_);
    sleep_time_ = std::min(sleep_time_, max_sleep_);
  }

  void reset() {
    sleep_time_ = init_sleep_;
  }

private:
  size_t sleep_time_;
  size_t max_sleep_;
  size_t init_sleep_;
  BackoffFn&& backoff_fn_;
};

template <typename BackoffFn = decltype(double_backoff)>
Backoff(size_t init_sleep = 1, size_t max_sleep = 100, BackoffFn&& backoff_fn = double_backoff) -> Backoff<BackoffFn>;

} // end BCL
