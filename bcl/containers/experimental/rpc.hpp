#pragma once

#include <bcl/containers/CircularQueue.hpp>
#include <type_traits>
#include <array>
#include <iostream>
#include <functional>
#include <cassert>
#include <list>

namespace BCL {

constexpr size_t max_rpc_size = 8;
constexpr size_t max_rpc_return_val_size = 8;
constexpr size_t rpc_queue_size = 8192;

constexpr size_t rpc_buffer_size = 200;
constexpr size_t return_val_buffer_size = 2048;

static_assert(rpc_buffer_size < rpc_queue_size && return_val_buffer_size < rpc_queue_size);

constexpr size_t rpc_kill_nonce = std::numeric_limits<size_t>::max();
constexpr size_t rpc_flush_nonce = rpc_kill_nonce-1;

void flush_rpc();

// Get a *position independent* function pointer
template <typename T>
std::uintptr_t get_pi_fnptr_(T* fn) {
  return reinterpret_cast<std::uintptr_t>(fn) - reinterpret_cast<std::uintptr_t>(init);
}

// Resolve a *position independent* function pointer
// to a local function pointer.
template <typename T = char>
T* resolve_pi_fnptr_(std::uintptr_t fn) {
  return reinterpret_cast<T*>(fn + reinterpret_cast<std::uintptr_t>(init));
}

template <typename Fn, typename... Args>
struct complete_rpc;

template <typename Fn, typename... Args>
complete_rpc(Fn&& fn, Args&&... args) -> complete_rpc<Fn, Args...>;

struct rpc_t {
  std::size_t rpc_id_;
  bool buffered_;
  std::size_t source_rank_;
  std::uintptr_t fn_;
  std::uintptr_t invoker_;
  std::array<char, max_rpc_size> data_;

  rpc_t() = default;
  rpc_t(const rpc_t&) = default;
  rpc_t& operator=(const rpc_t&) = default;

  rpc_t(rpc_t&&) = default;
  rpc_t& operator=(rpc_t&&) = default;

  rpc_t(std::size_t rpc_id, bool buffered = false, std::size_t source_rank = BCL::rank()) : rpc_id_(rpc_id), buffered_(buffered), source_rank_(source_rank) {}

  template <typename Fn, typename... Args>
  void load(Fn&& fn, Args&&... args) {
    using tuple_t = std::tuple<std::remove_reference_t<Args>...>;
    static_assert(sizeof(tuple_t) <= max_rpc_size, "Max RPC size too small." );

    this->fn_ = get_pi_fnptr_(reinterpret_cast<char*>(+fn));

    invoker_ = get_pi_fnptr_(
             reinterpret_cast<char*>(&decltype(
                  complete_rpc(std::forward<Fn>(fn), std::forward<Args>(args)...))::invoke));

    tuple_t& tuple = *reinterpret_cast<tuple_t*>(data_.data());
    new(&tuple) tuple_t();

    tuple = std::tie(args...);
  }
};

struct return_value_t {
  std::size_t rpc_id_;
  std::array<char, max_rpc_return_val_size> data_;

  return_value_t() = default;
  return_value_t(const return_value_t&) = default;
  return_value_t& operator=(const return_value_t&) = default;

  return_value_t(return_value_t&&) = default;
  return_value_t& operator=(return_value_t&&) = default;

  return_value_t(std::size_t rpc_id) : rpc_id_(rpc_id) {}

  template <typename T>
  void load(const T& value) {
    static_assert(sizeof(T) <= max_rpc_return_val_size, "Max return val size too small");
    *reinterpret_cast<T*>(data_.data()) = value;
  }
};


template <typename Fn, typename... Args>
struct complete_rpc {

  using return_value = decltype(std::declval<Fn>()(std::declval<Args>()...));

  static constexpr bool has_void_return_value = std::is_void<return_value>::value;

  complete_rpc(Fn&& fn, Args&&... args) {}

  template <std::size_t... I>
  static auto invoke_impl_(rpc_t& rpc_vals, std::index_sequence<I...>) {
    using fn_t = decltype(+std::declval<std::remove_reference_t<Fn>>());
    fn_t fn = reinterpret_cast<fn_t>(resolve_pi_fnptr_(rpc_vals.fn_));
    using tuple_t = std::tuple<std::remove_reference_t<Args>...>;
    tuple_t& args_ptr = *reinterpret_cast<tuple_t*>(rpc_vals.data_.data());

    if constexpr(has_void_return_value) {
      std::invoke(std::forward<fn_t>(fn),
                  std::forward<Args>(std::get<I>(args_ptr))...);
    } else {
      return std::invoke(std::forward<fn_t>(fn),
                         std::forward<Args>(std::get<I>(args_ptr))...);
    }
  }

  static return_value_t invoke(rpc_t& rpc_vals) {
    return_value_t rv;
    if constexpr(has_void_return_value) {
      invoke_impl_(rpc_vals, std::index_sequence_for<Args...>{});
    } else {
      rv.load(invoke_impl_(rpc_vals, std::index_sequence_for<Args...>{}));
    }
    return rv;
  }
};

// TODO: Add buffering.
//       Buffering will require careful thought about progress guarantees.

std::vector<BCL::CircularQueue<rpc_t>> rpc_requests_queue_;
std::vector<std::vector<rpc_t>> rpc_requests_buffer_;

std::vector<BCL::CircularQueue<return_value_t>> rpc_results_queue_;
std::vector<std::vector<return_value_t>> rpc_results_buffer_;

std::unordered_map<size_t, return_value_t> rpc_results_;
std::mutex rpc_mutex_;

std::thread rpc_thread_;

size_t rpc_nonce_;

void service_rpc();

void init_rpc() {
  rpc_nonce_ = 0;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    rpc_requests_queue_.emplace_back(rank, rpc_queue_size);
    rpc_results_queue_.emplace_back(rank, rpc_queue_size);
  }

  rpc_requests_buffer_.resize(BCL::nprocs());
  rpc_results_buffer_.resize(BCL::nprocs());

  rpc_thread_ = std::thread(service_rpc);

  BCL::barrier();
}

using invoker_type = return_value_t (*)(BCL::rpc_t &);

return_value_t run_rpc(rpc_t shtuff);

void send_rpc() {
  auto fn = [](int a, int b) -> void {
               std::cout << "Rank " << BCL::rank() << " sees " << a + b << std::endl;
            };
  int a = 12;
  int b = 12;

  rpc_t shtuff(0);

  shtuff.load(fn, a, b);

  return_value_t rv = run_rpc(shtuff);
}

return_value_t run_rpc(rpc_t shtuff) {
  invoker_type invoker = reinterpret_cast<invoker_type>
                                           (resolve_pi_fnptr_(shtuff.invoker_));
  return_value_t rv = invoker(shtuff);
  rv.rpc_id_ = shtuff.rpc_id_;
  return rv;
}


template <typename Fn, typename... Args>
auto rpc(size_t rank, Fn&& fn, Args&&... args) {
  rpc_t rpc_(rpc_nonce_);

  rpc_.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

  Backoff backoff;
  while (!rpc_requests_queue_[rank].push_atomic_impl_(rpc_, true)) {
    backoff.backoff();
  }

  backoff.reset();

  bool success = false;

  do {
    rpc_mutex_.lock();
    success = rpc_results_.find(rpc_nonce_) != rpc_results_.end();
    if (!success) {
      rpc_mutex_.unlock();
    }
    backoff.backoff();
  } while (!success);

  return_value_t rv_serialized_ = rpc_results_[rpc_nonce_];
  rpc_results_.erase(rpc_nonce_);
  rpc_mutex_.unlock();
  using return_value = decltype(std::declval<Fn>()(std::declval<Args>()...));

  rpc_nonce_++;

  if constexpr(!std::is_void<return_value>::value) {
    return *reinterpret_cast<return_value*>(rv_serialized_.data_.data());
  }
}

template <typename T>
struct rpc_future {
public:

  rpc_future(size_t rpc_id) : rpc_id_(rpc_id) {}

  T get() {
    Backoff backoff;

    bool success = false;
    do {
      rpc_mutex_.lock();
      success = rpc_results_.find(rpc_id_) != rpc_results_.end();
      if (!success) {
        rpc_mutex_.unlock();
      }
      backoff.backoff();
    } while (!success);

    return_value_t rv_serialized_ = rpc_results_[rpc_id_];
    rpc_results_.erase(rpc_id_);
    rpc_mutex_.unlock();

    if constexpr(!std::is_void<T>::value) {
      return *reinterpret_cast<T*>(rv_serialized_.data_.data());
    }
  }

  template <class Rep, class Period>
  std::future_status wait_for(const std::chrono::duration<Rep,Period>& timeout_duration) {
    std::lock_guard<std::mutex> guard(rpc_mutex_);
    if (rpc_results_.find(rpc_id_) != rpc_results_.end()) {
      return std::future_status::ready;
    } else {
      return std::future_status::timeout;
    }
  }

private:
  size_t rpc_id_;
};

template <typename Fn, typename... Args>
auto async_rpc(size_t rank, Fn&& fn, Args&&... args) {
  static_assert(std::is_invocable<Fn, Args...>::value, "Callable passed to async_rpc not valid with given arguments.");
  rpc_t rpc_(rpc_nonce_);

  rpc_.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

  // bool success = rpc_requests_queue_[rank].push(rpc_);
  // assert(success);

  Backoff backoff;
  while (!rpc_requests_queue_[rank].push_atomic_impl_(rpc_, true)) {
    backoff.backoff();
  }

  using return_value = decltype(std::declval<Fn>()(std::declval<Args>()...));

  return rpc_future<return_value>(rpc_nonce_++);
}

template <typename Fn, typename... Args>
auto buffered_rpc(size_t rank, Fn&& fn, Args&&... args) {
  static_assert(std::is_invocable<Fn, Args...>::value, "Callable passed to buffered_rpc not valid with given arguments.");
  rpc_t rpc_(rpc_nonce_, true);

  rpc_.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

  rpc_mutex_.lock();
  rpc_requests_buffer_[rank].push_back(rpc_);

  if (rpc_requests_buffer_[rank].size() >= rpc_buffer_size) {
    std::vector<rpc_t> send_buf = std::move(rpc_requests_buffer_[rank]);
    rpc_mutex_.unlock();
    Backoff backoff;
    while (!rpc_requests_queue_[rank].push_atomic_impl_(send_buf, true)) {
      backoff.backoff();
    }
  } else {
    rpc_mutex_.unlock();
  }

  using return_value = decltype(std::declval<Fn>()(std::declval<Args>()...));

  return rpc_future<return_value>(rpc_nonce_++);
}

void finalize_rpc() {
  BCL::barrier();
  rpc_t rpc_(rpc_kill_nonce);

  rpc_requests_queue_[BCL::rank()].push_atomic_impl_(rpc_, true);

  rpc_thread_.join();

  BCL::barrier();
}

void service_rpc() {
  using results_future_type = typename BCL::CircularQueue<return_value_t>::push_future;
  using requests_future_type = typename BCL::CircularQueue<rpc_t>::push_future;

  std::list<results_future_type> results_futures;
  std::list<requests_future_type> requests_futures;

  while (true) {
    bool idle = true;
    rpc_t rpc_;
    return_value_t rv_;
    if (rpc_requests_queue_[BCL::rank()].pop(rpc_)) {
      idle = false;
      if (rpc_.rpc_id_ == rpc_kill_nonce) {
        return;
      } else if (rpc_.rpc_id_ == rpc_flush_nonce) {
        // flush?
      } else {
        return_value_t rv = run_rpc(rpc_);
        if (!rpc_.buffered_) {
          auto fut = rpc_results_queue_[rpc_.source_rank_].async_push({rv});
          results_futures.push_back(std::move(fut));
        } else {
          rpc_mutex_.lock();
          rpc_results_buffer_[rpc_.source_rank_].push_back(rv);

          if (rpc_results_buffer_[rpc_.source_rank_].size() >= return_val_buffer_size) {
            std::vector<return_value_t> send_buf = std::move(rpc_results_buffer_[rpc_.source_rank_]);
            rpc_mutex_.unlock();
            auto fut = rpc_results_queue_[rpc_.source_rank_].async_push(std::move(send_buf));
          } else {
            rpc_mutex_.unlock();
          }
        }
      }
    }

    if (rpc_results_queue_[BCL::rank()].pop(rv_)) {
      std::lock_guard<std::mutex> guard(rpc_mutex_);
      rpc_results_[rv_.rpc_id_] = rv_;
      idle = false;
    }

    for (auto it = results_futures.begin(); it != results_futures.end(); ) {
      if ((*it).is_ready()) {
        auto new_val = it;
        new_val++;
        results_futures.erase(it);
        it = new_val;
      } else {
        it++;
      }
    }

    for (auto it = requests_futures.begin(); it != requests_futures.end(); ) {
      if ((*it).is_ready()) {
        auto new_val = it;
        new_val++;
        requests_futures.erase(it);
        it = new_val;
      } else {
        it++;
      }
    }

    if (idle) {
      usleep(100);
      // std::this_thread::yield();
    }
  }
}

void flush_signal() {
  for (size_t i = 0; i < rpc_results_buffer_.size(); i++) {
    rpc_mutex_.lock();
    Backoff backoff;
    std::vector<return_value_t> send_buf = std::move(rpc_results_buffer_[i]);
    rpc_mutex_.unlock();
    while (!rpc_results_queue_[i].push_atomic_impl_(send_buf, true)) {
      backoff.backoff();
    }
  }

  for (size_t i = 0; i < rpc_requests_buffer_.size(); i++) {
    Backoff backoff;
    rpc_mutex_.lock();
    std::vector<rpc_t> send_buf = std::move(rpc_requests_buffer_[i]);
    rpc_mutex_.unlock();
    while (!rpc_requests_queue_[i].push_atomic_impl_(send_buf, true)) {
      backoff.backoff();
    }
  }
}

// TODO: something more elaborate, a la HashMapBuffer
void flush_rpc() {
  flush_signal();
}

} // end BCL
