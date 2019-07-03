#pragma once

#include <bcl/containers/CircularQueue.hpp>
#include <type_traits>
#include <array>
#include <iostream>
#include <functional>
#include <cassert>
#include <list>

#define return_type(Fn, Args) decltype(std::declval<Fn>()(std::declval<Args>()...))
#define return_void(Fn, Args) std::is_void<decltype(std::declval<Fn>()(std::declval<Args>()...))>::value

/*
 * One-way RPC
 * drawback:
 * 1. increase bandwidth(result_value_t->rpc_t)
 */

namespace BCL {

  constexpr size_t max_rpc_size = 8; // argument/return value data size
  constexpr size_t rpc_queue_size = 8192;
  constexpr size_t rpc_buffer_size = 200;

  static_assert(rpc_buffer_size < rpc_queue_size, "rpc_buffer_size should be larger than rpc_queue_size");

  constexpr size_t rpc_kill_nonce = std::numeric_limits<size_t>::max();
  constexpr size_t rpc_flush_nonce = rpc_kill_nonce-1;
  constexpr size_t rpc_result_source_rank = std::numeric_limits<size_t>::max();

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

  struct rpc_t {
      std::size_t rpc_id_;
      bool buffered_;
      std::size_t source_rank_;
      std::uintptr_t fn_;
      std::uintptr_t invoker_;
      std::array<char, max_rpc_size> data_;

      rpc_t() = default;
      rpc_t(std::size_t rpc_id, bool buffered = false, std::size_t source_rank = BCL::rank()) : rpc_id_(rpc_id), buffered_(buffered), source_rank_(source_rank) {}

      template <typename Fn, typename... Args>
      void load(Fn&& fn, Args&&... args) {
        using tuple_t = std::tuple<std::remove_reference_t<Args>...>;
        static_assert(sizeof(tuple_t) <= max_rpc_size, "Max RPC size too small." );

        this->fn_ = get_pi_fnptr_(reinterpret_cast<char*>(+fn));

        invoker_ = get_pi_fnptr_(
                reinterpret_cast<char*>(&decltype(
                complete_rpc<Fn, Args...>())::invoke));

        tuple_t& tuple = *reinterpret_cast<tuple_t*>(data_.data());
        new(&tuple) tuple_t();

        tuple = std::tie(args...);
      }

      template <typename T>
      void load_result(const T& value) {
        static_assert(sizeof(T) <= max_rpc_size, "Max return val size too small");
        *reinterpret_cast<T*>(data_.data()) = value;
      }
  };

  // save the type Fn and Args
  template <typename Fn, typename... Args>
  struct complete_rpc {

      complete_rpc() {}

      template <std::size_t... I>
      static auto invoke_impl_(rpc_t& rpc_requests, std::index_sequence<I...>) {
        using fn_t = decltype(+std::declval<std::remove_reference_t<Fn>>());
        fn_t fn = reinterpret_cast<fn_t>(resolve_pi_fnptr_(rpc_requests.fn_));
        using tuple_t = std::tuple<std::remove_reference_t<Args>...>;
        tuple_t& args_ptr = *reinterpret_cast<tuple_t*>(rpc_requests.data_.data());

        if constexpr(return_void(Fn, Args)) {
          std::invoke(std::forward<fn_t>(fn),
                      std::forward<Args>(std::get<I>(args_ptr))...);
        } else {
          return std::invoke(std::forward<fn_t>(fn),
                             std::forward<Args>(std::get<I>(args_ptr))...);
        }
      }

      static rpc_t invoke(rpc_t& rpc_requests) {
        rpc_t rpc_result(rpc_requests.rpc_id_, rpc_requests.buffered_, rpc_result_source_rank);
        if constexpr(return_void(Fn, Args)) {
          invoke_impl_(rpc_requests, std::index_sequence_for<Args...>{});
        } else {
          rpc_result.load_result(invoke_impl_(rpc_requests, std::index_sequence_for<Args...>{}));
        }
        return rpc_result;
      }
  };

  std::vector<BCL::CircularQueue<rpc_t>> rpc_requests_queue_;
  std::vector<std::vector<rpc_t>> rpc_requests_buffer_; // this needs to be guarded by mutex

  using rpc_result_t = std::array<char, max_rpc_size>;
  std::unordered_map<size_t, rpc_result_t> rpc_results_; // this needs to be guarded by mutex

  std::mutex rpc_mutex_;
  std::thread rpc_thread_;

  size_t rpc_nonce_;

  void service_rpc();

  void init_rpc() {
    rpc_nonce_ = 0;

    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
      rpc_requests_queue_.emplace_back(rank, rpc_queue_size);
    }

    rpc_requests_buffer_.resize(BCL::nprocs());

    rpc_thread_ = std::thread(service_rpc);

    BCL::barrier();
  }

  rpc_t run_rpc(rpc_t& rpc_request) {
    using invoker_type = rpc_t (*)(BCL::rpc_t &);

    invoker_type invoker = reinterpret_cast<invoker_type>
            (resolve_pi_fnptr_(rpc_request.invoker_));
    rpc_t rpc_result = invoker(rpc_request);
    return rpc_result;
  }

  rpc_result_t wait_for_rpc_result(size_t rpc_nonce) {
    Backoff backoff;
    bool success = false;

    do {
      rpc_mutex_.lock();
      success = rpc_results_.find(rpc_nonce) != rpc_results_.end();
      if (!success) {
        rpc_mutex_.unlock();
      }
      backoff.backoff();
    } while (!success);

    rpc_result_t rpc_result = rpc_results_[rpc_nonce];
    rpc_results_.erase(rpc_nonce);
    rpc_mutex_.unlock();

    return rpc_result;
  }

  template <typename T>
  struct rpc_future {
  public:

      rpc_future(size_t rpc_id) : rpc_id_(rpc_id) {}

      T get() const {
        rpc_result_t rpc_result = wait_for_rpc_result(rpc_id_);

        if constexpr(!std::is_void<T>::value) {
          return *reinterpret_cast<T*>(rpc_result.data());
        }
      }

      template <class Rep, class Period>
      std::future_status wait_for(const std::chrono::duration<Rep,Period>& timeout_duration) const {
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
  auto rpc(size_t rank, Fn&& fn, Args&&... args) {
    static_assert(std::is_invocable<Fn, Args...>::value, "Callable passed to async_rpc not valid with given arguments.");
    rpc_t rpc_(rpc_nonce_++, false);

    rpc_.load(std::forward<Fn>(fn), std::forward<Args>(args)...);
    rpc_requests_queue_[rank].push_atomic_impl_(rpc_, true);

    return rpc_future<return_type(Fn, Args)>(rpc_.rpc_id_);
  }

  template <typename Fn, typename... Args>
  auto sync_rpc(size_t rank, Fn&& fn, Args&&... args) {
    static_assert(std::is_invocable<Fn, Args...>::value, "Callable passed to async_rpc not valid with given arguments.");

    auto fu = rpc(rank, fn, args...);

    if constexpr(return_void(Fn, Args)) {
      fu.get();
    } else {
      return fu.get();
    }
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
      rpc_requests_queue_[rank].push_atomic_impl_(send_buf, true);
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

  /*
   * service_rpc is running on another thread
   */
  void service_rpc() {
    using future_type = typename BCL::CircularQueue<rpc_t>::push_future;

    std::list<future_type> futures;

    while (true) {
      bool idle = true;
      rpc_t rpc_;
      if (rpc_requests_queue_[BCL::rank()].pop(rpc_)) {
        idle = false;
        if (rpc_.rpc_id_ == rpc_kill_nonce) {
          return;
        } else if (rpc_.rpc_id_ == rpc_flush_nonce) {
          // flush?
        } else {
          if (rpc_.source_rank_ == rpc_result_source_rank) {
            // rpc_result
            std::lock_guard<std::mutex> guard(rpc_mutex_);
            rpc_results_[rpc_.rpc_id_] = rpc_.data_;
          } else {
            // rpc_request
            rpc_t rpc_result = run_rpc(rpc_);
            if (!rpc_.buffered_) {
              auto fut = rpc_requests_queue_[rpc_.source_rank_].async_push({rpc_result});
              futures.push_back(std::move(fut));
            } else {
              rpc_mutex_.lock();
              rpc_requests_buffer_[rpc_.source_rank_].push_back(rpc_result);

              if (rpc_requests_buffer_[rpc_.source_rank_].size() >= rpc_buffer_size) {
                std::vector<rpc_t> send_buf = std::move(rpc_requests_buffer_[rpc_.source_rank_]);
                rpc_mutex_.unlock();
                auto fut = rpc_requests_queue_[rpc_.source_rank_].async_push(std::move(send_buf));
                futures.push_back(std::move(fut));
              } else {
                rpc_mutex_.unlock();
              }
            }
          }
        }
      }

      for (auto it = futures.begin(); it != futures.end(); ) {
        if ((*it).is_ready()) {
          auto new_val = it;
          new_val++;
          futures.erase(it);
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
    for (size_t i = 0; i < rpc_requests_buffer_.size(); i++) {
      rpc_mutex_.lock();
      std::vector<rpc_t> send_buf = std::move(rpc_requests_buffer_[i]);
      rpc_mutex_.unlock();
      rpc_requests_queue_[i].push_atomic_impl_(send_buf, true);
    }
  }

  // TODO: something more elaborate, a la HashMapBuffer
  void flush_rpc() {
    flush_signal();
  }

} // end BCL
