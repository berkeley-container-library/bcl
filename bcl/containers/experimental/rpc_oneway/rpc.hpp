#pragma once

#include <bcl/containers/CircularQueue.hpp>
#include <type_traits>
#include <array>
#include <iostream>
#include <functional>
#include <cassert>
#include <list>

#include "rpc_t.hpp"
/*
 * One-way RPC
 * drawback:
 * 1. increase bandwidth(result_value_t->rpc_t)
 */

namespace BCL {

  constexpr size_t rpc_queue_size = 8192;
  constexpr size_t rpc_buffer_size = 200;

  static_assert(rpc_buffer_size < rpc_queue_size, "rpc_buffer_size should be larger than rpc_queue_size");



  using rpc_result_t = rpc_t::rpc_result_t;

  std::vector<BCL::CircularQueue<rpc_t>> rpc_requests_queue_;
  std::vector<std::vector<rpc_t>> rpc_requests_buffer_; // this needs to be guarded by mutex
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

    return rpc_future<std::invoke_result_t<Fn, Args...>>(rpc_.rpc_id_);
  }

  template <typename Fn, typename... Args>
  auto sync_rpc(size_t rank, Fn&& fn, Args&&... args) {
    static_assert(std::is_invocable<Fn, Args...>::value, "Callable passed to async_rpc not valid with given arguments.");

    auto fu = rpc(rank, fn, args...);

    return fu.get();
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
    rpc_t rpc_(rpc_t::rpc_kill_nonce);
    rpc_requests_queue_[BCL::rank()].push_atomic_impl_(rpc_, true);

    rpc_thread_.join();

    BCL::barrier();
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

    /*
   * service_rpc is running on another thread
   */
    void service_rpc() {
      using future_type = typename BCL::CircularQueue<rpc_t>::push_future;

      std::list <future_type> futures;

      while (true) {
        bool idle = true;
        rpc_t rpc_;
        if (rpc_requests_queue_[BCL::rank()].pop(rpc_)) {
          idle = false;
          if (rpc_.rpc_id_ == rpc_t::rpc_kill_nonce) {
            return;
          } else if (rpc_.rpc_id_ == rpc_t::rpc_flush_nonce) {
            // flush?
          } else {
            if (rpc_.source_rank_ == rpc_t::rpc_result_source_rank) {
              // rpc_result
              std::lock_guard <std::mutex> guard(rpc_mutex_);
              rpc_results_[rpc_.rpc_id_] = rpc_.data_;
            } else {
              // rpc_request
              rpc_t rpc_result = rpc_.run();
              if (!rpc_.buffered_) {
                auto fut = rpc_requests_queue_[rpc_.source_rank_].async_push({rpc_result});
                futures.push_back(std::move(fut));
              } else {
                rpc_mutex_.lock();
                rpc_requests_buffer_[rpc_.source_rank_].push_back(rpc_result);

                if (rpc_requests_buffer_[rpc_.source_rank_].size() >= rpc_buffer_size) {
                  std::vector <rpc_t> send_buf = std::move(rpc_requests_buffer_[rpc_.source_rank_]);
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

        for (auto it = futures.begin(); it != futures.end();) {
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
} // end BCL
