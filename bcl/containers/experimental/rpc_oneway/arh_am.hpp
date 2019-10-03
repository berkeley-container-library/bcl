#ifndef ARH_AM_HPP
#define ARH_AM_HPP

#include "arh.hpp"
#include "arh_rpc_t.hpp"
#include "bcl/core/util/Backoff.hpp"

namespace ARH {

  extern size_t nprocs(void);
  extern void barrier(void);
  extern void process(void);

  std::atomic<size_t> rpc_nonce_ = 0;
  std::atomic<size_t> acknowledged = 0;
  std::atomic<size_t> requested = 0;

  gex_AM_Index_t hidx_generic_rpc_ackhandler_;
  gex_AM_Index_t hidx_generic_rpc_reqhandler_;

  using rpc_t = ARH::rpc_t;
  using rpc_result_t = ARH::rpc_t::rpc_result_t;
  std::unordered_map<size_t, rpc_result_t::payload_t> rpc_results_; // this needs to be guarded by mutex
  std::mutex rpc_mutex_;


  template<typename T>
  struct BufferPack {
    BufferPack(void *buf, size_t nbytes) {
      assert(nbytes % sizeof(T) == 0);
      len = nbytes / sizeof(T);
      pointer = reinterpret_cast<T *>(buf);
    }

    T &operator[](size_t index) {
      assert(index >= 0 && index < len);
      return *(pointer + index);
    }

    [[nodiscard]] size_t size() const {
      return len;
    }

  private:
    T* pointer;
    size_t len;
  };

  void generic_handler_reply_impl_(gex_Token_t token, std::vector<rpc_result_t> &&results) {
    gex_AM_ReplyMedium0(token, hidx_generic_rpc_ackhandler_, results.data(),
        results.size() * sizeof(rpc_result_t), GEX_EVENT_NOW, 0);
  }

  void generic_handler_request_impl_(size_t remote_proc, std::vector<rpc_t> &&rpcs) {
    gex_AM_RequestMedium0(BCL::tm, remote_proc, hidx_generic_rpc_reqhandler_, rpcs.data(),
    rpcs.size() * sizeof(rpc_t), GEX_EVENT_NOW, 0);
  }

  void generic_rpc_ackhandler_(gex_Token_t token, void *buf, size_t nbytes) {
    BufferPack<rpc_result_t> results(buf, nbytes);

    rpc_mutex_.lock();
    for (size_t i = 0; i < results.size(); ++i) {
      rpc_results_[results[i].rpc_id_] = results[i].data_;
    }
    rpc_mutex_.unlock();

    acknowledged += results.size();
  }

  void generic_rpc_reqhandler_(gex_Token_t token, void *buf, size_t nbytes) {
    BufferPack<rpc_t> rpcs(buf, nbytes);
    std::vector<rpc_result_t> results;

    for (size_t i = 0; i < rpcs.size(); ++i) {
      rpc_result_t result = rpcs[i].run();
      results.push_back(result);
    }

    generic_handler_reply_impl_(token, std::move(results));
  }

  void init_am() {
    size_t handler_num = GEX_AM_INDEX_BASE;

    hidx_generic_rpc_ackhandler_ = handler_num++;
    hidx_generic_rpc_reqhandler_ = handler_num;

    gex_AM_Entry_t htable[2] = {
        { hidx_generic_rpc_ackhandler_, (gex_AM_Fn_t) generic_rpc_ackhandler_,
          GEX_FLAG_AM_MEDIUM | GEX_FLAG_AM_REPLY,   0 },
        { hidx_generic_rpc_reqhandler_, (gex_AM_Fn_t) generic_rpc_reqhandler_,
          GEX_FLAG_AM_MEDIUM | GEX_FLAG_AM_REQUEST, 0 },
    };

    gex_EP_RegisterHandlers(BCL::ep, htable, sizeof(htable)/sizeof(gex_AM_Entry_t));
    barrier();
  }

  void flush_am() {
    while (acknowledged < requested) {
      gasnet_AMPoll();
    }
  }

  void flush_am_nopoll() {
    while (acknowledged < requested) {}
  }

  rpc_result_t::payload_t wait_for_rpc_result(size_t rpc_nonce) {
    bool success = false;

    do {
      rpc_mutex_.lock();
      success = rpc_results_.find(rpc_nonce) != rpc_results_.end();
      if (!success) {
        rpc_mutex_.unlock();
      }
      process();
    } while (!success);

    rpc_result_t::payload_t rpc_result = rpc_results_[rpc_nonce];
    rpc_results_.erase(rpc_nonce);
    rpc_mutex_.unlock();

    return rpc_result;
  }

  template<typename T>
  struct Future {

    Future(size_t rpc_id) : rpc_id_(rpc_id) {}

    T wait() const {
      rpc_result_t::payload_t rpc_result = wait_for_rpc_result(rpc_id_);

      if constexpr(!std::is_void<T>::value) {
        static_assert(sizeof(T) <= sizeof(rpc_result_t::payload_t));
        return *reinterpret_cast<T*>(&rpc_result);
      }
    }

    [[nodiscard]] std::future_status check() const {
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
  auto rpc(size_t remote_proc, Fn&& fn, Args&&... args) {
    assert(remote_proc < nprocs());

    rpc_nonce_++;
    rpc_t my_rpc(rpc_nonce_.load());
    my_rpc.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

    generic_handler_request_impl_(remote_proc, std::vector<rpc_t>(1, my_rpc));
    requested++;

    return Future<std::invoke_result_t<Fn, Args...>>(my_rpc.rpc_id_);
  }
} // end of arh

#endif