#pragma once

#include "bcl/bcl.hpp"
#include <cstring>
#include "rpc_t.hpp"
#include "arh.hpp"
#include "bcl/core/util/Backoff.hpp"

namespace ARH {

  extern gex_EP_t ep;

  template<typename T>
  struct gasnet_pack {
    static constexpr std::size_t gex_arg_size = sizeof(gex_AM_Arg_t);
    static constexpr std::size_t nargs = (sizeof(T) + gex_arg_size - 1) / gex_arg_size;

    gasnet_pack(const std::array<gex_AM_Arg_t, 16> &args) {
      for (size_t i = 0; i < nargs; i++) {
        buf[i] = args[i];
      }
    }

    gasnet_pack(const T &args) {
      std::memcpy(buf, &args, sizeof(T));
    }

    T &as_T() {
      return *reinterpret_cast<T *>(buf);
    }

    gex_AM_Arg_t buf[nargs];
  };

  std::atomic<size_t> rpc_nonce_ = 0;
  std::atomic<size_t> acknowledged = 0;
  std::atomic<size_t> requested = 0;

  size_t acknowledge_handler_id_;
  size_t generic_handler_id_;

  using rpc_result_t = BCL::rpc_t::rpc_result_t;
  std::unordered_map<size_t, rpc_result_t> rpc_results_; // this needs to be guarded by mutex
  std::mutex rpc_mutex_;

  template<std::size_t... I>
  void generic_handler_reply_impl_(gex_Token_t token, gasnet_pack<BCL::rpc_t> &pack, std::index_sequence<I...>) {
    gasnetc_AMReplyShortM(token, acknowledge_handler_id_, 0, sizeof...(I), pack.buf[I]...);
  }

  template<std::size_t... I>
  void generic_handler_request_impl_(size_t remote_proc, gasnet_pack<BCL::rpc_t> &pack, std::index_sequence<I...>) {
    gasnetc_AMRequestShortM(BCL::tm, remote_proc, generic_handler_id_, 0
    GASNETI_THREAD_GET, sizeof...(I), pack.buf[I]...);
  }

  void acknowledge_handler_(gex_Token_t token,
                            gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2,
                            gex_AM_Arg_t arg3, gex_AM_Arg_t arg4, gex_AM_Arg_t arg5,
                            gex_AM_Arg_t arg6, gex_AM_Arg_t arg7, gex_AM_Arg_t arg8,
                            gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
                            gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14,
                            gex_AM_Arg_t arg15) {
    gasnet_pack<BCL::rpc_t> packed_args({arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                                         arg9, arg10, arg11, arg12, arg13, arg14, arg15});
    BCL::rpc_t &rpc_result = packed_args.as_T();

    rpc_mutex_.lock();
    rpc_results_[rpc_result.rpc_id_] = rpc_result.data_;
    rpc_mutex_.unlock();

    acknowledged++;
  }

// could we use a variadic function parameter?
  void generic_handler_(gex_Token_t token,
                        gex_AM_Arg_t arg0, gex_AM_Arg_t arg1, gex_AM_Arg_t arg2,
                        gex_AM_Arg_t arg3, gex_AM_Arg_t arg4, gex_AM_Arg_t arg5,
                        gex_AM_Arg_t arg6, gex_AM_Arg_t arg7, gex_AM_Arg_t arg8,
                        gex_AM_Arg_t arg9, gex_AM_Arg_t arg10, gex_AM_Arg_t arg11,
                        gex_AM_Arg_t arg12, gex_AM_Arg_t arg13, gex_AM_Arg_t arg14,
                        gex_AM_Arg_t arg15) {
    gasnet_pack<BCL::rpc_t> packed_args({arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8,
                                         arg9, arg10, arg11, arg12, arg13, arg14, arg15});
    BCL::rpc_t &my_rpc = packed_args.as_T();

    BCL::rpc_t rpc_result = my_rpc.run();

    gasnet_pack<BCL::rpc_t> packed_result(rpc_result);

    generic_handler_reply_impl_(token, packed_result, std::make_index_sequence<gasnet_pack<BCL::rpc_t>::nargs>());
  }

  void init_am() {
    size_t handler_num = GEX_AM_INDEX_BASE;

    acknowledge_handler_id_ = handler_num++;

    gex_AM_Entry_t entry[2];
    entry[0].gex_index = acknowledge_handler_id_;
    entry[0].gex_fnptr = (gex_AM_Fn_t) acknowledge_handler_;
    entry[0].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
    entry[0].gex_nargs = gasnet_pack<BCL::rpc_t>::nargs;

    generic_handler_id_ = handler_num++;

    entry[1].gex_index = generic_handler_id_;
    entry[1].gex_fnptr = (gex_AM_Fn_t) generic_handler_;
    entry[1].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
    entry[1].gex_nargs = gasnet_pack<BCL::rpc_t>::nargs;

    gex_EP_RegisterHandlers(BCL::ep, entry, 2);
    BCL::barrier();
  }

  void flush_am() {
    while (acknowledged < requested) {
      gasnet_AMPoll();
    }
  }

  void flush_am_nopoll() {
    while (acknowledged < requested) {}
  }

  rpc_result_t wait_for_rpc_result(size_t rpc_nonce) {
    BCL::Backoff backoff;
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

  template<typename T>
  struct Future {

    Future(size_t rpc_id) : rpc_id_(rpc_id) {}

    T wait() const {
      rpc_result_t rpc_result = wait_for_rpc_result(rpc_id_);

      if constexpr(!std::is_void<T>::value) {
        return *reinterpret_cast<T*>(rpc_result.data());
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
    assert(remote_proc < BCL::nprocs());

    rpc_nonce_++;
    BCL::rpc_t my_rpc(rpc_nonce_.load(), false, 0);
    my_rpc.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

    gasnet_pack<BCL::rpc_t> packed_rpc(my_rpc);

    generic_handler_request_impl_(remote_proc, packed_rpc, std::make_index_sequence<gasnet_pack<BCL::rpc_t>::nargs>());
    requested++;

    return Future<std::invoke_result_t<Fn, Args...>>(my_rpc.rpc_id_);
  }
} // end of arh