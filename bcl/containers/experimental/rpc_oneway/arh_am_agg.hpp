#ifndef BCL_ARH_AM_AGGREGATE_HPP
#define BCL_ARH_AM_AGGREGATE_HPP

#include <vector>
#include "arh_am.hpp"
namespace ARH {
  std::vector<std::mutex> agg_locks;
  std::vector<std::vector<rpc_t>> agg_buffers;
  size_t max_agg_size;
  std::atomic<size_t> agg_size;

  void init_agg() {
    max_agg_size = MIN(
        gex_AM_MaxRequestMedium(BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_t),
        gex_AM_MaxReplyMedium  (BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_result_t)
        );
    agg_size = max_agg_size;

    agg_buffers.resize(nprocs());
    agg_locks = std::vector<std::mutex>(nprocs());
  }

  size_t set_agg_size(size_t custom_agg_size) {
#ifdef ARH_DEBUG
    assert(custom_agg_size > 0);
#endif
    agg_size = MIN(agg_size.load(), custom_agg_size);
    return agg_size.load();
  }

  size_t get_max_agg_size() {
    return max_agg_size;
  }

  size_t get_agg_size() {
    return agg_size.load();
  }

  void flush_agg_buffer() {
    for (size_t i = my_worker_local(); i < nprocs(); i += nworkers_local()) {
      agg_locks[i].lock();
      if (!agg_buffers[i].empty()) {
        std::vector<rpc_t> send_buf = std::move(agg_buffers[i]);
        agg_locks[i].unlock();
        requested += send_buf.size();
        generic_handler_request_impl_(i, std::move(send_buf));
      }
      else {
        agg_locks[i].unlock();
      }
    }
  }

  template <typename Fn, typename... Args>
  Future<std::invoke_result_t<Fn, Args...>>
  rpc_agg(size_t remote_worker, Fn&& fn, Args&&... args) {
    assert(remote_worker < nworkers());

    size_t remote_proc = remote_worker / nworkers_local();
    u_int8_t remote_worker_local = (u_int8_t) remote_worker % nworkers_local();

    Future<std::invoke_result_t<Fn, Args...>> future;
    rpc_t my_rpc(future.get_p(), remote_worker_local);
    my_rpc.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

    agg_locks[remote_proc].lock();
    agg_buffers[remote_proc].push_back(my_rpc);
    if (agg_buffers[remote_proc].size() >= agg_size) {
      std::vector<rpc_t> send_buf = std::move(agg_buffers[remote_proc]);
      agg_locks[remote_proc].unlock();
      requested += send_buf.size();
      generic_handler_request_impl_(remote_proc, std::move(send_buf));
    }
    else {
      agg_locks[remote_proc].unlock();
    }

    return std::move(future);
  }
}

#endif //BCL_ARH_AM_AGGREGATE_HPP
