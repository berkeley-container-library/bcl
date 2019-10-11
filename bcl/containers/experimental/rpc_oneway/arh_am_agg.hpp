#ifndef BCL_ARH_AM_AGGREGATE_HPP
#define BCL_ARH_AM_AGGREGATE_HPP

#include <vector>
#include "arh_am.hpp"
namespace ARH {
  std::mutex agg_lock;
  std::vector<std::vector<rpc_t>> agg_buffers;
  size_t max_agg_size;
  std::atomic<size_t> agg_size;

  void init_agg() {
    max_agg_size = MIN(
        gex_AM_MaxRequestMedium(BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_t),
        gex_AM_MaxReplyMedium  (BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_result_t)
        );

    agg_buffers.resize(nprocs());
  }

  size_t set_agg_size(size_t custom_agg_size) {
    agg_size = MIN(max_agg_size, custom_agg_size);
    return agg_size.load();
  }

  size_t get_max_agg_size() {
    return max_agg_size;
  }

  size_t get_agg_size() {
    return agg_size.load();
  }

  void flush_agg_buffer() {
    for (size_t i = my_local_worker(); i < nprocs(); i += num_workers_per_node) {
      agg_lock.lock();
      if (!agg_buffers[i].empty()) {
        std::vector<rpc_t> send_buf = std::move(agg_buffers[i]);
        agg_lock.unlock();
        requested += send_buf.size();
        generic_handler_request_impl_(i, std::move(send_buf));
      }
      else {
        agg_lock.unlock();
      }
    }
  }

  template <typename Fn, typename... Args>
  Future<std::invoke_result_t<Fn, Args...>>
  rpc_agg(size_t remote_proc, Fn&& fn, Args&&... args) {
    assert(remote_proc < nprocs());

    Future<std::invoke_result_t<Fn, Args...>> future;
    rpc_t my_rpc(future.get_p());
    my_rpc.load(std::forward<Fn>(fn), std::forward<Args>(args)...);

    agg_lock.lock();
    agg_buffers[remote_proc].push_back(my_rpc);
    if (agg_buffers[remote_proc].size() >= agg_size) {
      std::vector<rpc_t> send_buf = std::move(agg_buffers[remote_proc]);
      agg_lock.unlock();
      requested += send_buf.size();
      generic_handler_request_impl_(remote_proc, std::move(send_buf));
    }
    else {
      agg_lock.unlock();
    }

    return std::move(future);
  }
}

#endif //BCL_ARH_AM_AGGREGATE_HPP
