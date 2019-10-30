#ifndef BCL_ARH_AM_AGGREGATE_HPP
#define BCL_ARH_AM_AGGREGATE_HPP

#include <vector>
#include "arh_am.hpp"
#include "arh_agg_buffer.hpp"
#ifdef ARH_PROFILE
#include "arh_tools.hpp"
#endif

namespace ARH {
#ifdef ARH_PROFILE
  ARH::AverageTimer timer_load;
  ARH::AverageTimer timer_buf_npop;
  ARH::AverageTimer timer_buf_pop;
  ARH::AverageTimer timer_gex_req;
#endif
  std::vector<AggBuffer<rpc_t>> agg_buffers;
  size_t max_agg_size;
  std::atomic<size_t> agg_size;

  void init_agg() {
    max_agg_size = MIN(
        gex_AM_MaxRequestMedium(BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_t),
        gex_AM_MaxReplyMedium  (BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_result_t)
        );
    agg_size = max_agg_size;

    agg_buffers = std::vector<AggBuffer<rpc_t>>(nprocs());
    for (size_t i = 0; i < nprocs(); ++i) {
      agg_buffers[i].init(agg_size);
    }
  }

  size_t set_agg_size(size_t custom_agg_size) {
#ifdef ARH_DEBUG
    assert(custom_agg_size > 0);
#endif
    agg_size = MIN(agg_size.load(), custom_agg_size);
    for (size_t i = 0; i < nprocs(); ++i) {
      agg_buffers[i].init(agg_size);
    }
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
      std::vector<rpc_t> send_buf;
      agg_buffers[i].pop_nofull(send_buf);
      if (!send_buf.empty()) {
        requested += send_buf.size();
        generic_handler_request_impl_(i, std::move(send_buf));
      }
    }
  }

  template <typename Fn, typename... Args>
  Future<std::invoke_result_t<Fn, Args...>>
  rpc_agg(size_t remote_worker, Fn&& fn, Args&&... args) {
    assert(remote_worker < nworkers());

    size_t remote_proc = remote_worker / nworkers_local();
    u_int8_t remote_worker_local = (u_int8_t) remote_worker % nworkers_local();

#ifdef ARH_PROFILE
    timer_load.start();
#endif
    Future<std::invoke_result_t<Fn, Args...>> future;
    rpc_t my_rpc(future.get_p(), remote_worker_local);
    my_rpc.load(std::forward<Fn>(fn), std::forward<Args>(args)...);
#ifdef ARH_PROFILE
    timer_load.end_and_update();
    timer_buf_npop.start();
    timer_buf_pop.start();
#endif
    auto status = agg_buffers[remote_proc].push(std::move(my_rpc));
    while (status == AggBuffer<rpc_t>::status_t::FAIL) {
      progress();
      status = agg_buffers[remote_proc].push(std::move(my_rpc));
    }
    if (status == AggBuffer<rpc_t>::status_t::SUCCESS_AND_FULL) {
      std::vector<rpc_t> send_buf;
      agg_buffers[remote_proc].pop_full(send_buf);
      requested += send_buf.size();
#ifdef ARH_PROFILE
      timer_buf_pop.end_and_update();
      timer_gex_req.start();
#endif
      generic_handler_request_impl_(remote_proc, std::move(send_buf));
#ifdef ARH_PROFILE
      timer_gex_req.end_and_update();
#endif
    }
#ifdef ARH_PROFILE
    else {
      timer_buf_npop.end_and_update();
    }
#endif

    return std::move(future);
  }
}

#endif //BCL_ARH_AM_AGGREGATE_HPP
