#ifndef BCL_ARH_AM_AGGREGATE_HPP
#define BCL_ARH_AM_AGGREGATE_HPP

#include <vector>
#include "arh.hpp"
namespace ARH {
  std::vector<rpc_t> agg_buffer;
  size_t max_agg_size;

  void init_agg() {
    max_agg_size = MIN(
        gex_AM_MaxRequestMedium(BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_t),
        gex_AM_MaxReplyMedium  (BCL::tm,GEX_RANK_INVALID,GEX_EVENT_NOW,0,0) / sizeof(rpc_result_t)
        );

  }
}

#endif //BCL_ARH_AM_AGGREGATE_HPP
