#include <unordered_map>
#include <cstring>

#include "am_utils.hpp"

std::unordered_map<int, int> map;
size_t responses = 0;

void handler(gex_Token_t token, gex_AM_Arg_t rank, gex_AM_Arg_t flag) {
  map[rank] += flag;
  gex_AM_ReplyShort(token, GEX_AM_INDEX_BASE+1, 0);
}

void reply(gex_Token_t token) {
  responses++;
}

int main(int argc, char** argv) {
  BCL::init();

  size_t handler_num = GEX_AM_INDEX_BASE;
  size_t max_args = gex_AM_MaxArgs();

  gex_AM_Entry_t entry[2];
  entry[0].gex_index = handler_num++;
  entry[0].gex_fnptr = (gex_AM_Fn_t) handler;
  entry[0].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
  entry[0].gex_nargs = 2;

  entry[1].gex_index = handler_num++;
  entry[1].gex_fnptr = (gex_AM_Fn_t) reply;
  entry[1].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
  entry[1].gex_nargs = 0;

  int rv = gex_EP_RegisterHandlers(BCL::ep, entry, 2);
  size_t num_ams = 1000000;
  srand48(BCL::rank());
  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();
    int rv = gex_AM_RequestShort(BCL::tm, remote_proc, GEX_AM_INDEX_BASE, 0, 1, 1);
    while (responses < i+1) {
      gasnet_AMPoll();
    }
  }

  /*
  while (responses < num_ams) {
    gasnet_AMPoll();
  }
  */

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();
  double duration_us = 1e6*duration;

  double latency_us = duration_us / num_ams;

  BCL::print("Latency is %lf us per AM. (Finished in %lf s)\n", latency_us, duration);

  BCL::finalize();
  return 0;
}
