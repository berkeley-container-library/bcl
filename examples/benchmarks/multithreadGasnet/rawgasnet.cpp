#include <bcl/bcl.hpp>
#include <queue>
#include <mutex>

bool thread_run = true;

size_t issued = 0;
size_t received = 0;

void empty_handler(gex_Token_t token, gex_AM_Arg_t value) {
  {

  }
  gex_AM_ReplyShort(token, GEX_AM_INDEX_BASE+1, 0);
}

void reply_handler(gex_Token_t token) {
  received++;
}

void worker() {
  size_t num_ams = 100000;

  srand48(BCL::rank());
  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % (BCL::nprocs() - 1);
    if (remote_proc >= BCL::rank()) {
      remote_proc++;
    }

    issued++;
    int rv = gex_AM_RequestShort(BCL::tm, remote_proc, GEX_AM_INDEX_BASE, 0, BCL::rank());
    while (received < issued) {
      gasnet_AMPoll();
    }
  }

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double duration_us = 1e6 * duration;
  double latency_us = duration_us / num_ams;
  BCL::print("t = %lf\n", latency_us);
}

int main(int argc, char** argv) {

  BCL::init();

  size_t handler_num = GEX_AM_INDEX_BASE;
  size_t max_args = gex_AM_MaxArgs();

  gex_AM_Entry_t entry[2];
  entry[0].gex_index = handler_num++;
  entry[0].gex_fnptr = (gex_AM_Fn_t) empty_handler;
  entry[0].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
  entry[0].gex_nargs = 1;

  entry[1].gex_index = handler_num++;
  entry[1].gex_fnptr = (gex_AM_Fn_t) reply_handler;
  entry[1].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
  entry[1].gex_nargs = 0;

  int rv = gex_EP_RegisterHandlers(BCL::ep, entry, 2);

  worker();

  BCL::finalize();
  return 0;
}
