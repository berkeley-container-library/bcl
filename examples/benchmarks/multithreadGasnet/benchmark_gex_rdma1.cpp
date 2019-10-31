#include "bcl/containers/experimental/arh/arh.hpp"
#include "bcl/containers/experimental/arh/arh_gex.hpp"
#include "bcl/containers/experimental/arh/arh_tools.hpp"
#include <sched.h>
#include <queue>
#include <mutex>

std::atomic<size_t> issued = 0;
std::atomic<size_t> received = 0;
size_t hidx_rput_req;
size_t hidx_rput_rep;
size_t hidx_rget_req;
size_t hidx_rget_rep;

size_t shared_seg_size = 4 * 1e6;
void* shared_seg_ptr = NULL;

void init_shared_memory(size_t custom_seg_size = 4 * 1e6) {
  shared_seg_size = custom_seg_size;
  shared_seg_ptr = malloc(shared_seg_size);
}

void rput_req_handler(gex_Token_t token, gex_AM_Arg_t offset, gex_AM_Arg_t val32) {
  {
    if (offset >= shared_seg_size) {
      // do nothing
    }
    void* dest_ptr = shared_seg_ptr + offset;
    *reinterpret_cast<gex_AM_Arg_t*>(dest_ptr) = val32;
  }
  gex_AM_ReplyShort(token, hidx_rput_rep, 0);
}

void rput_rep_handler(gex_Token_t token) {
  received++;
}

void rget_req_handler(gex_Token_t token, gex_AM_Arg_t offset) {
  {
    // do nothing
  }
  gex_AM_ReplyShort(token, hidx_reply, 0);
}

void rget_rep_handler(gex_Token_t token, gex_AM_Arg_t val32) {
  received++;
}

void worker() {
  size_t num_ops = 1000;

  srand48(ARH::my_worker());
  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_proc = lrand48() % ARH::nprocs();
    while (remote_proc == ARH::my_proc()) {
      remote_proc = lrand48() % ARH::nprocs();
    }

    issued++;
    gex_AM_RequestShort(BCL::tm, remote_proc, hidx_empty_req, 0);

    while (received < issued) {
      gasnet_AMPoll();
    }
  }

  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e3;
  double latency = duration / num_ops;
  ARH::print("latency: %.2lf us\n", latency);

}

int main(int argc, char** argv) {

  ARH::init(15, 16);

  gex_AM_Entry_t entry[2];
  hidx_empty_req = ARH::handler_num++;
  hidx_reply = ARH::handler_num++;

  entry[0].gex_index = hidx_empty_req;
  entry[0].gex_fnptr = (gex_AM_Fn_t) empty_req_handler;
  entry[0].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
  entry[0].gex_nargs = 0;

  entry[1].gex_index = hidx_reply;
  entry[1].gex_fnptr = (gex_AM_Fn_t) reply_handler;
  entry[1].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
  entry[1].gex_nargs = 0;

  gex_EP_RegisterHandlers(BCL::ep, entry, 2);

  ARH::run(worker);
  ARH::finalize();

  return 0;
}