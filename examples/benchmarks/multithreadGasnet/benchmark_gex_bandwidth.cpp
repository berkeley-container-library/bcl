#include "bcl/containers/experimental/arh/arh.hpp"
#include "bcl/containers/experimental/arh/arh_tools.hpp"
#include <sched.h>
#include <queue>
#include <mutex>
#include "include/cxxopts.hpp"

struct ThreadObjects {
  std::atomic<size_t> issued = 0;
  std::atomic<size_t> received = 0;
};

ARH::GlobalObject<ThreadObjects> threadObjects;
size_t hidx_empty_req;
size_t hidx_reply;
bool single_worker_mode = false;
bool single_proc_mode = false;
bool all2all_mode = false;

void empty_req_handler(gex_Token_t token, gex_AM_Arg_t src_thread) {
  {
    // do nothing
  }
  gex_AM_ReplyShort(token, hidx_reply, 0, src_thread);
}

void reply_handler(gex_Token_t token, gex_AM_Arg_t src_thread) {
  size_t mContext = ARH::get_context();
  ARH::set_context(src_thread);
  threadObjects.get().received++;
  ARH::set_context(mContext);
}

void worker() {
  size_t num_ops = 10000;

  srand48(ARH::my_worker());

  bool run =
      (mode_command == "single_worker" && ARH::my_worker() == 0) ||
      (mode_command == "single_proc"   && ARH::my_proc()   == 0) ||
      mode_command == "all2all";

  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  if (run) {

    for (size_t i = 0; i < num_ops; i++) {
      size_t remote_proc = lrand48() % ARH::nprocs();
      while (remote_proc == ARH::my_proc()) {
        remote_proc = lrand48() % ARH::nprocs();
      }

      threadObjects.get().issued++;
      gex_AM_RequestShort(BCL::tm, remote_proc, hidx_empty_req, 0, ARH::get_context());
    }

  }

  while (threadObjects.get().received < threadObjects.get().issued) {
    gasnet_AMPoll();
  }
  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e3;
  double overhead = duration / num_ops;
  ARH::print("Setting: mode = %s; duration = %.2lf s; num_ops = %lu\n", mode_command.c_str(), duration / 1e6, num_ops);
  ARH::print("overhead: %.2lf us; throughput: %.2lf op/s\n", overhead, 1e6 / overhead);

}

int main(int argc, char** argv) {
  cxxopts::Options options("ARH Benchmark", "Benchmark of ARH system");
  options.add_options()
      ("mode", "Select mode: single_worker(default), single_proc, all2all", cxxopts::value<std::string>())
      ;
  auto result = options.parse(argc, argv);
  std::string mode_command;
  try {
    mode_command = result["mode"].as<std::string>();
  } catch (...) {
    mode_command = "single_worker";
  }
  if (mode_command != "single_worker" && mode_command != "single_proc" && mode_command != "all2all") {
    mode_command = "single_worker";
  }

  ARH::init(15, 16);
  threadObjects.init();

  gex_AM_Entry_t entry[2];
  hidx_empty_req = ARH::handler_num++;
  hidx_reply = ARH::handler_num++;

  entry[0].gex_index = hidx_empty_req;
  entry[0].gex_fnptr = (gex_AM_Fn_t) empty_req_handler;
  entry[0].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REQUEST;
  entry[0].gex_nargs = 1;

  entry[1].gex_index = hidx_reply;
  entry[1].gex_fnptr = (gex_AM_Fn_t) reply_handler;
  entry[1].gex_flags = GEX_FLAG_AM_SHORT | GEX_FLAG_AM_REPLY;
  entry[1].gex_nargs = 1;

  gex_EP_RegisterHandlers(BCL::ep, entry, 2);

  ARH::run(worker);
  ARH::finalize();

  return 0;
}