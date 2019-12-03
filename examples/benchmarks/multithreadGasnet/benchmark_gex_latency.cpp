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

ARH::WorkerObject<ThreadObjects> threadObjects;
size_t hidx_empty_req;
size_t hidx_reply;
std::string mode_command;

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
  ARH::AverageTimer timer;

  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  if (run) {

    for (size_t i = 0; i < num_ops; i++) {
      size_t remote_proc = lrand48() % ARH::nprocs();
      while (remote_proc == ARH::my_proc()) {
        remote_proc = lrand48() % ARH::nprocs();
      }

      threadObjects.get().issued++;
      timer.start();
      gex_AM_RequestShort(BCL::tm, remote_proc, hidx_empty_req, 0, ARH::get_context());
      timer.end_and_update();

      while (threadObjects.get().received < threadObjects.get().issued) {
        gasnet_AMPoll();
      }
    }

  }

  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e3;
  double latency = duration / num_ops;
  ARH::print("Setting: mode = %s; duration = %.2lf s; num_ops = %lu\n", mode_command.c_str(), duration / 1e6, num_ops);
  ARH::print("latency: %.2lf us\n", latency);
  timer.print_us();
}

int main(int argc, char** argv) {
  cxxopts::Options options("ARH Benchmark", "Benchmark of ARH system");
  options.add_options()
      ("mode", "Select mode: single_worker(default), single_proc, all2all", cxxopts::value<std::string>())
      ("w,worker", "worker number", cxxopts::value<size_t>())
      ("t,thread", "thread number", cxxopts::value<size_t>())
      ;
  auto result = options.parse(argc, argv);

  try {
    mode_command = result["mode"].as<std::string>();
  } catch (...) {
    mode_command = "single_worker";
  }
  if (mode_command != "single_worker" && mode_command != "single_proc" && mode_command != "all2all") {
    mode_command = "single_worker";
  }

  size_t thread_num;
  try {
    thread_num = result["t,thread"].as<size_t>();
  } catch (...) {
    thread_num = 16;
  }

  size_t worker_num;
  try {
    worker_num = result["w,worker"].as<size_t>();
  } catch (...) {
    worker_num = 15;
  }
  worker_num = MIN(worker_num, thread_num);

  ARH::init(worker_num, thread_num);
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