#ifdef GASNET_EX
//#define ARH_PROFILE
#include "bcl/containers/experimental/arh/arh.hpp"
#include "bcl/containers/experimental/arh/arh_tools.hpp"
#include <cassert>
#include "include/cxxopts.hpp"

bool use_agg = false;
size_t agg_size = 0;

void empty_handler() {
  // do nothing
}

void worker() {

  int num_ops = 100000;
  int total_num_ops = num_ops * (int) ARH::nworkers();
  double ticks_step = 0;
#ifdef ARH_PROFILE
  ARH::AverageTimer timer_rand;
  ARH::AverageTimer timer_rpc;
  ARH::AverageTimer timer_push;
  ARH::AverageTimer timer_barrier;
  ARH::AverageTimer timer_get;
#endif
  size_t my_rank = ARH::my_worker();
  size_t nworkers = ARH::nworkers();

  using rv = decltype(ARH::rpc(size_t(), empty_handler));
  std::vector<rv> futures;

  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  for (int i = 0; i < num_ops; i++) {
#ifdef ARH_PROFILE
    timer_rand.start();
#endif
    size_t target_rank = lrand48() % nworkers;
#ifdef ARH_PROFILE
    timer_rand.end_and_update();
    timer_rpc.start();
#endif
    rv f;
    if (use_agg) {
      f = ARH::rpc_agg(target_rank, empty_handler);
    } else {
      f = ARH::rpc(target_rank, empty_handler);
    }
#ifdef ARH_PROFILE
    timer_rpc.end_and_update();
    timer_push.start();
#endif
    futures.push_back(std::move(f));
#ifdef ARH_PROFILE
    timer_push.end_and_update();
#endif
  }

#ifdef ARH_PROFILE
  timer_barrier.start();
#endif
  // start ARH::barrier()
  ARH::threadBarrier.wait();
  ARH::flush_agg_buffer();
  ARH::flush_am();
  if (ARH::my_worker_local() == 0) {
    BCL::barrier();
  }
  ARH::threadBarrier.wait();
  // end ARH::barrier()
#ifdef ARH_PROFILE
  timer_barrier.end_and_update();
#endif

  for (int i = 0; i < num_ops; i++) {
#ifdef ARH_PROFILE
    timer_get.start();
#endif
    futures[i].get();
#ifdef ARH_PROFILE
    timer_get.end_and_update();
#endif
  }

  ARH::barrier();
  ARH::tick_t end_wait = ARH::ticks_now();

  double duration_total = ARH::ticks_to_ns(end_wait - start) / 1e3;
  ARH::print("Setting: agg_size = %lu; duration = %.2lf s; num_ops = %lu\n", agg_size, duration_total / 1e6, num_ops);
  ARH::print("Total bandwidth: %lu MB/s\n", (unsigned long) (sizeof(ARH::rpc_t) * num_ops * ARH::nworkers_local() * 2 / duration_total));
#ifdef ARH_PROFILE
  // fine-grained
  ARH::print("rand: %.3lf us\n", timer_rand.to_us());
  ARH::print("rpc/rpc_agg: %.3lf us\n", timer_rpc.to_us());
  ARH::print("push: %.3lf us\n", timer_push.to_us());
  ARH::print("barrier: %.3lf us (ave: %.3lf us)\n", timer_barrier.to_us(), timer_barrier.to_us() / num_ops);
  ARH::print("get: %.3lf us\n", timer_get.to_us());
  // rpc backend
  ARH::print("rpc/future preparation: %.3lf us\n", ARH::timer_load.to_us());
  ARH::print("agg buffer without pop: %.3lf us\n", ARH::timer_buf_npop.to_us());
  ARH::print("agg buffer with pop: %.3lf us\n", ARH::timer_buf_pop.to_us());
  ARH::print("agg buffer ave: %.3lf us\n", (ARH::timer_buf_pop.to_us() + ARH::timer_buf_npop.to_us() * MAX(agg_size - 1, 0)) / MAX(agg_size, 1));
  ARH::print("Gasnet_ex req: %.3lf us (ave: %.3lf us)\n", ARH::timer_gex_req.to_us(), ARH::timer_gex_req.to_us() / agg_size);
#endif
}

int main(int argc, char** argv) {
  // one process per node
  cxxopts::Options options("ARH Benchmark", "Benchmark of ARH system");
  options.add_options()
      ("size", "Aggregation size", cxxopts::value<size_t>())
      ;
  auto result = options.parse(argc, argv);
  try {
    agg_size = result["size"].as<size_t>();
  } catch (...) {
    agg_size = 0;
  }
  assert(agg_size >= 0);
  use_agg = (agg_size != 0);

  ARH::init(15, 16);
  if (use_agg) {
    agg_size = ARH::set_agg_size(agg_size);
  }

  ARH::run(worker);

  ARH::finalize();
}
#else
#include <iostream>
using namespace std;
int main() {
  cout << "Only run arh test with GASNET_EX" << endl;
  return 0;
}
#endif
