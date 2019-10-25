#ifdef GASNET_EX
#define ARH_PROFILE
#include "bcl/containers/experimental/rpc_oneway/arh.hpp"
#include <cassert>
#include "include/cxxopts.hpp"

bool use_agg = false;
size_t agg_size = 0;

struct ThreadObjects {
  std::vector<std::atomic<int>> v;
};

ARH::GlobalObject<ThreadObjects> mObjects;

void histogram_handler(int idx) {
  mObjects.get().v[idx] += 1;
}

void worker() {

  int local_range = 1000;
  int total_range = local_range * (int) ARH::nworkers();
  int num_ops = 100000;
  int total_num_ops = num_ops * (int) ARH::nworkers();
  double ticks_step = 0;

  size_t my_rank = ARH::my_worker_local();
  size_t nworkers = ARH::nworkers();

  mObjects.get().v = std::vector<std::atomic<int>>(local_range);

  using rv = decltype(ARH::rpc(size_t(), histogram_handler, int()));
  std::vector<rv> futures;

  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  for (int i = 0; i < num_ops; i++) {
    ARH::tick_t start_step = ARH::ticks_now();

    size_t target_rank = lrand48() % nworkers;
    int val = lrand48() % local_range;

    rv f;
    if (use_agg) {
      f = ARH::rpc_agg(target_rank, histogram_handler, val);
    } else {
      f = ARH::rpc(target_rank, histogram_handler, val);
    }
    futures.push_back(std::move(f));

    static int step = 0;
    ARH::tick_t end_step = ARH::ticks_now();
    if (ARH::my_worker_local() == 0) {
      ARH::update_average(ticks_step, end_step - start_step, ++step);
    }
  }

  ARH::barrier();
  ARH::tick_t end_req = ARH::ticks_now();

  for (int i = 0; i < num_ops; i++) {
    futures[i].wait();
  }

  ARH::barrier();
  ARH::tick_t end_wait = ARH::ticks_now();

  double duration_req = ARH::ticks_to_ns(end_req - start) / 1e3;
  double agg_overhead = duration_req / num_ops * MAX(agg_size, 1);
  double ave_overhead = duration_req / num_ops;
  double duration_total = ARH::ticks_to_ns(end_wait - start) / 1e3;
  ARH::print("Setting: agg_size = %lu; duration = %.2lf s\n", agg_size, duration_total / 1e6);
  ARH::print("ave_overhead = %.2lf us/op; agg_overhead = %.2lf us/op\n", ave_overhead, agg_overhead);
  ARH::print("Total throughput = %lu op/s\n", (unsigned long) (num_ops / (duration_req / 1e6)));
  // fine-grained
  double duration_load = ARH::ticks_to_ns(ARH::ticks_load) / 1e3;
  double duration_agg_buf_npop = ARH::ticks_to_ns(ARH::ticks_agg_buf_npop) / 1e3;
  double duration_agg_buf_pop = ARH::ticks_to_ns(ARH::ticks_agg_buf_pop) / 1e3;
  double duration_gex_req = ARH::ticks_to_ns(ARH::ticks_gex_req) / 1e3;
  ARH::print("rpc_agg load: overhead = %.3lf us;\n", duration_load);
  ARH::print("agg buffer without pop: overhead = %.3lf us;\n", duration_agg_buf_npop);
  ARH::print("agg buffer with pop: overhead = %.3lf us;\n", duration_agg_buf_pop);
  double duration_agg_buf = MAX(0, agg_size - 1) * duration_agg_buf_npop + duration_agg_buf_pop;
  ARH::print("agg buffer: overhead = %.3lf us;\n", duration_agg_buf);
  ARH::print("Gasnet_ex req: overhead = %.3lf us;\n", duration_gex_req);
  double duration_step = ARH::ticks_to_ns(ticks_step) / 1e3;
  ARH::print("per rpc overhead = %.3lf us;\n", duration_step);
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
  mObjects.init();
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
