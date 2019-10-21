#ifdef GASNET_EX
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

  size_t my_rank = ARH::my_worker_local();
  size_t nworkers = ARH::nworkers();

  mObjects.get().v = std::vector<std::atomic<int>>(local_range);

  using rv = decltype(ARH::rpc(size_t(), histogram_handler, int()));
  std::vector<rv> futures;

  ARH::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (int i = 0 ; i < num_ops; i++) {
    size_t target_rank = lrand48() % nworkers;
    int val = lrand48() % local_range;
    rv f;
    if (use_agg) {
      f = ARH::rpc_agg(target_rank, histogram_handler, val);
    } else {
      f = ARH::rpc(target_rank, histogram_handler, val);
    }
    futures.push_back(std::move(f));
  }

  auto time1 = std::chrono::high_resolution_clock::now();

  ARH::barrier();

  for (int i = 0 ; i < num_ops; i++) {
    futures[i].wait();
  }

  ARH::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(time1 - begin).count();
  double overhead = duration * (1e6 / num_ops * MAX(agg_size, 1));
  ARH::print("Request Phase: agg_size = %lu; duration = %.2lf s; overhead   = %.2lf us/op\n", agg_size, duration, overhead);

  duration = std::chrono::duration<double>(end - begin).count();
  ARH::print("Total:         agg_size = %lu; duration = %.2lf s; throughput = %d op/s\n", agg_size, duration, (int) (num_ops / duration));
}

int main(int argc, char** argv) {
  // one process per node
  cxxopts::Options options("ARH Benchmark", "Benchmark of ARH system");
  options.add_options()
      ("size", "Aggregation size", cxxopts::value<size_t>())
      ;
  auto result = options.parse(argc, argv);
  agg_size = result["size"].as<size_t>();
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
