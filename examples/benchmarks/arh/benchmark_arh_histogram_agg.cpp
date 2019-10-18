#ifdef GASNET_EX
#include "bcl/containers/experimental/rpc_oneway/arh.hpp"
#include <cassert>

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
    auto f = ARH::rpc_agg(target_rank, histogram_handler, val);
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
  ARH::print("The first duration %.2lf; %d ops / s\n", duration, (int) (num_ops / duration));

  duration = std::chrono::duration<double>(end - begin).count();
  ARH::print("Total duration %.2lf; %d ops / s\n", duration, (int) (num_ops / duration));
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init(15, 16);
  ARH::set_agg_size(100);
  mObjects.init();
//  std::printf("maximum aggregation size = %lu\n", ARH::get_max_agg_size());

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