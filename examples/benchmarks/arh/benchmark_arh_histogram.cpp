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
  int steps = 1000;

  size_t my_rank = ARH::my_worker_local();
  size_t nworkers = ARH::nworkers();

  mObjects.get().v = std::vector<std::atomic<int>>(local_range);

  using rv = decltype(ARH::rpc(size_t(), histogram_handler, int()));
  std::vector<rv> futures;

  ARH::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (int i = 0 ; i < steps; i++) {
    size_t target_rank = lrand48() % nworkers;
    int val = lrand48() % local_range;
    auto f = ARH::rpc(target_rank, histogram_handler, val);
    futures.push_back(std::move(f));
  }

  ARH::barrier();

  for (int i = 0 ; i < steps; i++) {
    futures[i].wait();
  }

  ARH::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();
  ARH::print("total duration %lf; %lf ops / s\n", duration, num_ops / duration);
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init(15, 16);
  mObjects.init();

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