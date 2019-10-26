#ifdef GASNET_EX
  #include "bcl/containers/experimental/arh/arh.hpp"
  #include <cassert>

struct ThreadObjects {
    std::vector<std::atomic<int>> v;
};

ARH::GlobalObject<ThreadObjects> mObjects;

void histogram_handler(int idx) {
  mObjects.get().v[idx] += 1;
}

void worker() {

  int local_range = 5;
  int total_range = local_range * (int) ARH::nworkers();

  size_t my_rank = ARH::my_worker_local();
  int nworkers = (int) ARH::nworkers();

  mObjects.get().v = std::vector<std::atomic<int>>(local_range);

  using rv = decltype(ARH::rpc(size_t(), histogram_handler, int()));
  std::vector<rv> futures;

  ARH::barrier();
  for (int i = 0 ; i < total_range; i++) {
    int idx = (i + local_range * (int)my_rank) % total_range;
    size_t target_rank = idx / local_range;
    int val = idx % local_range;
    auto f = ARH::rpc(target_rank, histogram_handler, val);
    futures.push_back(std::move(f));
  }

  ARH::barrier();

  for (int i = 0 ; i < total_range; i++) {
    futures[i].wait();
  }

  ARH::barrier();

  for (int i = 0; i < local_range; i++) {
    assert(mObjects.get().v[i] == nworkers);
  }
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