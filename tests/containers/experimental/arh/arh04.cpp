#ifdef GASNET_EX
  #include "bcl/bcl.hpp"
  #include "bcl/containers/experimental/arh/arh.hpp"
  #include <cassert>


void worker() {

  int my_rank = (int) ARH::my_worker();
  int steps = 10;

  auto fn = [](int a, int b) -> int {
    return a * b;
  };

  using rv = decltype(ARH::rpc(0, fn, my_rank, my_rank));
  std::vector<rv> futures;

  for (int i = 0 ; i < steps; i++) {
    size_t target_rank = rand() % ARH::nworkers();
    auto f = ARH::rpc_agg(target_rank, fn, my_rank, my_rank);
    futures.push_back(std::move(f));
  }

  ARH::barrier();

  for (int i = 0 ; i < steps; i++) {
    int val = futures[i].wait();
    assert(val == my_rank*my_rank);
  }
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init();
  ARH::set_agg_size(5);

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