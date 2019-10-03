#ifdef GASNET_EX
  #include "bcl/bcl.hpp"
  #include "bcl/containers/experimental/rpc_oneway/arh.hpp"
  #include <cassert>


int fn(int a, int b) {
  return a * b;
}

void worker() {

  size_t num_ops = 1000;

  ARH::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_rank = lrand48() % ARH::nprocs();
    auto f = ARH::rpc(remote_rank, [](int lel) { }, i);
    f.wait();
  }

  ARH::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();
  ARH::print("%lf total %lfus / op\n", duration, 1e6*duration / num_ops);
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init();

  ARH::run(worker, 8, 16);

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