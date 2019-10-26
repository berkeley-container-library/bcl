#ifdef GASNET_EX
  #include "bcl/bcl.hpp"
  #include "bcl/containers/experimental/arh/arh.hpp"
  #include <cassert>
  #include "bcl/containers/experimental/arh/arh_tools.hpp"


int fn(int a, int b) {
  return a * b;
}

void worker() {

  size_t num_ops = 100000;

  ARH::barrier();
  ARH::tick_t start = ARH::tick_now();

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_rank = lrand48() % ARH::nworkers();
    auto f = ARH::rpc(remote_rank, [](int lel) { }, i);
    f.wait();
  }

  ARH::barrier();
  ARH::tick_t end = ARH::tick_now();

  long duration = end - start;
  ARH::print("%lf total %lfus / op\n", duration / 1e6, duration / num_ops);
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init();

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