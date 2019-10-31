#ifdef GASNET_EX
#define ARH_PROFILE
#include "bcl/containers/experimental/arh/arh.hpp"
#include <cassert>
#include "include/cxxopts.hpp"

void worker() {

  int num_ops = 100000;
  int total_num_ops = num_ops * (int) ARH::nworkers();

  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  for (int i = 0; i < num_ops; i++) {
    ARH::barrier();
  }

  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e3;
  double ave_overhead = duration / num_ops;
  ARH::print("Setting: duration = %.2lf s; num_ops = %lu\n", duration / 1e6, num_ops);
  ARH::print("ave_overhead: %.2lf us\n", ave_overhead);
}

int main(int argc, char** argv) {
  ARH::init(15, 16);

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
