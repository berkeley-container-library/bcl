#ifdef GASNET_EX
#include "bcl/containers/experimental/arh/arh.hpp"
#include <cassert>
#include "include/cxxopts.hpp"

int num_ops = 100000;
const char* title = "broadcast";
void do_something(int i) {
  int val = ARH::my_worker();
  ARH::broadcast_node(val, i % ARH::nworkers_local());
}

void worker() {
  int total_num_ops = num_ops * (int) ARH::nworkers();

  ARH::print("Start benchmark %s on %lu ops\n", title, num_ops);
  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  for (int i = 0; i < num_ops; i++) {
    do_something(i);
  }

  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e3;
  double ave_overhead = duration / num_ops;
  ARH::print("%lu ops in %.2lf s\n", duration / 1e6, num_ops);
  ARH::print("%.2lf us/op\n", ave_overhead);
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
