#ifdef GASNET_EX
#define ARH_DEBUG
#include "bcl/containers/experimental/arh/arh.hpp"

void worker(int val) {
  std::printf("Hello, I am thread %lu/%lu! receive %d\n", ARH::my_worker(), ARH::nworkers(), val);
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init(15, 16);
  int val = 132;
  ARH::run(worker, val);

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