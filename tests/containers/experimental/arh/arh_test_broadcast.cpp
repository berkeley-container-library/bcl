#ifdef GASNET_EX
#define ARH_DEBUG
#include "bcl/containers/experimental/arh/arh.hpp"
#include <vector>

void worker() {
  std::vector<size_t> v(ARH::nworkers(), -1);
  v[ARH::my_worker()] = ARH::my_worker();
  for (int i = 0; i < ARH::nworkers(); ++i) {
    ARH::broadcast(v[i], i);
    assert(v[i] == i);
  }
}

int main(int argc, char** argv) {
  // one process per node
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