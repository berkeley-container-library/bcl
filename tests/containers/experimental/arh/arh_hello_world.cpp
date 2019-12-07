#ifdef GASNET_EX
#define ARH_DEBUG
#include "bcl/containers/experimental/arh/arh.hpp"

void worker() {
  std::printf("Hello, I am thread %lu/%lu!\n", ARH::my_worker(), ARH::nworkers());
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