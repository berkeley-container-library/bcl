#ifdef GASNET_EX
#include "bcl/bcl.hpp"
#include "bcl/containers/experimental/rpc_oneway/arh.hpp"

void worker() {
  std::printf("Hello, I am thread %lu/%lu!\n", ARH::my_worker(), ARH::nworkers());
}

int main(int argc, char** argv) {
  // one process per node
  BCL::init();
  ARH::run(worker);

  BCL::finalize();
}
#else
#include <iostream>
using namespace std;
int main() {
  cout << "Only run arh test with GASNET_EX" << endl;
  return 0;
}
#endif