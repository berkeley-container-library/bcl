#include "bcl/bcl.hpp"
#include "bcl/containers/experimental/rpc_oneway/ARH.hpp"

void worker() {
  std::printf("Hello, I am thread %lu/%lu", ARH::my_thread(), ARH::nthreads());
}

int main(int argc, char** argv) {
  // one process per node
  BCL::init();
  ARH::run(worker);

  BCL::finalize();
}