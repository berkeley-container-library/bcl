#ifdef GASNET_EX
#include "bcl/containers/experimental/rpc_oneway/arh.hpp"

void worker() {
  ARH::print("The number of processors configured is :%ld\n", sysconf(_SC_NPROCESSORS_CONF));
  ARH::print("Size of ARH::rpc_t %lu\n", sizeof(ARH::rpc_t));
  ARH::print("Size of ARH::rpc_t::rpc_result_t %lu\n", sizeof(ARH::rpc_t::rpc_result_t));
  ARH::print("Size of ARH::rpc_t::payload_t %lu\n", sizeof(ARH::rpc_t::payload_t));
  ARH::print("Size of ARH::rpc_t::rpc_result_t::payload_t %lu\n", sizeof(ARH::rpc_t::rpc_result_t::payload_t));
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