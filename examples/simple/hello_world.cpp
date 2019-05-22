#include <bcl/bcl.hpp>

int main(int argc, char** argv) {
  BCL::init();
  printf("Hello, BCL! I am rank %lu/%lu on host %s.\n",
         BCL::rank(), BCL::nprocs(), BCL::hostname().c_str());
  BCL::finalize();
  return 0;
}
