#include <bcl/bcl.hpp>

void say_rank(size_t rank) {
  printf("Hello, I am thread %lu executing on %lu\n", rank, BCL::rank());
}

int main(int argc, char** argv) {
  BCL::init(256, true, true);
  for (size_t i = 0; i < BCL::nprocs(); i++) {
    BCL::rpc(i, say_rank, BCL::rank());
  }
  BCL::finalize();
  return 0;
}
