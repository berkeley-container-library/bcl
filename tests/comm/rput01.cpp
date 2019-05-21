
#include <cassert>
#include <bcl/bcl.hpp>

int main(int argc, char** argv) {
  BCL::init();
  BCL::GlobalPtr<int> ptr = nullptr;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (BCL::rank() == rank) {
      ptr = BCL::alloc<int>(BCL::nprocs());
    }
    ptr = BCL::broadcast(ptr, rank);

    BCL::rput<int>(BCL::rank(), ptr + BCL::rank());
    BCL::barrier();

    if (BCL::rank() == rank) {
      for (size_t i = 0; i < BCL::nprocs(); i++) {
        size_t recvd = BCL::rget(ptr + i);
        size_t recvd_local = *(ptr.local() + i);
        assert(recvd == i);
        assert(recvd_local == i);
      }
      BCL::dealloc<int>(ptr);
    }
  }
  BCL::finalize();
  return 0;
}
