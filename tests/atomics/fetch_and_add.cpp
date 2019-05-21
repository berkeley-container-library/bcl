#include <cassert>

#include <bcl/bcl.hpp>

int main(int argc, char ** argv) {
  BCL::init();
  size_t n_adds = 100;
  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    BCL::GlobalPtr<int> ptr;

    if (BCL::rank() == rank) {
      ptr = BCL::alloc<int>(1);
      *ptr = 0;
    }

    ptr = BCL::broadcast(ptr, rank);

    for (size_t i = 0; i < n_adds; i++) {
      BCL::fetch_and_op<int>(ptr, BCL::rank(), BCL::plus<int>{});
    }

    BCL::barrier();

    if (BCL::rank() == 0) {
      int final_val = *ptr;

      int computed_val = 0;
      for (size_t i = 0; i < BCL::nprocs(); i++) {
        computed_val += n_adds*i;
      }

      assert(computed_val == final_val);
    }

    if (BCL::rank() == rank) {
      BCL::dealloc<int>(ptr);
    }
  }
  BCL::finalize();
  return 0;
}
