#include <cassert>

#include <bcl/bcl.hpp>

int main(int argc, char** argv) {
  BCL::init();

  float eps = 1e-5;

  size_t n_adds = 100;
  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    BCL::GlobalPtr<float> ptr;

    if (BCL::rank() == rank) {
      ptr = BCL::alloc<float>(1);
      *ptr = 0;
    }

    ptr = BCL::broadcast(ptr, rank);

    for (size_t i = 0; i < n_adds; i++) {
      BCL::fetch_and_op<float>(ptr, BCL::rank(), BCL::min<float>{});
    }

    BCL::barrier();

    if (BCL::rank() == 0) {
      float final_val = *ptr;

      float computed_val = 0;
      /*
      for (size_t i = 0; i < BCL::nprocs(); i++) {
        for (size_t j = 0; j < n_adds; j++) {
          computed_val += i;
        }
      }
      */

      assert((final_val - computed_val) < eps);
    }

    if (BCL::rank() == rank) {
      BCL::dealloc(ptr);
    }
  }
  BCL::finalize();
  return 0;
}
