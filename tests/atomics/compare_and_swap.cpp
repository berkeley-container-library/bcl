
#include <cassert>
#include <bcl/bcl.hpp>
#include <limits>

int main(int argc, char** argv) {
  BCL::init();
  BCL::GlobalPtr<int> ptr = nullptr;

  constexpr size_t n_repetitions = 10;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (BCL::rank() == rank) {
      ptr = BCL::alloc<int>(BCL::nprocs());
      for (size_t i = 0; i < BCL::nprocs(); i++) {
        ptr[i] = 0;
      }
    }
    ptr = BCL::broadcast(ptr, rank);

    for (size_t k = 0; k < n_repetitions; k++) {
      for (size_t i = 0; i < BCL::nprocs(); i++) {
        int rv = std::numeric_limits<int>::max();
        while (rv != BCL::nprocs()*k + BCL::rank()) {
          rv = BCL::compare_and_swap<int>(ptr + i, BCL::nprocs()*k + BCL::rank(),
                                          BCL::nprocs()*k + BCL::rank()+1);
        }
      }
    }

    BCL::barrier();

    if (BCL::rank() == rank) {
      BCL::dealloc<int>(ptr);
    }
  }

  BCL::finalize();
  return 0;
}
