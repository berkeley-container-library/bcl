#include <bcl/bcl.hpp>
#include <cassert>

/*
  The point of this example is to demonstrate basic mechanics
  of global pointers.  It shows off a couple different methods
  for reading and writing to remote pointers:

  1) Dereferencing a global pointer - ptr[12].
     * Can read and write.
     * int local_value = ptr[12];
     * ptr[12] = my_value;

  2) Calling ptr.local() to get a local pointer.

  3) BCL::memcpy()
     * Works like POSIX memcpy, except one
       of the arguments is a global pointer.

  4) rput and rget
     * Fairly similar to UPC++ rget and rput.
     * Like memcpy, but typed.
     * See /bcl/core/comm.hpp for definitions.
*/

int main(int argc, char** argv) {
  BCL::init();

  // Need to run this with at least two processes.
  assert(BCL::nprocs() >= 2);

  BCL::GlobalPtr<int> ptr = nullptr;

  if (BCL::rank() == 0) {
    ptr = BCL::alloc<int>(BCL::nprocs());
  }

  ptr = BCL::broadcast(ptr, 0);

  // 1) Using pointer dereference
  //    to write a value.
  ptr[BCL::rank()] = BCL::rank();

  BCL::barrier();

  if (BCL::rank() == 0) {
    // 2) Calling .local() to get a local pointer.
    int* local = ptr.local();

    printf("Rank 0 Sees:\n");
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      printf("%lu: %d\n", i, local[i]);
    }
  }

  BCL::barrier();

  if (BCL::rank() == 1) {
    std::vector<int> zeros(BCL::nprocs(), 0);
    // 3) Using BCL::memcpy.
    BCL::memcpy(ptr, zeros.data(), sizeof(int)*zeros.size());
  }

  BCL::barrier();

  // 4) Using rput
  BCL::rput((int) BCL::rank(), ptr + BCL::rank());

  BCL::barrier();

  if (BCL::rank() == 1) {
    printf("Rank 1 Sees:\n");
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      // 4) Using rget
      printf("%lu: %d\n", i, BCL::rget(ptr + i));
    }
  }

  BCL::finalize();
  return 0;
}
