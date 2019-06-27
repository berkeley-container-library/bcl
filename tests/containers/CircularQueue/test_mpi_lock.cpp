#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/CircularQueue.hpp>

#include <bcl/core/util/Backoff.hpp>

// XXX: Designed to test simultaneous queue pushes and pops of vectors.
// TODO: seems interminably slow without usleep().

int main(int argc, char** argv) {
  BCL::init();
  printf("after init: I am rank %lu out of %lu procs\n", BCL::rank(), BCL::nprocs());

  size_t n_pushes = 10;

  size_t rank = 0;

  printf("before queue: I am rank %lu out of %lu procs\n", BCL::rank(), BCL::nprocs());
  BCL::CircularQueue<int> queue(0, 100);

  if (BCL::rank() != rank) {
    for (size_t i = 0; i < n_pushes; i++) {
      int val = 1;
      bool success = queue.push_atomic_impl_(val, true);
      assert(success);
    }
  }

  if (BCL::rank() == rank) {
    std::unordered_map<int, int> counts;

    for (size_t i = 0; i < (BCL::nprocs()-1)*n_pushes; i++) {
      int val;
      while (!queue.pop(val)) {}
      assert(val == 1);
    }
  }
  BCL::barrier();
  printf("rank %lu done\n", BCL::rank());
  BCL::finalize();
  return 0;
}
