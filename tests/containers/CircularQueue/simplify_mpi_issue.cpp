#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/SafeChecksumQueue.hpp>

#include <bcl/core/util/Backoff.hpp>
/*
 * Be problematic only when rank=0
 * CircularQueue and SafeChecksumQueue both have the issue.
 */
int main(int argc, char** argv) {
  BCL::init();
  fprintf(stderr, "I am rank %lu out of %lu procs; tag1\n", BCL::rank(), BCL::nprocs());

  size_t n_pushes = 10;

  size_t rank = 0;
  fprintf(stderr, "I am rank %lu out of %lu procs; tag2\n", BCL::rank(), BCL::nprocs());
  BCL::ChecksumQueue<int> queue(0, 100);
  fprintf(stderr, "I am rank %lu out of %lu procs; tag3\n", BCL::rank(), BCL::nprocs());

  fprintf(stderr, "I am rank %lu out of %lu procs; tag4\n", BCL::rank(), BCL::nprocs());

  if (BCL::rank() != rank) {
    for (size_t i = 0; i < n_pushes; i++) {
      int val = 1;
      bool success = queue.push(val, true);
      assert(success);
    }
  }

//  BCL::barrier(); // adding this barrier fixes the issue

  if (BCL::rank() == rank) {
    std::unordered_map<int, int> counts;

    for (size_t i = 0; i < (BCL::nprocs()-1)*n_pushes; i++) {
      int val;
      BCL::Backoff backoff; // it doesn't relate to the issue
      while (!queue.pop(val)) {
        backoff.backoff();
      }
      assert(val == 1);
    }
  }

  fprintf(stderr, "rank %lu done\n", BCL::rank());
  BCL::finalize();
  return 0;
}
