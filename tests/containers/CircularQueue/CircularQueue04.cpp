#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/CircularQueue.hpp>

// XXX: Designed to test simultaneous multiple async_pushes and single pops.

int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 1000;
  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    BCL::CircularQueue<int> queue(rank, n_pushes * BCL::nprocs());

    if (BCL::rank() != rank) {
      for (size_t i = 0; i < n_pushes; i++) {
        auto future = queue.async_push({int(BCL::rank())});
        BCL::Backoff backoff;
        while (!future.is_ready()) {
          backoff.backoff();
        }
      }
    }

    std::unordered_map<int, int> counts;

    if (BCL::rank() == rank) {
      for (size_t i = 0; i < (BCL::nprocs() - 1)*n_pushes; i++) {
        bool success;
        int val;
        do {
          success = queue.pop(val);
        } while (!success);
        assert(val < BCL::nprocs() && val >= 0);
        counts[val]++;
        if (counts[val] > n_pushes) {
          throw std::runtime_error("BCL::CircularQueue04: " + std::to_string(rank)
                                   + " saw too many " +
                                   std::to_string(val) + "s");
        }
      }

      for (auto& c : counts) {
        if (c.second != n_pushes) {
          throw std::runtime_error("BCL::CircularQueue04: found " +
                                   std::to_string(c.second) + " != " +
                                   std::to_string(n_pushes) + " pushes for " +
                                   std::to_string(c.first));
        }
      }
    }
    BCL::barrier();
  }

  BCL::finalize();
  return 0;
}
