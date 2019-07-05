#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>

// XXX: Designed to test phasal queue pushes/pops.

int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 1000;
  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    BCL::ChecksumQueue<int> queue(rank, n_pushes * BCL::nprocs() + 1);

    for (size_t i = 0; i < n_pushes; i++) {
      bool success = queue.push(BCL::rank());
      assert(success);
    }

    BCL::barrier();

    std::unordered_map<int, int> counts;
    if (BCL::rank() == rank) {
      while (!queue.empty()) {
        int val;
        bool success = queue.pop(val);
        if (success) {
          // recovered val
          assert(val < BCL::nprocs() && val >= 0);
          counts[val]++;
          if (counts[val] > n_pushes) {
            throw std::runtime_error("BCL::ChecksumQueue01: too many " +
                                     std::to_string(val) + "s");
          }
        }
      }

      for (auto& c : counts) {
        assert(c.second == n_pushes);
      }
    }
  }
  BCL::finalize();
  return 0;
}
