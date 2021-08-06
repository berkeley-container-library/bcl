// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/CircularQueue.hpp>

// XXX: Designed to test phasal queue pushes/pops.

int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 1000;
  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    BCL::CircularQueue<int> queue(rank, n_pushes * BCL::nprocs());

    for (size_t i = 0; i < n_pushes; i++) {
      queue.push(BCL::rank(), BCL::CircularQueueAL::push);
    }

    BCL::barrier();

    std::unordered_map<int, int> counts;
    if (BCL::rank() == rank) {
      while (!queue.empty()) {
        int val;
        bool success = queue.pop(val, BCL::CircularQueueAL::none);
        if (success) {
          // recovered val
          assert(val < BCL::nprocs() && val >= 0);
          counts[val]++;
          if (counts[val] > n_pushes) {
            throw std::runtime_error("BCL::CircularQueue01: too many " +
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
