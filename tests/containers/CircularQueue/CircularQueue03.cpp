#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/CircularQueue.hpp>

#include <bcl/core/util/Backoff.hpp>

// XXX: Designed to test simultaneous multiple pushes and single pops of vectors.
// TODO: seems interminably slow without usleep().
// XXX: I don't think it is slow.

int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 1000;
  size_t push_size = 10;

  constexpr bool print_verbose = false;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (print_verbose) {
      BCL::print("Rank %lu\n", rank);
    }
    BCL::CircularQueue<int> queue(rank, 100);

    if (BCL::rank() != rank) {
      for (size_t i = 0; i < n_pushes; i++) {
        std::vector<int> vec(push_size, BCL::rank());
        BCL::Backoff backoff;
        while (!queue.push_atomic_impl_(vec, true)) {
          if (print_verbose) {
          fprintf(stderr, "(%lu): yielding with %lu/%lu pushed. "
                          "old_tail %d, but head_buf %d (%d, %d) -> (%d, %d)\n",
                          BCL::rank(), i, (BCL::nprocs()-1)*n_pushes*push_size,
                          BCL::fetch_and_op<int>(queue.tail, 0, BCL::plus<int>{}),
                          queue.head_buf,
                          BCL::fetch_and_op<int>(queue.head, 0, BCL::plus<int>{}),
                          BCL::fetch_and_op<int>(queue.reserved_head, 0, BCL::plus<int>{}),
                          BCL::fetch_and_op<int>(queue.tail, 0, BCL::plus<int>{}),
                          BCL::fetch_and_op<int>(queue.reserved_tail, 0, BCL::plus<int>{})
                          );
          }
          // backoff.backoff();
        }
      }
    }

    std::unordered_map<int, int> counts;

    if (BCL::rank() == rank) {
      for (size_t i = 0; i < (BCL::nprocs() - 1)*n_pushes*push_size; i++) {
        int val;
        BCL::Backoff backoff;
        while (!queue.pop(val)) {
          if (print_verbose) {
          fprintf(stderr, "(%lu): yielding with %lu/%lu popped. "
                          "old_head %d, but tail_buf %d (%d, %d) -> (%d, %d)\n",
                          BCL::rank(), i, (BCL::nprocs()-1)*n_pushes*push_size,
                          BCL::fetch_and_op<int>(queue.head, 0, BCL::plus<int>{}),
                          queue.tail_buf,
                          BCL::fetch_and_op<int>(queue.head, 0, BCL::plus<int>{}),
                          BCL::fetch_and_op<int>(queue.reserved_head, 0, BCL::plus<int>{}),
                          BCL::fetch_and_op<int>(queue.tail, 0, BCL::plus<int>{}),
                          BCL::fetch_and_op<int>(queue.reserved_tail, 0, BCL::plus<int>{})
                          );
          }
          // backoff.backoff();
        }
        assert(val < BCL::nprocs() && val >= 0);
        counts[val]++;
        if (counts[val] > n_pushes*push_size) {
          throw std::runtime_error("BCL::CircularQueue03: " + std::to_string(rank)
                                   + " saw too many " +
                                   std::to_string(val) + "s");
        }
      }

      for (auto& c : counts) {
        if (c.second != n_pushes*push_size) {
          throw std::runtime_error("BCL::CircularQueue03: found " +
                                   std::to_string(c.second) + " != " +
                                   std::to_string(n_pushes) + " pushes for " +
                                   std::to_string(c.first));
        }
      }
    }

    if (print_verbose) {
      fprintf(stderr, "(%lu) DONE\n", BCL::rank());
    }
    BCL::barrier();
  }

  BCL::finalize();
  return 0;
}
