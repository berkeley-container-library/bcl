#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>

#include <bcl/core/util/Backoff.hpp>

// XXX: Designed to test simultaneous queue pushes and pops of vectors.
// TODO: seems interminably slow without usleep().
// #NN: modified from CirculuarQueue03 for checksum queue.
int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 13;
  size_t push_size = 12;

  constexpr bool print_verbose = false;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    BCL::print("Rank %lu\n", rank);
    BCL::ChecksumQueue<int> queue(rank, 100);

    if (BCL::rank() != rank) {

		//usleep(50000);
      for (size_t i = 0; i < n_pushes; i++) {
        std::vector<int> vec(push_size, BCL::rank());
        BCL::Backoff backoff;
        while (!queue.push(vec, true)) {}
		//printf("Pushed from %d to %d\n", BCL::rank(), rank);
      }
    }

    std::unordered_map<int, int> counts;

    if (BCL::rank() == rank) {
		usleep(50000);
      for (size_t i = 0; i < (BCL::nprocs() - 1)*n_pushes*push_size; i++) {
        int val;
        BCL::Backoff backoff;
		//printf("Popping %d\n", i);
        while (!queue.pop(val)) {}
        assert(val < BCL::nprocs() && val >= 0);
        counts[val]++;
        if (counts[val] > n_pushes*push_size) {
          throw std::runtime_error("BCL::ChecksumQueue03: " + std::to_string(rank)
                                   + " saw too many " +
                                   std::to_string(val) + "s");
        }
      }

      for (auto& c : counts) {
        if (c.second != n_pushes*push_size) {
          throw std::runtime_error("BCL::ChecksumQueue03: found " +
                                   std::to_string(c.second) + " != " +
                                   std::to_string(n_pushes) + " pushes for " +
                                   std::to_string(c.first));
        }
      }
    }

    fprintf(stderr, "(%lu) DONE\n", BCL::rank());
    BCL::barrier();
	usleep(50000);
  }

  BCL::finalize();
  printf("OK");
  return 0;
}
