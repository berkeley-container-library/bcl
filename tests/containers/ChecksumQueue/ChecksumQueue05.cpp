#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>

#include <bcl/core/util/Backoff.hpp>

// #NN: modified from ChecksumQueue04 to use a small queue,
// but add waits to prevent overrun.
int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 132;
  size_t push_size = 12;

  constexpr bool print_verbose = false;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
	int qs = BCL::nprocs() * 2 * push_size + (push_size / 2);
	BCL::print("Rank %lu, queue size %d\n", rank, qs);
    BCL::ChecksumQueue<int> queue(rank, qs);

    if (BCL::rank() != rank) {
      for (size_t i = 0; i < n_pushes; i++) {
		//This wait is necessary, otherwise overruns occur.
		usleep(10000);
        std::vector<int> vec(push_size, BCL::rank());
        BCL::Backoff backoff;
        while (!queue.push(vec, true)) {}
		//printf("Pushed from %d to %d\n", BCL::rank(), rank);
      }
    }

    std::unordered_map<int, int> counts;

    if (BCL::rank() == rank) {
		std::vector<int> tmp;
      for (size_t i = 0; i < (BCL::nprocs() - 1)*n_pushes*push_size;) {
        BCL::Backoff backoff;
		size_t maxpop = (BCL::nprocs() - 1)*n_pushes*push_size - i;
		size_t pop_per = 23;
		while (!queue.pop(tmp, (maxpop < pop_per) ? maxpop : pop_per, true)) {
			backoff.backoff();
		}
		for (auto &val : tmp) {
			assert(val < BCL::nprocs() && val >= 0);
			counts[val]++;
			if (counts[val] > n_pushes*push_size) {
			  throw std::runtime_error("BCL::ChecksumQueue03: " + std::to_string(rank)
									   + " saw too many " +
									   std::to_string(val) + "s");
			}
			++i;
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
	usleep(5000);
	BCL::barrier();
  }

  BCL::finalize();
  fprintf(stderr, "(%lu) ALL OK\n", BCL::rank());
  return 0;
}
