#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>

#include <bcl/core/util/Backoff.hpp>

// #NN: modified from ChecksumQueue03 to test batch-pops.
int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 132;
  size_t push_size = 12;
  size_t pop_per = 23;

  constexpr bool print_verbose = false;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    BCL::print("Rank %lu\n", rank);
	//Concurrent push/pop on full queue is safe with reserve.
    BCL::ChecksumQueue<int> queue(rank, BCL::nprocs()*10*push_size, pop_per);

    if (BCL::rank() != rank) {
		usleep(200);
      for (size_t i = 0; i < n_pushes; i++) {
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
		while (!queue.pop(tmp, (maxpop < pop_per) ? maxpop : pop_per, false)) {
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
    BCL::barrier();
	usleep(50000);
  }

  BCL::finalize();
  fprintf(stderr, "(%lu) ALL OK\n", BCL::rank());
  return 0;
}
