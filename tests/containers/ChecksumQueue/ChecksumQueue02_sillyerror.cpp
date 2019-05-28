#include <cassert>
#include <unordered_map>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>
#include <unistd.h>

// XXX: Designed to test simultaneous queue pushes and pops.
// #NN: modified from CirculuarQueue02 for checksum queue.

int main(int argc, char** argv) {
  BCL::init();

  size_t n_pushes = 3;

  //Repeat in order to verify that senses disambiguate
  for (int rep = 0; rep < 3; rep++) {
	  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
		BCL::ChecksumQueue<int> queue(rank, n_pushes * BCL::nprocs());

		if (BCL::rank() != rank) {
		  for (size_t i = 0; i < n_pushes; i++) {
			printf("Pushing from %d to %d\n", BCL::rank(), rank);
			bool success = queue.push(BCL::rank());
			assert(success);
		  }
		}

		std::unordered_map<int, int> counts;

		if (BCL::rank() == rank) {
		  for (size_t i = 0; i < (BCL::nprocs() - 1)*n_pushes; i++) {
			bool success;
			int val;
			do {
			  success = queue.pop(val);
			  //printf("Pop %d on rank %d succeeded: %d\n", i, rank, success);
			  if (!success) {
				  usleep(200000);
			  }
			} while (!success);
			assert(val < BCL::nprocs() && val >= 0);
			counts[val]++;
			if (counts[val] > n_pushes) {
			  throw std::runtime_error("BCL::ChecksumQueue02: " + std::to_string(rank)
									   + " saw too many " +
									   std::to_string(val) + "s");
			}
		  }

		  for (auto& c : counts) {
			if (c.second != n_pushes) {
			  throw std::runtime_error("BCL::ChecksumQueue02: found " +
									   std::to_string(c.second) + " != " +
									   std::to_string(n_pushes) + " pushes for " +
									   std::to_string(c.first));
			}
		  }
		}
		BCL::barrier();
	  }
  }

  BCL::finalize();
  printf("Success\n");
  return 0;
}
