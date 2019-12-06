#include <cstdlib>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <chrono>
#include <stdexcept>

#include <BCL.hpp>

#include <containers/CircularQueue.hpp>

int main (int argc, char **argv) {
  BCL::init(1024);

  // Sort 1M integers
  const size_t n_vals = 1024*1024*1;

  const size_t bucket_size = n_vals*2;

  // Generate random values between [0, 1G)
  const int range = 1000000000;

  int partition_size = (range + BCL::nprocs() - 1) / BCL::nprocs();

  // Tuning parameter
  const int message_size = 1024*1;

  std::vector<BCL::CircularQueue<int>> queues;

  for (int rank = 0; rank < BCL::nprocs(); rank++) {
    queues.push_back(BCL::CircularQueue<int>(rank, bucket_size));
  }

  srand48(BCL::rank());

  std::vector<int> vals;

  for (int i = 0; i < n_vals; i++) {
    vals.push_back(lrand48() % range);
  }

  BCL::barrier();

  std::vector <std::vector<int>> message_queues(BCL::nprocs());

  for (size_t i = 0; i < message_queues.size(); i++) {
    message_queues.reserve(message_size);
  }

  // Insert values into buffers, flushing full buffers
  // into the remote queues as necessary.

  for (const auto& val : vals) {
    int my_node = val / partition_size;

    message_queues[my_node].push_back(val);

    if (message_queues[my_node].size() >= message_size) {
      bool success = queues[my_node].push(message_queues[my_node]);
      if (!success) {
        throw std::runtime_error("error: Queue on " + std::to_string(my_node) +
          "out of space");
      }
      message_queues[my_node].clear();
      message_queues[my_node].reserve(message_size);
    }
  }

  // Flush any remaining queues.

  for (int rank = 0; rank < message_queues.size(); rank++) {
    queues[rank].push(message_queues[rank]);
    message_queues[rank].clear();
  }

  BCL::barrier();

  std::vector <int> my_bucket = queues[BCL::rank()].as_vector();

  std::sort(my_bucket.begin(), my_bucket.end());

  BCL::finalize();
  return 0;
}
