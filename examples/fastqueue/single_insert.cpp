#include <algorithm>

#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>

int main(int argc, char** argv) {
  BCL::init();

  size_t n_to_insert = 100;

  size_t queue_size = n_to_insert*2;

  std::vector<BCL::FastQueue<int>> queues;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    queues.push_back(BCL::FastQueue<int>(rank, queue_size));
  }

  srand48(BCL::rank());

  for (size_t i = 0; i < n_to_insert; i++) {
    size_t dst_rank = lrand48() % BCL::nprocs();
    queues[dst_rank].push(lrand48());
  }

  BCL::barrier();

  // Sort local queue in place
  std::sort(queues[BCL::rank()].begin().local(), queues[BCL::rank()].end().local());

  // Pop out of queue
  size_t count = 0;
  while (!queues[BCL::rank()].empty()) {
    int value;
    bool success = queues[BCL::rank()].pop(value);

    if (success) {
      count++;
    }
  }

  size_t total_count = BCL::allreduce<size_t>(count, std::plus<size_t>{});

  BCL::print("Popped %lu values total.\n", total_count);

  BCL::finalize();
  return 0;
}
