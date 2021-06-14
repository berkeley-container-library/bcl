
/*
  Perform the ISx benchmark.
*/

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <list>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include "bucket_count.hpp"

#include <bcl/bcl.hpp>

#include <bcl/containers/FastQueue.hpp>

int main (int argc, char **argv) {
  BCL::init(1024);

  if (argc < 2) {
    BCL::print("usage: [launcher] ./isx_benchmark [message_size]\n");
    BCL::finalize();
    return 0;
  }

  // Sort 1M integers
  constexpr size_t n_vals = 16777216;

  constexpr size_t bucket_size = 2*n_vals;

  // Generate random values between [0, 1G)
  constexpr size_t range = 1 << 28;

  size_t partition_size = (range + BCL::nprocs() - 1) / BCL::nprocs();

  // Message size (in number of ints)
  const size_t message_size = std::atoi(argv[1]);

  BCL::print("Sorting %lu integers among %lu ranks with message size %lu\n",
             n_vals, BCL::nprocs(), message_size);

  std::vector<BCL::FastQueue<int>> queues;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    queues.push_back(BCL::FastQueue<int>(rank, bucket_size));
  }

  BCL::print("Warming up queues...\n");
  BCL::barrier();
  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    queues[rank].warmup_();
  }
  BCL::barrier();
  BCL::print("Finished warming up.\n");

  srand48(BCL::rank());

  // Generate local keys to be sorted.
  std::vector<int> vals;

  for (int i = 0; i < n_vals; i++) {
    vals.push_back(lrand48() % range);
  }

  BCL::barrier();

  // Allocate memory for message queues.
  std::vector <std::vector<int>> message_queues(BCL::nprocs());

  for (size_t i = 0; i < message_queues.size(); i++) {
    message_queues.reserve(message_size);
  }

  BCL::barrier();

  // Begin sorting

  auto begin = std::chrono::high_resolution_clock::now();
  auto rbegin = begin;

  // Insert values into buffers, flushing full buffers
  // into the remote queues as they fill up.

  std::list<BCL::future<std::vector<int>>> futures;

  for (const auto& val : vals) {
    int my_node = val / partition_size;

    message_queues[my_node].push_back(val);

    if (message_queues[my_node].size() >= message_size) {
      auto future = queues[my_node].push(std::move(message_queues[my_node]));
      if (!future.has_value()) {
        throw std::runtime_error("error: Queue on " + std::to_string(my_node) +
          "out of space");
      }
      message_queues[my_node].reserve(message_size);
      futures.emplace_back(std::move(future.value()));
    }
  }

  // Flush any remaining queues.

  for (int rank = 0; rank < message_queues.size(); rank++) {
  auto future = queues[rank].push(std::move(message_queues[rank]));
    if (!future.has_value()) {
      throw std::runtime_error("error: Queue on " + std::to_string(rank) +
        "out of space");
    }
    futures.emplace_back(std::move(future.value()));
  }

  for (auto& future: futures) {
    future.get();
  }

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double send_duration = std::chrono::duration<double>(end - begin).count();

  begin = std::chrono::high_resolution_clock::now();

  // For MPI + OpenMP, replace with __gnu_parallel::sort or your favorite
  // parallel sort.
  std::sort(queues[BCL::rank()].begin().local(), queues[BCL::rank()].end().local());

  // Depending on problem size, it may be beneficial to do a "histogram sort" instead.
  // std::vector<int> counts = bucket_count(queues[BCL::rank()], partition_size, BCL::rank()*partition_size);

  end = std::chrono::high_resolution_clock::now();
  auto rend = end;


  double sort_duration = std::chrono::duration<double>(end - begin).count();

  BCL::print("%lf seconds sending, %lf seconds sorting.\n", send_duration, sort_duration);
  double duration = std::chrono::duration<double>(rend - rbegin).count();
  double total_duration = BCL::allreduce(duration, std::plus<double>{});
  double avg_duration = total_duration / BCL::nprocs();
  BCL::print("%lf seconds total\n", duration);
  BCL::print("%lf seconds average\n", avg_duration);

  BCL::finalize();
  return 0;
}
