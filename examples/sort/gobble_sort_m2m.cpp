
/*
  Perhaps a poorly-worded example: I call this "Gobble Sort"
  because it uses the FastQueue::push(T&& vals) method to
  perform (more) asynchronous pushes.
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

#include <bcl/containers/ManyToManyDistributor.hpp>

int main (int argc, char **argv) {
  BCL::init(1024);

  if (argc < 2) {
    BCL::print("usage: [launcher] ./gobble_sort [message_size]\n");
    BCL::finalize();
    return 0;
  }

  // Sort 1M integers
  constexpr size_t n_vals = 16777216;

  constexpr size_t bucket_size = 2*n_vals;

  // Generate random values between [0, 1G)
  constexpr size_t range = 1 << 28;

  size_t partition_size = (range + BCL::nprocs() - 1) / BCL::nprocs();

  // Tuning parameter
  const size_t message_size = std::atoi(argv[1]);

  BCL::print("Sorting %lu integers among %lu ranks with message size %lu\n",
             n_vals, BCL::nprocs(), message_size);

  BCL::ManyToManyDistributor<int> dist(bucket_size, message_size);

  srand48(BCL::rank());

  std::vector<int> vals;

  for (size_t i = 0; i < n_vals; i++) {
    vals.push_back(lrand48() % range);
  }

  BCL::barrier();

  auto begin = std::chrono::high_resolution_clock::now();
  auto rbegin = begin;

  // Insert values into buffers, flushing full buffers
  // into the remote queues as necessary.

  for (const auto& val : vals) {
    size_t my_node = val / partition_size;
    dist.insert(val, my_node);
  }

  dist.flush();
  auto end = std::chrono::high_resolution_clock::now();

  double send_duration = std::chrono::duration<double>(end - begin).count();

  begin = std::chrono::high_resolution_clock::now();

  // std::vector <int> my_bucket = queues[BCL::rank()].as_vector();

  // std::sort(my_bucket.begin(), my_bucket.end());
  // std::sort(queues[BCL::rank()].begin().local(), queues[BCL::rank()].end().local());
  std::vector<int> counts = bucket_count(dist, partition_size, BCL::rank()*partition_size);

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
