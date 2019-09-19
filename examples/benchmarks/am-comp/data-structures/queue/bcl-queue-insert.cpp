#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>

int main(int argc, char** argv) {
  BCL::init(16);
  size_t num_ops = 100000;

  std::vector<BCL::FastQueue<int>> queues;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    queues.push_back(BCL::FastQueue<int>(rank, num_ops*2));
  }

  srand48(BCL::rank());
  BCL::barrier();

  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();

    queues[remote_proc].push(BCL::rank());
  }
  
  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double duration_us = 1e6*duration;
  double latency_us = duration_us / num_ops;

  BCL::print("Latency is %lf us per AM. (Finished in %lf s)\n", latency_us, duration);

  BCL::finalize();
  return 0;
}
