#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <cassert>

#include <bcl/bcl.hpp>

int main(int argc, char** argv) {
  BCL::init();
  using T = int;

  std::vector<BCL::GlobalPtr<T>> ptrs(BCL::nprocs(), nullptr);

  // Global data size, in bytes.
  size_t global_data_size = 256*1024*size_t(1024);
  // Number of ops ("hash table accesses") to perform, per processor
  size_t num_ops = 1000;

  size_t global_size = global_data_size / sizeof(T);
  size_t local_size = (global_size + BCL::nprocs() - 1) / BCL::nprocs();

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    if (BCL::rank() == i) {
      ptrs[i] = BCL::alloc<int>(local_size);
    }
    ptrs[i] = BCL::broadcast(ptrs[i], i);

    if (ptrs[i] == nullptr) {
      throw std::runtime_error("Ran out of memory.");
    }
  }

  srand48(BCL::rank());

  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ops; i++) {
    // Pick a random processor p'
    size_t dest_rank = lrand48() % BCL::nprocs();
    size_t rand_loc = lrand48() % local_size;

    // Perform a global atomic `compare_and_swap()` on an integer variable
    BCL::compare_and_swap<int>(ptrs[dest_rank] + rand_loc, 0, 1);
  }

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  double latency = duration / num_ops;
  double latency_us = latency*1e6;

  BCL::print("Atomics latency benchmark completed in %lfs.\n", duration);
  BCL::print("Measured latency %lf us\n", latency_us);

  BCL::finalize();
  return 0;
}
