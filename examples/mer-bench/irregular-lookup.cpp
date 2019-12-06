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
  // Number of "lookups" to perform, per processor
  size_t num_lookups = 1000;
  // Transfer Size `S` of each "insert", in bytes.
  size_t transfer_data_size = 1000;

  size_t global_size = global_data_size / sizeof(T);
  size_t local_size = (global_size + BCL::nprocs() - 1) / BCL::nprocs();
  size_t transfer_size = transfer_data_size / sizeof(T);

  assert(transfer_size > 0);

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

  for (size_t i = 0; i < num_lookups; i++) {
    // Pick a random processor p'
    size_t dest_rank = lrand48() % BCL::nprocs();

    size_t rand_loc = lrand48() % (local_size - transfer_size);

    auto fut = BCL::arget(ptrs[dest_rank] + rand_loc, transfer_size);
    fut.get();
  }

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  size_t data_transferred = transfer_size*num_lookups*BCL::nprocs();
  double bw = data_transferred / duration;
  double bw_gb = bw*1e-9;

  BCL::print("Irregular Lookup benchmark completed in %lfs.\n", duration);
  BCL::print("Total bandwidth %lf GB/s\n", bw_gb);
  BCL::print("Bandwidth/process %lf GB/s\n", bw_gb/BCL::nprocs());

  BCL::finalize();
  return 0;
}
