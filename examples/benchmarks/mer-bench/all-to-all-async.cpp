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
  std::vector<BCL::GlobalPtr<int>> counters(BCL::nprocs(), nullptr);

  // Global data size, in bytes.
  size_t global_data_size = 256*1024*size_t(1024);
  // Number of "inserts" to perform, per processor
  size_t num_inserts = 1000;
  // Transfer Size `S` of each "insert", in bytes.
  size_t transfer_data_size = 1000;

  size_t global_size = global_data_size / sizeof(T);
  size_t local_size = (global_size + BCL::nprocs() - 1) / BCL::nprocs();
  size_t transfer_size = transfer_data_size / sizeof(T);

  assert(transfer_size > 0);

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    if (BCL::rank() == i) {
      ptrs[i] = BCL::alloc<int>(local_size);
      counters[i] = BCL::alloc<int>(1);
    }
    ptrs[i] = BCL::broadcast(ptrs[i], i);
    counters[i] = BCL::broadcast(counters[i], i);

    if (ptrs[i] == nullptr || counters[i] == nullptr) {
      throw std::runtime_error("Ran out of memory.");
    }
  }

  *counters[BCL::rank()].local() = 0;

  std::vector<T> src(transfer_size);
  for (auto& val : src) {
    val = lrand48();
  }

  srand48(BCL::rank());

  std::vector<BCL::request> requests;

  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_inserts; i++) {
    // Pick a random processor p'
    size_t dest_rank = lrand48() % BCL::nprocs();

    // Perform a remote atomic `fetch_and_add()` on p's integer variable.
    BCL::fetch_and_op<int>(counters[dest_rank], 1, BCL::plus<int>{});

    size_t rand_loc = lrand48() % (local_size - transfer_size);

    auto request = BCL::arput(ptrs[dest_rank] + rand_loc, src.data(), transfer_size);
    requests.emplace_back(std::move(request));
  }

  for (auto& request : requests) {
    request.wait();
  }

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  size_t data_transferred = transfer_size*num_inserts*BCL::nprocs();
  double bw = data_transferred / duration;
  double bw_gb = bw*1e-9;

  BCL::print("All-to-all benchmark completed in %lfs.\n", duration);
  BCL::print("Total bandwidth %lf GB/s\n", bw_gb);
  BCL::print("Bandwidth/process %lf GB/s\n", bw_gb/BCL::nprocs());

  BCL::finalize();
  return 0;
}
