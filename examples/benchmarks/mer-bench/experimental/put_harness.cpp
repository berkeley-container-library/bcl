#include <bcl/bcl.hpp>
#include <chrono>
#include "queue_impls.hpp"

int main(int argc, char** argv) {
  BCL::init(1024);

  size_t write_size_bytes;
  if (argc >= 2) {
    write_size_bytes = std::atoll(argv[1]);
  } else {
    BCL::print("usage: ./put_harness [write size in bytes]\n");
    BCL::finalize();
    return 0;
  }

  using T = int;

  std::vector<BCL::GlobalPtr<T>> ptrs(BCL::nprocs());
  std::vector<BCL::GlobalPtr<int>> counters(BCL::nprocs());

  size_t local_segment_size_bytes = 512*1024*1024;

  size_t local_segment_size = (sizeof(T) + local_segment_size_bytes - 1) / sizeof(T);
  size_t inserts_per_proc = 256*1024;
  size_t write_size = (sizeof(T) + write_size_bytes - 1) / sizeof(T);

  BCL::print("Creating local segment size of %lu, write size %lu (%lu bytes), number of writes %lu\n",
             local_segment_size, write_size, write_size*sizeof(T), inserts_per_proc);

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    if (BCL::rank() == i) {
      ptrs[i] = BCL::alloc<T>(local_segment_size);
      counters[i] = BCL::alloc<int>(1);
      *counters[i].local() = 0;

      if (ptrs[i] == nullptr || counters[i] == nullptr) {
        throw std::runtime_error("Ran out of memory!");
      }
    }
    ptrs[i] = BCL::broadcast(ptrs[i], i);
    counters[i] = BCL::broadcast(counters[i], i);
  }

  std::vector<T> buffer(write_size, BCL::rank());

  srand48(BCL::rank());

  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  relaxed_send(ptrs, counters, buffer,
              local_segment_size, inserts_per_proc, write_size);

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double data_sent = inserts_per_proc*sizeof(T)*write_size*BCL::nprocs();
  double data_sent_gb = 1e-9*data_sent;

  double bw_gb = data_sent_gb / duration;

  BCL::print("%lf GB/s\n", bw_gb);
  BCL::print("Runtime %lf s\n", duration);

  BCL::finalize();
  return 0;
}
