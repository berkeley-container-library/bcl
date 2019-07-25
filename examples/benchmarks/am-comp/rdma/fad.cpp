#include <bcl/bcl.hpp>
#include <chrono>

int main(int argc, char** argv) {
  BCL::init(16);

  printf("%lu / %lu on %s\n", BCL::rank(), BCL::nprocs(), BCL::hostname().c_str());

  BCL::barrier();
  fflush(stdout);
  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  constexpr size_t num_ops = 100000;
  constexpr size_t local_size = 100;

  std::vector<BCL::GlobalPtr<int>> ptr(BCL::nprocs(), nullptr);

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (BCL::rank() == rank) {
      ptr[rank] = BCL::alloc<int>(local_size);
    }
    ptr[rank] = BCL::broadcast(ptr[rank], rank);
  }

  for (size_t i = 0; i < local_size; i++) {
    ptr[BCL::rank()].local()[i] = 0;
  }

  srand48(BCL::rank());

  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_rank = lrand48() % BCL::nprocs();
    size_t remote_offset = lrand48() % local_size;

    auto ptr_ = ptr[remote_rank] + remote_offset;

    int rv = BCL::fetch_and_op(ptr_, 1, BCL::plus<int>{});
  }

  BCL::barrier();

  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double latency = duration / num_ops;
  double latency_us = latency*1e6;

  BCL::print("Latency is %lf us / op.\n", latency_us);
  BCL::print("Completed in %lf s.\n", duration);

  BCL::finalize();
  return 0;
}
