#include <bcl/bcl.hpp>
#include <chrono>

int main(int argc, char** argv) {
  BCL::init(16);

  size_t num_ops = 10000;
  size_t local_size = 100;
  if(argc == 1)
    BCL::print("./cas -n <number of put operations = 10000> -s <local buffer size = 100>\n");
  else if(argc > 1)
  {
    for(int i=1; i<argc; i++)
    {
       if(std::string(argv[i]) == "-n")
	 num_ops = std::stoll(argv[i+1]);
       else if(std::string(argv[i]) == "-s")
	 local_size = std::stoll(argv[i+1]);
    }
  }
  BCL::print("run with: num_ops %llu, local_size %llu\n", num_ops, local_size);

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

    int new_val = BCL::rank();
    int old_val = 0;
    // Attempt to change from old_val -> new_val
    int return_val = old_val;
    do {
      old_val = return_val;
      return_val = BCL::compare_and_swap<int>(ptr_, old_val, new_val);
    } while (return_val != old_val);
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
