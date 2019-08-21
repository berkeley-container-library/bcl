#include <bcl/bcl.hpp>
#include <chrono>

int main(int argc, char** argv) {
  BCL::init(16);

  size_t num_ops = 1000000;
  constexpr size_t local_size = 100000;
  constexpr size_t message_size_bytes = 4;
  constexpr size_t message_size = (message_size_bytes + sizeof(int) - 1) / sizeof(int);

  if(argc == 1)
    BCL::print("./get -n <number of put operations = 10000>\n");
  else if(argc > 1)
  {
    for(int i=1; i<argc; i++)
    {
       if(std::string(argv[i]) == "-n")
	 num_ops = std::stoll(argv[i+1]);
    }
  }
  BCL::print("run with: num_ops %llu, local_size %llu, message_size_bytes %llu\n", num_ops, local_size, message_size_bytes);


  std::vector<BCL::GlobalPtr<int>> ptr(BCL::nprocs(), nullptr);

  std::vector<int> local_buffer(message_size, BCL::rank());

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
    size_t remote_offset = lrand48() % (local_size - message_size + 1);

    auto ptr_ = ptr[remote_rank] + remote_offset;

    BCL::rget(ptr_, local_buffer.data(), message_size);
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
