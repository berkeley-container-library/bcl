#include <bcl/bcl.hpp>

#include <unordered_map>
#include <chrono>

// NOTE: this will only compile with the GASNet-EX BCL backend.

std::unordered_map<int, int> map;

int main(int argc, char** argv) {
  BCL::init();

  BCL::gas::init_am();

  // Register active message handler.

  auto caller = BCL::gas::register_am([](int key, int val) -> void {
    map[key] += val;
  }, int(), int());

  size_t num_ams = 100000;
  if(argc == 1)
    BCL::print("./unordered map add -n <number of put operations = 10000>\n");
  else if(argc > 1)
  {
    for(int i=1; i<argc; i++)
    {
       if(std::string(argv[i]) == "-n")
	 num_ams = std::stoll(argv[i+1]);
    }
  }
  BCL::print("run with: num_ops %llu\n", num_ams);

  srand48(BCL::rank());
  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();

    caller.launch(remote_proc, 1, 1);
    BCL::gas::flush_am();
  }
  
  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double duration_us = 1e6*duration;

  double latency_us = duration_us / num_ams;

  if (BCL::rank() == 0) {
    printf("Printing:\n");
    for (auto val : map) {
      std::cout << val.first << " " << val.second << std::endl;
    }
  }

  BCL::print("Latency is %lf us per AM. (Finished in %lf s)\n", latency_us, duration);

  BCL::finalize();
  return 0;
}
