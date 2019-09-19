#include <bcl/bcl.hpp>
#include <queue>

std::queue<int> queue;

int main(int argc, char** argv) {
  BCL::init();

  BCL::gas::init_am();

  auto insert = BCL::gas::register_am([](int value) -> void {
    queue.push(value);
  }, int());

  size_t num_ams = 100000;
  srand48(BCL::rank());
  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();

    insert.launch(remote_proc, BCL::rank());
    BCL::gas::flush_am();
  }
  
  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double duration_us = 1e6*duration;
  double latency_us = duration_us / num_ams;

  BCL::print("Latency is %lf us per AM. (Finished in %lf s)\n", latency_us, duration);

  BCL::finalize();
  return 0;
}
