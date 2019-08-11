#include <bcl/bcl.hpp>
#include <queue>
#include <limits>
#include <experimental/random>

using value_type = std::size_t;

std::unordered_map<value_type, value_type> map;

int main(int argc, char** argv) {
  BCL::init();

  BCL::gas::init_am();


  auto insert = BCL::gas::register_am([](value_type key, value_type value) -> void {
    map[key] = value;
  }, value_type(), value_type());

  size_t num_ams = 100000;
  size_t key_space = 1000000;
  size_t keys_per_n = (key_space + BCL::nprocs() - 1) / BCL::nprocs();

  srand48(BCL::rank());

  size_t keys[num_ams];
  for (size_t i = 0; i < num_ams; i++) {
    keys[i] = lrand48() % key_space;
  }

  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ams; i++) {
    size_t value = keys[i];
    size_t remote_proc = value / keys_per_n;

    insert.launch(remote_proc, value, value);
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
