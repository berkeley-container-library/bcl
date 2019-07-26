#include <bcl/bcl.hpp>
#include <bcl/containers/HashMap.hpp>
#include <limits>

int main(int argc, char** argv) {
  BCL::init();

  size_t num_ops = 10000;
  size_t key_space = std::numeric_limits<int>::max();

  BCL::HashMap<int, int, BCL::djb2_hash<int>> map(2*num_ops*BCL::nprocs());

  srand48(BCL::rank());
  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ops; i++) {
    size_t value = lrand48() % key_space;

    map.insert_nonatomic_impl_(value, value);
  }

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double duration_us = 1e6*duration;
  double latency_us = duration_us / num_ops;

  BCL::print("Latency is %lf us per AM. (Finished in %lf s)\n", latency_us, duration);

  BCL::finalize();
  return 0;
}
