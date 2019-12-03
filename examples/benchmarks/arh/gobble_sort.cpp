#ifdef GASNET_EX
#define ARH_PROFILE
#include "bcl/containers/experimental/arh/arh.hpp"
#include <cassert>
#include "include/cxxopts.hpp"
#include <random>

struct ThreadObjects {
  std::vector<std::atomic<int>> v;
};

ARH::WorkerObject<ThreadObjects> mObjects;

void histogram_handler(int idx) {
  mObjects.get().v[idx] += 1;
}

void worker() {
  // Sort 16M integers
  constexpr size_t n_vals = 1 << 24;

  // Generate random values between [0, 256M)
  constexpr size_t range = 1 << 28;
  size_t partition_size = (range + ARH::nworkers() - 1) / ARH::nworkers();

  ARH::print("Sorting %lu integers among %lu workers with aggregation size %lu\n",
          n_vals, ARH::nworkers(), ARH::get_agg_size());

  // Set up the random engine and generate values
  std::default_random_engine generator(ARH::nworkers());
  std::uniform_int_distribution<int> distribution(1, range);

  std::vector<int> vals;

  for (int i = 0; i < n_vals; i++) {
    vals.push_back(distribution(generator) % range);
  }

  // Initialize the buckets
  mObjects.get().v = std::vector<std::atomic<int>>(partition_size);

  std::vector<ARH::Future<void> > futures;

  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  for (const auto& val : vals) {
    size_t target_rank = val / partition_size;

    auto future = ARH::rpc_agg(target_rank, histogram_handler, val);
    futures.push_back(std::move(future));
  }

  ARH::barrier();

  for (auto& future: futures) {
    future.get();
  }

  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e9;
  ARH::print("%lf seconds\n", duration);
}

int main(int argc, char** argv) {
  // one process per node
  size_t agg_size = 0;
  cxxopts::Options options("ARH Benchmark", "Benchmark of ARH system");
  options.add_options()
      ("size", "Aggregation size", cxxopts::value<size_t>())
      ;
  auto result = options.parse(argc, argv);
  try {
    agg_size = result["size"].as<size_t>();
  } catch (...) {
    agg_size = 0;
  }
  ARH_Assert(agg_size >= 0, "");

  ARH::init(15, 16);
  mObjects.init();
  agg_size = ARH::set_agg_size(agg_size);

  ARH::run(worker);

  ARH::finalize();
}
#else
#include <iostream>
using namespace std;
int main() {
  cout << "Only run arh test with GASNET_EX" << endl;
  return 0;
}
#endif
