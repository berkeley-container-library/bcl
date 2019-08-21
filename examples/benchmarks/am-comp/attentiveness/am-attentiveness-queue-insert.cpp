#include <bcl/bcl.hpp>
#include <queue>

void compute_by_time(double time_us);
void compute_by_work(double time_us);

std::queue<int> queue;

int main(int argc, char** argv) {
  size_t num_ams = 100000;
  double compute_array[] = {0, 0, 0.5, 1, 2, 4, 8, 16, 32};  BCL::init();

  BCL::gas::init_am();

  for (auto compute_us : compute_array) {
    auto insert = BCL::gas::register_am([](int value) -> void {
      queue.push(value);
    }, int());

    srand48(BCL::rank());
    BCL::barrier();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_ams; i++) {
      size_t remote_proc = lrand48() % BCL::nprocs();

      insert.launch(remote_proc, BCL::rank());
      BCL::gas::flush_am();

      compute_by_time(compute_us);
      // usleep(compute_us);
    }

    BCL::barrier();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();

    double duration_us = 1e6 * duration;
    double latency_us = (duration_us - compute_us*num_ams) / num_ams;

    BCL::print("Latency is %lf us per op. compute time is %lf us per op. (Finished in %lf s)\n",
               latency_us, compute_us, duration);
  }
  BCL::finalize();
  return 0;
}

void compute_by_time(double time_us) {
  double time = time_us / 1e6;
  auto begin = std::chrono::high_resolution_clock::now();
  auto now = begin;
  double duration = std::chrono::duration<double>(now - begin).count();
  while (duration < time) {
    now = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration<double>(now - begin).count();
  }
}

void compute_by_work(long workload) {
  for (long i = 0; i < workload; ++i);
}