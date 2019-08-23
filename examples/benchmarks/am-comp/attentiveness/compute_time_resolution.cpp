#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>

void compute_by_time(double time_us);
void compute_by_work(double time_us);
void warmup(size_t num_ops);

int main(int argc, char** argv) {
  size_t num_ops = 50000;
  double compute_array[] = {0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  BCL::init(16);

  warmup(num_ops);

  for (auto compute_us : compute_array) {
    std::vector<BCL::FastQueue<int>> queues;

    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
      queues.push_back(BCL::FastQueue<int>(rank, num_ops*2));
    }

    srand48(BCL::rank());
    BCL::barrier();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_ops; i++) {
//      size_t remote_proc = lrand48() % BCL::nprocs();

//      queues[remote_proc].push(BCL::rank());

      compute_by_time(compute_us);
    }

    BCL::barrier();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    double duration_us = 1e6*duration;
    double latency_us = duration_us / num_ops - compute_us;

    BCL::print("Compute time is %.2lf us per op\n", compute_us);
    BCL::print("Overhead is %lf us per op. (Finished in %lf s)\n",
               latency_us, duration);
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

void warmup(size_t num_ops) {
  std::vector<BCL::FastQueue<int>> queues;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    queues.push_back(BCL::FastQueue<int>(rank, num_ops*2));
  }

  srand48(BCL::rank());
  BCL::barrier();

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();
    queues[remote_proc].push(BCL::rank());

    compute_by_time(0);
  }

  BCL::barrier();
}