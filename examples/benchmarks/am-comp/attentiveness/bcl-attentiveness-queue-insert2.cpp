#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>

void compute_by_time(double time_us);
long compute_by_work(double time_us);
void warmup(size_t num_ops);
double calculate_workload_us(double workload);

int main(int argc, char** argv) {
  size_t num_ops = 100000;
  double compute_array[] = {0, 0.5, 1, 2, 4, 8, 16, 32, 64};
  BCL::init(16);

  warmup(num_ops);

  for (auto compute_workload : compute_array) {
    std::vector<BCL::FastQueue<int>> queues;
    long t = 0;
    // calculate compute time
//    double workload_us = calculate_workload_us(compute_workload);

    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
      queues.push_back(BCL::FastQueue<int>(rank, num_ops*2));
    }

    srand48(BCL::rank());
    BCL::barrier();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_ops; i++) {
      size_t remote_proc = lrand48() % BCL::nprocs();

      queues[remote_proc].push(BCL::rank());

      t = compute_by_work(compute_workload);
    }

    BCL::barrier();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    double duration_us = 1e6*duration;
//    double latency_us = (duration_us - workload_us*num_ops) / num_ops;
    double latency_us = duration_us / num_ops;

//    BCL::print("Compute time is %.2lf us per op. t = %ld\n", workload_us, t);
    BCL::print("Compute workload is %.2lf. t = %ld\n", compute_workload, t);
    BCL::print("Latency is %.2lf us per op. (Finished in %.2lf s)\n",
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

long compute_by_work(double workload) {
  long workload_unit = 1000;
  long a = 1, b = 1, c = 0;
  for (long i = 0; i < workload_unit*workload; ++i) {
    c = a + b;
    a = b;
    b = c;
  }
  return b;
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
  }

  BCL::barrier();
}

double calculate_workload_us(double workload) {
  size_t num_ops = 100000;
  long t = 0;
  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  for (size_t i = 0; i < num_ops; i++) {
    t = compute_by_work(workload);
  }

  BCL::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double duration_us = 1e6 * duration;
  double latency_us = duration_us / num_ops;
  BCL::print("t = %ld\n", t);

  return latency_us;
}