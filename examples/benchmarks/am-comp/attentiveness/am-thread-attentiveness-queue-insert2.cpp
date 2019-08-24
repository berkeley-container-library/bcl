#include <bcl/bcl.hpp>
#include <queue>

long compute_by_work(double workload);
double calculate_workload_us(double workload);

template <typename AM>
void warmup(size_t num_ams, AM& insert);

bool thread_run = true;
std::mutex queue_mutex;
std::queue<int> queue;

void service_ampoll() {
  while (thread_run) {
    gasnet_AMPoll();
  }
}

int main(int argc, char** argv) {
  size_t num_ams = 100000;
  double compute_array[] = {0, 0.5, 1, 2, 4, 8, 16, 32, 64};

  BCL::init();
  BCL::gas::init_am();

  auto thread_ampool = std::thread(service_ampoll);
  auto insert = BCL::gas::register_am([](int value) -> void {
      std::lock_guard<std::mutex> guard(queue_mutex);
      queue.push(value);
  }, int());

  warmup(num_ams, insert);

  for (auto compute_workload : compute_array) {
    long t = 0;
    // calculate compute time
//    double workload_us = calculate_workload_us(compute_workload);

    srand48(BCL::rank());
    BCL::barrier();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_ams; i++) {
      size_t remote_proc = lrand48() % BCL::nprocs();

      insert.launch(remote_proc, BCL::rank());
      BCL::gas::flush_am_nopoll();

      t = compute_by_work(compute_workload);
    }
    BCL::gas::flush_am();
    BCL::barrier();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    double duration_us = 1e6*duration;
//    double latency_us = (duration_us - workload_us*num_ams) / num_ams;
    double latency_us = duration_us / num_ams;

//    BCL::print("Compute time is %.2lf us per op. t = %ld\n", workload_us, t);
    BCL::print("Compute workload is %.2lf. t = %ld\n", compute_workload, t);
    BCL::print("Latency is %.2lf us per op. (Finished in %.2lf s)\n",
               latency_us, duration);
  }

  thread_run = false;
  thread_ampool.join();
  BCL::finalize();
  return 0;
}

long compute_by_work(double workload) {
  long workload_unit = 1000;
  long a = 1, b = 1, c = 0;
  for (long i = 0; i < workload * workload_unit; ++i) {
    c = a + b;
    a = b;
    b = c;
  }
  return b;
}

template <typename AM>
void warmup(size_t num_ams, AM& insert) {
  srand48(BCL::rank());
  BCL::barrier();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();

    insert.launch(remote_proc, BCL::rank());
    BCL::gas::flush_am_nopoll();
  }
  BCL::gas::flush_am();
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