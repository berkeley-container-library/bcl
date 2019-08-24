#include <bcl/bcl.hpp>
#include <queue>
#include <mutex>

void compute_by_time(double time_us);

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
  size_t num_ams = 50000;
  double compute_array[] = {0, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

  BCL::init();
  BCL::gas::init_am();

  auto thread_ampool = std::thread(service_ampoll);
  auto insert = BCL::gas::register_am([](int value) -> void {
      std::lock_guard<std::mutex> guard(queue_mutex);
      queue.push(value);
  }, int());
  warmup(num_ams, insert);

  BCL::barrier();

  for (auto compute_workload : compute_array) {

    srand48(BCL::rank());
    BCL::barrier();

    auto begin = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < num_ams; i++) {
      size_t remote_proc = lrand48() % BCL::nprocs();

      insert.launch(remote_proc, BCL::rank());
      BCL::gas::flush_am_nopoll();

      compute_by_time(compute_workload);
    }

    BCL::gas::flush_am();
    BCL::barrier();
    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double>(end - begin).count();
    double duration_us = 1e6*duration;
    double latency_us = (duration_us - compute_workload*num_ams) / num_ams;

    BCL::print("Compute time is %.2lf us per op.\n", compute_workload);
    BCL::print("Latency is %.2lf us per op. (Finished in %.2lf s)\n",
               latency_us, duration);
  }

  thread_run = false;
  thread_ampool.join();
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

template <typename AM>
void warmup(size_t num_ams, AM& insert) {

  srand48(BCL::rank());
  BCL::barrier();

  for (size_t i = 0; i < num_ams; i++) {
    size_t remote_proc = lrand48() % BCL::nprocs();

    insert.launch(remote_proc, BCL::rank());
    BCL::gas::flush_am_nopoll();
//    BCL::gas::flush_am();

    compute_by_time(0);
  }
  BCL::gas::flush_am();
  BCL::barrier();
}
