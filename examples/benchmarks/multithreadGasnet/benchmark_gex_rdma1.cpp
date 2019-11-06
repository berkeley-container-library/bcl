#include "bcl/containers/experimental/arh/arh.hpp"
#include "bcl/containers/experimental/arh/arh_gex.hpp"
#include "bcl/containers/experimental/arh/arh_tools.hpp"
#include <sched.h>
#include <queue>
#include <mutex>
#include <memory>
#include "include/cxxopts.hpp"

size_t shared_seg_size = 4 * 1e6;
void* shared_seg_ptr = NULL;
bool full_mode = false;

void init_shared_memory(size_t custom_seg_size = 4 * 1e6) {
  shared_seg_size = custom_seg_size;
  shared_seg_ptr = malloc(shared_seg_size);
}

auto rput_int_handler = [](size_t offset, int val) {
  if (offset >= shared_seg_size) {
    // do nothing
  }
  char* dest_ptr = (char*) shared_seg_ptr + offset;
  *reinterpret_cast<int*>(dest_ptr) = val;
};

auto rget_int_handler = [](size_t offset) -> int {
  if (offset >= shared_seg_size) {
    return 0;
  }
  char* dest_ptr = (char*) shared_seg_ptr + offset;
  return *reinterpret_cast<int*>(dest_ptr);
};

// using rput_int_handler_t = BCL::gas::launch_am<decltype(rput_int_handler), size_t(), int()>;
using rput_int_handler_t = decltype(BCL::gas::register_am(rput_int_handler, size_t(), int()));
// using rget_int_handler_t = BCL::gas::launch_2wayam<decltype(rget_int_handler), size_t()>;
using rget_int_handler_t = decltype(BCL::gas::register_2wayam(rget_int_handler, size_t()));
rput_int_handler_t* rput_ptr;
rget_int_handler_t* rget_ptr;

void bench_rput() {
  size_t num_ops = 1000;
  size_t rank = ARH::my_worker();

  srand48(ARH::my_worker());
  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  rput_int_handler_t& rput = *rput_ptr;
  rget_int_handler_t& rget = *rget_ptr;

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_proc = lrand48() % ARH::nprocs();
    while (remote_proc == ARH::my_proc()) {
      remote_proc = lrand48() % ARH::nprocs();
    }

    rput.launch(remote_proc, rank * sizeof(int), (int)rank);
    BCL::gas::flush_am();
  }

  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e3;
  double latency = duration / num_ops;
  ARH::print("rput Setting: full_mode = %d; duration = %.2lf s; num_ops = %lu\n", full_mode, duration / 1e6, num_ops);
  ARH::print("latency: %.2lf us\n", latency);
}

void bench_rget() {
  size_t num_ops = 1000;
  size_t rank = ARH::my_worker();

  rput_int_handler_t& rput = *rput_ptr;
  rget_int_handler_t& rget = *rget_ptr;

  srand48(ARH::my_worker());
  ARH::barrier();
  ARH::tick_t start = ARH::ticks_now();

  for (size_t i = 0; i < num_ops; i++) {
    size_t remote_proc = lrand48() % ARH::nprocs();
    while (remote_proc == ARH::my_proc()) {
      remote_proc = lrand48() % ARH::nprocs();
    }

    auto fut = rget.launch(remote_proc, rank * sizeof(int));
    BCL::gas::flush_am();
    int val = fut.get();
    assert(val == rank);
  }

  ARH::barrier();
  ARH::tick_t end = ARH::ticks_now();

  double duration = ARH::ticks_to_ns(end - start) / 1e3;
  double latency = duration / num_ops;
  ARH::print("rget Setting: full_mode = %d; duration = %.2lf s; num_ops = %lu\n", full_mode, duration / 1e6, num_ops);
  ARH::print("latency: %.2lf us\n", latency);
}

void worker() {
  bench_rput();
  bench_rget();
}

int main(int argc, char** argv) {
  cxxopts::Options options("ARH Benchmark", "Benchmark of ARH system");
  options.add_options()
      ("full", "Enable full mode")
      ;
  auto result = options.parse(argc, argv);
  try {
    full_mode = result.count("full");
  } catch (...) {
    full_mode = false;
  }

  ARH::init(15, 16);
  BCL::gas::init_am(ARH::handler_num);
  BCL::gas::init_2wayam();

  init_shared_memory();
  rput_ptr = new rput_int_handler_t(BCL::gas::register_am(rput_int_handler, size_t(), int()));
  rget_ptr = new rget_int_handler_t(BCL::gas::register_2wayam(rget_int_handler, size_t()));

  ARH::run(worker);
  ARH::finalize();

  return 0;
}
