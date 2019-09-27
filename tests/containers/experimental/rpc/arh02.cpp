#include "bcl/bcl.hpp"
#include "bcl/containers/experimental/rpc_oneway/arh.hpp"
#include <cassert>

int fn(int a, int b) {
  return a * b;
}

void worker() {

  int my_rank = (int) ARH::my_worker();

//  auto fn = [](int a, int b) -> int {
//    return a * b;
//  };
//  int a = 7;
//  int b = 7;

  using rv = decltype(ARH::rpc(0, fn, my_rank, my_rank));
  std::vector<rv> futures;

  for (int i = 0 ; i < 10; i++) {
    size_t target_rank = rand() % ARH::nprocs();
    auto f = ARH::rpc(target_rank, fn, my_rank, my_rank);
    futures.push_back(std::move(f));
  }

  ARH::flush_am();

  for (auto& f : futures) {
    int val = f.wait();
    assert(val == my_rank*my_rank);
  }
  std::printf("worker %lu finished\n", ARH::my_worker());
}

int main(int argc, char** argv) {
  // one process per node
  BCL::init();
  ARH::init_am();

  ARH::run(worker);

  BCL::finalize();
}