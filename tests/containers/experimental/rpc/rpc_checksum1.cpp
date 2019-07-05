#include <string>
#include <cassert>

#include<stdio.h>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/rpc_checksum.hpp>

template <typename Future>
bool ready(std::vector<Future>& futures) {
  for (auto& future : futures) {
    if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      return false;
    }
  }
  return true;
}

template <typename Future>
void future_barrier(std::vector<Future>& futures) {
  bool success = false;
  do {
    BCL::flush_rpc();
    size_t success_count = ready(futures);
    success_count = BCL::allreduce<size_t>(success_count, std::plus<size_t>{});
    success = success_count == BCL::nprocs();
  } while (!success);
}

int main(int argc, char** argv) {
  BCL::init();
  BCL::init_rpc();


  auto fn = [](int a, int b) -> int {
               return a * b;
            };
  int a = 7;
  int b = 7;


  using rv = decltype(BCL::buffered_rpc(0, fn, a, b));
  std::vector<rv> futures;
  if (BCL::rank() == 1) {
    for (int i = 0 ; i < 1000; i++) {
      auto f = BCL::buffered_rpc(0, fn, a, b);
      futures.push_back(std::move(f));
    }
  }

  BCL::flush_signal(); 

  future_barrier(futures);

  for (auto& f : futures) {
    int val = f.get();
    assert(val == a*b);
  }

  BCL::finalize_rpc();
  BCL::finalize();

  return 0;
}
