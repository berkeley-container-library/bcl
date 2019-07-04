#include <string>
#include <cassert>
#include <cstdio>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/rpc_oneway/rpc.hpp>

template <typename Future>
bool ready(const std::vector<Future>& futures) {
  for (auto& future : futures) {
    if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      return false;
    }
  }
  return true;
}

template <typename Future>
void future_barrier(const std::vector<Future>& futures) {
  bool success = false;
  do {
    BCL::flush_rpc();
    size_t success_count = ready(futures);
    success_count = BCL::allreduce<size_t>(success_count, std::plus<>{});
    success = (success_count == BCL::nprocs());
  } while (!success);
}

/*
 * test buffered_rpc
 */
int main(int argc, char** argv) {
  BCL::init();
  BCL::init_rpc();

  srand(time(NULL) + BCL::rank());
  auto fn = [](int a, int b) -> int {
      return a * b;
  };

  int my_rank = BCL::rank();

  using rv = decltype(BCL::buffered_rpc(0, fn, my_rank, my_rank));
  std::vector<rv> futures;

  for (int i = 0 ; i < 1000; i++) {
    size_t target_rank = rand() % BCL::nprocs();
    auto f = BCL::buffered_rpc(target_rank, fn, my_rank, my_rank);
    futures.push_back(std::move(f));
  }

  future_barrier(futures);

  for (auto& f : futures) {
    int val = f.get();
    assert(val == my_rank*my_rank);
  }

  BCL::finalize_rpc();
  BCL::finalize();
  return 0;
}
