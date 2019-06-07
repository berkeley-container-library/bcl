#include <string>
#include <cassert>
#include <cstdio>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/rpc.hpp>

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
    for (int i = 0 ; i < 10000; i++) {
      auto f = BCL::buffered_rpc(0, fn, a, b);
      futures.push_back(std::move(f));
    }
  }

  for (auto& f : futures) {
    int val = f.get();
    assert(val == a*b);
  }

  BCL::finalize_rpc();
  BCL::finalize();
  return 0;
}
