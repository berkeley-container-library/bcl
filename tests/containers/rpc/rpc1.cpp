#include <cassert>

#include<stdio.h>

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


  // basic buffered RPC test
  // if (BCL::rank() == 0) {
  for (int i = 0 ; i < 400; i++) {
    BCL::buffered_rpc(i % 2, fn, a, b);
  }
    // BCL::buffered_rpc(1, fn, a, b);
    // BCL::buffered_rpc(1, fn, a, b);
    // BCL::buffered_rpc(1, fn, a, b);
  // }

  // BCL::flush_signal();
  // basic RPC test
  // rpc_t test_rpc(0);
  // test_rpc.load(fn, a, b);
  // run_rpc(test_rpc);


  BCL::barrier();

  BCL::finalize_rpc();
  BCL::finalize();
  return 0;
}
