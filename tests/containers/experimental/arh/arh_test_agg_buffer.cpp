#ifdef GASNET_EX
#define ARH_DEBUG
#include "bcl/bcl.hpp"
#include "bcl/containers/experimental/arh/arh.hpp"
#include "bcl/containers/experimental/arh/arh_agg_buffer.hpp"
#include <sstream>

ARH::AggBuffer<int> aggBuffer;
size_t buf_size = 5;

const int MAX_VAL = 10;
const size_t N_STEPS = 5;
std::vector<std::atomic<int>> counts(MAX_VAL);

const bool print_verbose = false;

void worker() {
  using status_t = ARH::AggBuffer<int>::status_t;

  ARH::print("Initializing...\n");

  for (int i = 0; i < N_STEPS; ++i) {
    int val = lrand48() % MAX_VAL;
    status_t status = aggBuffer.push(val);
    while (status == status_t::FAIL) {
      status = aggBuffer.push(val);
    }
    // successful push
    int count = counts[val]++;
    if (print_verbose) {
      printf("Rank %lu push val: (%d, %d), buf_size = %lu\n",
             ARH::my_worker(), val, count, aggBuffer.size());
    }
    if (status == status_t::SUCCESS_AND_FULL) {
      std::vector<int> buf;
      aggBuffer.pop_all(buf);
      if (!print_verbose)
        for (auto val: buf) {
          counts[val]--;
        }
      else {
        std::ostringstream ostr;
        ostr << "Rank " << ARH::my_worker() << " pop vec: ";
        for (auto val: buf) {
          int count = --counts[val];
          ostr << "(" << val << ", " << count << "), ";
        }
        ostr << "buf_size = " << aggBuffer.size();
        std::cout << ostr.str() << std::endl;
      }
    }
  }

  ARH::barrier();
  ARH::print("Finish pushing...\n");

  if (ARH::my_worker_local() == 0) {
    std::vector<int> buf;
    size_t len = aggBuffer.pop_all(buf);
    if (!print_verbose) {
      for (int i = 0; i < len; ++i) {
        int val = buf[i];
        counts[val]--;
      }
    } else {
      std::ostringstream ostr;
      ostr << "Rank " << ARH::my_worker() << " pop vec: ";
      for (int i = 0; i < len; ++i) {
        int val = buf[i];
        int count = --counts[val];
        ostr << "(" << val << ", " << count << "), ";
      }
      ostr << "buf_size = " << aggBuffer.size();
      std::cout << ostr.str() << std::endl;
    }

    bool success = true;
    for (int i = 0; i < MAX_VAL; ++i) {
      int sum = counts[i].load();
      if (print_verbose && sum != 0) {
        printf("Error! Proc %lu, val = %d, sum = %d\n", ARH::my_proc(), i, sum);
        success = false;
      } else {
        assert(sum == 0);
      }
    }
    if (print_verbose && success) {
      printf("Pass!\n");
    } else {
      printf("Pass!\n");
    }
  }
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init(15, 16);
  aggBuffer.init(buf_size);
  for (int i = 0; i < MAX_VAL; ++i) {
    counts[i] = 0;
  }
  ARH::run(worker);

  ARH::finalize();
}
#else
#include <iostream>
using namespace std;
int main() {
  cout << "Only run arh test with GASNET_EX" << endl;
  return 0;
}
#endif