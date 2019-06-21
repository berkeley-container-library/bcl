#include <cassert>
#include <unordered_map>
#include <stdlib.h>     /* srand, rand */

#include <bcl/bcl.hpp>
#include <bcl/containers/CircularQueue.hpp>
#include <bcl/containers/experimental/ChecksumQueue.hpp>

#include <bcl/core/util/Backoff.hpp>

// XXX: Designed to test simultaneous multiple pushes and multiple pops.
// wait_on_overrun = false
const int PUSH_VAL = 0;
const int POP_VAL = 1;
const int PUSH_VEC = 2;
const int POP_VEC = 3;
const int PUSH_ASYNC = 4;

const size_t QUEUE_SIZE = 10;
const size_t N_STEPS = 20;
const int MAX_VAL = 5;
const size_t MAX_VEC_SIZE = 5; // must less than QUEUE_SIZE

std::vector<int> generate_rand_vec(size_t size) {
  std::vector<int> vec;
  for (int i = 0; i < size; ++i) {
    vec.push_back(rand() % MAX_VAL);
  }
  return std::move(vec);
}

int main(int argc, char** argv) {
  BCL::init();

  assert(QUEUE_SIZE >= MAX_VEC_SIZE);
  srand(time(NULL));
  constexpr bool print_verbose = true;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (print_verbose) {
      BCL::print("Rank %lu 's turn\n", rank);
    }
    BCL::ChecksumQueue<int> queue(rank, QUEUE_SIZE);
    std::unordered_map<int, int> counts;

    for (size_t i = 0; i < N_STEPS; i++) {
      int method = rand() % 2;

      switch (method) {
        case PUSH_VAL : {
          int val = rand() % MAX_VAL;
          bool success = queue.push(val);
          if (success) {
            counts[val]++;
            if (print_verbose) {
              printf("Rank %lu push %d, count %d, queue_size = %lu\n",
                      BCL::rank(), val, counts[val], queue.size());
            }
          }
          break;
        }
        case POP_VAL: {
          int val;
          bool success = queue.pop(val);
          if (success) {
            counts[val]--;
            if (print_verbose) {
              printf("Rank %lu pop %d, count %d, queue_size = %lu\n",
                      BCL::rank(), val, counts[val], queue.size());
            }
          }
          break;
        }
//        case PUSH_VEC: {
//          int size = rand() % MAX_VEC_SIZE;
//          std::vector<int> vec = generate_rand_vec(size);
//          bool success = queue.push(vec);
//          if (success) {
//            for (auto val: vec) {
//              counts[val]++;
//            }
//          }
//          break;
//        }
//        case POP_VEC: {
//          std::vector<int> vec;
//          bool success = queue.pop(vec, MAX_VEC_SIZE, true);
//          if (success) {
//            for (auto val: vec) {
//              counts[val]--;
//            }
//          }
//          break;
//        }
//        case PUSH_ASYNC:
//          break;
        default:
          break;
      }
    }
    BCL::barrier();
    printf("Rank %lu barrier\n", BCL::rank());

    while (!queue.empty()) {
      int val;
      bool success = queue.pop(val);
      if (success) {
        counts[val]--;
        if (print_verbose) {
          printf("Rank %lu pop %d, count %d, queue_size = %lu\n",
                     BCL::rank(), val, counts[val], queue.size());
        }
      }
    }

    for (int i = 0; i < MAX_VAL; ++i) {
      int count = counts[i];
      int tmp;
      int sum = 0;
      for (int j = 0; j < BCL::nprocs(); ++j) {
        if (BCL::rank() == j) {
          tmp = count;
        }
        BCL::broadcast(tmp, j);
        sum += tmp;
      }
      if (BCL::rank() == 0) {
        if (print_verbose && sum != 0) {
          BCL::print("Error! Rank %lu, val = %d, sum = %d\n", rank, i, sum);
        } else {
          assert(sum == 0);
        }
      }
    }

    if (print_verbose) {
      fprintf(stderr, "(%lu) DONE\n", BCL::rank());
    }
    BCL::barrier();
  }

  BCL::finalize();
  return 0;
}
