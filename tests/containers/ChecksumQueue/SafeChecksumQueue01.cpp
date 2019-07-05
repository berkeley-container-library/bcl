#include <cassert>
#include <unordered_map>
#include <stdlib.h>     /* srand, rand */
#include <sstream>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/SafeChecksumQueue.hpp>

#include <bcl/core/util/Backoff.hpp>

// XXX: Designed to test simultaneous multiple pushes and multiple pops.

const int TEST_OP = 2;
const int PUSH_VAL = 0;
const int POP_VAL = 1;
const int PUSH_VEC = 2;
const int POP_VEC = 3;
const int PUSH_ASYNC = 4;

const size_t QUEUE_SIZE = 100;
const size_t N_STEPS = 5;
const int MAX_VAL = 10;
const size_t MAX_VEC_SIZE = 20; // must less than QUEUE_SIZE
const bool print_verbose = false;

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
  srand(time(NULL) + BCL::rank());

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (print_verbose) {
      BCL::print("Rank %lu 's turn\n", rank);
    }
    BCL::ChecksumQueue<int> queue(rank, QUEUE_SIZE);
    std::vector<int> counts(MAX_VAL, 0);

    for (size_t i = 0; i < N_STEPS; i++) {
      int method = rand() % TEST_OP;

      switch (method) {
        case PUSH_VAL : {
          int val = rand() % MAX_VAL;
          bool success = queue.push(val);
          if (success) {
            counts[val]++;
            if (print_verbose) {
              printf("Rank %lu push val: (%d, %d), queue_size = %lu\n",
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
              printf("Rank %lu pop val: (%d, %d), queue_size = %lu\n",
                      BCL::rank(), val, counts[val], queue.size());
            }
          }
          break;
        }
        case PUSH_VEC: {
          int size = rand() % MAX_VEC_SIZE;
          std::vector<int> vec = generate_rand_vec(size);
          bool success = queue.push(vec);
          if (success) {
            if (!print_verbose)
              for (auto val: vec) {
                counts[val]++;
              }
            else {
              std::ostringstream ostr;
              ostr << "Rank " << BCL::rank() << " push vec: ";
              for (auto val: vec) {
                counts[val]++;
                ostr << "(" << val << ", " << counts[val] << "), ";
              }
              ostr << "queue_size = " << queue.size();
              std::cout << ostr.str() << std::endl;
            }
          }
          break;
        }
        case POP_VEC: {
          std::vector<int> vec;
          bool success = queue.pop(vec, MAX_VEC_SIZE); // take fewer is unsafe on multi-pops
          if (success) {
            if (!print_verbose)
              for (auto val: vec) {
                counts[val]--;
              }
            else {
              std::ostringstream ostr;
              ostr << "Rank " << BCL::rank() << " pop vec: ";
              for (auto val: vec) {
                counts[val]--;
                ostr << "(" << val << ", " << counts[val] << "), ";
              }
              ostr << "queue_size = " << queue.size();
              std::cout << ostr.str() << std::endl;
            }
          }
          break;
        }
        case PUSH_ASYNC:
          break;
        default:
          break;
      }
    }
    BCL::barrier();
    if (print_verbose) {
      printf("Rank %lu barrier\n", BCL::rank());
    }

    while (true) {
      int val;
      bool success = queue.pop(val);
      if (success) {
        counts[val]--;
        if (print_verbose) {
          printf("Rank %lu pop val: (%d, %d), queue_size = %lu\n",
                 BCL::rank(), val, counts[val], queue.size());
        }
      }
      else {
        break;
      }
    }
    BCL::barrier();
    if (print_verbose) {
      printf("Rank %lu enter counting\n", BCL::rank());
    }

    for (int i = 0; i < MAX_VAL; ++i) {
      int count = counts[i];
      int tmp = 0;
      int sum = 0;
      for (int j = 0; j < BCL::nprocs(); ++j) {
        if (BCL::rank() == j) {
          tmp = count;
        }
        tmp = BCL::broadcast(tmp, j);
        sum += tmp;
      }
      if (BCL::rank() == 0) {
        if (print_verbose && sum != 0) {
          printf("Error! Rank %lu, val = %d, sum = %d\n", rank, i, sum);
        } else {
          assert(sum == 0);
        }
      }
    }

    if (print_verbose) {
      fprintf(stderr, "(%lu) DONE\n", BCL::rank());
    }
//    BCL::barrier();
  }

  BCL::finalize();
  return 0;
}
