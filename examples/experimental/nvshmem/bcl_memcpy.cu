#include <bcl/bcl.hpp>
#include <bcl/backends/experimental/nvshmem/backend.hpp>

void print_values(std::vector<BCL::cuda::ptr<int>>& ptrs) {
  if (BCL::rank() == 0) {
    printf("Process 0 sees:\n");
    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
      printf("%lu:", rank);
      for (size_t i = 0; i < BCL::nprocs(); i++) {
        int val;
        BCL::cuda::memcpy(&val, ptrs[rank] + i, sizeof(int));
        printf(" %d", val);
      }
      printf("\n");
    }
  }
}

int main(int argc, char** argv) {
  BCL::init();

  printf("Hello, world! I am rank %lu/%lu\n",
         BCL::rank(), BCL::nprocs());

  BCL::cuda::init();

  std::vector<BCL::cuda::ptr<int>> ptrs(BCL::nprocs());

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    if (BCL::rank() == i) {
      ptrs[i] = BCL::cuda::alloc<int>(BCL::nprocs());
    }
    ptrs[i] = BCL::broadcast(ptrs[i], i);
  }

  BCL::cuda::barrier();

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    int val = BCL::rank();
    BCL::cuda::memcpy(ptrs[BCL::rank()] + i, &val, sizeof(int));
  }

  BCL::cuda::barrier();

  print_values(ptrs);

  BCL::cuda::barrier();

  for (size_t origin_rank = 0; origin_rank < BCL::nprocs(); origin_rank++) {
    BCL::print("Origin rank is %lu\n", origin_rank);
    if (BCL::rank() == origin_rank) {
      for (size_t i = 0; i < BCL::nprocs(); i++) {
        int val = BCL::rank();
        BCL::cuda::memcpy(ptrs[BCL::rank()] + i, &val, sizeof(int));
      }
      for (size_t dst_rank = 0; dst_rank < BCL::nprocs(); dst_rank++) {
        // XXX: does not currently work
        BCL::cuda::memcpy(ptrs[dst_rank], ptrs[origin_rank], sizeof(int)*BCL::nprocs());
      }
    }
    BCL::cuda::barrier();
    print_values(ptrs);
    BCL::cuda::barrier();
  }

  BCL::finalize();
  return 0;
}
