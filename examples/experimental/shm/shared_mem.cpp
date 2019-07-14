#include <bcl/bcl.hpp>

#include <iostream>

#include "shm_segment.hpp"
#include "naive_allocator.hpp"

int main(int argc, char** argv) {
  BCL::init();

  std::string shm_key = "bcl_shmem";
  size_t segment_size = 8*1024*1024;

  shm::init_shm(shm_key, segment_size);
  shm::init_basic_malloc();

  BCL::barrier();

  using vector_type = std::vector<int, shm::allocator<int>>;

  using vector_ptr_type = decltype(shm::allocator<vector_type>{}.allocate(1));

  vector_ptr_type vector_ptr;

  if (BCL::rank() == 0) {
    vector_ptr = shm::allocator<vector_type>{}.allocate(1);
    shm::allocator<vector_type>{}.construct(vector_ptr);
  }

  BCL::barrier();

  vector_ptr->operator[](BCL::rank()) = BCL::rank();

  /*
  using map_type = std::unordered_map<int, int, std::hash<int>, std::equal_to<int>,
                     shm::allocator<std::pair<const int, int>>>;

  using map_ptr_type = decltype(shm::allocator<map_type>{}.allocate(1));

  map_ptr_type map_ptr;

  if (BCL::rank() == 0) {
    map_ptr = shm::allocator<map_type>{}.allocate(1);
    new (map_ptr.local()) map_type;

    for (size_t i = 0; i < BCL::nprocs(); i++) {
      map_ptr->operator[](i) = i;
    }
  }

  map_ptr = BCL::broadcast(map_ptr, 0);

  BCL::barrier();

  if (BCL::rank() == 1) {
    auto iter = map_ptr->find(BCL::rank());
    if (iter == map_ptr->end()) {
      std::cout << "Not found!" << std::endl;
    } else {
      std::cout << "Found " << iter->first << " " << iter->second << std::endl;
    }
  }
  */

  BCL::barrier();
  shm::finalize_shm();

  BCL::finalize();
  return 0;
}
