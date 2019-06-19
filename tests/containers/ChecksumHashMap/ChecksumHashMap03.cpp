#include <string>
#include <cassert>
#include <array>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumHashMap.hpp>

int main(int argc, char** argv) {
  BCL::init();

//  srand48(BCL::rank());
//
//  using array_t = std::array<int, 100>;
//
//  BCL::ChecksumHashMap<int, array_t> map(1000);
//
//  array_t my_array;
//
//  my_array.fill(BCL::rank());
//
//  bool success = map.insert(BCL::rank(), my_array);
//  assert(success);
//
//  BCL::barrier();
//
//  for (size_t i = 0; i < BCL::nprocs(); i++) {
//    auto result = map.find((int) i);
//    assert(result != map.end());
//    array_t remote_array = *result;
//
//    for (size_t i = 0; i < remote_array.size(); i++) {
//      if (i > 0) {
//        assert(remote_array[i] == remote_array[i-1]);
//        if (remote_array[i] != remote_array[i-1]) {
//          fprintf(stderr, "Remote array not consistent.\n");
//        }
//      }
//    }
//
//    success = map.insert(i, my_array).second;
//    assert(success);
//  }

  BCL::finalize();
  return 0;
}
