#include <cstdio>
#include <cstdlib>

#include <bcl/bcl.hpp>
#include <bcl/containers/HashMap.hpp>
#include <bcl/core/detail/hash_functions.hpp>

int main(int argc, char **argv) {
  BCL::init();

  BCL::HashMap<int, std::string, BCL::djb2_hash<int>> map(BCL::nprocs()*2);

  map[BCL::rank()] = std::to_string(BCL::rank());

  BCL::barrier();

  for (auto it = map.local_begin(); it != map.local_end(); it++) {
    typename decltype(map)::value_type local = *it;
    std::cout << BCL::rank() << " sees " << std::get<0>(local) << " " << std::get<1>(local) << std::endl;
  }

  if (BCL::rank() == 0) {
    auto it = map.find(BCL::nprocs()+1);
    if (it == map.end()) {
      printf("Not found!\n");
    } else {
      printf("Found!\n");
    }
  }

  BCL::finalize();
  return 0;
}
