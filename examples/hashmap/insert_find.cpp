#include <string>
#include <cassert>

#include <bcl/bcl.hpp>
#include <bcl/containers/HashMap.hpp>

int main(int argc, char** argv) {
  BCL::init();

  BCL::HashMap<std::string, int> map(1000);

  map[std::to_string(BCL::rank())] = BCL::rank();

  BCL::barrier();

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      int value = map[std::to_string(i)];

      printf("%lu: %d\n", i, value);
    }
  }

  BCL::finalize();
  return 0;
}
