#include <string>
#include <cassert>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumHashMap.hpp>

int main(int argc, char** argv) {
  const int steps = 10000;

  BCL::init();
  BCL::ChecksumHashMap<std::string, int> map(steps * BCL::nprocs() * 2);

  for (int i = 0; i < steps; i++) {
    bool success = map.insert(std::to_string(BCL::rank()) + '_' + std::to_string(i), BCL::rank());
    assert(success);
  }

  BCL::barrier();

  for (size_t i = 0; i < steps; i++) {
    int value;
    bool success = map.find_value(std::to_string(BCL::rank()) + '_' + std::to_string(i), value);
    assert(success && value == BCL::rank());
  }
  std::cout << "test pass!" << std::endl;

  BCL::finalize();
  return 0;
}
