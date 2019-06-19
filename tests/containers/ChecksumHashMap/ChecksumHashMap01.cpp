#include <string>
#include <cassert>

#include <bcl/bcl.hpp>
#include <bcl/containers/experimental/ChecksumHashMap.hpp>

int main(int argc, char** argv) {
  BCL::init();

  BCL::ChecksumHashMap<std::string, int> map(1000);

  bool success = map.insert(std::to_string(BCL::rank()), BCL::rank());
  assert(success);

  BCL::barrier();

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      int value;
      success = map.find_value(std::to_string(i), value);
      assert(success && value == i);
    }
    std::cout << "test pass!" << std::endl;
  }

  BCL::finalize();
  return 0;
}
