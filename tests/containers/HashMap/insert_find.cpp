#include <string>
#include <cassert>

#include <bcl/bcl.hpp>
#include <bcl/containers/HashMap.hpp>

int main(int argc, char** argv) {
  BCL::init();

  BCL::HashMap<std::string, int> map(1000);

  auto result = map.insert_or_assign(std::to_string(BCL::rank()), BCL::rank());
  bool success = result.second;
  assert(success);

  BCL::barrier();

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < BCL::nprocs(); i++) {
      int value;
      auto val = map.find(std::to_string(i));
      assert(success);
    }
  }

  BCL::finalize();
  return 0;
}
