#include <bcl/bcl.hpp>

#include <unordered_map>
#include <cstring>

#include "am_utils.hpp"

std::unordered_map<int, double> map;

int main(int argc, char** argv) {
  BCL::init();

  init_am();

  auto caller = register_am([](size_t key, double val) -> void {
    printf("Received %lu, %lf\n", key, val);
    map[key] += val;
  }, size_t(), double());

  caller.launch(0, 12, 0.5);

  flush_am();
  BCL::barrier();
  fflush(stdout);
  BCL::barrier();

  if (BCL::rank() == 0) {
    printf("Printing:\n");
    for (auto val : map) {
      std::cout << val.first << " " << val.second << std::endl;
    }
  }

  BCL::finalize();
  return 0;
}
