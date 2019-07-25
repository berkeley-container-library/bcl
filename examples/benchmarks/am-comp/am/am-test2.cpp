#include <bcl/bcl.hpp>

#include <unordered_map>
#include <cstring>

// NOTE: this will only compile with the GASNet-EX BCL backend.

std::unordered_map<int, double> map;

int main(int argc, char** argv) {
  BCL::init();

  BCL::gas::init_am();

  auto caller = BCL::gas::register_am([](size_t key, double val) -> void {
    printf("Received %lu, %lf\n", key, val);
    map[key] += val;
  }, size_t(), double());

  caller.launch(0, 12, 0.5);

  BCL::gas::flush_am();
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
