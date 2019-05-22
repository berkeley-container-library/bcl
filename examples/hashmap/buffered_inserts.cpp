#include <string>
#include <cassert>

#include <bcl/bcl.hpp>
#include <bcl/containers/HashMap.hpp>
#include <bcl/containers/HashMapBuffer.hpp>

int main(int argc, char** argv) {
  BCL::init();

  size_t n_to_insert = 10;

  double load_factor = 0.5;
  BCL::HashMap<std::string, int> map(n_to_insert*BCL::nprocs() / load_factor);

  BCL::HashMapBuffer<std::string, int> buffer(map,
                                              n_to_insert*2,
                                              std::max<size_t>(n_to_insert/BCL::nprocs(), 1));

  for (size_t i = 0; i < n_to_insert; i++) {
    bool success = buffer.insert(std::to_string(n_to_insert*BCL::rank() + i), i);
    assert(success);
  }

  buffer.flush();

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < BCL::nprocs(); i++) {

      for (size_t j = 0; j < n_to_insert; j++) {
        auto iter = map.find(std::to_string(n_to_insert*i + j));
        int value = *iter;
        printf("%lu: %d\n", n_to_insert*BCL::rank() + i, value);
      }
    }
  }

  BCL::finalize();
  return 0;
}
