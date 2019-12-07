#ifdef GASNET_EX
#define ARH_DEBUG
#include "bcl/containers/experimental/arh/arh.hpp"
#include <string>

void worker() {
  size_t num_ops = 5;
  size_t hashmap_size = ARH::nworkers() * num_ops * 2;
  ARH::HashMap<size_t, size_t> hashmap(hashmap_size);

  for (int i = 0; i < num_ops; ++i) {
    size_t my_val = ARH::my_worker();
    size_t my_key = my_val + i * ARH::nworkers();
    hashmap.insert(my_key, my_val).get();
  }

  ARH::barrier();

  for (int i = 0; i < num_ops; ++i) {
    size_t my_val = (ARH::my_worker() + 1) % ARH::nworkers();
    size_t my_key = my_val + i * ARH::nworkers();
    size_t result = hashmap.find(my_key).get();
    if (result != my_val) {
      printf("Error! key %lu, %lu != %lu\n", my_key, result, my_val);
      abort();
    }
  }
  ARH::print("Pass!\n");
}

int main(int argc, char** argv) {
  // one process per node
  ARH::init(2, 3);
  ARH::run(worker);

  ARH::finalize();
}
#else
#include <iostream>
using namespace std;
int main() {
  cout << "Only run arh test with GASNET_EX" << endl;
  return 0;
}
#endif