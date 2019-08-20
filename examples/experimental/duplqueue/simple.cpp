#include <bcl/bcl.hpp>
#include <bcl/containers/DuplQueue.hpp>

int main(int argc, char** argv) {
  BCL::init(1024);
  size_t num_inserts = 100;
  BCL::DuplQueue<int> queue(0, 1024);

  for (size_t i = 0; i < num_inserts; i++) {
    queue.push_nonatomic(BCL::rank());
  }

  BCL::barrier();

  if (BCL::rank() == 0) {
    std::unordered_map<int, size_t> counts;
    int value;
    size_t total_popped = 0;
    while (queue.pop(value)) {
      assert(value >= 0 && value < BCL::nprocs());
      counts[value] += 1;
      total_popped++;
    }

    assert(total_popped == num_inserts*BCL::nprocs());

    for (size_t i = 0; i < BCL::nprocs(); i++) {
      if (counts[i] != num_inserts) {
        fprintf(stderr, "%lu: %lu != %lu inserts\n", i, counts[i], num_inserts);
      }
      assert(counts[i] == num_inserts);
    }
  }

  BCL::finalize();
  return 0;
}
