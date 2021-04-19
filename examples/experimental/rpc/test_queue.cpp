#include <bcl/bcl.hpp>
#include <string>
#include <cstdio>
#include <bcl/containers/CircularQueue.hpp>

int main(int argc, char** argv) {
  BCL::init(256, true, true);
  BCL::CircularQueue<std::string> queue(0, 10*BCL::nprocs());

  if (BCL::rank() != 0) {
    for (size_t i = 0; i < 10; i++) {
      queue.push(std::string("this is the ") + std::to_string(i) + "th long string from " + std::to_string(BCL::rank()));
    }
  } else {
    for (size_t i = 0; i < (BCL::nprocs()-1)*10; i++) {
      std::string buf;
      bool success = false;
      while (!success) { success = queue.pop(buf); usleep(100); }
      printf("Popped \"%s\"\n", buf.c_str());
    }
  }

  BCL::finalize();
  return 0;
}
