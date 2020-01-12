#include <upcxx/upcxx.hpp>
#include <chrono>
#include <vector>

std::vector<int> queue_;

int main(int argc, char** argv) {
  upcxx::init();

  size_t write_size_bytes;
  if (argc >= 2) {
    write_size_bytes = std::atoll(argv[1]);
  } else {
    if (upcxx::rank_me() == 0) {
      printf("usage: ./put_harness [write size in bytes]\n");
    }
    upcxx::finalize();
    return 0;
  }

  using T = int;

  size_t inserts_per_proc = 256*1024;
  size_t write_size = (sizeof(T) + write_size_bytes - 1) / sizeof(T);

  std::vector<int> buffer(write_size, upcxx::rank_me());

  srand48(upcxx::rank_me());

  upcxx::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  size_t concurrency_level = 100;
  std::vector<upcxx::future<int>> futures;

  for (size_t i = 0; i < inserts_per_proc; i++) {
    size_t rand_proc = lrand48() % upcxx::rank_n();
    auto f = upcxx::rpc(rand_proc, [](std::vector<int> buffer) -> int {
                                     queue_.insert(queue_.end(),
                                                   buffer.begin(),
                                                   buffer.end());
                                     return 0;
                          }, buffer);

    futures.push_back(std::move(f));
  }

  for (auto& f : futures) {
    f.wait();
  }

  upcxx::barrier();
  auto end = std::chrono::high_resolution_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();

  double data_sent = inserts_per_proc*sizeof(T)*write_size*upcxx::rank_n();
  double data_sent_gb = 1e-9*data_sent;

  double bw_gb = data_sent_gb / duration;

  if (upcxx::rank_me() == 0) {
    printf("%lf GB/s\n", bw_gb);
    printf("Runtime %lf s\n", duration);
  }

  upcxx::finalize();
  return 0;
}
