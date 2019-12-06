#include <bcl/bcl.hpp>
#include <vector>
#include <cstdlib>
#include <cstdio>
#include <iostream>

#include <futar/fn_wrapper.hpp>
#include <futar/pools/ChainPool.hpp>

int main(int argc, char** argv) {
  BCL::init(1024);
  using T = int;
  std::vector<BCL::GlobalPtr<T>> ptrs(BCL::nprocs(), nullptr);
  std::vector<BCL::GlobalPtr<int>> counters(BCL::nprocs(), nullptr);

  if (argc < 5) {
    BCL::print("usage: ./put_bench [hash_size (bytes)] [message_size (bytes)] [num_inserts] [concurrency]\n");
    BCL::finalize();
    return 0;
  }

  size_t hash_size = std::atoll(argv[1]);
  hash_size = (hash_size + sizeof(T) - 1) / sizeof(T);
  size_t message_size = std::atoll(argv[2]);
  message_size = (message_size + sizeof(T) - 1) / sizeof(T);
  size_t num_inserts = std::atoll(argv[3]);

  size_t concurrency = std::atoll(argv[4]);

  size_t local_size = (hash_size + BCL::nprocs() - 1) / BCL::nprocs();

  BCL::print("%lu bytes \"Hash Table\", %lu elements, %lu local size\n",
             hash_size*sizeof(T), hash_size, local_size);
  BCL::print("%lu byte (%lu element) message, %lu inserts/rank\n",
             message_size*sizeof(T), message_size, num_inserts);
  BCL::print("Concurrency: %lu\n", concurrency);

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    if (BCL::rank() == i) {
      counters[i] = BCL::alloc<int>(1);
      ptrs[i] = BCL::alloc<T>(local_size);
      if (ptrs[i] == nullptr || counters[i] == nullptr) {
        throw std::runtime_error("AGH! Could not allocate my segment.");
      }
    }
    ptrs[i] = BCL::broadcast(ptrs[i], i);
    counters[i] = BCL::broadcast(counters[i], i);
  }

  srand48(BCL::rank());

  T* local_seg = ptrs[BCL::rank()].local();
  for (size_t i = 0; i < local_size; i++) {
    local_seg[i] = lrand48();
  }

  *counters[BCL::rank()].local() = 0;

  std::vector<T> src(message_size);

  for (auto& val : src) {
    val = lrand48();
  }

  futar::FuturePool<int> pool(concurrency);

  double issue_fad = 0;
  double issue_put = 0;

  BCL::barrier();

  auto begin = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < num_inserts; i++) {
    size_t dest_rank = lrand48() % BCL::nprocs();

    /*
    auto fut = BCL::arfetch_and_op<int>(counters[dest_rank], 1, BCL::plus<int>{});

    fut.get();

    size_t rand_loc = lrand48() % (local_size - message_size);

    auto request = BCL::arput(src.data(), ptrs[dest_rank] + rand_loc, message_size);
    request.wait();
    */

    auto begin = std::chrono::high_resolution_clock::now();
    auto fut = BCL::arfetch_and_op<int>(counters[dest_rank], 1, BCL::plus<int>{});
    auto end = std::chrono::high_resolution_clock::now();

    issue_fad += std::chrono::duration<double>(end - begin).count();

    auto future = futar::call(
      [&](int val, size_t dest_rank_) {
        size_t rand_loc = lrand48() % (local_size - message_size);
        auto begin = std::chrono::high_resolution_clock::now();
        auto request = BCL::arput(src.data(), ptrs[dest_rank_] + rand_loc, message_size);
        auto end = std::chrono::high_resolution_clock::now();
        issue_put += std::chrono::duration<double>(end - begin).count();
        return BCL::future<int>(val, request);
      }, std::move(fut), dest_rank);

    pool.push_back(std::move(future));
  }

  pool.drain();

  BCL::barrier();

  BCL::print("%lf issue FAD, %lf issue put (%lf total issue)\n",
             issue_fad, issue_put, issue_fad + issue_put);

  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  double nbytes = 1e-9*message_size*num_inserts*BCL::nprocs();

  double bw = nbytes / duration;

  BCL::print("Finished in %lfs. (%lf GB/s)\n", duration, bw);

  BCL::finalize();
  return 0;
}
