#pragma once

#include <bcl/bcl.hpp>
#include <futar/fn_wrapper.hpp>
#include <futar/pools/ChainPool.hpp>

template <typename T>
void strict_send(std::vector<BCL::GlobalPtr<T>>& ptrs,
                 std::vector<BCL::GlobalPtr<int>>& counters,
                 std::vector<T>& buffer,
                 size_t local_segment_size,
                 size_t inserts_per_proc,
                 size_t write_size) {
  for (size_t i = 0; i < inserts_per_proc; i++) {
    size_t rand_proc = lrand48() % BCL::nprocs();
    size_t rand_loc = lrand48() % (local_segment_size - write_size);

    auto ptr = ptrs[rand_proc] + rand_loc;

    BCL::fetch_and_op(counters[rand_proc], int(write_size), BCL::plus<int>{});
    BCL::memcpy(ptr, buffer.data(), sizeof(T)*write_size);
    BCL::flush();
  }
}

template <typename T>
void relaxed_send(std::vector<BCL::GlobalPtr<T>>& ptrs,
                  std::vector<BCL::GlobalPtr<int>>& counters,
                  std::vector<T>& buffer,
                  size_t local_segment_size,
                  size_t inserts_per_proc,
                  size_t write_size) {
  for (size_t i = 0; i < inserts_per_proc; i++) {
    size_t rand_proc = lrand48() % BCL::nprocs();
    size_t rand_loc = lrand48() % (local_segment_size - write_size);

    auto ptr = ptrs[rand_proc] + rand_loc;

    BCL::fetch_and_op(counters[rand_proc], int(write_size), BCL::plus<int>{});
    BCL::memcpy(ptr, buffer.data(), sizeof(T)*write_size);
  }
}

template <typename T>
void future_send(std::vector<BCL::GlobalPtr<T>>& ptrs,
                 std::vector<BCL::GlobalPtr<int>>& counters,
                 std::vector<T>& buffer,
                 size_t local_segment_size,
                 size_t inserts_per_proc,
                 size_t write_size) {
  size_t concurrency = 100;
  futar::FuturePool<int> pool(concurrency);
  for (size_t i = 0; i < inserts_per_proc; i++) {
    size_t rand_proc = lrand48() % BCL::nprocs();
    size_t rand_loc = lrand48() % (local_segment_size - write_size);

    auto ptr = ptrs[rand_proc] + rand_loc;

    /*
    if (BCL::rank() == 0) {
      printf("Arfetch %lu -> (%lu)\n",
             BCL::rank(), rand_proc);
    }
    */
    auto fut = BCL::arfetch_and_op<int>(counters[rand_proc], int(write_size), BCL::plus<int>{});

    auto future = futar::call(
      [&](int val, BCL::GlobalPtr<int> ptr_) {
      /*
        if (BCL::rank() == 0) {
          printf("Arput %lu -> (%lu, %d)\n",
                 BCL::rank(), ptr_.rank, val);
        }
        */
        auto request = BCL::arput(ptr_, buffer.data(), write_size);
        return BCL::future<int>(val, request);
      }, std::move(fut), ptr);

    // pool.push_back(std::move(future));
    pool.push_back(std::move(future));
  }
  pool.drain();
}

template <typename T>
void async_send(std::vector<BCL::GlobalPtr<T>>& ptrs,
                std::vector<BCL::GlobalPtr<int>>& counters,
                std::vector<T>& buffer,
                size_t local_segment_size,
                size_t inserts_per_proc,
                size_t write_size) {
  std::vector<std::tuple<BCL::request, size_t>> requests;
  std::vector<BCL::future<int>> atomic_requests;
  size_t iter_manage = 10;
  for (size_t i = 0; i < inserts_per_proc; i++) {
    size_t rand_proc = lrand48() % BCL::nprocs();
    size_t rand_loc = lrand48() % (local_segment_size - write_size);

    auto ptr = ptrs[rand_proc] + rand_loc;

    BCL::request request = arput(ptr, buffer.data(), write_size);

    requests.push_back({request, rand_proc});

    if (i % iter_manage == 0) {
      for (size_t i = 0; i < requests.size(); ) {
        if (std::get<0>(requests[i]).check()) {
          auto request = BCL::arfetch_and_op(counters[std::get<1>(requests[i])], int(write_size), BCL::plus<int>{});
          atomic_requests.push_back(std::move(request));
          requests.erase(requests.begin() + i);
        } else {
          i++;
        }
      }
      for (size_t i = 0; i < atomic_requests.size(); ) {
        if (atomic_requests[i].check()) {
          atomic_requests.erase(atomic_requests.begin() + i);
        } else {
          i++;
        }
      }
    }
  }

  for (size_t i = 0; i < requests.size(); i++) {
    std::get<0>(requests[i]).wait();
    auto request = BCL::arfetch_and_op(counters[std::get<1>(requests[i])], int(write_size), BCL::plus<int>{});
    atomic_requests.push_back(std::move(request));
  }

  for (size_t i = 0; i < atomic_requests.size(); i++) {
    atomic_requests[i].wait();
  }
}
