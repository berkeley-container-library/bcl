#ifndef BCL_ARH_HPP
#define BCL_ARH_HPP

#include "bcl/bcl.hpp"
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>

namespace ARH {

  std::unordered_map<std::thread::id, size_t> ids;
  size_t num_workers_per_node = 30;

  inline void run(const std::function<void(void)> &worker, size_t custom_num_workers_per_node = 30) {
//    std::vector<std::thread> thread_pool(num_workers_per_node, std::thread(worker));
    num_workers_per_node = custom_num_workers_per_node;
    std::vector<std::thread> thread_pool;
    for (size_t i = 0; i < num_workers_per_node; ++i) {
      auto t = std::thread(worker);
      ids[t.get_id()] = i + BCL::rank() * num_workers_per_node;
      thread_pool.push_back(std::move(t));
    }

    for (size_t i = 0; i < num_workers_per_node; ++i) {
      thread_pool[i].join();
    }
  }

  inline size_t my_thread() {
    return ids[std::this_thread::get_id()];
  }

  inline size_t nthreads() {
    return BCL::nprocs() * num_workers_per_node;
  }

}

#endif //BCL_ARH_HPP
