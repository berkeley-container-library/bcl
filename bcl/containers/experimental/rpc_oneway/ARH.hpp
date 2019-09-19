#ifndef BCL_ARH_HPP
#define BCL_ARH_HPP

#include "bcl/bcl.hpp"
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <atomic>

namespace ARH {

  std::unordered_map<std::thread::id, size_t> worker_ids;
  std::unordered_map<std::thread::id, size_t> progress_ids;
  size_t num_threads_per_node = 32;
  size_t num_workers_per_node = 30;
  std::atomic<bool> worker_run = true;

  void progress() {
    while (worker_run.load()) {
      // poll the queue
    }
  }

  void run(const std::function<void(void)> &worker, size_t custom_num_workers_per_node = 30,
      size_t custom_num_threads_per_node = 32) {
//    std::vector<std::thread> thread_pool(num_workers_per_node, std::thread(worker));
    num_workers_per_node = custom_num_workers_per_node;
    num_threads_per_node = custom_num_threads_per_node;

    std::vector<std::thread> worker_pool;

    for (size_t i = 0; i < num_workers_per_node; ++i) {
      auto t = std::thread(worker);
      worker_ids[t.get_id()] = i + BCL::rank() * num_workers_per_node;
      worker_pool.push_back(std::move(t));
    }

    std::vector<std::thread> progress_pool;
    size_t num_progress_per_node = num_threads_per_node - num_workers_per_node;
    for (size_t i = 0; i < num_progress_per_node; ++i) {
      auto t = std::thread(progress);
      progress_ids[t.get_id()] = i + BCL::rank() * num_workers_per_node;
      progress_pool.push_back(std::move(t));
    }

    for (size_t i = 0; i < num_workers_per_node; ++i) {
      worker_pool[i].join();
    }

    worker_run = false;

    for (size_t i = 0; i < num_progress_per_node; ++i) {
      progress_pool[i].join();
    }
  }

  inline size_t my_worker() {
    return worker_ids[std::this_thread::get_id()];
  }

  inline size_t nworkers() {
    return BCL::nprocs() * num_workers_per_node;
  }

}

#endif //BCL_ARH_HPP
