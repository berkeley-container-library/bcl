//
// Created by Jiakun Yan on 10/1/19.
//

#ifndef BCL_ARH_BASE_HPP
#define BCL_ARH_BASE_HPP

#include "bcl/bcl.hpp"
#include "arh_am.hpp"
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

  inline size_t my_worker() {
    return worker_ids[std::this_thread::get_id()];
  }

  inline size_t nworkers() {
    return BCL::nprocs() * num_workers_per_node;
  }

  inline size_t my_proc() {
    return BCL::rank();
  }

  inline size_t nprocs() {
    return BCL::nprocs();
  }

  inline void barrier() {
    BCL::barrier();
  }

  void progress() {
    flush_am();
  }

  // progress thread
  void progress_thread() {
    while (worker_run.load()) {
      progress();
    }
  }

  inline void init(uint64_t shared_segment_size = 256) {
    BCL::init(shared_segment_size, true);
    init_am();
  }

  inline void finalize() {
    BCL::finalize();
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
      auto t = std::thread(progress_thread);
      progress_ids[t.get_id()] = i + BCL::rank() * num_workers_per_node;
      progress_pool.push_back(std::move(t));
    }

    for (size_t i = 0; i < num_workers_per_node; ++i) {
      worker_pool[i].join();
    }

    worker_run = false;
//    std::printf("process %lu finished\n", my_proc());

    for (size_t i = 0; i < num_progress_per_node; ++i) {
      progress_pool[i].join();
    }
  }

}

#endif //BCL_ARH_BASE_HPP
