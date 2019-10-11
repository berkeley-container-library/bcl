//
// Created by Jiakun Yan on 10/1/19.
//

#ifndef BCL_ARH_BASE_HPP
#define BCL_ARH_BASE_HPP

#include "bcl/bcl.hpp"
#include "arh_am.hpp"
#include "arh_threadbarrier.hpp"
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <sched.h>
#include <pthread.h>

namespace ARH {
  extern void init_am(void);
  extern void flush_am(void);
  extern void init_agg(void);
  extern void flush_agg_buffer(void);

  std::unordered_map<std::thread::id, size_t> worker_ids;
  std::unordered_map<std::thread::id, size_t> progress_ids;
  size_t num_threads_per_node = 32;
  size_t num_workers_per_node = 30;
  std::atomic<bool> worker_run = true;
  ThreadBarrier threadBarrier;

  inline size_t my_worker() {
    return worker_ids[std::this_thread::get_id()];
  }

  inline size_t nworkers() {
    return BCL::nprocs() * num_workers_per_node;
  }

  inline size_t my_worker_local() {
    return my_worker() % num_workers_per_node;
  }

  inline size_t nworkers_local() {
    return num_workers_per_node;
  }

  inline size_t my_proc() {
    return BCL::rank();
  }

  inline size_t nprocs() {
    return BCL::nprocs();
  }

  inline void barrier() {
    threadBarrier.wait();
    flush_agg_buffer();
    flush_am();
    if (my_worker_local() == 0) {
      BCL::barrier();
    }
    threadBarrier.wait();
  }

  void progress() {
    gasnet_AMPoll();
  }

  // progress thread
  void progress_handler() {
    while (worker_run.load()) {
      progress();
    }
  }

  void worker_handler(const std::function<void(void)>& worker) {
    barrier();
    worker();
    barrier();
  }

  inline void init(size_t shared_segment_size = 256) {
    BCL::init(shared_segment_size, true);
    init_am();
    init_agg();
  }

  inline void finalize() {
    BCL::finalize();
  }

  void run(const std::function<void(void)> &worker, size_t custom_num_workers_per_node = 30,
           size_t custom_num_threads_per_node = 32) {
    num_workers_per_node = custom_num_workers_per_node;
    num_threads_per_node = custom_num_threads_per_node;
    threadBarrier.init(num_workers_per_node, progress);

    std::vector<std::thread> worker_pool;
    std::vector<std::thread> progress_pool;

    size_t proc_num = sysconf(_SC_NPROCESSORS_CONF);
    cpu_set_t cpuset;

    for (size_t i = 0; i < num_workers_per_node; ++i) {
      auto t = std::thread(worker_handler, worker);

      CPU_ZERO(&cpuset);
      CPU_SET(i % proc_num, &cpuset);
      int rv = pthread_setaffinity_np(t.native_handle(), sizeof(cpuset), &cpuset);
      if (rv != 0) {
        throw std::runtime_error("ERROR thread affinity didn't work.");
      }

      worker_ids[t.get_id()] = i + BCL::rank() * num_workers_per_node;
      worker_pool.push_back(std::move(t));
    }

    for (size_t i = num_workers_per_node; i < num_threads_per_node; ++i) {
      auto t = std::thread(progress_handler);

      CPU_ZERO(&cpuset);
      CPU_SET(i % proc_num, &cpuset);
      int rv = pthread_setaffinity_np(t.native_handle(), sizeof(cpuset), &cpuset);
      if (rv != 0) {
        throw std::runtime_error("ERROR thread affinity didn't work.");
      }

      progress_ids[t.get_id()] = i + BCL::rank() * num_workers_per_node; // might break
      progress_pool.push_back(std::move(t));
    }

    for (size_t i = 0; i < num_workers_per_node; ++i) {
      worker_pool[i].join();
    }

    worker_run = false;

    for (size_t i = 0; i < num_threads_per_node - num_workers_per_node; ++i) {
      progress_pool[i].join();
    }
  }

  template <typename ...Args>
  void print(std::string format, Args... args) {
    fflush(stdout);
    ARH::barrier();
    if (ARH::my_worker() == 0) {
      std::printf(format.c_str(), args...);
    }
    fflush(stdout);
    ARH::barrier();
  }

}

#endif //BCL_ARH_BASE_HPP
