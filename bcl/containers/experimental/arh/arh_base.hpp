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

  std::unordered_map<std::thread::id, size_t> thread_ids;
  std::vector<size_t> thread_contexts;
  size_t num_threads_per_proc = 16;
  size_t num_workers_per_proc = 15;
  std::atomic<bool> worker_run = true;
  ThreadBarrier threadBarrier;

  inline size_t get_thread_id() {
    return thread_ids[std::this_thread::get_id()];
  }

  inline void set_context(size_t mContext) {
    thread_contexts[get_thread_id()] = mContext;
  }

  inline size_t get_context() {
    return thread_contexts[get_thread_id()];
  }

  inline size_t my_worker_local() {
    return get_context();
  }

  inline size_t nworkers_local() {
    return num_workers_per_proc;
  }

  inline size_t my_proc() {
    return BCL::rank();
  }

  inline size_t nprocs() {
    return BCL::nprocs();
  }

  inline size_t my_worker() {
    return my_worker_local() + my_proc() * num_workers_per_proc;
  }

  inline size_t nworkers() {
    return BCL::nprocs() * num_workers_per_proc;
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

  inline void init(size_t custom_num_workers_per_proc = 15,
      size_t custom_num_threads_per_proc = 16, size_t shared_segment_size = 256) {
    num_workers_per_proc = custom_num_workers_per_proc;
    num_threads_per_proc = custom_num_threads_per_proc;
    thread_contexts.resize(num_threads_per_proc, 0);
    threadBarrier.init(num_workers_per_proc, progress);

    BCL::init(shared_segment_size, true);
    init_am();
    init_agg();
  }

  inline void finalize() {
    BCL::finalize();
  }

  void set_affinity(pthread_t pthread_handler, size_t target) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(target, &cpuset);
    int rv = pthread_setaffinity_np(pthread_handler, sizeof(cpuset), &cpuset);
    if (rv != 0) {
      throw std::runtime_error("ERROR thread affinity didn't work.");
    }
  }

  void run(const std::function<void(void)> &worker) {
    std::vector<std::thread> worker_pool;
    std::vector<std::thread> progress_pool;

    size_t cpuoffset;
    int my_cpu = sched_getcpu();
    if ((my_cpu >= 0 && my_cpu < 16) || (my_cpu >= 32 && my_cpu < 48)) {
      cpuoffset = 0;
    } else {
      cpuoffset = 16;
    }

    for (size_t i = 0; i < num_workers_per_proc; ++i) {
      auto t = std::thread(worker_handler, worker);

      set_affinity(t.native_handle(), i + cpuoffset);

      thread_ids[t.get_id()] = i;
      thread_contexts[i] = i;
      worker_pool.push_back(std::move(t));
    }

    for (size_t i = num_workers_per_proc; i < num_threads_per_proc; ++i) {
      auto t = std::thread(progress_handler);

      set_affinity(t.native_handle(), i + cpuoffset);

      thread_ids[t.get_id()] = i;
      thread_contexts[i] = i;
      progress_pool.push_back(std::move(t));
    }

    for (size_t i = 0; i < num_workers_per_proc; ++i) {
      worker_pool[i].join();
    }

    worker_run = false;

    for (size_t i = 0; i < num_threads_per_proc - num_workers_per_proc; ++i) {
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
