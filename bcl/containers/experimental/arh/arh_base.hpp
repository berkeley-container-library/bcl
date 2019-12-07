//
// Created by Jiakun Yan on 10/1/19.
//

#ifndef BCL_ARH_BASE_HPP
#define BCL_ARH_BASE_HPP

namespace ARH {
  extern void init_am(void);
  extern void flush_am(void);
  extern void init_agg(void);
  extern void flush_agg_buffer(void);

  alignas(alignof_cacheline) std::unordered_map<std::thread::id, size_t> thread_ids;
  struct ThreadContext {
    alignas(alignof_cacheline) size_t val;
  };
  alignas(alignof_cacheline) std::vector<ThreadContext> thread_contexts;
  alignas(alignof_cacheline) ThreadBarrier threadBarrier;
  alignas(alignof_cacheline) std::atomic<bool> thread_run = false;
  alignas(alignof_cacheline) std::atomic<bool> worker_exit = false;
  alignas(alignof_cacheline) size_t num_threads_per_proc = 16;
  alignas(alignof_cacheline) size_t num_workers_per_proc = 15;

  inline size_t get_thread_id() {
    return thread_ids[std::this_thread::get_id()];
  }

  inline void set_context(size_t mContext) {
    thread_contexts[get_thread_id()].val = mContext;
  }

  inline size_t get_context() {
    return thread_contexts[get_thread_id()].val;
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
    while (!thread_run) {
      usleep(1);
    }
    while (!worker_exit) {
      progress();
    }
  }

  template <typename Fn, typename... Args>
  void worker_handler(Fn &&fn, Args &&... args) {
    while (!thread_run) {
      usleep(1);
    }
    barrier();
    std::invoke(std::forward<Fn>(fn),
                std::forward<Args>(args)...);
    barrier();
  }

  inline void init(size_t custom_num_workers_per_proc = 15,
      size_t custom_num_threads_per_proc = 16, size_t shared_segment_size = 256) {
    num_workers_per_proc = custom_num_workers_per_proc;
    num_threads_per_proc = custom_num_threads_per_proc;
    thread_contexts.resize(num_threads_per_proc);
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

  template <typename Fn, typename... Args>
  void run(Fn &&fn, Args &&... args) {
    using FP = decltype(&fn); // TODO: Is there a clear way to do this?
    std::vector<std::thread> worker_pool;
    std::vector<std::thread> progress_pool;

#ifdef ARH_THREAD_PIN
    int numberOfProcessors = sysconf(_SC_NPROCESSORS_ONLN);
    size_t cpuoffset;
    int my_cpu = sched_getcpu();
    if ((my_cpu >= 0 && my_cpu < 16) || (my_cpu >= 32 && my_cpu < 48)) {
      cpuoffset = 0;
    } else {
      cpuoffset = 16;
    }
#endif

    for (size_t i = 0; i < num_workers_per_proc; ++i) {
      std::thread t(worker_handler<FP, Args...>,
                    std::forward<Fn>(fn),
                    std::forward<Args>(args)...);
#ifdef ARH_THREAD_PIN
      set_affinity(t.native_handle(), (i + cpuoffset) % numberOfProcessors);
#endif
      thread_ids[t.get_id()] = i;
      thread_contexts[i].val = i;
      worker_pool.push_back(std::move(t));
    }

    for (size_t i = num_workers_per_proc; i < num_threads_per_proc; ++i) {
      std::thread t(progress_handler);
#ifdef ARH_THREAD_PIN
      set_affinity(t.native_handle(), (i + cpuoffset) % numberOfProcessors);
#endif
      thread_ids[t.get_id()] = i;
      thread_contexts[i].val = i;
      progress_pool.push_back(std::move(t));
    }
    thread_run = true;

    for (size_t i = 0; i < num_workers_per_proc; ++i) {
      worker_pool[i].join();
    }

    worker_exit = true;

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
