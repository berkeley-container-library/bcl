//
// Created by Jiakun Yan on 12/6/19.
//

#ifndef ARH_COLLECTIVE_HPP
#define ARH_COLLECTIVE_HPP

namespace ARH {
  template <typename T>
  inline T broadcast_node(T& val, uint64_t root) {
    ARH_Assert(root < ARH::nworkers_local(), "");
    static T shared_val;
    if (my_worker_local() == root) {
      shared_val = val;
      threadBarrier.wait();
    } else {
      threadBarrier.wait();
      val = shared_val;
    }
    threadBarrier.wait();
    return val;
  }

  template <typename T>
  inline T broadcast(T& val, uint64_t root) {
    ARH_Assert(root < ARH::nworkers(), "");
    if (my_worker_local() == root % nworkers_local()) {
      val = BCL::broadcast(val, root / nworkers_local());
    }
    broadcast_node(val, root % nworkers_local());
    return val;
  }
}

#endif //ARH_COLLECTIVE_HPP
