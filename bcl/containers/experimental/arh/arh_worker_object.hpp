#ifndef BCL_ARH_WORKER_OBJECT_HPP
#define BCL_ARH_WORKER_OBJECT_HPP

#include "arh_base.hpp"
#include <vector>

namespace ARH {

  template <typename T>
  struct WorkerObject {
#ifdef ARH_DEBUG
    size_t len = -1;
#endif
    struct align_T {
      alignas(alignof_cacheline) T _val;
      align_T() {
        ARH_Assert_Align(_val, alignof_cacheline);
      }
      explicit align_T(T&& val): _val(std::forward(val)) {
        ARH_Assert_Align(_val, alignof_cacheline);
      }
    };

    std::vector<align_T> objects;

    void init() {
#ifdef ARH_DEBUG
      len = nworkers_local();
#endif
      objects = std::vector<align_T>(nworkers_local());
    }

    // T must have copy constructor
    void init(T&& val) {
#ifdef ARH_DEBUG
      len = nworkers_local();
#endif
      objects = std::vector<align_T>(nworkers_local(), val);
    }

    T &get() {
#ifdef ARH_DEBUG
      ARH_Assert(len != 1, "Use before calling init!");
      ARH_Assert(my_worker_local() >= len, "Index out of scope!");
#endif
      return objects[my_worker_local()]._val;
    }
  };

}

#endif //BCL_ARH_WORKER_OBJECT_HPP
