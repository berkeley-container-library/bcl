#ifndef BCL_ARH_GLOBAL_OBJECT_HPP
#define BCL_ARH_GLOBAL_OBJECT_HPP

#include "arh_base.hpp"
#include <vector>

namespace ARH {

  template <typename T>
  struct GlobalObject {
#if ARH_DEBUG
    size_t len = -1;
#endif
    std::vector<T> objects;

    void init() {
#if ARH_DEBUG
      len = nworkers_local();
#endif
      objects = std::vector<T>(nworkers_local());
    }

    void init(T val) {
#if ARH_DEBUG
      len = nworkers_local();
#endif
      objects = std::vector<T>(nworkers_local(), val);
    }

    T &get() {
#if ARH_DEBUG
      if (len == -1) {
        std::printf("GlobalObject::get(): you didn't initialize me!");
      }
      size_t idx = my_worker_local();
      if (idx >= len) {
        std::printf("GlobalObject::get(): oops, out of scope. len = %lu, idx = %lu", len, idx);
      }
#endif
      return objects[my_worker_local()];
    }
  };

}

#endif //BCL_ARH_GLOBAL_OBJECT_HPP
