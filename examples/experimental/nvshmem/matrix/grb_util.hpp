#pragma once

namespace BCL {

template <typename T>
struct max {
  T operator()(const T& a, const T& b) const {
    if (a < b) {
      return b;
    } else {
      return a;
    }
  }
};

template <typename T>
struct min {
  T operator()(const T& a, const T& b) const {
    if (a < b) {
      return a;
    } else {
      return b;
    }
  }
};


namespace cuda {



/*
struct alloc_t {
  char* ptr_;
  size_t size_;

  template <typename T>
  alloc_t(T* ptr, size_t size) : ptr_((char *) ptr), size_(size*sizeof(T)) {}

  char* begin() {
    return ptr_;
  }

  char* end() {
    return ptr_ + size_;
  }
};

bool check_overlap(alloc_t a, alloc_t b) {
  return a.begin() <= b.end() && a.end() >= b.begin();
}

bool check_overlap(const std::vector<alloc_t>& allocations) {
  for (size_t i = 0; i < allocations.size(); i++) {
    for (size_t j = i+1; j < allocations.size(); j++) {
      if (check_overlap(allocations[i], allocations[j])) {
        fprintf(stderr, "%p overlaps with %p\n", allocations[i].ptr_, allocations[j].ptr_);
        return true;
      }
    }
  }
  return false;
}
*/

} // end cuda	

} // end BCL
