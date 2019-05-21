#pragma once

#include <bcl/bcl.hpp>

#include <bcl/containers/Array.hpp>

namespace BCL {

  namespace distribution {
    struct blocked {
    };
  }

  template <typename T,
            typename TSerialize = BCL::serialize <T>>
  class DArray {
  public:
    size_t my_size = 0;
    size_t local_size = 0;

    std::vector <BCL::Array <T, TSerialize>> data;

    size_t size() const noexcept {
      return my_size;
    }

    DArray(size_t size) : my_size(size) {
      local_size = (size + BCL::nprocs() - 1) / BCL::nprocs();
      for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
        data.push_back(BCL::Array <T, TSerialize> (rank, local_size));
      }
    }

    ArrayRef <T, TSerialize> operator[](size_t idx) {
      uint64_t node = idx / local_size;
      uint64_t node_slot = idx - node*local_size;
      return data[node][node_slot];
    }
  };
}
