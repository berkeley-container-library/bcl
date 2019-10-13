#pragma once

#include <bcl/bcl.hpp>
#include <bcl/containers/SPMatrix.hpp>

namespace BCL {

template <
          typename T,
          typename index_type,
          typename Mult = std::multiplies<T>,
          typename Plus = std::plus<T>
          >
void gemm(const BCL::SPMatrix<T, index_type>& a,
          const BCL::SPMatrix<T, index_type>& b,
                BCL::SPMatrix<T, index_type>& c) {
  // accumulated C's: a map of grid coordinates to sparse
  //                  matrices (for now, also maps)
  //

  std::unordered_map<
                     std::pair<size_t, size_t>,
                     // BCL::SparseSPAAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     BCL::SparseHashAccumulator<T, index_type, Plus, BCL::bcl_allocator<T>>,
                     // BCL::SparseHeapAccumulator<T, index_type, BCL::bcl_allocator<T>>,
                     // BCL::CombBLASAccumulator<T, index_type>,
                     // BCL::EagerMKLAccumulator<T, index_type>,
                     BCL::djb2_hash<std::pair<size_t, size_t>>
                     >
                     accumulated_cs;

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_locale(i, j) == BCL::rank()) {

        size_t k_offset = j % a.grid_shape()[1];

        auto buf_a = a.arget_tile(i, k_offset % a.grid_shape()[1]);
        auto buf_b = b.arget_tile(k_offset % a.grid_shape()[1], j);

        for (size_t k_ = 0; k_ < a.grid_shape()[1]; k_++) {
          size_t k = (k_ + k_offset) % a.grid_shape()[1];


          auto my_a = buf_a.get();
          auto my_b = buf_b.get();

          if (k_+1 < a.grid_shape()[1]) {
            buf_a = a.arget_tile(i, (k+1) % a.grid_shape()[1]);
            buf_b = b.arget_tile((k+1) % a.grid_shape()[1], j);
          }

          auto c = my_a. template dot<Mult, Plus>(my_b);

          accumulated_cs[{i, j}].accumulate(std::move(c));
        }
      }
    }
  }

  BCL::barrier();

  for (size_t i = 0; i < c.grid_shape()[0]; i++) {
    for (size_t j = 0; j < c.grid_shape()[1]; j++) {
      if (c.tile_locale(i, j) == BCL::rank()) {
        auto cmatrix = accumulated_cs[{i, j}].get_matrix(c.tile_shape(i, j)[0], c.tile_shape(i, j)[1]);
        c.assign_tile(i, j, cmatrix);
      }
    }
  }

  BCL::barrier();
  c.rebroadcast_tiles();
}

} // end BCL
