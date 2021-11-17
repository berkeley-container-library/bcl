// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <bcl/bcl.hpp>

bool is_set(BCL::GlobalPtr<int> ptr, size_t size) {
  int* ptr_ = ptr.local();
  for (size_t i = 0; i < BCL::nprocs()*size; i++) {
    if (ptr_[i] != i / size) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  BCL::init();
  BCL::GlobalPtr<int> ptr = nullptr;

  size_t size = 8*1024;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (BCL::rank() == rank) {
      ptr = BCL::alloc<int>(BCL::nprocs() * size);
    }
    ptr = BCL::broadcast(ptr, rank);

    for (size_t i = size*BCL::rank(); i < size*(BCL::rank()+1); i++) {
      ptr[i] = BCL::rank();
    }

    if (BCL::rank() == rank) {
      while (!is_set(ptr, size)) {}
    }

    BCL::barrier();
  }

  BCL::finalize();
  return 0;
}
