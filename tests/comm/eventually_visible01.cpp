// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#include <cassert>
#include <bcl/bcl.hpp>

bool is_set(BCL::GlobalPtr<int> ptr) {
  for (size_t i = 0; i < BCL::nprocs(); i++) {
    int value = ptr[i];
    if (value != i) {
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  BCL::init();
  BCL::GlobalPtr<int> ptr = nullptr;

  for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
    if (BCL::rank() == rank) {
      ptr = BCL::alloc<int>(BCL::nprocs());
    }
    ptr = BCL::broadcast(ptr, rank);

    ptr[BCL::rank()] = BCL::rank();

    while (!is_set(ptr)) {}

    BCL::barrier();
  }

  BCL::finalize();
  return 0;
}
