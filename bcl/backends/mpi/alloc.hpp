// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <stdexcept>
#include <mpi.h>
#include "backend.hpp"

namespace BCL {
  extern bool progress_running;
  extern uint64_t nprocs();

  template <typename T>
  GlobalPtr <T> ralloc(const size_t size, const uint64_t rank) {
    if (rank == BCL::rank()) {
      return BCL::alloc <T> (size);
    } else {
      return BCL::rpc(rank, &BCL::alloc <T>, size);
    }
  }

  // TODO: Decay or bind args or something?
  template <typename T>
  void rdealloc(GlobalPtr <T> &ptr) {
    if (ptr.rank == BCL::rank()) {
      BCL::dealloc <T> (ptr);
    } else {
      BCL::rpc(ptr.rank, &BCL::dealloc <T>, ptr);
    }
  }
}
