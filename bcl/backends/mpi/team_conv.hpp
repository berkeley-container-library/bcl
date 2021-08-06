// SPDX-FileCopyrightText: 2021 Benjamin Brock
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cassert>
#include <mpi.h>
#include <stdexcept>
#include <bcl/core/teams.hpp>

namespace BCL {

namespace backend {

struct MPICommWrapper {
  MPI_Comm comm_ = MPI_COMM_NULL;

  MPICommWrapper(const MPICommWrapper&) = delete;
  MPICommWrapper() = default;

  MPICommWrapper(MPICommWrapper&& other) {
    comm_ = other.comm_;
    other.comm_ = MPI_COMM_NULL;
  }

  MPICommWrapper& operator=(MPICommWrapper&& other) {
    comm_ = other.comm_;
    other.comm_ = MPI_COMM_NULL;
    return *this;
  }

  MPICommWrapper(const BCL::Team& team) {
    std::vector<int> ranks;
    ranks.reserve(team.nprocs());

    for (size_t i = 0; i < team.nprocs(); i++) {
      ranks.push_back(team.to_world(i));
    }

    MPI_Group world_group, group;
    int rv = MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    assert(rv == MPI_SUCCESS);

    rv = MPI_Group_incl(world_group, ranks.size(), ranks.data(), &group);
    assert(rv == MPI_SUCCESS);

    rv = MPI_Comm_create(MPI_COMM_WORLD, group, &comm_);
    assert(rv == MPI_SUCCESS);

    MPI_Group_free(&world_group);
    MPI_Group_free(&group);
  }

  ~MPICommWrapper() {
    if (comm_ != MPI_COMM_NULL) {
      int finalized;
      MPI_Finalized(&finalized);
      if (!finalized) {
        MPI_Comm_free(&comm_);
      }
    }
  }

  MPI_Comm comm() const {
    return comm_;
  }
};

}
}
