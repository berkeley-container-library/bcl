#pragma once

#include <mpi.h>
#include <stdexcept>
#include <bcl/core/teams.hpp>

namespace BCL {

namespace backend {

struct MPICommWrapper {
  MPI_Comm comm_;

  MPICommWrapper(const MPICommWrapper&) = delete;

  MPICommWrapper() = delete;

  MPICommWrapper(const BCL::Team& team) {
    std::vector<int> ranks;
    ranks.reserve(team.nprocs());

    for (size_t i = 0; i < team.nprocs(); i++) {
      ranks.push_back(team.to_world(i));
    }

    MPI_Group world_group, group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    MPI_Group_incl(world_group, ranks.size(), ranks.data(), &group);

    MPI_Comm_create(MPI_COMM_WORLD, group, &comm_);

    MPI_Group_free(&world_group);
    MPI_Group_free(&group);
  }

  ~MPICommWrapper() {
    if (comm_ != MPI_COMM_NULL) {
      MPI_Comm_free(&comm_);
    }
  }

  MPI_Comm comm() const {
    return comm_;
  }
};

}
}
