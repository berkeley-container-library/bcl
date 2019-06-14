#pragma once

#include <bcl/containers/FastQueue.hpp>

#include <vector>
#include <list>

namespace BCL {

template <typename T, typename Serialize = BCL::serialize<T>>
class ManyToManyDistributor {
  std::vector<std::vector<T>> buffers;
  std::vector<BCL::FastQueue<T, Serialize>> queues;

  std::list<BCL::future<std::vector<T>>> futures;

  size_t message_size_;
  size_t queue_size_;

  std::unique_ptr<BCL::Team> team_ptr_;

public:

  using value_type = T;

  ManyToManyDistributor(const ManyToManyDistributor&) = delete;
  ManyToManyDistributor(ManyToManyDistributor&&) = default;

  const BCL::Team& team() const {
    return *team_ptr_;
  }

  template <typename TeamType>
  ManyToManyDistributor(size_t queue_size, size_t message_size,
                        const TeamType& team) :
                        message_size_(message_size), queue_size_(queue_size),
                        team_ptr_(team.clone()) {
    buffers.resize(BCL::nprocs(this->team()));

    if (this->team().in_team()) {
      for (size_t i = 0; i < BCL::nprocs(this->team()); i++) {
        buffers[i].reserve(message_size);
      }
    }

    for (size_t rank = 0; rank < BCL::nprocs(this->team()); rank++) {
      queues.push_back(BCL::FastQueue<T, Serialize>(this->team().to_world(rank), queue_size));
    }
  }

  ManyToManyDistributor(size_t queue_size, size_t message_size) :
                        message_size_(message_size), queue_size_(queue_size),
                        team_ptr_(new BCL::WorldTeam()) {
    buffers.resize(BCL::nprocs(team()));

    if (team().in_team()) {
      for (size_t i = 0; i < BCL::nprocs(team()); i++) {
        buffers[i].reserve(message_size);
      }
    }

    for (size_t rank = 0; rank < BCL::nprocs(team()); rank++) {
      queues.push_back(BCL::FastQueue<T, Serialize>(team().to_world(rank), queue_size));
    }
  }

  // Insert value into the queue owned by rank,
  // where rank is the rank of the desired process
  // in the distributor's team.
  bool insert(const T& value, size_t rank) {
    assert(rank < buffers.size());
    buffers[rank].push_back(value);

    if (buffers[rank].size() >= message_size_) {
      auto future = queues[rank].push(std::move(buffers[rank]));
      if (!future.has_value()) {
        return false;
      }
      futures.emplace_back(std::move(future.value()));
      buffers[rank].reserve(message_size_);
    }
    return true;
  }

  bool flush() {
    for (size_t rank = 0; rank < buffers.size(); rank++) {
      auto future = queues[rank].push(std::move(buffers[rank]));
      if (!future.has_value()) {
        return false;
      }
      futures.emplace_back(std::move(future.value()));
    }

    for (auto& future : futures) {
      future.get();
    }
    // futures.clear();

    BCL::barrier();
    return true;
  }

  std::vector<T> local_as_vector() {
    return queues[BCL::rank(team())].as_vector();
  }

  auto begin() const {
    return queues[BCL::rank(team())].begin();
  }

  auto end() const {
    return queues[BCL::rank(team())].end();
  }
};

}
