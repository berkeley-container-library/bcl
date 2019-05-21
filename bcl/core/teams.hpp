#pragma once

#include <vector>
#include <unordered_map>
#include <algorithm>

namespace BCL {

namespace backend {
  extern uint64_t rank();
  extern uint64_t nprocs();
};

// TODO: use CRTP to ensure no vtable overhead?
struct Team {
  virtual size_t resolve(size_t rank) const = 0;
  virtual size_t nprocs() const noexcept = 0;
  virtual bool in_team(size_t rank = BCL::backend::rank()) const noexcept = 0;
  virtual size_t to_world(size_t rank) const = 0;
  virtual Team* clone() const = 0;
};

struct WorldTeam final : virtual Team {
  size_t resolve(size_t rank) const override {
    return rank;
  }

  size_t nprocs() const noexcept override {
    return BCL::backend::nprocs();
  }

  bool in_team(size_t rank = BCL::backend::rank()) const noexcept override {
    return true;
  }

  size_t to_world(size_t rank) const override {
    return rank;
  }

  Team* clone() const override {
    return new WorldTeam();
  }

  WorldTeam() = default;
  WorldTeam(const WorldTeam&) = default;
};

struct UserTeam final : virtual Team {
  std::vector<size_t> members_;
  std::unordered_map<size_t, size_t> mapping_;

  UserTeam(const UserTeam&) = default;

  UserTeam(const std::vector<size_t>& members) : members_(members) {
    std::sort(members_.begin(), members_.end());
    members_.erase(std::unique(members_.begin(), members_.end()), members_.end());
    size_t idx = 0;
    for (auto& member : members_) {
      mapping_[member] = idx++;
    }
  }

  size_t nprocs() const noexcept override {
    return members_.size();
  }

  size_t resolve(size_t rank) const override {
    const auto memb = mapping_.find(rank);
    if (memb == mapping_.end()) {
      throw std::runtime_error("SQUAWK!!! Error resolving team member (UserTeam)");
    }
    return memb->second;
  }

  bool in_team(size_t rank = BCL::backend::rank()) const noexcept override {
    const auto memb = mapping_.find(rank);
    return memb != mapping_.end();
  }

  size_t to_world(size_t rank) const override {
    if (rank >= members_.size()) {
      throw std::runtime_error("SQUAWK!!! Error resolving team number");
    }
    return members_[rank];
  }

  Team* clone() const override {
    return new UserTeam(*this);
  }
};

struct RangeTeam final : virtual Team {
  size_t bottom_, top_;

  RangeTeam(size_t bottom, size_t top) : bottom_(bottom), top_(top) {}

  size_t resolve(size_t rank) const override {
    if (rank < bottom_ || rank >= top_) {
      throw std::runtime_error("SQUAWK!! Error resolving team member (RangeTeam)");
    }
    return rank - bottom_;
  }

  size_t nprocs() const noexcept override {
    return top_ - bottom_;
  }

  size_t to_world(size_t rank) const override {
    if (rank >= BCL::backend::nprocs()) {
      throw std::runtime_error("SQUAWK!! Ruh roh");
    }
    return bottom_ + rank;
  }

  bool in_team(size_t rank = BCL::backend::rank()) const noexcept override {
    if (rank < bottom_ || rank >= top_) {
      return false;
    }
    return true;
  }

  Team* clone() const override {
    return new RangeTeam(*this);
  }
};

inline std::vector<BCL::RangeTeam> split_world(size_t c) {
  size_t team_size = (BCL::backend::nprocs() + c - 1) / c;
  std::vector<BCL::RangeTeam> teams;
  for (size_t i = 0; i < BCL::backend::nprocs(); i += team_size) {
    BCL::RangeTeam team(i, std::min<size_t>(i+team_size, BCL::backend::nprocs()));
    teams.push_back(team);
  }
  return teams;
}

inline bool in_team(const BCL::Team& team, size_t rank = BCL::backend::rank()) {
  return team.in_team(rank);
}

inline size_t rank(const BCL::Team& team) {
  return team.resolve(BCL::backend::rank());
}

inline size_t rank() {
  return BCL::backend::rank();
}

inline size_t nprocs(const BCL::Team& team) {
  return team.nprocs();
}

inline size_t nprocs() {
  return BCL::backend::nprocs();
}

}
