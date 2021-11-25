#pragma once

namespace bcl {
inline constexpr std::size_t block_stride = std::numeric_limits<std::size_t>::max();
namespace execution {

template <typename LocalExecutionPolicy = std::execution::parallel_unsequenced_policy>
class sequential_policy {
public:
  constexpr sequential_policy(std::size_t rank = 0, LocalExecutionPolicy local_policy = std::execution::par_unseq)
    : rank_(rank), local_policy_(local_policy) {}

  constexpr std::size_t rank() const noexcept {
    return rank_;
  }

  constexpr std::size_t stride() const noexcept {
    return bcl::block_stride;
  }

  constexpr std::size_t stride(std::size_t count) const noexcept {
    return count;
  }

  constexpr BCL::RangeTeam team() const noexcept {
    return BCL::RangeTeam(rank_, rank_+1);
  }

  constexpr LocalExecutionPolicy local_policy() const noexcept {
    return local_policy_;
  }

private:
  std::size_t rank_;
  LocalExecutionPolicy local_policy_;
};

template <typename TeamType = BCL::WorldTeam,
          typename LocalExecutionPolicy = std::execution::parallel_unsequenced_policy>
class parallel_policy {
public:
  constexpr parallel_policy(std::size_t stride = bcl::block_stride,
                            TeamType&& team = TeamType(),
                            LocalExecutionPolicy local_policy = std::execution::par_unseq)
    : stride_(stride), team_(team), local_policy_(local_policy) {}

  constexpr std::size_t stride() const noexcept {
    return stride_;
  }

  std::size_t stride(std::size_t count) const noexcept {
    if (stride() == bcl::block_stride) {
      return (count + team().nprocs() - 1) / team().nprocs();
    } else {
      return stride();
    }
  }

  constexpr const TeamType& team() const noexcept {
    return team_;
  }

  constexpr LocalExecutionPolicy local_policy() const noexcept {
    return local_policy_;
  }

private:
  std::size_t stride_;
  TeamType team_;
  LocalExecutionPolicy local_policy_;
};

template <typename LocalExecutionPolicy = std::execution::parallel_unsequenced_policy>
class parallel_local_policy {
public:
  constexpr parallel_local_policy(LocalExecutionPolicy local_policy = std::execution::par_unseq)
    : local_policy_(local_policy) {}

  constexpr LocalExecutionPolicy local_policy() const noexcept {
    return local_policy_;
  }
private:
  LocalExecutionPolicy local_policy_;
};

inline constexpr sequential_policy seq = sequential_policy();
inline constexpr parallel_policy par = parallel_policy();
inline constexpr parallel_local_policy par_local = parallel_local_policy();

}
}
