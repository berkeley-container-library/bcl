#include <bcl/bcl.hpp>
#include <iostream>
#include <limits>
#include <execution>
#include <bcl/core/teams.hpp>
#include <ranges>

#ifndef CPP20
namespace std {
inline constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

struct identity {
  template <typename T>
  constexpr T&& operator()( T&& t ) const noexcept { std::forward<T>(t); }
};
}
#else
#include <span>
#include <functional>
#endif

template <typename T,
          std::size_t Extent = std::dynamic_extent>
struct remote_span_storage_impl_ {
  BCL::GlobalPtr<T> ptr_;
  remote_span_storage_impl_(BCL::GlobalPtr<T> ptr, std::size_t size) : ptr_(ptr) {}
  constexpr std::size_t size() const noexcept {
    return Extent;
  }
};

template <typename T>
struct remote_span_storage_impl_<T, std::dynamic_extent> {
  BCL::GlobalPtr<T> ptr_;
  std::size_t size_;

  remote_span_storage_impl_(BCL::GlobalPtr<T> ptr, std::size_t size) : ptr_(ptr), size_(size) {}

  constexpr std::size_t size() const noexcept {
    return size_;
  }
};

template <typename T,
          std::size_t Extent = std::dynamic_extent>
class remote_span {
public:
  using element_type = T;
  using value_type = std::remove_cv_t<T>;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  using pointer = BCL::GlobalPtr<T>;
  using const_pointer = BCL::GlobalPtr<const T>;

  using reference = BCL::GlobalRef<T>;
  using const_reference = BCL::GlobalRef<const T>;

  using iterator = pointer;

  constexpr remote_span(BCL::GlobalPtr<T> first, size_type count) : storage_(first, count) {}

  constexpr remote_span(BCL::GlobalPtr<T> first, BCL::GlobalPtr<T> last)
    : storage_(first, last - first) {}

  constexpr size_type size() const noexcept {
    return storage_.size();
  }

  constexpr size_type size_bytes() const noexcept {
    return size()*sizeof(element_type);
  }

  constexpr pointer data() const noexcept {
    return storage_.ptr_;
  }

  constexpr iterator begin() const noexcept {
    return storage_.ptr_;
  }

  constexpr iterator end() const noexcept {
    return data() + size();
  }

  T* local_begin() const noexcept {
    if (BCL::rank() == data().rank) {
      return begin().local();
    } else {
      return nullptr;
    }
  }

  T* local_end() const noexcept {
    if (BCL::rank() == data().rank) {
      return end().local();
    } else {
      return nullptr;
    }
  }

  constexpr reference front() const {
    return data();
  }

  constexpr reference back() const {
    return data() + (size() - 1);
  }

  constexpr reference operator[](size_type idx) const {
    return data()[idx];
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return size() == 0;
  }

  template <std::size_t Count>
  constexpr remote_span<element_type, Count> first() const {
    return remote_span<element_type, Count>(data(), Count);
  }
      
  constexpr remote_span<element_type, std::dynamic_extent> first(size_type Count) const {
    return remote_span<element_type, std::dynamic_extent>(data(), Count);
  }

  template <std::size_t Count>
  constexpr remote_span<element_type, Count> last() const {
    return remote_span<element_type, Count>(data() + size() - Count, Count);
  }
      
  constexpr remote_span<element_type, std::dynamic_extent> last( size_type Count ) const {
    return remote_span<element_type, std::dynamic_extent>(data() + size() - Count, Count);
  }

  template<std::size_t Offset,
           std::size_t Count = std::dynamic_extent>
  constexpr auto subspan() const {
    if constexpr(Count != std::dynamic_extent) {
      return remote_span<element_type, Count>(data() + Offset, Count);
    } else if constexpr(Extent != std::dynamic_extent) {
      return remote_span<element_type, Extent - Offset>(data() + Offset, Count);
    } else {
      return remote_span<element_type, std::dynamic_extent>(data() + Offset, size() - Offset);
    }
  }

  constexpr remote_span<element_type, std::dynamic_extent>
  subspan(size_type Offset, size_type Count = std::dynamic_extent) const {
    if (Count == std::dynamic_extent) {
      return remote_span<element_type, std::dynamic_extent>(data() + Offset, size() - Offset);
    } else {
      return remote_span<element_type, std::dynamic_extent>(data() + Offset, Count);
    }
  }

private:

  remote_span_storage_impl_<T, Extent> storage_;
};

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

// Limited to random access ranges
// Not locality-aware
template <typename ExecutionPolicy, typename R, typename Fun, typename Proj = std::identity>
void for_each(ExecutionPolicy policy, R&& r, Fun f, Proj proj = {}) {
  if constexpr(!std::is_same_v<ExecutionPolicy, bcl::execution::parallel_local_policy<>>) {
    if (policy.team().in_team()) {
      std::size_t stride = policy.stride(r.size());
      std::size_t chunk_size = stride*policy.team().nprocs();
      std::size_t num_chunks = (r.size() + chunk_size - 1) / chunk_size;

      for (size_t chunk = 0; chunk < num_chunks; chunk++) {
        std::size_t offset = chunk*chunk_size + policy.team().rank()*stride;
        std::size_t size = std::min(stride, r.size() - offset);
        // XXX: apparent bug in STL impl. when:
        //  1) passing policy as rvalue reference
        //  2) using random_access_iterator, but not continguous_iterator
        // Copying to new local variable as workaround.
        auto local_policy = policy.local_policy();
        std::for_each(local_policy, r.begin() + offset, r.begin() + offset + size, f);
      }
    }
  } else {
    auto local_policy = policy.local_policy();
    std::for_each(policy.local_policy(), r.local_begin(), r.local_end(), f);
  }
}

template <std::ranges::range R, typename Fun, typename Proj = std::identity>
void for_each(R&& r, Fun f, Proj proj = {}) {
  for_each(bcl::execution::par_local, std::forward<R>(r), f, proj);
}

int main(int argc, char** argv) {
  BCL::init();

  constexpr std::size_t size_per_proc = 10;

  BCL::GlobalPtr<int> data = nullptr;

  srand48(BCL::rank());

  if (BCL::rank() == 0) {
    data = BCL::alloc<int>(size_per_proc*BCL::nprocs());
  }

  data = BCL::broadcast(data, 0);

  remote_span<int> span(data, size_per_proc*BCL::nprocs());

  printf("Span is size %lu\n", span.size());

  auto my_span = span.subspan(size_per_proc*BCL::rank(), size_per_proc);

  printf("My Span is size %lu\n", span.size());

  std::transform(my_span.begin(), my_span.end(), my_span.begin(),
                 [](auto ref) { return lrand48() % 100; });

  BCL::barrier();

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < span.size(); i++) {
      int v = span[i];
      std::cout << "(" << v << ")";
    }
    std::cout << std::endl;
  }

  if (BCL::rank() == 0) {
    std::sort(span.begin(), span.end());
  }

  if (BCL::rank() == 0) {
    for (size_t i = 0; i < span.size(); i++) {
      int v = span[i];
      std::cout << "(" << v << ")";
    }
    std::cout << std::endl;
  }

  BCL::barrier();
  fflush(stdout);
  fflush(stderr);
  usleep(100);
  BCL::barrier();

  for_each(bcl::execution::parallel_policy(1),
           span, [](auto f) { printf("(%lu) Hello, fam! got %d\n", BCL::rank(), (int) f); });

/*
  BCL::barrier();
  fflush(stdout);
  fflush(stderr);
  usleep(100);
  BCL::barrier();

  for_each(bcl::execution::par_local,
           span, [](auto f) { printf("(%lu) Hello, fam! got %d\n", BCL::rank(), (int) f); });
           */

  BCL::finalize();
  return 0;
}