#pragma once

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
