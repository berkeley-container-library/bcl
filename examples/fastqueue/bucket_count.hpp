#pragma once

#include <cassert>
#include <vector>
#include <bcl/bcl.hpp>

template <typename Range>
std::vector<typename Range::value_type> bucket_count(const Range& data, size_t bucket_width, size_t bucket_start) {
  using T = typename Range::value_type;
  std::vector<T> counts(bucket_width, 0);
  for (auto it = data.begin().local(); it != data.end().local(); it++) {
    const auto val = *it;
    // assert(val >= bucket_start);
    // assert(val < bucket_start+bucket_width);
    counts[val - bucket_start] += 1;
  }
  return counts;
}

template <typename T>
size_t total_count(std::vector<T>& vals) {
  size_t count = 0;
  for (const auto& val : vals) {
    count += val;
  }

  size_t total_count = BCL::allreduce<size_t>(count, std::plus<size_t>{});

  return total_count;
}
