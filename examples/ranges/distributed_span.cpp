#include <bcl/bcl.hpp>
#include "remote_span.hpp"
#include "execution_policy.hpp"
#include "algorithm.hpp"
#include "distributed_span.hpp"
#include "iterator_adaptor.hpp"
#include <iostream>
#include <fmt/core.h>

template <typename T, std::size_t Extent = std::dynamic_extent>
remote_span<T, Extent> create_span_coll(std::size_t size, std::size_t rank) {
  BCL::GlobalPtr<T> ptr = nullptr;
  if (BCL::rank() == rank) {
    ptr = BCL::alloc<T>(size);

    for (size_t i = 0; i < size; i++) {
      ptr[i] = rank;
    }
  }
  ptr = BCL::broadcast(ptr, rank);

  return remote_span<T, Extent>(ptr, size);
}

int main(int argc, char** argv) {
  BCL::init();

  std::size_t size = 5;
  std::vector<remote_span<int>> spans;

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    spans.push_back(create_span_coll<int>(size, i));
  }

  distributed_span<int> dspan(spans);

  if (BCL::rank() == 0) {
    std::vector<int> v{1, 5, 65, 23, 14, 2, 12, 3, 5, 10, 12, 13};
    const std::vector<int>& cv = v;

    auto f = [](auto v) { return v*2; };
    auto f_inverse = [](auto v) { return v / 2; };

    filter_view view(zip_view(iota_view<std::size_t>(), transform_view(v, f, f_inverse)),
                     [](auto tup) { auto&& [idx, v] = tup; return idx < 5; });

    for (auto&& tup : view) {
      auto&& [idx, v] = tup;
      fmt::print("{}, {}\n", idx, v);
    }
  }

  BCL::finalize();
  return 0;
}
