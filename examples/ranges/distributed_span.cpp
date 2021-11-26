#include <bcl/bcl.hpp>
#include "remote_span.hpp"
#include "execution_policy.hpp"
#include "algorithm.hpp"
#include "distributed_span.hpp"
#include "iterator_adaptor.hpp"
#include <iostream>

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

  for_each(bcl::execution::parallel_policy(1),
           dspan, [](auto f) { printf("(%lu) Hello, fam! got %d\n", BCL::rank(), (int) f); });

  if (BCL::rank() == 0) {
    auto begin = dspan.begin();
    auto end = dspan.end();

    bool nequal = begin != end;
  }

  BCL::finalize();
  return 0;
}