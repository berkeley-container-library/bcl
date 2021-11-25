#include <bcl/bcl.hpp>
#include "remote_span.hpp"
#include "execution_policy.hpp"
#include "algorithm.hpp"
#include <iostream>

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