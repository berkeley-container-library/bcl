#include <bcl/bcl.hpp>
#include <cassert>
#include <numeric>

int main(int argc, char** argv) {
  BCL::init();

  size_t vec_size = 1000;

  std::vector<BCL::GlobalPtr<int>> ptrs(BCL::nprocs());

  ptrs[BCL::rank()] = BCL::alloc<int>(vec_size);

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    ptrs[i] = BCL::broadcast(ptrs[i], i);
  }

  for (size_t i_ = 0; i_ < BCL::nprocs(); i_++) {
    size_t i = (i_ + BCL::rank()) % BCL::nprocs();

    std::vector<int> vec(vec_size);

    std::iota(vec.begin(), vec.end(), i_);

    auto fut = arput(ptrs[i], std::move(vec));

    vec = fut.get();

    BCL::barrier();

    assert(vec.size() == vec_size);

    for (size_t j = 0; j < vec_size; j++) {
      assert(ptrs[BCL::rank()].local()[j] == vec[j]);
    }

    BCL::barrier();
  }

  BCL::finalize();
  return 0;
}
