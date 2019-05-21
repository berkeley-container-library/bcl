#include <bcl/bcl.hpp>
#include <cassert>

int main(int argc, char** argv) {
  BCL::init();

  size_t vec_size = 1000;

  std::vector<BCL::GlobalPtr<int>> ptrs(BCL::nprocs());

  ptrs[BCL::rank()] = BCL::alloc<int>(vec_size);

  for (size_t i = 0; i < vec_size; i++) {
    ptrs[BCL::rank()].local()[i] = i;
  }

  for (size_t i = 0; i < BCL::nprocs(); i++) {
    ptrs[i] = BCL::broadcast(ptrs[i], i);
  }

  for (size_t i_ = 0; i_ < BCL::nprocs(); i_++) {
    size_t i = (i_ + BCL::rank()) % BCL::nprocs();

    auto f = BCL::arget(ptrs[i], vec_size);

    auto vec = f.get();

    assert(vec.size() == vec_size);

    for (size_t j = 0; j < vec.size(); j++) {
      assert(vec[j] == j);
    }

    for (size_t j = 0; j < vec_size; j++) {
      auto f = BCL::arget(&ptrs[i][j]);
      assert(f.get() == j);
    }
  }

  BCL::finalize();
  return 0;
}
