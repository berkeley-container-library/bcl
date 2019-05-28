#include <bcl/bcl.hpp>
#include <bcl/containers/HashMap.hpp>
#include <bcl/core/detail/hash_functions.hpp>

#include <bcl/containers/experimental/rpc.hpp>

using hash = std::hash<int>;

BCL::HashMap<int, int, hash> *map_ptr;

template <typename Future>
bool ready(std::vector<Future>& futures) {
  for (auto& future : futures) {
    if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      return false;
    }
  }
  return true;
}

template <typename Future>
size_t count_ready(std::vector<Future>& futures) {
  size_t num_ready = 0;
  for (auto& future : futures) {
    if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
    } else {
      num_ready++;
    }
  }
  return num_ready;
}

template <typename Future>
void future_barrier(std::vector<Future>& futures) {
  bool success = false;
  do {
    BCL::flush_rpc();
    size_t success_count = ready(futures);
    success_count = BCL::allreduce<size_t>(success_count, std::plus<size_t>{});
    success = success_count == BCL::nprocs();
  } while (!success);
}

template <typename Key, typename T, typename Hash>
auto buffered_write(Key key, T value) {
  BCL::HashMap<Key, T, Hash>& map = *map_ptr;
  size_t hash = map.hash_fn_(key);
  size_t probe = 0;
  size_t slot = (hash + map.get_probe(probe++)) % map.capacity_;

  size_t rank = map.slot_ptr(slot).rank;

  return BCL::buffered_rpc(rank, [](Key key, T value) {
    BCL::HashMap<Key, T, Hash>& map = *map_ptr;
    return map.insert_or_assign(key, value);
  }, key, value);
}

template <typename Key, typename T, typename Hash>
auto buffered_read(Key key) {
  BCL::HashMap<Key, T, Hash>& map = *map_ptr;
  size_t hash = map.hash_fn_(key);
  size_t probe = 0;
  size_t slot = (hash + map.get_probe(probe++)) % map.capacity_;

  size_t rank = map.slot_ptr(slot).rank;

  return BCL::buffered_rpc(rank, [](Key key) {
    BCL::HashMap<Key, T, Hash>& map = *map_ptr;
    T val;
    bool success = map.find_nonatomic_impl_(key, val);
    return std::make_pair(val, success);
  }, key);
}

int main(int argc, char** argv) {
  BCL::init(256, true);

  BCL::print("init...\n");

  BCL::init_rpc();

  size_t n_to_insert = 100;
  size_t map_size = n_to_insert*BCL::nprocs()*2;
  BCL::print("Each process will insert %lu elements (%lu total).\n", n_to_insert, n_to_insert*BCL::nprocs());
  BCL::print("Making hash map of size %lu\n", map_size);

  map_ptr = new BCL::HashMap<int, int, hash>(map_size);

  auto& map = *map_ptr;

  for (size_t i = BCL::rank()*n_to_insert; i < (BCL::rank()+1)*n_to_insert; i++) {
    auto rv = map.insert_or_assign((int) i, (int) BCL::rank());
    if (!rv.second) {
      fprintf(stderr, "%lu could not insert %lu\n", BCL::rank(), i);
    }
    assert(rv.second);
  }

  BCL::print("Done inserting...\n");

  BCL::barrier();

  using future_type = decltype(buffered_read<int, int, hash>(int()));

  constexpr bool buffered = false;
  BCL::barrier();
  auto begin = std::chrono::high_resolution_clock::now();

  if (buffered) {
    std::vector<future_type> futures;

    for (size_t i = 0; i < BCL::nprocs()*n_to_insert; i++) {
      futures.emplace_back(buffered_read<int, int, hash>(i));
    }

    BCL::print("Flushing RPCs...\n");

    future_barrier(futures);
  } else {
    for (size_t i = 0; i < BCL::nprocs()*n_to_insert; i++) {
      int val;
      bool success = map.find_nonatomic_impl_(i, val);
      assert(success);
    }
  }

  auto end = std::chrono::high_resolution_clock::now();

  double duration = std::chrono::duration<double>(end - begin).count();

  BCL::print("Aha! done in %lf\n", duration);

  BCL::finalize_rpc();
  BCL::finalize();
  return 0;
}
