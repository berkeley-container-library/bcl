#pragma once

#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"

struct HashMap {
  std::vector<upcxx::global_ptr<kmer_pair>> data;
  std::vector<upcxx::global_ptr<int>> used;

  size_t my_size;
  size_t total_size;

  size_t size() const noexcept;

  HashMap(size_t size);
  ~HashMap();

  // Most important functions: insert and retrieve
  // k-mers from the hash table.
  bool insert(const kmer_pair &kmer);
  bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

  // Helper functions

  // Write and read to a logical data slot in the table.
  void write_slot(uint64_t slot, const kmer_pair &kmer);
  kmer_pair read_slot(uint64_t slot);

  // Request a slot or check if it's already used.
  bool request_slot(uint64_t slot);
  bool slot_used(uint64_t slot);
};

HashMap::HashMap(size_t size) {
  my_size = (size + upcxx::rank_n() - 1) / upcxx::rank_n();
  total_size = my_size * upcxx::rank_n();
  data.resize(upcxx::rank_n(), nullptr);
  used.resize(upcxx::rank_n(), nullptr);
  data[upcxx::rank_me()] = upcxx::new_array<kmer_pair>(my_size);
  used[upcxx::rank_me()] = upcxx::new_array<int>(my_size);
  for (size_t i = 0; i < upcxx::rank_n(); i++) {
    data[i] = upcxx::broadcast(data[i], i).wait();
    used[i] = upcxx::broadcast(used[i], i).wait();
  }
}

HashMap::~HashMap() {
  upcxx::delete_array(data[upcxx::rank_me()]);
  upcxx::delete_array(used[upcxx::rank_me()]);
}

bool HashMap::insert(const kmer_pair &kmer) {
  uint64_t hash = kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % total_size;
    if ((success = request_slot(slot))) {
      write_slot(slot, kmer);
    }
  } while (!success && probe < total_size);
  return success;
}

bool HashMap::find(const pkmer_t &key_kmer, kmer_pair &val_kmer) {
  uint64_t hash = key_kmer.hash();
  uint64_t probe = 0;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % total_size;
    if (slot_used(slot)) {
      val_kmer = read_slot(slot);
      if (val_kmer.kmer == key_kmer) {
        success = true;
      }
    }
  } while (!success && probe < total_size);
  if (!success) {
    uint64_t slot = (hash + probe - 1) % total_size;
    printf("total size: %d, local size: %d, probe: %d\n", total_size, size(), probe);
    printf("global slot: %d, slot rank: %d, local slot: %d\n",
           slot, slot / size(), slot % size());
  }
  return success;
}

bool HashMap::slot_used(uint64_t slot) {
  int rank = slot / size();
  int offset = slot % size();
  upcxx::atomic_domain<int> ld({upcxx::atomic_op::load});
  bool slot_used = ld.load(used[rank] + offset, std::memory_order_relaxed).wait();
  ld.destroy(upcxx::entry_barrier::none);
  return slot_used;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair &kmer) {
  int rank = slot / size();
  int offset = slot % size();
  upcxx::rput(kmer, data[rank] + offset).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) {
  int rank = slot / size();
  int offset = slot % size();
  return upcxx::rget(data[rank] + offset).wait();
}

bool HashMap::request_slot(uint64_t slot) {
  int rank = slot / size();
  int offset = slot % size();
  upcxx::atomic_domain<int> cx({upcxx::atomic_op::compare_exchange});
  // atomic CAS
  bool origin = cx.compare_exchange(used[rank] + offset, 0, 1, std::memory_order_relaxed).wait();
  // seems to have to destroy like this
  cx.destroy(upcxx::entry_barrier::none);
  return !origin;
}

size_t HashMap::size() const noexcept {
  return my_size;
}