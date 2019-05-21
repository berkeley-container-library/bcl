#pragma once

#include <cstdlib>
#include <cstdio>
#include <vector>

#include <bcl/bcl.hpp>
#include <bcl/containers/FastQueue.hpp>

namespace BCL {

template <
  typename Key,
  typename T,
  typename Hash = std::hash<Key>,
  typename KeySerialize = BCL::serialize <Key>,
  typename ValSerialize = BCL::serialize <T>
>
struct HashMapBuffer {
  using hashmap_type = BCL::HashMap<Key, T, Hash, KeySerialize, ValSerialize>;
  using HME = typename hashmap_type::HME;

  using key_type = Key;
  using mapped_type = T;
  using value_type = std::pair<const Key, T>;

  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  hashmap_type *hashmap;

  std::vector <BCL::FastQueue <HME>> queues;
  std::vector <std::vector <HME>> buffers;
  std::vector <BCL::future <std::vector <HME>>> futures;

  HashMapBuffer(const HashMapBuffer&) = delete;
  HashMapBuffer& operator=(const HashMapBuffer&) = delete;

  HashMapBuffer(HashMapBuffer&&) = default;
  HashMapBuffer& operator=(HashMapBuffer&&) = default;

  HashMapBuffer() = delete;
  ~HashMapBuffer() = default;

  size_t buffer_size;

  HashMapBuffer(hashmap_type& hashmap, size_t queue_capacity,
                size_t buffer_size) {
    this->hashmap = &hashmap;

    this->buffer_size = buffer_size;

    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
      queues.emplace_back(rank, queue_capacity);
    }

    for (size_t rank = 0; rank < BCL::nprocs(); rank++) {
      buffers.emplace_back();
    }

    for (auto& buffer : buffers) {
      buffer.reserve(buffer_size);
    }
  }

  bool insert(const Key& key, const T& val) {
    size_t hash = hashmap->hash_fn_(key);
    size_t slot = hash % hashmap->capacity();
    size_t node = slot / hashmap->local_capacity();
    buffers[node].push_back(HME(key, val));

    if (buffers[node].size() >= buffer_size) {
      auto future = queues[node].push(std::move(buffers[node]));
      if (bool(future)) {
        futures.emplace_back(std::move(future.value()));
        buffers[node].reserve(buffer_size);
        return true;
      } else {
        return false;
      }
    } else {
      return true;
    }
  }

  bool flush() {
    bool full_queues = false;
    bool success = true;
    do {
      // Flush local buffers to remote queues.
      // Only returns false if remote queues
      // are full.
      full_queues = !flush_buffers();

      for (auto& future : futures) {
        future.get();
      }
      futures.clear();
      BCL::barrier();
      // Flush local queues to hash table.
      // Only returns false if hash table is full.
      success = flush_queues();
    } while (success && full_queues);
    return success;
  }

  // Flush HMEs sitting in local queues
  bool flush_queues() {
    std::vector <HME> failed_inserts;
    int num = 0;
    bool success;
    HME entry;
    do {
      num++;
      success = queues[BCL::rank()].local_nonatomic_pop(entry);
      if (success) {
        bool inserted = hashmap->local_nonatomic_insert(entry);
        if (!inserted) {
          failed_inserts.push_back(entry);
        }
      }
    } while (success);
    BCL::barrier();

    success = true;
    for (HME &entry : failed_inserts) {
      if (!hashmap->insert_atomic_impl_(entry.get_key(), entry.get_val())) {
        success = false;
        break;
      }
    }

    int success_ = (success) ? 0 : 1;
    success_ = BCL::allreduce(success_, std::plus <int> ());

    return (success_ == 0);
  }

  // Flush local HME buffers to remote queues
  bool flush_buffers() {
    bool success = true;
    for (int rank = 0; rank < buffers.size(); rank++) {
      auto future = queues[rank].push(std::move(buffers[rank]));
      if (bool(future)) {
        futures.emplace_back(std::move(future.value()));
      } else {
        success = false;
      }
    }

    int success_ = (success) ? 0 : 1;
    success_ = BCL::allreduce(success_, std::plus <int> ());

    return (success_ == 0);
  }
};

} // end BCL
