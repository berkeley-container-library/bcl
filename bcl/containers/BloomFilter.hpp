#pragma once

#include <cstdlib>
#include <cstdio>
#include <cmath>

#include <set>

#include <bcl/bcl.hpp>

namespace BCL {
  template <typename T>
  struct BloomFilter {
    std::vector <BCL::GlobalPtr <uint64_t>> data;
    size_t num_buckets;
    size_t local_size;

    // "Number of hash functions"
    size_t k = 0;

    std::hash <T> hash_fn;

    // TODO: Better hash function
    uint64_t hash(uint64_t key) {
      key = (~key) + (key << 21); // key = (key << 21) - key - 1;
      key = key ^ (key >> 24);
      key = (key + (key << 3)) + (key << 8); // key * 265
      key = key ^ (key >> 14);
      key = (key + (key << 2)) + (key << 4); // key * 21
      key = key ^ (key >> 28);
      key = key + (key << 31);
      return key;
    }

    BloomFilter() {}

    // n - expected size of data
    // p - desired false positive rate
    BloomFilter(size_t n, double p = 0.01) {
      double bits_per_elem = -log2(p) / log(2);
      double elems_per_bucket = (8*sizeof(uint64_t)) / bits_per_elem;
      elems_per_bucket /= -log(p);
      num_buckets = (n / elems_per_bucket) + 0.5;

      k = (- (log(p) / log(2))) + 0.5;

      if (k < 1) {
        k = 1;
      }

      /*
      BCL::print("%d uniue elements, %lf bits per elem, %lf elems per 64-bit bucket, %d buckets, %d bits total, %lf (%d) hash fns\n",
        n, bits_per_elem, elems_per_bucket, num_buckets, num_buckets*8*sizeof(uint64_t), (-(log(p) / log(2))), k);
      */

      local_size = (num_buckets + BCL::nprocs() - 1) / BCL::nprocs();

      num_buckets = local_size * BCL::nprocs();

      /*
      BCL::print("local buckets %d, num_buckets now %d\n", local_size, num_buckets);
      */

      data.resize(BCL::nprocs(), nullptr);
      for (int rank = 0; rank < BCL::nprocs(); rank++) {
        if (BCL::rank() == rank) {
          data[rank] = BCL::alloc <uint64_t> (local_size);

          for (int i = 0; i < local_size; i++) {
            data[rank].local()[i] = 0x0;
          }
        }
        data[rank] = BCL::broadcast(data[rank], rank);
        if (data[rank] == nullptr) {
          throw std::runtime_error("BloomFilter: ran out of space.");
        }
      }
    }

    ~BloomFilter() {
      if (!BCL::bcl_finalized) {
        if (BCL::rank() < data.size() && data[BCL::rank()] != nullptr) {
          BCL::dealloc(data[BCL::rank()]);
        }
      }
    }

    bool insert(const T &val) {
      size_t my_hash = hash(hash_fn(val));
      size_t my_filter = my_hash % num_buckets;

      uint64_t my_bits = 0x0;

      for (int i = 0; i < k; i++) {
        my_hash = hash(my_hash);
        int my_bit = my_hash % (8*sizeof(uint64_t));
        my_bits |= 0x1 << my_bit;
      }

      size_t node = my_filter / local_size;
      size_t node_slot = my_filter - node*local_size;

      uint64_t old_filter = BCL::uint64_atomic_fetch_or(data[node] + node_slot, my_bits);

      return ((old_filter & my_bits) == my_bits);
    }

    bool find(const T &val) {
      size_t my_hash = hash(hash_fn(val));
      size_t my_filter = my_hash % num_buckets;

      uint64_t my_bits = 0x0;

      for (int i = 0; i < k; i++) {
        my_hash = hash(my_hash);
        int my_bit = my_hash % (8*sizeof(uint64_t));
        my_bits |= 0x1 << my_bit;
      }

      size_t node = my_filter / local_size;
      size_t node_slot = my_filter - node*local_size;

      uint64_t filter = BCL::rget(data[node] + node_slot);

      return ((filter & my_bits) == my_bits);
    }

    BloomFilter(const BloomFilter &bloom_filter) = delete;

    operator=(BloomFilter &&bloom_filter) {
      this->data = std::move(bloom_filter.data);
      this->num_buckets = bloom_filter.num_buckets;
      this->local_size = bloom_filter.local_size;
      this->k = bloom_filter.k;
    }

    BloomFilter(BloomFilter &&bloom_filter) {
      this->data = std::move(bloom_filter.data);
      this->num_buckets = bloom_filter.num_buckets;
      this->local_size = bloom_filter.local_size;
      this->k = bloom_filter.k;
    }
  };
}
